#!/usr/bin/env python3
"""Model-level flex_attention timing test.

Patches all attention layers in Qwen2.5-7B to use flex_attention,
then measures full model prefill time for full vs ddi_qf masks.
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"

import types
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from transformers import AutoModelForCausalLM
from ddi.masks import build_mask, to_additive

MODEL = "results/checkpoints/Qwen__Qwen2.5-7B-Instruct__ddi_qf__8L"
CELLS = [(3, 512), (9, 512), (25, 512), (25, 2048)]
Q = 64
N_REPEATS = 5
WARMUP = 3

_flex_compiled = None


def get_flex_compiled():
    global _flex_compiled
    if _flex_compiled is None:
        _flex_compiled = torch.compile(flex_attention)
    return _flex_compiled


def make_ddi_block_mask(doc_lengths, query_len, device="cuda"):
    N = len(doc_lengths)
    doc_ends = torch.cumsum(torch.tensor(doc_lengths, device=device, dtype=torch.long), dim=0)
    total_doc = int(doc_ends[-1].item())
    T = total_doc + query_len

    doc_id = torch.empty(T, dtype=torch.long, device=device)
    for i in range(N):
        start = int(doc_ends[i - 1].item()) if i > 0 else 0
        end = int(doc_ends[i].item())
        doc_id[start:end] = i
    doc_id[total_doc:] = N

    def mask_mod(b, h, q_idx, kv_idx):
        q_doc = doc_id[q_idx]
        k_doc = doc_id[kv_idx]
        q_is_query = q_doc == N
        k_is_query = k_doc == N
        same_doc = (q_doc == k_doc) & ~q_is_query & ~k_is_query
        q_sees_k = q_is_query & ~k_is_query
        q_query_causal = q_is_query & k_is_query & (q_idx >= kv_idx)
        return same_doc | q_sees_k | q_query_causal

    return create_block_mask(mask_mod, B=1, H=None, Q_LEN=T, KV_LEN=T, device=device)


def make_full_causal_block_mask(T, device="cuda"):
    def mask_mod(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx
    return create_block_mask(mask_mod, B=1, H=None, Q_LEN=T, KV_LEN=T, device=device)


def patch_attention_layers(model, block_mask):
    """Replace each Qwen2Attention.forward with a flex_attention version."""
    flex_fn = get_flex_compiled()

    for layer in model.model.layers:
        attn = layer.self_attn
        original_forward = attn.forward

        def make_flex_forward(attn_module, orig_fwd):
            def flex_forward(hidden_states, position_embeddings, attention_mask=None,
                             past_key_value=None, cache_position=None, **kwargs):
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, attn_module.head_dim)

                query_states = attn_module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                key_states = attn_module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                value_states = attn_module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                cos, sin = position_embeddings
                from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

                if past_key_value is not None:
                    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                    key_states, value_states = past_key_value.update(
                        key_states, value_states, attn_module.layer_idx, cache_kwargs
                    )

                # flex_attention: [B, H, T, D] — Q has all heads, K/V have kv_heads (GQA)
                num_kv_groups = attn_module.config.num_attention_heads // attn_module.config.num_key_value_heads
                if num_kv_groups > 1:
                    key_states = key_states.repeat_interleave(num_kv_groups, dim=1)
                    value_states = value_states.repeat_interleave(num_kv_groups, dim=1)

                attn_output = flex_fn(query_states, key_states, value_states, block_mask=block_mask)
                attn_output = attn_output.transpose(1, 2).reshape(hidden_states.shape)
                attn_output = attn_module.o_proj(attn_output)

                return attn_output, None

            return flex_forward

        attn.forward = make_flex_forward(attn, original_forward)


def measure_model_prefill(model, input_ids, n_repeats, warmup):
    times = []
    for i in range(warmup + n_repeats):
        torch.cuda.empty_cache()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            model(input_ids=input_ids, use_cache=False)
        end.record()
        end.synchronize()
        if i >= warmup:
            times.append(start.elapsed_time(end))
    times.sort()
    return sum(times) / len(times)


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"torch: {torch.__version__}")
    print(f"Model: {MODEL}")

    results = []

    for N, L in CELLS:
        T = N * L + Q
        doc_lengths = [L] * N
        print(f"\n=== N={N} L={L} T={T} ===")

        # --- sdpa baseline (full) ---
        print(f"  loading model (sdpa)...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to("cuda").eval()

        input_ids = torch.randint(0, 152064, (1, T), device="cuda")

        # sdpa full (no mask = causal by default)
        sdpa_full_ms = measure_model_prefill(model, input_ids, N_REPEATS, WARMUP)
        print(f"  sdpa full = {sdpa_full_ms:.2f}ms")

        # sdpa ddi_qf (4D additive mask)
        allow = build_mask(doc_lengths, query_len=Q, answer_len=0, variant="ddi_qf")
        additive = to_additive(allow, dtype=torch.bfloat16).to("cuda").unsqueeze(0).unsqueeze(0)
        def _sdpa_ddi():
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=additive, use_cache=False)
        times = []
        for i in range(WARMUP + N_REPEATS):
            torch.cuda.empty_cache()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(); _sdpa_ddi(); e.record(); e.synchronize()
            if i >= WARMUP: times.append(s.elapsed_time(e))
        sdpa_ddi_ms = sum(sorted(times)) / len(times)
        print(f"  sdpa ddi_qf = {sdpa_ddi_ms:.2f}ms")

        del model
        torch.cuda.empty_cache()

        # --- flex_attention ---
        print(f"  loading model (flex)...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.bfloat16, attn_implementation="eager"
        ).to("cuda").eval()

        # flex full causal
        print(f"  creating full causal block mask...")
        bm_full = make_full_causal_block_mask(T, "cuda")
        patch_attention_layers(model, bm_full)
        print(f"  warming up flex full...")
        with torch.no_grad():
            model(input_ids=input_ids, use_cache=False)
        torch.cuda.synchronize()
        flex_full_ms = measure_model_prefill(model, input_ids, N_REPEATS, WARMUP)
        print(f"  flex full = {flex_full_ms:.2f}ms")

        # flex ddi_qf
        print(f"  creating ddi_qf block mask...")
        bm_ddi = make_ddi_block_mask(doc_lengths, Q, "cuda")
        # Re-patch with new block mask
        model2 = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.bfloat16, attn_implementation="eager"
        ).to("cuda").eval()
        patch_attention_layers(model2, bm_ddi)
        print(f"  warming up flex ddi_qf...")
        with torch.no_grad():
            model2(input_ids=input_ids, use_cache=False)
        torch.cuda.synchronize()
        flex_ddi_ms = measure_model_prefill(model2, input_ids, N_REPEATS, WARMUP)
        print(f"  flex ddi_qf = {flex_ddi_ms:.2f}ms")

        sparsity = 1.0 - allow.float().mean().item()
        sdpa_ratio = sdpa_full_ms / sdpa_ddi_ms if sdpa_ddi_ms > 0 else 0
        flex_ratio = flex_full_ms / flex_ddi_ms if flex_ddi_ms > 0 else 0
        analytical = 1.0 / (1.0 - sparsity) if sparsity > 0 else 1.0

        print(f"\n  SUMMARY N={N} L={L}:")
        print(f"  sdpa: full={sdpa_full_ms:.1f}ms ddi={sdpa_ddi_ms:.1f}ms ratio={sdpa_ratio:.2f}x")
        print(f"  flex: full={flex_full_ms:.1f}ms ddi={flex_ddi_ms:.1f}ms ratio={flex_ratio:.2f}x")
        print(f"  sparsity={sparsity:.3f} analytical={analytical:.1f}x")

        results.append({
            "N": N, "L": L, "T": T, "sparsity": sparsity,
            "sdpa_full": sdpa_full_ms, "sdpa_ddi": sdpa_ddi_ms, "sdpa_ratio": sdpa_ratio,
            "flex_full": flex_full_ms, "flex_ddi": flex_ddi_ms, "flex_ratio": flex_ratio,
            "analytical": analytical,
        })

        del model, model2
        torch.cuda.empty_cache()

    print("\n=== FINAL SUMMARY ===")
    print(f"{'N':>4} {'L':>5} {'T':>6}  {'sdpa':>8} {'flex':>8} {'analyt':>8} {'sparsity':>8}")
    print("-" * 55)
    for r in results:
        print(f"{r['N']:>4} {r['L']:>5} {r['T']:>6}  {r['sdpa_ratio']:>8.2f} {r['flex_ratio']:>8.2f} "
              f"{r['analytical']:>8.1f} {r['sparsity']:>8.3f}")


if __name__ == "__main__":
    main()
