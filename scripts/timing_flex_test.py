#!/usr/bin/env python3
"""Test flex_attention for realizing block-diagonal mask sparsity.

flex_attention (torch 2.5+) compiles a mask_mod function into a block-sparse
attention kernel that SKIPS masked QK^T blocks entirely. This is exactly what
sdpa cannot do — sdpa computes all QK^T then applies the mask.

DDI block-diagonal pattern:
- Each document attends only to itself (block-diagonal)
- Query attends to all documents
- Answer attends to query + all documents (causal within answer)
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from ddi.masks import build_mask, to_additive

MODEL = "results/checkpoints/Qwen__Qwen2.5-7B-Instruct__ddi_qf__8L"
CELLS = [
    (3, 512),
    (9, 512),
    (25, 512),
    (25, 2048),
]
Q = 64
N_REPEATS = 5
WARMUP = 3


def ddi_qf_mask_mod(b, h, q_idx, kv_idx, doc_lengths, query_len):
    """Block-diagonal mask: docs see only own tokens, query sees all docs."""
    # Build cumulative doc boundaries
    cumsum = torch.cumsum(doc_lengths, dim=0)
    T = cumsum[-1] + query_len

    # For each position, determine which "block" it belongs to
    # Doc i: [cumsum[i-1], cumsum[i])
    # Query: [cumsum[-1], cumsum[-1]+query_len)

    # Find which doc a position belongs to (or if it's query)
    def doc_id(pos):
        # Returns doc index or -1 for query
        for i in range(len(cumsum)):
            start = cumsum[i] - doc_lengths[i] if i > 0 else 0
            end = cumsum[i]
            if start <= pos < end:
                return i
        return -1  # query

    # This is slow in Python — need vectorized version
    pass


def make_ddi_block_mask(doc_lengths, query_len, device="cuda"):
    """Create a flex_attention BlockMask for the DDI qf pattern.

    Pattern:
    - Doc i tokens attend only to doc i tokens (block-diagonal)
    - Query tokens attend to ALL doc tokens
    - (No answer tokens in prefill-only timing)
    """
    N = len(doc_lengths)
    doc_ends = torch.cumsum(torch.tensor(doc_lengths, device=device, dtype=torch.long), dim=0)
    total_doc = int(doc_ends[-1].item())
    T = total_doc + query_len

    # Precompute doc_id for each position: doc i -> i, query -> N (special marker)
    doc_id = torch.empty(T, dtype=torch.long, device=device)
    for i in range(N):
        start = int(doc_ends[i - 1].item()) if i > 0 else 0
        end = int(doc_ends[i].item())
        doc_id[start:end] = i
    doc_id[total_doc:] = N  # query positions get doc_id = N

    def mask_mod(b, h, q_idx, kv_idx):
        q_doc = doc_id[q_idx]
        k_doc = doc_id[kv_idx]
        q_is_query = q_doc == N
        k_is_query = k_doc == N

        # Both in docs: same doc only
        same_doc = (q_doc == k_doc) & ~q_is_query & ~k_is_query
        # q in query, k in docs: allowed
        q_sees_k = q_is_query & ~k_is_query
        # Both in query: causal
        q_query_causal = q_is_query & k_is_query & (q_idx >= kv_idx)

        return same_doc | q_sees_k | q_query_causal

    block_mask = create_block_mask(
        mask_mod,
        B=1,
        H=None,
        Q_LEN=T,
        KV_LEN=T,
        device=device,
    )
    return block_mask


def make_full_causal_block_mask(T, device="cuda"):
    """Full causal mask for flex_attention baseline."""
    def mask_mod(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    return create_block_mask(
        mask_mod,
        B=1,
        H=None,
        Q_LEN=T,
        KV_LEN=T,
        device=device,
    )


def measure_prefill_flex(model, input_ids, block_mask, n_repeats, warmup):
    """Measure prefill with flex_attention."""
    times = []
    for i in range(warmup + n_repeats):
        torch.cuda.empty_cache()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            _ = model(input_ids=input_ids, block_mask=block_mask, use_cache=False)
        end.record()
        end.synchronize()
        if i >= warmup:
            times.append(start.elapsed_time(end))
    times.sort()
    return sum(times) / len(times)


def measure_prefill_sdpa(model, input_ids, additive_mask, n_repeats, warmup):
    """Measure prefill with sdpa (standard 4D mask)."""
    from ddi.model import forward_with_mask
    times = []
    for i in range(warmup + n_repeats):
        torch.cuda.empty_cache()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            forward_with_mask(model, input_ids, additive_mask)
        end.record()
        end.synchronize()
        if i >= warmup:
            times.append(start.elapsed_time(end))
    times.sort()
    return sum(times) / len(times)


def patch_model_for_flex(model):
    """Patch model's attention layers to use flex_attention when block_mask is provided."""
    original_forward = model.forward

    def flex_forward(self, input_ids=None, block_mask=None, attention_mask=None, use_cache=False, **kwargs):
        if block_mask is not None:
            # Temporarily patch each attention layer
            return _flex_forward_impl(self, input_ids, block_mask, use_cache, **kwargs)
        return original_forward(input_ids=input_ids, attention_mask=attention_mask, use_cache=use_cache, **kwargs)

    model.forward = flex_forward.__get__(model)
    return model


def _flex_forward_impl(model, input_ids, block_mask, use_cache, **kwargs):
    """Run model forward with flex_attention by patching attention layers."""
    # This is complex — we need to intercept the attention computation
    # For now, let's test at the raw attention level instead
    raise NotImplementedError("Model-level flex requires patching attention layers")


def test_raw_attention_timing(model, N, L):
    """Test flex_attention vs sdpa at the raw attention level.

    Extract one attention layer from the model and time it directly.
    """
    T = N * L + Q

    # Get hidden dim from model config
    hidden = model.config.hidden_size
    n_heads = model.config.num_attention_heads
    head_dim = hidden // n_heads
    dtype = model.dtype
    device = "cuda"

    # Create random Q, K, V
    q = torch.randn(1, n_heads, T, head_dim, dtype=dtype, device=device)
    k = torch.randn(1, n_heads, T, head_dim, dtype=dtype, device=device)
    v = torch.randn(1, n_heads, T, head_dim, dtype=dtype, device=device)

    doc_lengths = [L] * N

    # --- sdpa with full causal mask ---
    def sdpa_full():
        causal = torch.ones(T, T, dtype=torch.bool, device=device).tril()
        causal_4d = causal.unsqueeze(0).unsqueeze(0)
        neg_inf = torch.finfo(dtype).min
        add_mask = torch.where(causal_4d, 0.0, neg_inf).to(dtype)
        with torch.no_grad():
            F.scaled_dot_product_attention(q, k, v, attn_mask=add_mask)

    # --- sdpa with ddi_qf mask ---
    def sdpa_ddi():
        allow = build_mask(doc_lengths, query_len=Q, answer_len=0, variant="ddi_qf")
        add_mask = to_additive(allow, dtype=dtype).to(device).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            F.scaled_dot_product_attention(q, k, v, attn_mask=add_mask)

    # --- flex_attention with full causal ---
    flex_full_compiled = None
    def flex_full():
        nonlocal flex_full_compiled
        if flex_full_compiled is None:
            flex_full_compiled = torch.compile(flex_attention)
        bm = make_full_causal_block_mask(T, device)
        with torch.no_grad():
            flex_full_compiled(q, k, v, block_mask=bm)

    # --- flex_attention with ddi_qf ---
    flex_ddi_compiled = None
    def flex_ddi():
        nonlocal flex_ddi_compiled
        if flex_ddi_compiled is None:
            flex_ddi_compiled = torch.compile(flex_attention)
        bm = make_ddi_block_mask(doc_lengths, Q, device)
        with torch.no_grad():
            flex_ddi_compiled(q, k, v, block_mask=bm)

    def measure(fn, n_repeats=N_REPEATS, warmup=WARMUP):
        times = []
        for i in range(warmup + n_repeats):
            torch.cuda.empty_cache()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            if i >= warmup:
                times.append(start.elapsed_time(end))
        times.sort()
        return sum(times) / len(times)

    print(f"\n  N={N} L={L} T={T} (hidden={hidden}, heads={n_heads}, head_dim={head_dim})")

    # Warmup flex (compilation)
    print(f"  compiling flex_attention...")
    try:
        bm = make_ddi_block_mask(doc_lengths, Q, device)
        flex_compiled = torch.compile(flex_attention)
        with torch.no_grad():
            flex_compiled(q, k, v, block_mask=bm)
        torch.cuda.synchronize()
    except Exception as e:
        print(f"  flex_attention FAILED: {e}")
        return None

    sdpa_full_ms = measure(sdpa_full)
    sdpa_ddi_ms = measure(sdpa_ddi)

    bm_full = make_full_causal_block_mask(T, device)
    flex_full_fn = lambda: flex_compiled(q, k, v, block_mask=bm_full)
    flex_full_ms = measure(flex_full_fn)

    bm_ddi = make_ddi_block_mask(doc_lengths, Q, device)
    flex_ddi_fn = lambda: flex_compiled(q, k, v, block_mask=bm_ddi)
    flex_ddi_ms = measure(flex_ddi_fn)

    allow = build_mask(doc_lengths, query_len=Q, answer_len=0, variant="ddi_qf")
    sparsity = 1.0 - allow.float().mean().item()

    print(f"  sdpa  full={sdpa_full_ms:8.3f}ms  ddi_qf={sdpa_ddi_ms:8.3f}ms  ratio={sdpa_full_ms/sdpa_ddi_ms:.2f}x")
    print(f"  flex  full={flex_full_ms:8.3f}ms  ddi_qf={flex_ddi_ms:8.3f}ms  ratio={flex_full_ms/flex_ddi_ms:.2f}x")
    print(f"  sparsity={sparsity:.3f}  analytical={1.0/(1.0-sparsity):.1f}x")

    return {
        "N": N, "L": L, "T": T, "sparsity": sparsity,
        "sdpa_full": sdpa_full_ms, "sdpa_ddi": sdpa_ddi_ms,
        "flex_full": flex_full_ms, "flex_ddi": flex_ddi_ms,
        "sdpa_ratio": sdpa_full_ms / sdpa_ddi_ms,
        "flex_ratio": flex_full_ms / flex_ddi_ms,
    }


def main():
    from ddi.inference import load

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"torch: {torch.__version__}")

    model, _ = load(MODEL, backend="sdpa")
    model.eval()
    model = model.to("cuda")

    results = []
    for N, L in CELLS:
        r = test_raw_attention_timing(model, N, L)
        if r:
            results.append(r)

    print("\n=== SUMMARY ===")
    print(f"{'N':>4} {'L':>5} {'T':>6}  {'sdpa':>8} {'flex':>8} {'analyt':>8} {'sparsity':>8}")
    print("-" * 55)
    for r in results:
        print(f"{r['N']:>4} {r['L']:>5} {r['T']:>6}  {r['sdpa_ratio']:>8.2f} {r['flex_ratio']:>8.2f} "
              f"{1.0/(1.0-r['sparsity']):>8.1f} {r['sparsity']:>8.3f}")


if __name__ == "__main__":
    main()
