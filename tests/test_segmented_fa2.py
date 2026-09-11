"""CPU tests for the segmented-FA2 wiring: mocked kernels, real split/assembly.

The FA2 kernels cannot run without CUDA, so the mocks compute the same math
densely and the tests verify call structure (cu_seqlens, GQA head counts,
bottom-right query call) and end-to-end agreement with the unpatched SDPA
reference on a tiny eager model.
"""

from types import SimpleNamespace

import pytest
import torch

from ddi.segmented_fa2 import (
    _segmented_prefill,
    clear_segmented_state,
    patch_model_for_segmented_fa2,
    set_segmented_state,
)


def _model(backend: str):
    try:
        from transformers import Qwen2Config, Qwen2ForCausalLM
    except Exception as error:  # Local optional vision stack can make Qwen lazy import unavailable.
        pytest.skip(f"tiny Qwen runtime unavailable: {error}")
    from ddi.model import patch_model

    torch.manual_seed(61)
    config = Qwen2Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        attention_dropout=0.0,
        pad_token_id=0,
    )
    config._attn_implementation = backend
    return patch_model(Qwen2ForCausalLM(config).eval())


def _dense_varlen_mock(calls):
    """Compute block-diagonal causal varlen densely, recording call metadata."""

    def varlen(
        q, k, v, *, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal, softmax_scale
    ):
        assert causal is True
        calls.append(
            {
                "cu_q": cu_seqlens_q.tolist(),
                "cu_k": cu_seqlens_k.tolist(),
                "q_tokens": q.shape[0],
                "k_tokens": k.shape[0],
                "q_heads": q.shape[1],
                "k_heads": k.shape[1],
            }
        )
        qf, kf, vf = q.float(), k.float(), v.float()
        groups = qf.shape[1] // kf.shape[1]
        if groups > 1:
            kf = kf.repeat_interleave(groups, dim=1)
            vf = vf.repeat_interleave(groups, dim=1)
        out = torch.empty_like(qf)
        n_segments = len(cu_seqlens_q) - 1
        for segment in range(n_segments):
            q0, q1 = cu_seqlens_q[segment], cu_seqlens_q[segment + 1]
            k0, k1 = cu_seqlens_k[segment], cu_seqlens_k[segment + 1]
            qs, ks, vs = qf[q0:q1], kf[k0:k1], vf[k0:k1]
            scores = torch.einsum("qhd,khd->hqk", qs, ks) * softmax_scale
            allow = torch.ones(q1 - q0, k1 - k0, dtype=torch.bool)
            if q1 - q0 == k1 - k0:
                allow = torch.tril(allow)
            elif q1 - q0 < k1 - k0:
                # bottom-right alignment
                offset = (k1 - k0) - (q1 - q0)
                allow = torch.zeros(q1 - q0, k1 - k0, dtype=torch.bool)
                for row in range(q1 - q0):
                    allow[row, : offset + row + 1] = True
            probs = torch.softmax(scores.masked_fill(~allow[None], float("-inf")), dim=-1)
            out[q0:q1] = torch.einsum("hqk,khd->qhd", probs, vs)
        return out.to(q.dtype)

    return varlen


def _dense_full_mock(calls):
    def func(q, k, v, *, causal, softmax_scale):
        assert causal is True
        assert q.dim() == 4  # [batch, seqlen, heads, dim] — batch kept for flash_attn_func
        calls.append(
            {"mode": "full", "tokens": q.shape[1], "q_heads": q.shape[2], "k_heads": k.shape[2]}
        )
        qf = q.float()[0]
        kf = k.float()[0]
        vf = v.float()[0]
        groups = qf.shape[1] // kf.shape[1]
        if groups > 1:
            kf = kf.repeat_interleave(groups, dim=1)
            vf = vf.repeat_interleave(groups, dim=1)
        scores = torch.einsum("thd,shd->hts", qf, kf) * softmax_scale
        allow = torch.tril(torch.ones(qf.shape[0], kf.shape[0], dtype=torch.bool))
        probs = torch.softmax(scores.masked_fill(~allow[None], float("-inf")), dim=-1)
        out = torch.einsum("htk,khd->thd", probs, vf)
        return out.unsqueeze(0).to(q.dtype)

    return func


@pytest.mark.parametrize("variant", ["ddi_qf", "blind"])
def test_segmented_prefill_call_structure_and_math(variant):
    import ddi.segmented_fa2 as module

    calls: list = []
    original = dict(module._FLASH)
    module._FLASH["varlen"] = _dense_varlen_mock(calls)
    module._FLASH.pop("func", None)
    try:
        torch.manual_seed(0)
        doc_lengths, query_len = [3, 4], 2
        heads, kv_heads, dim = 4, 2, 8
        tokens = sum(doc_lengths) + query_len
        q = torch.randn(1, heads, tokens, dim)
        k = torch.randn(1, kv_heads, tokens, dim)
        v = torch.randn(1, kv_heads, tokens, dim)
        state = {"doc_lengths": doc_lengths, "query_len": query_len, "variant": variant}
        out = _segmented_prefill(q, k, v, state, dim**-0.5)
    finally:
        module._FLASH.clear()
        module._FLASH.update(original)

    assert out.shape == q.shape
    assert len(calls) == 2
    doc_call, query_call = calls
    assert doc_call["cu_q"] == doc_call["cu_k"] == [0, 3, 7]
    assert doc_call["q_heads"] == heads and doc_call["k_heads"] == kv_heads  # native GQA
    assert doc_call["q_tokens"] == 7
    if variant == "ddi_qf":
        assert query_call["cu_q"] == [0, 2] and query_call["cu_k"] == [0, 9]
        assert query_call["k_tokens"] == 9  # docs + query
    else:
        assert query_call["cu_q"] == [0, 2] and query_call["cu_k"] == [0, 2]
        assert query_call["k_tokens"] == 2  # query only


def test_full_variant_single_call():
    import ddi.segmented_fa2 as module

    calls: list = []
    original = dict(module._FLASH)
    module._FLASH["func"] = _dense_full_mock(calls)
    module._FLASH.pop("varlen", None)
    try:
        torch.manual_seed(0)
        q = torch.randn(1, 4, 9, 8)
        k = torch.randn(1, 2, 9, 8)
        v = torch.randn(1, 2, 9, 8)
        state = {"doc_lengths": [3, 4], "query_len": 2, "variant": "full"}
        out = _segmented_prefill(q, k, v, state, 8**-0.5)
    finally:
        module._FLASH.clear()
        module._FLASH.update(original)
    assert out.shape == q.shape
    assert len(calls) == 1 and calls[0]["mode"] == "full"


def test_state_helpers_validate():
    model = SimpleNamespace()
    set_segmented_state(model, [3, 4], 2, variant="ddi_qf")
    assert model._segmented_fa2_state == {
        "doc_lengths": [3, 4],
        "query_len": 2,
        "variant": "ddi_qf",
    }
    clear_segmented_state(model)
    assert model._segmented_fa2_state is None
    with pytest.raises(ValueError):
        set_segmented_state(model, [3, 4], 2, variant="nope")
    with pytest.raises(ValueError):
        set_segmented_state(model, [], 2, variant="ddi_qf")


def test_tiny_model_segmented_matches_sdpa_reference():
    """End-to-end: patched tiny model with mocked kernels == SDPA additive-mask path."""
    import ddi.segmented_fa2 as module

    model = _model("eager")
    patch_model_for_segmented_fa2(model)
    torch.manual_seed(3)
    doc_lengths, query_len = [5, 7], 3
    ids = torch.randint(1, 90, (1, sum(doc_lengths) + query_len))
    from ddi.masks import build_mask, to_additive
    from ddi.positions import build_positions

    additive = to_additive(build_mask(doc_lengths, query_len, 0, "ddi_qf"), model.dtype)
    positions = build_positions(doc_lengths, query_len, 0, "restart").unsqueeze(0)

    with torch.no_grad():
        reference = model(
            input_ids=ids,
            attention_mask=additive[None, None],
            position_ids=positions,
            use_cache=False,
        ).logits

    calls: list = []
    original = dict(module._FLASH)
    module._FLASH["varlen"] = _dense_varlen_mock(calls)
    module._FLASH.pop("func", None)
    try:
        set_segmented_state(model, doc_lengths, query_len, variant="ddi_qf")
        with torch.no_grad():
            candidate = model(
                input_ids=ids,
                attention_mask=additive[None, None],
                position_ids=positions,
                use_cache=False,
            ).logits
    finally:
        clear_segmented_state(model)
        module._FLASH.clear()
        module._FLASH.update(original)

    assert len(calls) == 2 * model.config.num_hidden_layers  # doc + query call per layer
    assert candidate.shape == reference.shape
    assert torch.allclose(candidate, reference, atol=1e-4, rtol=1e-4)


def test_decode_falls_back_to_sdpa():
    model = _model("eager")
    patch_model_for_segmented_fa2(model)
    set_segmented_state(model, [5, 7], 3, variant="ddi_qf")
    # A single-token forward (decode shape) must not enter the segmented path:
    # hidden length 1 != doc_total + query_len, and the forward must succeed.
    ids = torch.randint(1, 90, (1, 1))
    positions = torch.tensor([[15]])
    additive = torch.zeros(1, 1, 1, 1, dtype=model.dtype)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=additive, position_ids=positions, use_cache=False)
    assert out.logits.shape[-1] == model.config.vocab_size
    clear_segmented_state(model)
