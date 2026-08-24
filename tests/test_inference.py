"""Tests for `ddi.inference` — the sole HF-interaction surface (issue #84).

Pure-unit tests (no HF download) cover the small orchestration invariants:
dtype resolution, backend name mapping, `encode` output shapes/layouts, and
config validation. The end-to-end `generate` / `forward` /
mask-variant cross-check tests use `hf-internal-testing/tiny-random-LlamaForCausalLM`
(same tiny checkpoint the rest of the suite uses) and are gated behind
`DDI_INTEGRATION=1` so CPU-only CI does not download weights or run the real
attention path.
"""

from __future__ import annotations

import os

import pytest
import torch

from ddi.inference import (
    BatchEncoding,
    ForwardOutput,
    _resolve_dtype,
    encode,
    forward,
    generate,
    load,
)

# ---------------------------------------------------------------------------
# Pure-unit tests — no HF download, no CUDA.
# ---------------------------------------------------------------------------


def test_resolve_dtype_maps_supported_strings() -> None:
    assert _resolve_dtype("bf16") is torch.bfloat16
    assert _resolve_dtype("fp16") is torch.float16
    assert _resolve_dtype("fp32") is torch.float32


def test_resolve_dtype_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown dtype"):
        _resolve_dtype("int8")


def test_load_rejects_unknown_dtype_without_hf_download() -> None:
    """dtype validation must happen before any HF call."""
    with pytest.raises(ValueError, match="unknown dtype"):
        load("hf-internal-testing/tiny-random-LlamaForCausalLM", dtype="int8")


def test_encode_rejects_unknown_layout() -> None:
    class _StubTokenizer:
        def __call__(self, *a, **k):  # noqa: ARG002
            raise AssertionError("tokenizer must not be called if layout is invalid")

    with pytest.raises(ValueError, match="unknown layout"):
        encode(_StubTokenizer(), docs=["x"], query="y", layout="zigzag")


class _FakeTokenizerOutput:
    def __init__(self, ids: list[int]) -> None:
        self.input_ids = torch.tensor([ids], dtype=torch.long)


class _FakeTokenizer:
    """Fixed-length whitespace-split tokenizer; enough to drive `encode` end-to-end."""

    def __call__(self, text, return_tensors=None, add_special_tokens=None):  # noqa: ARG002
        return _FakeTokenizerOutput([abs(hash(w)) % 100 + 1 for w in text.split()])


def test_encode_shape_and_doc_lengths() -> None:
    tok = _FakeTokenizer()
    docs = ["one two three", "four five", "six"]
    query = "seven eight"
    batch = encode(tok, docs, query, layout="sequential")

    assert isinstance(batch, BatchEncoding)
    assert batch.doc_lengths == [3, 2, 1]
    assert batch.query_len == 2
    total = sum(batch.doc_lengths) + batch.query_len
    assert batch.input_ids.shape == (1, total)
    assert batch.position_ids.shape == (1, total)
    assert batch.layout == "sequential"
    # Sequential layout must be [0, 1, 2, ...] regardless of doc structure.
    assert torch.equal(batch.position_ids[0], torch.arange(total, dtype=torch.long))


def test_encode_restart_layout_restarts_positions_per_doc() -> None:
    tok = _FakeTokenizer()
    batch = encode(tok, docs=["one two three", "four five"], query="six", layout="restart")

    # doc_lengths = [3, 2]; positions restart within each doc, then the query starts at max(3, 2).
    expected = torch.tensor([0, 1, 2, 0, 1, 3], dtype=torch.long)
    assert torch.equal(batch.position_ids[0], expected)


def test_encode_no_docs() -> None:
    tok = _FakeTokenizer()
    batch = encode(tok, docs=[], query="one two", layout="sequential")
    assert batch.doc_lengths == []
    assert batch.query_len == 2
    assert batch.input_ids.shape == (1, 2)


# ---------------------------------------------------------------------------
# Integration tests — opt-in via DDI_INTEGRATION=1.
# ---------------------------------------------------------------------------


_TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"
_SKIP_INTEGRATION = pytest.mark.skipif(
    os.environ.get("DDI_INTEGRATION") != "1",
    reason="integration test; set DDI_INTEGRATION=1 to enable",
)


@_SKIP_INTEGRATION
def test_load_returns_compatible_model_and_tokenizer() -> None:
    """`load` must return an HF model + tokenizer usable together on CPU with fp32."""
    model, tokenizer = load(_TINY_MODEL, dtype="fp32", backend="eager")
    model.eval()
    assert model.config.vocab_size > 0
    ids = tokenizer("hello world", return_tensors="pt", add_special_tokens=False).input_ids
    assert ids.dim() == 2 and ids.shape[0] == 1


@_SKIP_INTEGRATION
def test_forward_returns_expected_logits_shape() -> None:
    """`forward` must return logits with shape [1, T, V]."""
    model, tokenizer = load(_TINY_MODEL, dtype="fp32", backend="eager")
    model.eval()

    batch = encode(tokenizer, docs=["alpha bravo", "charlie"], query="delta")
    out = forward(
        model,
        batch.input_ids,
        mask_variant="ddi_qf",
        doc_lengths=batch.doc_lengths,
        query_len=batch.query_len,
    )

    assert isinstance(out, ForwardOutput)
    T = batch.input_ids.shape[-1]
    assert out.logits.shape == (1, T, model.config.vocab_size)
    assert out.mask_variant == "ddi_qf"
    # Requested via output_hidden_states=True — must be non-empty for a real HF model.
    assert out.hidden_states is not None and len(out.hidden_states) >= 1


@_SKIP_INTEGRATION
def test_generate_returns_list_of_strings() -> None:
    """`generate` returns a list of length batch_size (=1 in Phase 1)."""
    model, tokenizer = load(_TINY_MODEL, dtype="fp32", backend="eager")
    model.eval()

    batch = encode(tokenizer, docs=["alpha bravo charlie"], query="delta")
    outputs = generate(model, tokenizer, batch, mask_variant="ddi_qf", max_new_tokens=5)

    assert isinstance(outputs, list)
    assert len(outputs) == 1
    assert isinstance(outputs[0], str)


@_SKIP_INTEGRATION
def test_generate_bridge_k_requires_bridge_kwarg() -> None:
    """`ddi_bridge_k` without `bridge_k` in mask_kwargs must surface the mask's own error."""
    model, tokenizer = load(_TINY_MODEL, dtype="fp32", backend="eager")
    model.eval()

    batch = encode(tokenizer, docs=["alpha bravo", "charlie"], query="delta")
    with pytest.raises(ValueError, match="bridge_k"):
        generate(model, tokenizer, batch, mask_variant="ddi_bridge_k", max_new_tokens=3)


@_SKIP_INTEGRATION
def test_mask_variant_actually_affects_forward_output() -> None:
    """Two variants on the same input must yield different logits — proves the mask flows through.

    Uses two structurally distinct variants (`full` vs `ddi_strict`) on a multi-doc
    encoding where DDI's cross-doc gating actually changes the attention geometry.
    """
    model, tokenizer = load(_TINY_MODEL, dtype="fp32", backend="eager")
    model.eval()

    batch = encode(tokenizer, docs=["alpha bravo", "charlie delta"], query="echo foxtrot")

    out_full = forward(
        model,
        batch.input_ids,
        mask_variant="full",
        doc_lengths=batch.doc_lengths,
        query_len=batch.query_len,
    )
    out_strict = forward(
        model,
        batch.input_ids,
        mask_variant="ddi_strict",
        doc_lengths=batch.doc_lengths,
        query_len=batch.query_len,
    )

    assert out_full.logits.shape == out_strict.logits.shape
    # If the mask were silently ignored the two forwards would be bit-identical.
    assert not torch.allclose(out_full.logits, out_strict.logits, atol=1e-6)


@_SKIP_INTEGRATION
def test_generate_next_token_logits_differ_across_mask_variants() -> None:
    """Cross-check for `generate`: the prefill's next-token logits must differ per variant.

    Comparing decoded strings under greedy decoding is fragile on a tiny random model
    (argmax often collapses to the same id even when the logit landscapes differ). What
    the mask must change is the prefill logit distribution the greedy loop reads from —
    that is what this test pins.
    """
    model, tokenizer = load(_TINY_MODEL, dtype="fp32", backend="eager")
    model.eval()

    batch = encode(tokenizer, docs=["alpha bravo", "charlie delta"], query="echo foxtrot")

    shared = dict(doc_lengths=batch.doc_lengths, query_len=batch.query_len)
    logits_full = forward(model, batch.input_ids, mask_variant="full", **shared).logits[0, -1]
    logits_blind = forward(model, batch.input_ids, mask_variant="blind", **shared).logits[0, -1]

    # `blind` cuts query->doc attention entirely; the last-token logits must diverge.
    assert not torch.allclose(logits_full, logits_blind, atol=1e-6)
