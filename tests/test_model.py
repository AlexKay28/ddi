"""Tests for the HF model wrapper and mask-preservation patch (issue #5, Phase 0.4).

Unit tests use a mock module in the shape `load_model` returns (outer `nn.Module` with an inner
`.model` attribute that owns `_update_causal_mask` and a `forward` that records what it saw), so
the CPU test path exercises the patch's real logic without any HF download. The integration test
loads a genuinely tiny public checkpoint and is opt-in via `DDI_INTEGRATION=1`.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ddi.masks import build_mask, to_additive
from ddi.model import forward_with_mask, load_model, patch_model, unpatch_model


# ---------------------------------------------------------------------------
# Mocks that mimic the surface `load_model` returns
# ---------------------------------------------------------------------------


class _MockInner(nn.Module):
    """Stands in for `LlamaModel` / `Qwen2Model` — owns `_update_causal_mask`."""

    _SENTINEL = "unset"

    def __init__(self) -> None:
        super().__init__()
        self.called_with: object = self._SENTINEL
        self.forwarded_with: torch.Tensor | None = None

    def _update_causal_mask(self, attention_mask, *args, **kwargs):  # noqa: ARG002
        self.called_with = attention_mask
        if attention_mask is None:
            return None
        return torch.zeros_like(attention_mask, dtype=torch.float32)


class _MockModel(nn.Module):
    """Stands in for `LlamaForCausalLM` — outer wrapper with `.model` inner."""

    def __init__(self) -> None:
        super().__init__()
        self.model = _MockInner()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,  # noqa: ARG002
    ):
        # Route through inner so tests can assert the mask reached the layer that would run
        # attention. We also invoke the (possibly patched) `_update_causal_mask` here so the
        # short-circuit is exercised on the real call path.
        prepared = self.model._update_causal_mask(attention_mask)
        self.model.forwarded_with = prepared
        return SimpleNamespace(logits=torch.zeros(input_ids.shape + (10,)))


class _BareModel(nn.Module):
    """No `_update_causal_mask` anywhere — patch_model must reject this."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()


# ---------------------------------------------------------------------------
# patch_model / unpatch_model
# ---------------------------------------------------------------------------


def test_patch_finds_inner_update_causal_mask() -> None:
    mock = _MockModel()
    original_func = _MockInner._update_causal_mask

    patch_model(mock)

    stored = mock.model.__ddi_original_update_causal_mask__
    assert stored.__func__ is original_func
    assert stored.__self__ is mock.model
    # After patch, `_update_causal_mask` is an instance attribute (types.MethodType), not the
    # class descriptor → its `__func__` is our DDI wrapper, not the original.
    assert mock.model._update_causal_mask.__func__ is not original_func


def test_patch_short_circuits_4d_float_mask() -> None:
    mock = _MockModel()
    patch_model(mock)

    mask4d = torch.zeros(1, 1, 4, 4, dtype=torch.float32)
    out = mock.model._update_causal_mask(mask4d)

    assert out is mask4d
    # `_SENTINEL` still there → the original was not invoked at all.
    assert mock.model.called_with == _MockInner._SENTINEL


def test_patch_delegates_for_2d() -> None:
    mock = _MockModel()
    patch_model(mock)

    mask2d = torch.ones(4, 4, dtype=torch.float32)
    mock.model._update_causal_mask(mask2d)

    assert mock.model.called_with is mask2d


def test_patch_delegates_for_bool_4d() -> None:
    """Bool 4D is HF's own path; only float 4D is DDI's additive mask."""
    mock = _MockModel()
    patch_model(mock)

    mask4d_bool = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    mock.model._update_causal_mask(mask4d_bool)

    assert mock.model.called_with is mask4d_bool


def test_patch_delegates_for_none() -> None:
    mock = _MockModel()
    patch_model(mock)

    out = mock.model._update_causal_mask(None)
    assert out is None
    # The original was invoked (sentinel was overwritten with None).
    assert mock.model.called_with is None


def test_unpatch_restores() -> None:
    mock = _MockModel()
    original_func = _MockInner._update_causal_mask
    patch_model(mock)
    unpatch_model(mock)

    # Bound method compares equal when `.__func__` and `.__self__` match — the point is that
    # `_update_causal_mask` now resolves to the class descriptor again, not to the DDI wrapper.
    assert mock.model._update_causal_mask.__func__ is original_func
    assert not hasattr(mock.model, "__ddi_original_update_causal_mask__")

    # And behavior is back to the original: calling with a 4D float mask does NOT short-circuit.
    mask4d = torch.zeros(1, 1, 3, 3, dtype=torch.float32)
    out = mock.model._update_causal_mask(mask4d)
    assert out is not mask4d
    assert mock.model.called_with is mask4d


def test_patch_raises_when_no_method_exists() -> None:
    bare = _BareModel()
    with pytest.raises(AttributeError, match="_update_causal_mask"):
        patch_model(bare)


def test_patch_is_per_instance() -> None:
    """Two loaded models must not share patch state."""
    a = _MockModel()
    b = _MockModel()
    patch_model(a)

    # b remains pristine
    assert not hasattr(b.model, "__ddi_original_update_causal_mask__")

    mask4d = torch.zeros(1, 1, 3, 3, dtype=torch.float32)
    out_a = a.model._update_causal_mask(mask4d)
    out_b = b.model._update_causal_mask(mask4d)

    assert out_a is mask4d  # patched → short-circuit
    assert out_b is not mask4d  # original → returns zeros_like copy
    assert b.model.called_with is mask4d


def test_patch_on_outer_when_method_lives_there() -> None:
    """Some HF classes put `_update_causal_mask` on the top-level model."""

    class _OuterOwns(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.called_with: torch.Tensor | None = None

        def _update_causal_mask(self, attention_mask, *args, **kwargs):  # noqa: ARG002
            self.called_with = attention_mask
            return attention_mask

    outer = _OuterOwns()
    original_func = _OuterOwns._update_causal_mask
    patch_model(outer)

    stored = outer.__ddi_original_update_causal_mask__
    assert stored.__func__ is original_func
    assert stored.__self__ is outer

    mask4d = torch.zeros(1, 1, 2, 2, dtype=torch.float32)
    assert outer._update_causal_mask(mask4d) is mask4d
    assert outer.called_with is None


# ---------------------------------------------------------------------------
# forward_with_mask
# ---------------------------------------------------------------------------


def test_forward_with_mask_rejects_non_4d() -> None:
    mock = _MockModel()
    patch_model(mock)
    input_ids = torch.zeros(1, 4, dtype=torch.long)
    mask2d = torch.zeros(4, 4, dtype=torch.float32)

    with pytest.raises(ValueError, match="4D"):
        forward_with_mask(mock, input_ids, mask2d)


def test_forward_with_mask_rejects_non_float() -> None:
    mock = _MockModel()
    patch_model(mock)
    input_ids = torch.zeros(1, 4, dtype=torch.long)
    mask_bool = torch.ones(1, 1, 4, 4, dtype=torch.bool)

    with pytest.raises(ValueError, match="floating-point"):
        forward_with_mask(mock, input_ids, mask_bool)


def test_forward_with_mask_passes_mask_through() -> None:
    mock = _MockModel()
    patch_model(mock)

    input_ids = torch.zeros(1, 4, dtype=torch.long)
    mask4d = torch.zeros(1, 1, 4, 4, dtype=torch.float32)

    logits = forward_with_mask(mock, input_ids, mask4d)

    # The 4D float mask survived HF's `_update_causal_mask` (mock short-circuits) and reached
    # the point where attention would consume it. Identity — same object, not a copy.
    assert mock.model.forwarded_with is mask4d
    assert logits.shape == (1, 4, 10)


# ---------------------------------------------------------------------------
# Integration — opt-in
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("DDI_INTEGRATION") != "1",
    reason="set DDI_INTEGRATION=1 to run against a real model",
)
def test_real_model_end_to_end() -> None:
    model, tokenizer = load_model(
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()

    doc_lengths = [3, 4]
    query_len, answer_len = 2, 1
    total = sum(doc_lengths) + query_len + answer_len

    allow = build_mask(doc_lengths, query_len, answer_len, "ddi_qf")
    additive = to_additive(allow, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    vocab = model.config.vocab_size
    input_ids = torch.arange(total, dtype=torch.long).remainder(vocab).unsqueeze(0)

    with torch.no_grad():
        logits = forward_with_mask(model, input_ids, additive)

    assert logits.shape[:2] == input_ids.shape
    assert logits.shape[-1] == vocab

    # Cleanup so subsequent loads of the same object stay clean.
    unpatch_model(model)
    assert tokenizer is not None
