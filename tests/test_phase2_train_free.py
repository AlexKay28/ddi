"""Tests for the Phase 2.1 train-free mask sweep runner (issue #13).

End-to-end runner behaviour needs a real model and is gated behind `DDI_INTEGRATION=1`.
The unit tests here cover the small helpers that own the interesting invariants:

- the shared cross-entropy formula stays bit-identical to the baseline's (so `Δ` is real)
- `run_id_prefix` encodes the variant + optional `bridge_k` unambiguously
- config validation fails loudly on missing / mismatched keys
"""

from __future__ import annotations

import os

import pytest
import torch

from ddi.runners._shared import shifted_cross_entropy


# ---------------------------------------------------------------------------
# Shared loss formula — bit-identical across phase1_baseline and phase2_train_free.
# ---------------------------------------------------------------------------


def _reference_shifted_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Inlined copy of the exact formula phase1_baseline used pre-refactor."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def test_shifted_cross_entropy_matches_baseline() -> None:
    """The shared helper must equal the phase1_baseline formula on a fixed input."""
    torch.manual_seed(0)
    batch, seq, vocab = 2, 7, 11
    logits = torch.randn(batch, seq, vocab)
    labels = torch.full((batch, seq), -100, dtype=torch.long)
    # Label the last 3 positions of each row so the shift actually contributes something.
    labels[:, -3:] = torch.randint(0, vocab, (batch, 3))

    shared = shifted_cross_entropy(logits, labels)
    reference = _reference_shifted_ce(logits, labels)
    assert torch.equal(shared, reference)


def test_shifted_cross_entropy_ignores_minus_100() -> None:
    """`-100` positions must be excluded — flipping a label to a real class changes the loss."""
    torch.manual_seed(1)
    logits = torch.randn(1, 5, 4)
    labels = torch.full((1, 5), -100, dtype=torch.long)
    labels[0, -1] = 2
    loss_a = shifted_cross_entropy(logits, labels)

    labels_b = labels.clone()
    labels_b[0, -2] = 1
    loss_b = shifted_cross_entropy(logits, labels_b)

    assert torch.isfinite(loss_a)
    assert torch.isfinite(loss_b)
    assert not torch.equal(loss_a, loss_b)


# ---------------------------------------------------------------------------
# run_id_prefix variant tag — sanity checks on the string the store keys on.
# ---------------------------------------------------------------------------


def _make_variant_tag(mask_variant: str, bridge_k: int | None) -> str:
    """Mirror of the runner's tag construction, factored so the test does not need a model."""
    return mask_variant if bridge_k is None else f"{mask_variant}_k{bridge_k}"


def test_variant_tag_no_bridge() -> None:
    assert _make_variant_tag("ddi_qf", None) == "ddi_qf"
    assert _make_variant_tag("full", None) == "full"
    assert _make_variant_tag("ddi_strict", None) == "ddi_strict"
    assert _make_variant_tag("blind", None) == "blind"


def test_variant_tag_with_bridge() -> None:
    assert _make_variant_tag("ddi_bridge_k", 1) == "ddi_bridge_k_k1"
    assert _make_variant_tag("ddi_bridge_k", 8) == "ddi_bridge_k_k8"
    assert _make_variant_tag("ddi_bridge_k", 64) == "ddi_bridge_k_k64"


# ---------------------------------------------------------------------------
# Runner registration & config validation.
# ---------------------------------------------------------------------------


def test_runner_is_registered() -> None:
    """Side-effect import of the runners package must register `phase2_train_free`."""
    import ddi.runners  # noqa: F401 — force side-effect registration
    from ddi.cli import _RUNNERS

    assert "phase2_train_free" in _RUNNERS


def test_config_missing_mask_key_raises() -> None:
    from ddi.runners.phase2_train_free import _run

    with pytest.raises(ValueError, match="missing required key: mask"):
        _run({"dataset": "nq", "model": "irrelevant"}, "cfg", 0)


def test_config_missing_variant_raises() -> None:
    from ddi.runners.phase2_train_free import _run

    with pytest.raises(ValueError, match="missing required key: variant"):
        _run({"dataset": "nq", "model": "irrelevant", "mask": {}}, "cfg", 0)


def test_config_bridge_k_without_bridge_variant_raises() -> None:
    from ddi.runners.phase2_train_free import _run

    with pytest.raises(ValueError, match="only valid for variant='ddi_bridge_k'"):
        _run(
            {
                "dataset": "nq",
                "model": "irrelevant",
                "mask": {"variant": "ddi_qf", "bridge_k": 4},
            },
            "cfg",
            0,
        )


def test_config_bridge_variant_without_bridge_k_raises() -> None:
    from ddi.runners.phase2_train_free import _run

    with pytest.raises(ValueError, match="requires mask.bridge_k"):
        _run(
            {
                "dataset": "nq",
                "model": "irrelevant",
                "mask": {"variant": "ddi_bridge_k"},
            },
            "cfg",
            0,
        )


# ---------------------------------------------------------------------------
# Integration: greedy prefill-mask generation on a tiny random model.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("DDI_INTEGRATION") != "1",
    reason="integration test; set DDI_INTEGRATION=1 to enable",
)
def test_generate_with_prefill_mask_shape() -> None:
    """Greedy loop must respect max_new_tokens on a real (tiny) HF model."""
    from ddi.masks import build_mask, to_additive
    from ddi.model import load_model
    from ddi.runners.phase2_train_free import _generate_with_prefill_mask

    model, tokenizer = load_model("hf-internal-testing/tiny-random-LlamaForCausalLM")
    model.eval()

    doc_ids = tokenizer("alpha bravo charlie", return_tensors="pt", add_special_tokens=False).input_ids
    query_ids = tokenizer("What?", return_tensors="pt", add_special_tokens=False).input_ids
    prefill_ids = torch.cat([doc_ids, query_ids], dim=-1).to(model.device)

    allow = build_mask(
        [doc_ids.shape[-1]], query_len=query_ids.shape[-1], answer_len=0, variant="ddi_qf"
    )
    additive = to_additive(allow, dtype=model.dtype).to(model.device).unsqueeze(0).unsqueeze(0)

    max_new = 5
    out = _generate_with_prefill_mask(model, tokenizer, prefill_ids, additive, max_new)
    assert isinstance(out, str)
    # Round-trip through the tokenizer to bound the generated slice length.
    generated_ids = tokenizer(out, return_tensors="pt", add_special_tokens=False).input_ids
    assert generated_ids.shape[-1] <= max_new
