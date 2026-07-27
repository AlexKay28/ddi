"""Tests for the Phase 2.2 SFT loop (issue #14).

Unit tests exercise the small pure pieces — the shifted-CE formula, the TrainConfig
defaults, and the deterministic run-tag hash. The training loop itself is only smoke-
tested behind `DDI_INTEGRATION=1` against a tiny public checkpoint; running it on CPU
by default would make the test suite unusable, and mocking a Transformer well enough
to actually validate the loop is more mock than test.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from ddi.provenance import Provenance
from ddi.train import TrainConfig, _run_tag, _shifted_cross_entropy


# ---------------------------------------------------------------------------
# TrainConfig defaults
# ---------------------------------------------------------------------------


def test_trainconfig_defaults() -> None:
    """Defaults must match the values documented in the runner + config — a silent
    drift here would change the meaning of every SFT run that omits a key."""
    cfg = TrainConfig(model_name="m", dataset="d", mask_variant="ddi_qf")
    assert cfg.bridge_k is None
    assert cfg.split == "train"
    assert cfg.lr == 1e-5
    assert cfg.warmup_steps == 50
    assert cfg.max_steps == 500
    assert cfg.grad_accum_steps == 8
    assert cfg.batch_size == 1
    assert cfg.gradient_checkpointing is True
    assert cfg.validate_every == 100
    assert cfg.output_dir == Path("checkpoints/phase2_finetune")


def test_trainconfig_is_frozen() -> None:
    cfg = TrainConfig(model_name="m", dataset="d", mask_variant="ddi_qf")
    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
        cfg.lr = 1e-4  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _shifted_cross_entropy
# ---------------------------------------------------------------------------


def test_shifted_cross_entropy_matches_hand_computed() -> None:
    """The helper must reproduce F.cross_entropy applied to the shifted logits/labels
    exactly — this is the same formula phase1_baseline uses inline. If issue #13
    factors it into ddi/runners/_shared.py, that module should import from here (or
    we should adopt _shared.py) so scoring stays in one place.
    """
    torch.manual_seed(0)
    logits = torch.randn(1, 5, 7)
    # -100 mask on the first two positions; real labels afterwards.
    labels = torch.tensor([[-100, -100, 3, 5, 1]])

    got = _shifted_cross_entropy(logits, labels)

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    expected = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    assert torch.allclose(got, expected)


def test_shifted_cross_entropy_all_masked_returns_nan() -> None:
    logits = torch.randn(1, 3, 4)
    labels = torch.full((1, 3), -100, dtype=torch.long)
    got = _shifted_cross_entropy(logits, labels)
    # cross_entropy with all-ignored targets is NaN by convention; callers use
    # torch.isfinite to route around this.
    assert torch.isnan(got)


# ---------------------------------------------------------------------------
# _run_tag
# ---------------------------------------------------------------------------


def test_train_run_tag_stable() -> None:
    """Given a fixed cfg + seed, the run tag is deterministic — otherwise a repeated
    launch would write to a different directory and the store rows could not be
    matched to a checkpoint on disk."""
    cfg = TrainConfig(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        dataset="nq",
        mask_variant="ddi_qf",
        lr=1e-5,
        max_steps=500,
    )
    assert _run_tag(cfg, 42) == _run_tag(cfg, 42)


def test_train_run_tag_differs_on_seed() -> None:
    cfg = TrainConfig(model_name="m", dataset="d", mask_variant="ddi_qf")
    assert _run_tag(cfg, 1) != _run_tag(cfg, 2)


def test_train_run_tag_differs_on_mask() -> None:
    a = TrainConfig(model_name="m", dataset="d", mask_variant="ddi_qf")
    b = TrainConfig(model_name="m", dataset="d", mask_variant="ddi_strict")
    assert _run_tag(a, 0) != _run_tag(b, 0)


def test_train_run_tag_slash_safe() -> None:
    """HF model names contain `/` — the tag must be a legal single directory name."""
    cfg = TrainConfig(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        dataset="a/b",
        mask_variant="ddi_qf",
    )
    tag = _run_tag(cfg, 0)
    assert "/" not in tag


# ---------------------------------------------------------------------------
# Integration — opt-in
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("DDI_INTEGRATION") != "1",
    reason="set DDI_INTEGRATION=1 to run against a real (tiny) model",
)
def test_train_two_steps_saves_reloadable_checkpoint(tmp_path: Path) -> None:
    from ddi.model import load_model
    from ddi.train import train

    cfg = TrainConfig(
        model_name="hf-internal-testing/tiny-random-LlamaForCausalLM",
        dataset="synthetic_qa",
        mask_variant="ddi_qf",
        lr=1e-4,
        warmup_steps=1,
        max_steps=2,
        grad_accum_steps=1,
        gradient_checkpointing=False,
        validate_every=0,
        output_dir=tmp_path / "ckpts",
    )
    prov = Provenance(
        git_sha="0" * 40,
        dirty=False,
        config_hash="cafecafecafecafe",
        seed=0,
        model_revision=cfg.model_name,
        dataset_revision=cfg.dataset,
        gpu_model="cpu",
        driver_version="unknown",
        cuda_version="cpu",
        torch_version=torch.__version__,
        timestamp_utc="1970-01-01T00:00:00+00:00",
    )
    ckpt = train(cfg, seed=0, provenance=prov)

    assert ckpt.exists()
    assert ckpt.is_dir()
    # Re-load must succeed via the same path production uses.
    reloaded, tok = load_model(str(ckpt), dtype=torch.float32, attn_implementation="eager")
    assert reloaded is not None
    assert tok is not None
