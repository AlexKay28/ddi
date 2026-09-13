"""Tests for `ddi.finetune` (issue #120).

Pure-unit tests cover the small combinatorial helpers (`_mix_records`,
`_holdout_split`) — no HF download. The end-to-end 2-step training smoke
test uses `hf-internal-testing/tiny-random-LlamaForCausalLM` and is gated on
``DDI_INTEGRATION=1``, matching every other integration test in the suite.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from ddi import finetune as ddi_finetune

# ---------------------------------------------------------------------------
# Pure-unit: record mixing + holdout split invariants.
# ---------------------------------------------------------------------------


def _stub_loader_factory(
    monkeypatch: pytest.MonkeyPatch, records_by_ds: dict[str, list[dict]]
) -> None:
    """Monkey-patch ddi.finetune's `load_dataset` to yield from the given fixture."""

    def _fake_load(name, split, seed):  # noqa: ARG001
        return iter(records_by_ds.get(name, []))

    monkeypatch.setattr(ddi_finetune, "load_dataset", _fake_load)


def test_mix_records_roundrobin(monkeypatch: pytest.MonkeyPatch) -> None:
    records_by_ds = {
        "a": [{"id": f"a{i}"} for i in range(3)],
        "b": [{"id": f"b{i}"} for i in range(2)],
        "c": [{"id": f"c{i}"} for i in range(4)],
    }
    _stub_loader_factory(monkeypatch, records_by_ds)
    mixed = ddi_finetune._mix_records(["a", "b", "c"], seed=0, max_records=None)
    # Round-robin: a0, b0, c0, a1, b1, c1, a2, c2, c3.
    assert [r["id"] for r in mixed] == ["a0", "b0", "c0", "a1", "b1", "c1", "a2", "c2", "c3"]


def test_mix_records_max_records_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    records_by_ds = {
        "a": [{"id": f"a{i}"} for i in range(5)],
        "b": [{"id": f"b{i}"} for i in range(5)],
    }
    _stub_loader_factory(monkeypatch, records_by_ds)
    mixed = ddi_finetune._mix_records(["a", "b"], seed=0, max_records=3)
    assert [r["id"] for r in mixed] == ["a0", "b0", "a1"]


def test_holdout_split_tail() -> None:
    records = [{"id": str(i)} for i in range(10)]
    train, test = ddi_finetune._holdout_split(records, held_out_frac=0.2)
    assert [r["id"] for r in train] == [str(i) for i in range(8)]
    assert [r["id"] for r in test] == ["8", "9"]


def test_holdout_split_zero_frac() -> None:
    records = [{"id": str(i)} for i in range(4)]
    train, test = ddi_finetune._holdout_split(records, held_out_frac=0.0)
    assert len(train) == 4 and test == []


def test_holdout_split_rejects_full_holdout() -> None:
    records = [{"id": str(i)} for i in range(2)]
    with pytest.raises(ValueError, match="no training records"):
        ddi_finetune._holdout_split(records, held_out_frac=0.99)


def test_holdout_split_rejects_bad_frac() -> None:
    records = [{"id": "x"}]
    with pytest.raises(ValueError, match="held_out_frac"):
        ddi_finetune._holdout_split(records, held_out_frac=1.0)


def test_linear_warmup_lr() -> None:
    # Below warmup: fraction of base lr.
    assert ddi_finetune._linear_warmup_lr(0, warmup_steps=10, base_lr=1.0) == pytest.approx(0.1)
    assert ddi_finetune._linear_warmup_lr(4, warmup_steps=10, base_lr=1.0) == pytest.approx(0.5)
    # At/after warmup: full base_lr.
    assert ddi_finetune._linear_warmup_lr(10, warmup_steps=10, base_lr=1.0) == 1.0
    assert ddi_finetune._linear_warmup_lr(100, warmup_steps=10, base_lr=1.0) == 1.0
    # warmup_steps=0 short-circuits to base_lr.
    assert ddi_finetune._linear_warmup_lr(0, warmup_steps=0, base_lr=0.5) == 0.5


def test_answer_kl_is_zero_for_identical_logits() -> None:
    logits = torch.tensor([[[1.0, 2.0], [3.0, -1.0]]])
    assert ddi_finetune._answer_kl(logits, logits, temperature=2.0) == pytest.approx(
        0.0, abs=1e-7
    )


def test_teacher_first_token_ce_uses_teacher_argmax() -> None:
    student = torch.tensor([[[0.0, 4.0, 0.0], [9.0, 0.0, 0.0]]])
    teacher = torch.tensor([[[0.0, 3.0, 1.0], [0.0, 0.0, 8.0]]])
    expected = torch.nn.functional.cross_entropy(
        student[:, 0, :], torch.tensor([1])
    )
    assert ddi_finetune._teacher_first_token_ce(student, teacher) == expected


@pytest.mark.parametrize(
    "name",
    ["teacher_first_token_weight", "full_anchor_weight", "blind_anchor_weight"],
)
def test_finetune_rejects_negative_objective_weight(
    tmp_path: Path, name: str
) -> None:
    from ddi.provenance import Provenance

    prov = Provenance(
        git_sha="x", dirty=False, config_hash="x", seed=0,
        model_revision="x", dataset_revision="x",
        gpu_model="cpu", driver_version="x", cuda_version="x",
        torch_version="x", timestamp_utc="x",
    )
    with pytest.raises(ValueError, match=f"{name} must be >= 0"):
        ddi_finetune.finetune(
            "irrelevant", ["ds"], tmp_path / "out", seed=0,
            provenance=prov, store_path=tmp_path / "runs.jsonl", **{name: -1.0},
        )


def test_finetune_rejects_empty_datasets(tmp_path: Path) -> None:
    from ddi.provenance import Provenance

    prov = Provenance(
        git_sha="x", dirty=False, config_hash="x", seed=0,
        model_revision="x", dataset_revision="x",
        gpu_model="cpu", driver_version="x", cuda_version="x",
        torch_version="x", timestamp_utc="x",
    )
    with pytest.raises(ValueError, match="`datasets` must be non-empty"):
        ddi_finetune.finetune(
            "irrelevant",
            [],
            tmp_path / "out",
            seed=0,
            provenance=prov,
            store_path=tmp_path / "runs.jsonl",
        )


def _rec(rid: str, answers=("x",), n_docs: int = 2) -> dict:
    return {
        "id": rid,
        "possible_answers": list(answers),
        "info_contexts": [f"doc {i}" for i in range(n_docs)],
    }


def test_prefilter_records_drops_empty_possible_answers() -> None:
    records = [
        _rec("a"),
        _rec("b", answers=()),          # dropped: no answer span -> no loss
        _rec("c", answers=("y", "z")),
        {"id": "d", "info_contexts": ["doc 0", "doc 1"]},  # dropped (missing key)
    ]
    kept = ddi_finetune._prefilter_records(records)
    assert [r["id"] for r in kept] == ["a", "c"]


def test_prefilter_records_drops_zero_gradient_single_doc_records() -> None:
    """`build_mask` collapses ddi_qf -> full at <=1 doc, so student == teacher, KL == 0.

    Those records cost two 4B forward passes and contribute exactly nothing. The
    kilt_nq train split is 100% of them (VIBEQUEUE-2 review, issue R4).
    """
    records = [_rec("two", n_docs=2), _rec("one", n_docs=1), _rec("none", n_docs=0)]
    kept = ddi_finetune._prefilter_records(records)
    assert [r["id"] for r in kept] == ["two"]


def test_prefilter_records_min_docs_one_restores_old_behaviour() -> None:
    records = [_rec("two", n_docs=2), _rec("one", n_docs=1), _rec("none", n_docs=0)]
    kept = ddi_finetune._prefilter_records(records, min_docs=1)
    assert [r["id"] for r in kept] == ["two", "one"]


def test_init_ddp_single_gpu_returns_zeros_even_under_torchrun_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # num_gpus=1 must ignore any leftover torchrun env — the single-device path
    # stays unchanged (issue #123).
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "2")
    assert ddi_finetune._init_ddp(1) == (0, 1, 0)


def test_init_ddp_multi_gpu_without_torchrun_env_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Missing WORLD_SIZE/RANK/LOCAL_RANK → error with a torchrun hint.
    for k in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        monkeypatch.delenv(k, raising=False)
    # Fake CUDA availability so we hit the env-check branch, not the "no CUDA" branch.
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(RuntimeError, match="torchrun"):
        ddi_finetune._init_ddp(2)


def test_init_ddp_multi_gpu_without_cuda_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        ddi_finetune._init_ddp(2)


def test_init_ddp_world_size_mismatch_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    # Guard against torch.cuda.set_device — no real device in the unit-test env.
    monkeypatch.setattr(torch.cuda, "set_device", lambda idx: None)
    with pytest.raises(RuntimeError, match="WORLD_SIZE=4"):
        ddi_finetune._init_ddp(2)


def test_finetune_rejects_num_gpus_zero(tmp_path: Path) -> None:
    from ddi.provenance import Provenance

    prov = Provenance(
        git_sha="x", dirty=False, config_hash="x", seed=0,
        model_revision="x", dataset_revision="x",
        gpu_model="cpu", driver_version="x", cuda_version="x",
        torch_version="x", timestamp_utc="x",
    )
    with pytest.raises(ValueError, match="num_gpus must be >= 1"):
        ddi_finetune.finetune(
            "irrelevant",
            ["ds"],
            tmp_path / "out",
            seed=0,
            num_gpus=0,
            provenance=prov,
            store_path=tmp_path / "runs.jsonl",
        )


# ---------------------------------------------------------------------------
# Frozen trunk (`trainable_layers`) — what makes the 27B fit on a 141 GB card.
# ---------------------------------------------------------------------------


def _stub_decoder(n_layers: int, width: int) -> torch.nn.Module:
    """A module shaped like a decoder: `model.layers` of `width`-sized linears + a norm."""
    inner = torch.nn.Module()
    inner.layers = torch.nn.ModuleList(
        torch.nn.Linear(width, width) for _ in range(n_layers)
    )
    inner.norm = torch.nn.LayerNorm(width)
    outer = torch.nn.Module()
    outer.model = inner
    outer.lm_head = torch.nn.Linear(width, 7)
    return outer


def test_freeze_trunk_trains_only_the_top_layers() -> None:
    model = _stub_decoder(6, 8)
    ddi_finetune._freeze_trunk(model, 2)

    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    # top two layers (4, 5) + the stack's sibling norm; nothing below, no lm_head
    assert {n for n in trainable if "layers." in n} == {
        "model.layers.4.weight", "model.layers.4.bias",
        "model.layers.5.weight", "model.layers.5.bias",
    }
    assert "model.norm.weight" in trainable
    assert not any(n.startswith("lm_head") for n in trainable)


def test_freeze_trunk_rejects_more_layers_than_the_model_has() -> None:
    model = _stub_decoder(4, 8)
    with pytest.raises(ValueError, match="between 1 and 4"):
        ddi_finetune._freeze_trunk(model, 5)


def test_decoder_layers_picks_the_biggest_stack_not_the_first() -> None:
    """Qwen3.6-27B is multimodal: its vision tower has a `layers` list too.

    Traversal order is a transformers implementation detail, so picking the first match
    could train the vision encoder and leave all 64 text layers frozen — a run that would
    report a falling loss and save what is effectively the base model.
    """
    model = torch.nn.Module()
    vision = torch.nn.Module()
    vision.layers = torch.nn.ModuleList(torch.nn.Linear(4, 4) for _ in range(3))
    text = torch.nn.Module()
    text.layers = torch.nn.ModuleList(torch.nn.Linear(64, 64) for _ in range(5))
    # registered vision-first, so `named_modules` yields it first
    model.vision_tower = vision
    model.language_model = text

    name, layers = ddi_finetune._decoder_layers(model)
    assert name == "language_model.layers"
    assert len(layers) == 5


# ---------------------------------------------------------------------------
# Divergence guards. Without these a single gradient spike takes the weights to
# NaN, training continues for thousands of steps, and the run saves and uploads a
# checkpoint that scores zero everywhere while reporting success (2026-08-07, 7B).
# ---------------------------------------------------------------------------


def _one_param_optimizer(value: float, grad: float):
    p = torch.nn.Parameter(torch.tensor([value]))
    p.grad = torch.tensor([grad])
    return p, torch.optim.SGD([p], lr=0.1)


def test_flush_step_skips_when_gradients_are_not_finite() -> None:
    p, opt = _one_param_optimizer(1.0, float("nan"))
    before = p.detach().clone()
    _lr, stepped = ddi_finetune._flush_optimizer_step(
        opt, warmup_steps=0, base_lr=1e-3, step_before=0, params=[p], max_grad_norm=1.0
    )
    assert stepped is False
    # the whole point: a poisoned gradient must not reach the weights
    assert torch.equal(p.detach(), before)
    assert torch.isfinite(p).all()


def test_flush_step_takes_a_normal_step() -> None:
    p, opt = _one_param_optimizer(1.0, 0.5)
    _lr, stepped = ddi_finetune._flush_optimizer_step(
        opt, warmup_steps=0, base_lr=1e-3, step_before=0, params=[p], max_grad_norm=1.0
    )
    assert stepped is True
    assert not torch.equal(p.detach(), torch.tensor([1.0]))
    assert torch.isfinite(p).all()


def test_flush_step_clips_large_gradients() -> None:
    """A big-but-finite gradient is clipped, not skipped."""
    p, opt = _one_param_optimizer(0.0, 1000.0)
    _lr, stepped = ddi_finetune._flush_optimizer_step(
        opt, warmup_steps=0, base_lr=1e-3, step_before=0, params=[p], max_grad_norm=1.0
    )
    assert stepped is True
    # clipped to norm 1.0, so SGD(lr=0.1) moves it by at most 0.1
    assert abs(float(p.detach())) <= 0.1 + 1e-6


def test_assert_finite_weights_raises_on_nan() -> None:
    m = torch.nn.Linear(3, 3)
    with torch.no_grad():
        m.weight[0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="not finite"):
        ddi_finetune._assert_finite_weights(m, "test")


def test_assert_finite_weights_passes_on_a_healthy_model() -> None:
    ddi_finetune._assert_finite_weights(torch.nn.Linear(3, 3), "test")


# ---------------------------------------------------------------------------
# Integration: real 2-step training on tiny random Llama.
# ---------------------------------------------------------------------------


_TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
_INTEG = pytest.mark.skipif(
    os.environ.get("DDI_INTEGRATION") != "1",
    reason="integration test; set DDI_INTEGRATION=1 to enable",
)


def _read_rows(store: Path) -> list[dict]:
    return [json.loads(line) for line in store.read_text().splitlines() if line.strip()]


@_INTEG
def test_finetune_smoke_writes_checkpoint_and_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """2-step smoke: checkpoint dir + at least one loss row + per-epoch pplx rows."""
    from ddi.data import register_loader
    from ddi.provenance import capture

    ds_name = "tiny_finetune_ds"

    def _loader(split, seed):  # noqa: ARG001
        for i in range(6):
            yield {
                "id": f"r{i}",
                "info_contexts": ["alpha beta", "gamma delta"],
                "dialog": [
                    {"role": "user", "content": f"question {i}?"},
                    {"role": "assistant", "content": f"answer{i}"},
                ],
                "possible_answers": [f"answer{i}"],
            }

    try:
        register_loader(ds_name)(_loader)
    except ValueError:
        pass  # already registered

    output_dir = tmp_path / "ckpt"
    store = tmp_path / "runs.jsonl"
    prov = capture(config_hash="test", seed=0, model_revision=_TINY, dataset_revision=ds_name)

    ddi_finetune.finetune(
        _TINY,
        [ds_name],
        output_dir,
        seed=0,
        max_records=4,
        held_out_frac=0.5,   # 2 test / 2 train
        epochs=1,
        lr=1e-4,
        grad_accum_steps=1,
        log_every=1,
        provenance=prov,
        store_path=store,
    )

    # Checkpoint dir created + populated with HF-loadable files.
    assert output_dir.is_dir()
    files = {p.name for p in output_dir.iterdir()}
    assert "config.json" in files
    # At least one of these weight formats must be present (varies by transformers version).
    assert any(name in files for name in ("pytorch_model.bin", "model.safetensors")) or any(
        name.startswith("model") for name in files
    )

    rows = _read_rows(store)
    assert rows, "finetune wrote no rows"
    metrics = {r["metric"] for r in rows}
    assert "kl_loss" in metrics
    assert "lr" in metrics
    # Both baseline and post-epoch pplx rows should be present.
    assert "perplexity_student_ddi_qf_pre" in metrics
    assert "perplexity_teacher_full" in metrics
    assert "perplexity_student_ddi_qf_post" in metrics
    # Every row carries the finetune: prefix.
    assert all(r["run_id"].startswith("finetune:") for r in rows)
