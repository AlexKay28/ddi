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
