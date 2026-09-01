"""Tests for the append-only results store (issue #7)."""

from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

import pytest

from ddi.predictions import append as append_prediction
from ddi.provenance import Provenance
from ddi.store import append


@pytest.fixture
def prov() -> Provenance:
    return Provenance(
        git_sha="0" * 40,
        dirty=False,
        config_hash="cfg-hash",
        seed=7,
        model_revision="model@rev",
        dataset_revision="data@rev",
        gpu_model="cpu",
        driver_version="unknown",
        cuda_version="cpu",
        torch_version="2.4.0",
        timestamp_utc="2026-07-27T14:33:12+00:00",
    )


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines(keepends=True)


def test_append_writes_one_line(tmp_path: Path, prov: Provenance) -> None:
    store = tmp_path / "runs.jsonl"
    append("run-1", "em", 0.75, prov, store=store)

    lines = _read_lines(store)
    assert len(lines) == 1
    assert lines[0].endswith("\n")

    row = json.loads(lines[0])
    assert row["run_id"] == "run-1"
    assert row["metric"] == "em"
    assert row["value"] == 0.75
    assert isinstance(row["provenance"], dict)
    assert row["provenance"]["git_sha"] == prov.git_sha
    assert row["provenance"]["seed"] == 7
    assert set(row["provenance"].keys()) == {
        "git_sha",
        "dirty",
        "config_hash",
        "seed",
        "model_revision",
        "dataset_revision",
        "gpu_model",
        "driver_version",
        "cuda_version",
        "torch_version",
        "timestamp_utc",
    }


def test_two_appends_two_lines(tmp_path: Path, prov: Provenance) -> None:
    store = tmp_path / "runs.jsonl"
    append("run-1", "em", 0.5, prov, store=store)
    append("run-1", "f1", 0.6, prov, store=store)

    lines = _read_lines(store)
    assert len(lines) == 2
    metrics = [json.loads(line)["metric"] for line in lines]
    assert metrics == ["em", "f1"]


def test_append_creates_missing_parent(tmp_path: Path, prov: Provenance) -> None:
    store = tmp_path / "nested" / "deeper" / "runs.jsonl"
    assert not store.parent.exists()
    append("run-1", "em", 0.5, prov, store=store)
    assert store.exists()
    assert len(_read_lines(store)) == 1


@pytest.mark.parametrize("bad", [{"k": 1}, [1, 2, 3], None, 1.5 + 0j])
def test_rejects_unsupported_value_types(tmp_path: Path, prov: Provenance, bad) -> None:
    store = tmp_path / "runs.jsonl"
    with pytest.raises(TypeError):
        append("run-1", "m", bad, prov, store=store)
    assert not store.exists()


def test_accepts_int_float_bool_str(tmp_path: Path, prov: Provenance) -> None:
    store = tmp_path / "runs.jsonl"
    append("r", "int_m", 3, prov, store=store)
    append("r", "float_m", 3.14, prov, store=store)
    append("r", "bool_m", True, prov, store=store)
    append("r", "str_m", "label", prov, store=store)

    values = [json.loads(line)["value"] for line in _read_lines(store)]
    assert values == [3, 3.14, True, "label"]


def test_append_prediction_persists_one_complete_example(tmp_path: Path, prov: Provenance) -> None:
    store = tmp_path / "quality_predictions.jsonl"
    append_prediction(
        "quality:full:example-1",
        "example-1",
        "answer",
        ["answer", "alternate"],
        {"em": 1.0, "f1": 1.0},
        prov,
        store,
    )

    rows = [json.loads(line) for line in _read_lines(store)]
    assert rows == [
        {
            "run_id": "quality:full:example-1",
            "record_id": "example-1",
            "prediction": "answer",
            "possible_answers": ["answer", "alternate"],
            "metrics": {"em": 1.0, "f1": 1.0},
            "provenance": {
                "git_sha": "0" * 40,
                "dirty": False,
                "config_hash": "cfg-hash",
                "seed": 7,
                "model_revision": "model@rev",
                "dataset_revision": "data@rev",
                "gpu_model": "cpu",
                "driver_version": "unknown",
                "cuda_version": "cpu",
                "torch_version": "2.4.0",
                "timestamp_utc": "2026-07-27T14:33:12+00:00",
            },
        }
    ]


def _writer(store_path: str, run_id: str, n: int) -> None:
    prov = Provenance(
        git_sha="0" * 40,
        dirty=False,
        config_hash="cfg",
        seed=0,
        model_revision="m",
        dataset_revision="d",
        gpu_model="cpu",
        driver_version="unknown",
        cuda_version="cpu",
        torch_version="2.4.0",
        timestamp_utc="2026-07-27T14:33:12+00:00",
    )
    for i in range(n):
        append(run_id, "metric", i, prov, store=Path(store_path))


def test_concurrent_writes_do_not_interleave(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    n = 50
    ctx = mp.get_context("fork")
    p1 = ctx.Process(target=_writer, args=(str(store), "worker-a", n))
    p2 = ctx.Process(target=_writer, args=(str(store), "worker-b", n))
    p1.start()
    p2.start()
    p1.join(timeout=30)
    p2.join(timeout=30)
    assert p1.exitcode == 0
    assert p2.exitcode == 0

    lines = _read_lines(store)
    assert len(lines) == 2 * n
    for line in lines:
        row = json.loads(line)
        assert row["metric"] == "metric"
        assert row["run_id"] in {"worker-a", "worker-b"}
        assert isinstance(row["value"], int)

    run_ids = [json.loads(line)["run_id"] for line in lines]
    assert run_ids.count("worker-a") == n
    assert run_ids.count("worker-b") == n
