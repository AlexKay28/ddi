"""Tests for the report generator (issue #8)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ddi import report


@pytest.fixture(autouse=True)
def _reset_renderers():
    saved = dict(report._RENDERERS)
    report._RENDERERS.clear()
    yield
    report._RENDERERS.clear()
    report._RENDERERS.update(saved)


def _write_store(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _prov(**overrides) -> dict:
    base = {
        "git_sha": "abc123",
        "dirty": False,
        "config_hash": "cfg",
        "seed": 0,
        "model_revision": "m@rev",
        "dataset_revision": "d@rev",
        "gpu_model": "cpu",
        "driver_version": "unknown",
        "cuda_version": "cpu",
        "torch_version": "2.4.0",
        "timestamp_utc": "2026-07-27T14:33:12+00:00",
    }
    base.update(overrides)
    return base


# ---- registry + dispatch ------------------------------------------------------------------


def test_register_and_dispatch_table(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    _write_store(store, [])

    @report.register("X1")
    def _x1(store_path: Path) -> str:
        return f"rendered from {store_path.name}"

    assert "X1" in report.list_reports()
    assert report.render("X1", store) == "rendered from runs.jsonl"


def test_list_reports_sorted() -> None:
    report.register("T9")(lambda s: "")
    report.register("T2")(lambda s: "")
    report.register("F3")(lambda s, o: None)
    assert report.list_reports() == ["F3", "T2", "T9"]


def test_duplicate_registration_rejected() -> None:
    report.register("T7")(lambda s: "")
    with pytest.raises(ValueError, match="already registered"):
        report.register("T7")(lambda s: "")


def test_unknown_report_raises_key_error(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    _write_store(store, [])
    with pytest.raises(KeyError, match="unknown report"):
        report.render("does_not_exist", store)


def test_figure_requires_output(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    _write_store(store, [])

    @report.register("F9")
    def _f9(store_path: Path, output: Path) -> None:
        output.write_bytes(b"x")

    with pytest.raises(ValueError, match="requires an output path"):
        report.render("F9", store)


# ---- store reader --------------------------------------------------------------------------


def test_iter_rows_preserves_file_order(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    rows = [
        {"run_id": "r1", "metric": "em", "value": 0.5, "provenance": _prov()},
        {"run_id": "r2", "metric": "f1", "value": 0.7, "provenance": _prov()},
        {"run_id": "r1", "metric": "loss", "value": 0.3, "provenance": _prov()},
    ]
    _write_store(store, rows)

    got = list(report.iter_rows(store))
    assert got == rows


def test_iter_rows_skips_blank_lines(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    row = {"run_id": "r", "metric": "m", "value": 1, "provenance": _prov()}
    with open(store, "w", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
        f.write("\n")
        f.write(json.dumps(row) + "\n")

    got = list(report.iter_rows(store))
    assert len(got) == 2


def test_filter_rows_by_metric(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    rows = [
        {"run_id": "r1", "metric": "em", "value": 0.5, "provenance": _prov()},
        {"run_id": "r1", "metric": "f1", "value": 0.6, "provenance": _prov()},
        {"run_id": "r2", "metric": "em", "value": 0.7, "provenance": _prov()},
    ]
    _write_store(store, rows)

    got = list(report.filter_rows(report.iter_rows(store), metric="em"))
    assert [r["value"] for r in got] == [0.5, 0.7]


def test_filter_rows_by_run_id(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    rows = [
        {"run_id": "r1", "metric": "em", "value": 0.5, "provenance": _prov()},
        {"run_id": "r2", "metric": "em", "value": 0.7, "provenance": _prov()},
        {"run_id": "r2", "metric": "f1", "value": 0.6, "provenance": _prov()},
    ]
    _write_store(store, rows)

    got = list(report.filter_rows(report.iter_rows(store), run_id="r2"))
    assert [r["metric"] for r in got] == ["em", "f1"]


def test_filter_rows_by_provenance_subset(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    rows = [
        {"run_id": "r1", "metric": "em", "value": 0.5, "provenance": _prov(git_sha="aaa")},
        {"run_id": "r2", "metric": "em", "value": 0.6, "provenance": _prov(git_sha="bbb")},
        {
            "run_id": "r3",
            "metric": "em",
            "value": 0.7,
            "provenance": _prov(git_sha="aaa", seed=1),
        },
    ]
    _write_store(store, rows)

    got = list(report.filter_rows(report.iter_rows(store), provenance={"git_sha": "aaa"}))
    assert [r["run_id"] for r in got] == ["r1", "r3"]

    got = list(
        report.filter_rows(
            report.iter_rows(store), provenance={"git_sha": "aaa", "seed": 1}
        )
    )
    assert [r["run_id"] for r in got] == ["r3"]


def test_filter_rows_combined(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    rows = [
        {"run_id": "r1", "metric": "em", "value": 0.5, "provenance": _prov(seed=0)},
        {"run_id": "r1", "metric": "em", "value": 0.6, "provenance": _prov(seed=1)},
        {"run_id": "r1", "metric": "f1", "value": 0.7, "provenance": _prov(seed=0)},
    ]
    _write_store(store, rows)

    got = list(
        report.filter_rows(
            report.iter_rows(store),
            metric="em",
            run_id="r1",
            provenance={"seed": 0},
        )
    )
    assert len(got) == 1
    assert got[0]["value"] == 0.5


# ---- helpers -------------------------------------------------------------------------------


def test_median_with_iqr_odd() -> None:
    assert report.median_with_iqr([1, 2, 3, 4, 5]) == (3.0, 2.0, 4.0)


def test_median_with_iqr_unsorted_input() -> None:
    assert report.median_with_iqr([5, 1, 3, 2, 4]) == (3.0, 2.0, 4.0)


def test_median_with_iqr_single_value() -> None:
    assert report.median_with_iqr([7.5]) == (7.5, 7.5, 7.5)


def test_median_with_iqr_empty_raises() -> None:
    with pytest.raises(ValueError):
        report.median_with_iqr([])


def test_group_by_single_key() -> None:
    rows = [
        {"run_id": "r1", "metric": "em", "value": 0.5},
        {"run_id": "r1", "metric": "f1", "value": 0.6},
        {"run_id": "r2", "metric": "em", "value": 0.7},
    ]
    got = report.group_by(rows, ["run_id"])
    assert got == {
        ("r1",): [rows[0], rows[1]],
        ("r2",): [rows[2]],
    }


def test_group_by_dotted_path() -> None:
    rows = [
        {"run_id": "r1", "metric": "em", "provenance": {"seed": 0}},
        {"run_id": "r2", "metric": "em", "provenance": {"seed": 1}},
        {"run_id": "r3", "metric": "em", "provenance": {"seed": 0}},
    ]
    got = report.group_by(rows, ["provenance.seed"])
    assert got == {
        (0,): [rows[0], rows[2]],
        (1,): [rows[1]],
    }


def test_group_by_multiple_keys() -> None:
    rows = [
        {"metric": "em", "provenance": {"seed": 0}},
        {"metric": "em", "provenance": {"seed": 1}},
        {"metric": "f1", "provenance": {"seed": 0}},
    ]
    got = report.group_by(rows, ["metric", "provenance.seed"])
    assert got == {
        ("em", 0): [rows[0]],
        ("em", 1): [rows[1]],
        ("f1", 0): [rows[2]],
    }


# ---- idempotency ---------------------------------------------------------------------------


def test_render_table_is_byte_identical(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    rows = [
        {"run_id": "r2", "metric": "f1", "value": 0.7, "provenance": _prov()},
        {"run_id": "r1", "metric": "em", "value": 0.5, "provenance": _prov()},
        {"run_id": "r1", "metric": "f1", "value": 0.6, "provenance": _prov()},
        {"run_id": "r2", "metric": "em", "value": 0.8, "provenance": _prov()},
    ]
    _write_store(store, rows)

    @report.register("Xdump")
    def _xdump(store_path: Path) -> str:
        got = list(report.iter_rows(store_path))
        got.sort(key=lambda r: (r["run_id"], r["metric"]))
        lines = [f"{r['run_id']}\t{r['metric']}\t{r['value']}" for r in got]
        return "\n".join(lines) + "\n"

    first = report.render("Xdump", store)
    second = report.render("Xdump", store)
    assert first == second
    assert first.encode("utf-8") == second.encode("utf-8")


# ---- figure rendering without matplotlib --------------------------------------------------


def test_figure_renderer_no_matplotlib_import(tmp_path: Path) -> None:
    """Figure renderers must not force a matplotlib import at module load time."""
    store = tmp_path / "runs.jsonl"
    _write_store(store, [])
    output = tmp_path / "fig.bin"
    payload = b"\x89PNG-ish-fake-bytes\x00"

    @report.register("Ffake")
    def _ffake(store_path: Path, out: Path) -> None:
        out.write_bytes(payload)

    result = report.render("Ffake", store, output=output)
    assert result is None
    assert output.exists()
    assert output.read_bytes() == payload
