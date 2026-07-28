"""Tests for ddi/reports/_provenance.py — footer helper.

Renderer-side coverage that the footer marker actually lands in each table's
output lives in each `test_reports_*` file next to the assertions on that
table's data columns; here we only exercise the helper directly.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pytest

from ddi.reports._provenance import FOOTER_MARKER, render_provenance_footer


def _mk_row(run_id: str, model: str, seed: int) -> dict:
    return {
        "run_id": run_id,
        "metric": "f1",
        "value": 0.5,
        "provenance": {"model_revision": model, "seed": seed},
    }


def test_footer_shape(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    store.write_text('{"run_id": "x"}\n', encoding="utf-8")
    rows = [_mk_row("phase2_train_free:m:d:v:seed=0:sha:ex_0", "Qwen/Qwen2.5-7B", 0)]
    out = render_provenance_footer(store, rows)

    assert out.startswith(FOOTER_MARKER + "\n")
    assert "**Generated:**" in out
    assert "git `" in out
    assert "store `" in out
    assert "(sha256 `" in out
    assert "**Backing rows:**" in out
    assert "**Seeds:**" in out
    assert out.endswith("\n")


def test_footer_timestamp_iso8601(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    store.write_text("{}\n", encoding="utf-8")
    out = render_provenance_footer(store, [])

    m = re.search(r"\*\*Generated:\*\* (\S+)", out)
    assert m is not None, out
    ts = m.group(1)
    # Normalize `Z` back to `+00:00` so fromisoformat can parse it.
    parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None


def test_footer_git_sha_length(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    store.write_text("{}\n", encoding="utf-8")
    out = render_provenance_footer(store, [])

    m = re.search(r"git `([^`]+)`", out)
    assert m is not None, out
    sha = m.group(1)
    # Inside the repo -> 40-char hex. Outside -> the sentinel. Both acceptable.
    assert sha == "not-a-git-repo" or re.fullmatch(r"[0-9a-f]{40}", sha)


def test_footer_store_hash_length(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    store.write_text("some bytes\n", encoding="utf-8")
    out = render_provenance_footer(store, [])

    m = re.search(r"sha256 `([^`]+)`", out)
    assert m is not None, out
    h = m.group(1)
    assert re.fullmatch(r"[0-9a-f]{8}", h), h


def test_footer_missing_store_marks_hash(tmp_path: Path) -> None:
    # Not created — helper must degrade gracefully rather than raise.
    store = tmp_path / "does_not_exist.jsonl"
    out = render_provenance_footer(store, [])

    assert "sha256 `missing`" in out


def test_footer_counts_correct(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    store.write_text("x\n", encoding="utf-8")
    rows = [
        _mk_row("phase2_train_free:a:b:c:seed=0:sha:1", "Qwen/A", 0),
        _mk_row("phase2_train_free:a:b:c:seed=0:sha:2", "Qwen/A", 0),
        _mk_row("phase2_train_free:a:b:d:seed=1:sha:3", "Qwen/A", 1),
    ]
    out = render_provenance_footer(store, rows)

    # 3 backing rows, 3 distinct run_ids (all different suffixes), 1 model, 2 seeds.
    assert "**Backing rows:** 3 rows across 3 run_ids" in out
    assert "models: Qwen/A" in out
    assert "**Seeds:** 0, 1" in out


def test_footer_handles_missing_provenance(tmp_path: Path) -> None:
    # A row without a `provenance` key must not crash the footer.
    store = tmp_path / "runs.jsonl"
    store.write_text("x\n", encoding="utf-8")
    rows = [{"run_id": "phase3_timing:m:n:s:sha", "metric": "ttft", "value": 1.0}]
    out = render_provenance_footer(store, rows)

    assert "**Backing rows:** 1 rows across 1 run_ids" in out
    assert "models: unknown" in out
    assert "**Seeds:** unknown" in out


def test_footer_empty_rows(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    store.write_text("x\n", encoding="utf-8")
    out = render_provenance_footer(store, [])

    assert "**Backing rows:** 0 rows across 0 run_ids" in out
    # Rendering with zero rows is legal — callers may want to stamp an empty table.
