"""Tests for `ddi.audit_cli` (issue #10, Phase 1.2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ddi import audit_cli, hop_labels
from ddi.data import manifest


@pytest.fixture(autouse=True)
def _isolate_heuristics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hop_labels, "_HEURISTICS", dict(hop_labels._HEURISTICS))


@pytest.fixture(autouse=True)
def _isolate_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "splits"
    monkeypatch.setattr(manifest, "DEFAULT_ROOT", root)
    return root


def test_sample_writes_expected_shape(tmp_path: Path) -> None:
    out = tmp_path / "audit.jsonl"
    rc = audit_cli.main(
        [
            "sample",
            "--dataset", "synthetic_qa",
            "--split", "val",
            "--seed", "0",
            "--n", "5",
            "--out", str(out),
        ]
    )
    assert rc == 0
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 5
    for line in lines:
        row = json.loads(line)
        assert set(row) == {"id", "dialog", "info_contexts", "possible_answers", "hop_type"}
        assert row["hop_type"] is None
        assert isinstance(row["dialog"], str) and row["dialog"]
        assert isinstance(row["info_contexts"], list) and row["info_contexts"]
        assert isinstance(row["possible_answers"], list) and row["possible_answers"]


def test_score_prints_agreement(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # synthetic_qa's native hop_type is "single" for every record → auto = "single" everywhere
    # (no heuristic registered). Build 4 rows: 2 correct, 1 wrong, 1 unfilled.
    sampled = hop_labels.sample_for_audit("synthetic_qa", "val", seed=0, n=4)
    ids = [r["id"] for r in sampled]

    human_rows = [
        {"id": ids[0], "hop_type": "single"},
        {"id": ids[1], "hop_type": "single"},
        {"id": ids[2], "hop_type": "multi"},
        {"id": ids[3], "hop_type": None},
    ]
    human_path = tmp_path / "human.jsonl"
    human_path.write_text("\n".join(json.dumps(r) for r in human_rows) + "\n")

    rc = audit_cli.main(
        [
            "score",
            "--dataset", "synthetic_qa",
            "--split", "val",
            "--seed", "0",
            "--n", "4",
            "--human", str(human_path),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "n_reviewed: 3" in out
    assert "66.7%" in out
    assert "2 / 3" in out
    assert "skipped (unfilled human labels): 1" in out
    assert "disagreements" in out
    assert ids[2] in out
