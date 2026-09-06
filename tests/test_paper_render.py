from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from paper.render import (
    DEFAULT_EVIDENCE,
    DEFAULT_OUTPUT,
    GENERATED_FILES,
    EvidenceError,
    load_evidence,
    render,
    verify_sources,
)


def test_checked_in_artifacts_match_renderer() -> None:
    evidence = load_evidence()
    rendered = render(evidence)

    assert tuple(rendered) == GENERATED_FILES
    for name, content in rendered.items():
        assert (DEFAULT_OUTPUT / name).read_text(encoding="utf-8") == content


def test_all_available_sources_match_preserved_hashes() -> None:
    assert verify_sources(load_evidence()) == []


def test_missing_required_source_fails(tmp_path: Path) -> None:
    evidence = load_evidence()
    evidence["sources"] = {
        "missing": {
            "path": "missing.jsonl",
            "sha256": "0" * 64,
            "bytes": 1,
            "required": True,
        }
    }

    with pytest.raises(EvidenceError, match="source is missing"):
        verify_sources(evidence, tmp_path)


def test_available_historical_source_hash_mismatch_fails(tmp_path: Path) -> None:
    (tmp_path / "raw.jsonl").write_text("changed\n", encoding="utf-8")
    evidence = {
        "sources": {
            "raw": {
                "path": "raw.jsonl",
                "sha256": "0" * 64,
                "bytes": 8,
                "required": False,
            }
        }
    }

    with pytest.raises(EvidenceError, match="SHA-256 mismatch"):
        verify_sources(evidence, tmp_path)


def test_malformed_evidence_fails_loudly(tmp_path: Path) -> None:
    evidence = json.loads(DEFAULT_EVIDENCE.read_text(encoding="utf-8"))
    del evidence["artifacts"]["T5"]["rows"]
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")

    with pytest.raises(EvidenceError, match=r"artifacts.T5.rows"):
        load_evidence(path)


def test_incomplete_seed_evidence_fails_loudly() -> None:
    evidence = copy.deepcopy(load_evidence())
    evidence["artifacts"]["F5"]["rows"][0][1] = [1.0, 2.0]

    with pytest.raises(EvidenceError, match="needs three seed values"):
        render(evidence)
