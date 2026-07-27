"""Tests for `ddi.hop_labels` (issue #10, Phase 1.2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ddi import hop_labels
from ddi.data import manifest


@pytest.fixture(autouse=True)
def _isolate_heuristics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Snapshot + restore the heuristics registry so tests don't leak."""
    monkeypatch.setattr(hop_labels, "_HEURISTICS", dict(hop_labels._HEURISTICS))


@pytest.fixture(autouse=True)
def _isolate_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the split manifest root so tests can call the real synthetic loader freely."""
    root = tmp_path / "splits"
    monkeypatch.setattr(manifest, "DEFAULT_ROOT", root)
    return root


# ---------------------------------------------------------------------------
# hop_type
# ---------------------------------------------------------------------------


def test_hop_type_defaults_unknown() -> None:
    assert hop_labels.hop_type({}) == "unknown"
    assert hop_labels.hop_type({"hop_type": "multi"}) == "multi"
    assert hop_labels.hop_type({"hop_type": "single"}) == "single"
    # Unrecognised values collapse to "unknown" rather than propagating garbage downstream.
    assert hop_labels.hop_type({"hop_type": "triple"}) == "unknown"


# ---------------------------------------------------------------------------
# register_heuristic / apply_heuristic
# ---------------------------------------------------------------------------


def test_register_and_apply_heuristic() -> None:
    @hop_labels.register_heuristic("fake_ds")
    def _label(_rec: dict) -> hop_labels.Hop:
        return "multi"

    records = [
        {"id": "a", "hop_type": "unknown"},
        {"id": "b", "hop_type": "single"},
        {"id": "c", "hop_type": "multi"},
        {"id": "d"},  # no hop_type field at all → treated as unknown
    ]
    out = list(hop_labels.apply_heuristic("fake_ds", records))
    assert [r["hop_type"] for r in out] == ["multi", "single", "multi", "multi"]

    # No heuristic registered → records pass through unchanged.
    out_other = list(hop_labels.apply_heuristic("no_heuristic_ds", records))
    assert [r.get("hop_type") for r in out_other] == ["unknown", "single", "multi", None]


def test_register_duplicate_raises() -> None:
    hop_labels.register_heuristic("dup_ds")(lambda _r: "single")
    with pytest.raises(ValueError, match="already registered"):
        hop_labels.register_heuristic("dup_ds")(lambda _r: "multi")


def test_apply_heuristic_yields_copies() -> None:
    @hop_labels.register_heuristic("copy_ds")
    def _label(_rec: dict) -> hop_labels.Hop:
        return "multi"

    source = [{"id": "a", "hop_type": "unknown"}]
    out = list(hop_labels.apply_heuristic("copy_ds", source))
    out[0]["hop_type"] = "single"
    out[0]["extra"] = "mutated"
    assert source[0] == {"id": "a", "hop_type": "unknown"}


# ---------------------------------------------------------------------------
# sample_for_audit
# ---------------------------------------------------------------------------


def test_sample_for_audit_deterministic() -> None:
    a = hop_labels.sample_for_audit("synthetic_qa", "val", seed=0, n=5)
    b = hop_labels.sample_for_audit("synthetic_qa", "val", seed=0, n=5)
    assert [r["id"] for r in a] == [r["id"] for r in b]
    ids = [r["id"] for r in a]
    assert ids == sorted(ids)


def test_sample_for_audit_n_greater_than_available() -> None:
    # synthetic_qa's val split has 20 records.
    got = hop_labels.sample_for_audit("synthetic_qa", "val", seed=0, n=10_000)
    assert len(got) == 20
    ids = [r["id"] for r in got]
    assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# load_audit
# ---------------------------------------------------------------------------


def test_load_audit(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    rows = [
        {"id": "a", "hop_type": "single"},
        {"id": "b", "hop_type": "multi"},
        {"id": "c", "hop_type": None},  # unfilled — skip
        {"id": "d"},  # missing hop_type entirely — skip
        {"id": "e", "hop_type": "unknown"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    loaded = hop_labels.load_audit(path)
    assert loaded == {"a": "single", "b": "multi", "e": "unknown"}


# ---------------------------------------------------------------------------
# agreement_rate
# ---------------------------------------------------------------------------


def test_agreement_rate_basic() -> None:
    auto = {"a": "single", "b": "multi", "c": "single"}
    human = {"a": "single", "b": "multi", "c": "multi"}
    assert hop_labels.agreement_rate(auto, human) == pytest.approx(2 / 3)


def test_agreement_rate_empty_human_raises() -> None:
    with pytest.raises(ValueError):
        hop_labels.agreement_rate({"a": "single"}, {})
