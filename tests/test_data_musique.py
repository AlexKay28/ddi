"""Tests for the MuSiQue-ans loader (issue #44, Phase 1.1).

Unit tests run offline. The integration tests hit HuggingFace and are gated on
`DDI_INTEGRATION=1` so `pytest` in CI stays hermetic.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ddi.data import list_datasets, load, manifest, musique


@pytest.fixture(autouse=True)
def manifest_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the manifest root away from the real results/splits/ for every test."""
    root = tmp_path / "splits"
    monkeypatch.setattr(manifest, "DEFAULT_ROOT", root)
    return root


def test_registered() -> None:
    assert "musique" in list_datasets()


_INTEGRATION = pytest.mark.skipif(
    os.environ.get("DDI_INTEGRATION") != "1",
    reason="MuSiQue loader hits HuggingFace; opt in with DDI_INTEGRATION=1",
)


@_INTEGRATION
def test_common_record_shape() -> None:
    stream = load("musique", "val", seed=0)
    rec = next(iter(stream))

    assert isinstance(rec["id"], str) and rec["id"]
    assert isinstance(rec["info_contexts"], list)
    assert all(isinstance(d, str) for d in rec["info_contexts"])
    assert len(rec["dialog"]) == 2
    assert rec["dialog"][0]["role"] == "user"
    assert rec["dialog"][1]["role"] == "assistant"
    assert isinstance(rec["possible_answers"], list) and rec["possible_answers"]
    assert all(isinstance(a, str) for a in rec["possible_answers"])
    # First possible_answer is always the canonical answer, aliases follow.
    assert rec["possible_answers"][0] == rec["dialog"][1]["content"]
    assert rec["hop_type"] == "multi"
    assert rec["gold_position"] is None
    assert isinstance(rec["meta"], dict)
    assert "n_hops" in rec["meta"]
    assert "supporting_idxs" in rec["meta"]


@_INTEGRATION
def test_hop_type_is_multi() -> None:
    for rec in load("musique", "val", seed=0):
        assert rec["hop_type"] == "multi"


@_INTEGRATION
def test_n_hops_in_range() -> None:
    # MuSiQue's whole point: every question decomposes into 2, 3, or 4 sub-questions.
    for rec in load("musique", "val", seed=0):
        assert rec["meta"]["n_hops"] in {2, 3, 4}, (
            f"{rec['id']}: n_hops={rec['meta']['n_hops']!r} outside {{2, 3, 4}}"
        )


@_INTEGRATION
def test_paragraph_count() -> None:
    # Canonical MuSiQue ships 20 paragraphs per record, but a handful of edge cases in
    # mirror-uploads have dropped a few. Assert >= 10 to leave headroom without letting a
    # regression to single-doc records slide through.
    for rec in load("musique", "val", seed=0):
        assert len(rec["info_contexts"]) >= 10, (
            f"{rec['id']}: only {len(rec['info_contexts'])} paragraphs"
        )


@_INTEGRATION
def test_deterministic() -> None:
    a = [r["id"] for r in load("musique", "val", seed=0)]
    b = [r["id"] for r in load("musique", "val", seed=0)]
    assert a == b


# ---------------------------------------------------------------------------
# offline-safe: cache short-circuits the HF fetch on the second call
# ---------------------------------------------------------------------------


def test_second_call_hits_cache_not_hf(monkeypatch: pytest.MonkeyPatch) -> None:
    """Warm cache must bypass ``_load_hf_split`` entirely — offline-container safe."""
    from ddi.data.records_cache import write_split

    fake = [{"id": "mq-0", "info_contexts": [], "meta": {"n_hops": 2}}]
    for s in ("train", "val", "test"):
        write_split("musique", 0, s, fake)

    def boom(*_a: object, **_kw: object) -> list[dict]:
        raise AssertionError("HF fetch invoked despite warm cache")

    monkeypatch.setattr(musique, "_load_hf_split", boom)

    got = list(load("musique", "val", seed=0))
    assert got == fake
