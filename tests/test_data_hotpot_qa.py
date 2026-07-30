"""Tests for the HotpotQA distractor loader (issue #43)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ddi.data import hotpot_qa, list_datasets, load, manifest

_INTEGRATION = os.environ.get("DDI_INTEGRATION") == "1"
_INTEGRATION_REASON = "HotpotQA loader hits HuggingFace; opt in with DDI_INTEGRATION=1"


@pytest.fixture(autouse=True)
def manifest_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the manifest root away from the real results/splits/ for every test."""
    root = tmp_path / "splits"
    monkeypatch.setattr(manifest, "DEFAULT_ROOT", root)
    return root


def test_registered() -> None:
    assert "hotpot_qa" in list_datasets()


@pytest.mark.skipif(not _INTEGRATION, reason=_INTEGRATION_REASON)
def test_common_record_shape() -> None:
    stream = load("hotpot_qa", "val", seed=0)
    rec = next(iter(stream))

    assert isinstance(rec["id"], str) and rec["id"]

    assert isinstance(rec["info_contexts"], list)
    assert all(isinstance(d, str) for d in rec["info_contexts"])

    assert isinstance(rec["dialog"], list)
    assert len(rec["dialog"]) == 2
    assert rec["dialog"][0]["role"] == "user"
    assert rec["dialog"][1]["role"] == "assistant"
    assert all(
        isinstance(m, dict) and set(m) == {"role", "content"} for m in rec["dialog"]
    )
    assert all(isinstance(m["content"], str) for m in rec["dialog"])

    assert isinstance(rec["possible_answers"], list)
    assert rec["possible_answers"]
    assert all(isinstance(a, str) for a in rec["possible_answers"])

    assert rec["hop_type"] == "multi"
    assert rec["gold_position"] is None

    assert isinstance(rec["meta"], dict)
    assert rec["meta"]["type"] in {"bridge", "comparison"}
    assert rec["meta"]["level"] in {"easy", "medium", "hard"}
    assert isinstance(rec["meta"]["supporting_titles"], list)
    assert all(isinstance(t, str) for t in rec["meta"]["supporting_titles"])


@pytest.mark.skipif(not _INTEGRATION, reason=_INTEGRATION_REASON)
def test_hop_type_is_multi() -> None:
    for rec in load("hotpot_qa", "val", seed=0):
        assert rec["hop_type"] == "multi"


@pytest.mark.skipif(not _INTEGRATION, reason=_INTEGRATION_REASON)
def test_info_contexts_length() -> None:
    for rec in load("hotpot_qa", "val", seed=0):
        assert len(rec["info_contexts"]) == 10


@pytest.mark.skipif(not _INTEGRATION, reason=_INTEGRATION_REASON)
def test_deterministic() -> None:
    a = [r["id"] for r in load("hotpot_qa", "val", seed=0)]
    b = [r["id"] for r in load("hotpot_qa", "val", seed=0)]
    assert a == b


# ---------------------------------------------------------------------------
# offline-safe: cache short-circuits the HF fetch on the second call
# ---------------------------------------------------------------------------


def test_second_call_hits_cache_not_hf(monkeypatch: pytest.MonkeyPatch) -> None:
    """Once the cache is warm, load() must not call _load_hf_split again.

    The first call is faked by pre-populating the record cache; if the loader
    still reached for HF it would trip ``boom`` and fail loudly.
    """
    from ddi.data.records_cache import write_split

    # Pre-populate every split so has_cached() returns True.
    fake = [{"id": "hq-0", "info_contexts": [], "meta": {}}]
    for s in ("train", "val", "test"):
        write_split("hotpot_qa", 0, s, fake)

    def boom(*_a: object, **_kw: object) -> list[dict]:
        raise AssertionError("HF fetch invoked despite warm cache")

    monkeypatch.setattr(hotpot_qa, "_load_hf_split", boom)

    got = list(load("hotpot_qa", "val", seed=0))
    assert got == fake
