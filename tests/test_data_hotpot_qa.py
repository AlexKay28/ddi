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
    # `distractor` pads each question to 10 paragraphs (2 gold + 8 distractors) — but a
    # small upstream tail ships fewer, and one such record is enough to fail an
    # ==10-for-every-record assertion. Measured on val/seed0: 3668 of 3702 have exactly
    # 10, the remainder spread over 2..9. Assert the invariants instead: never more than
    # the distractor cap, never fewer than the two gold docs multi-hop requires, and 10
    # still overwhelmingly the norm.
    counts = [len(rec["info_contexts"]) for rec in load("hotpot_qa", "val", seed=0)]
    assert counts
    assert all(2 <= n <= 10 for n in counts), f"out of range: {sorted(set(counts))}"
    assert sum(n == 10 for n in counts) / len(counts) > 0.95


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


# --- DDI_HOTPOT_N_PARAGRAPHS: the document-count axis ---------------------------------
#
# These are pure functions over lists, so unlike the rest of this file they need neither
# HuggingFace nor a GPU. They had no coverage at all until 2026-08-06, while rows 20/23/25
# — nine GPU runs across three model sizes — read their output as *the* document-count
# axis. A wrong selection here does not crash; it returns a plausible number, which is the
# failure mode this ticket exists to eliminate.


def _set_n(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("DDI_HOTPOT_N_PARAGRAPHS", raising=False)
    else:
        monkeypatch.setenv("DDI_HOTPOT_N_PARAGRAPHS", str(value))


def test_n_paragraphs_unset_means_all(monkeypatch) -> None:
    _set_n(monkeypatch, None)
    assert hotpot_qa.n_paragraphs() is None


def test_n_paragraphs_parses_and_validates(monkeypatch) -> None:
    _set_n(monkeypatch, 5)
    assert hotpot_qa.n_paragraphs() == 5

    _set_n(monkeypatch, "not-a-number")
    with pytest.raises(ValueError, match="must be an integer"):
        hotpot_qa.n_paragraphs()

    # Below 2 the DDI mask family collapses to plain `full` (ddi/masks.py), so the record
    # would measure nothing at all. Failing loudly beats silently running a no-op sweep.
    _set_n(monkeypatch, 1)
    with pytest.raises(ValueError, match="must be >= 2"):
        hotpot_qa.n_paragraphs()


def test_select_keeps_everything_when_unset_or_under_limit(monkeypatch) -> None:
    ctx = ["a", "b", "c"]
    titles = ["A", "B", "C"]

    _set_n(monkeypatch, None)
    assert hotpot_qa._select_paragraphs(ctx, titles, ["A"]) == [0, 1, 2]

    _set_n(monkeypatch, 10)
    assert hotpot_qa._select_paragraphs(ctx, titles, ["A"]) == [0, 1, 2]


def test_select_keeps_every_supporting_paragraph(monkeypatch) -> None:
    """Gold must survive truncation, or the record silently becomes unanswerable.

    That is the confound this axis exists to avoid: an unanswerable record looks exactly
    like a document-count effect while actually being an answerability effect.
    """
    ctx = [f"p{i}" for i in range(10)]
    titles = [f"T{i}" for i in range(10)]
    supporting = ["T3", "T7"]

    _set_n(monkeypatch, 5)
    keep = hotpot_qa._select_paragraphs(ctx, titles, supporting)

    assert len(keep) == 5
    assert 3 in keep and 7 in keep
    assert keep == sorted(keep), "original relative order must be preserved"


def test_select_at_two_is_gold_only(monkeypatch) -> None:
    """At the bottom of the sweep every kept paragraph is gold, by construction.

    Worth pinning because it changes what the sweep measures: the gold:distractor ratio
    moves from 2:0 at n=2 to 2:8 at n=10, so the curve is 'fewer distractors', not merely
    'fewer documents'. For multi-hop that also makes n=2 the *most* stringent test of the
    mask -- both documents are needed and neither may see the other.
    """
    ctx = [f"p{i}" for i in range(10)]
    titles = [f"T{i}" for i in range(10)]

    _set_n(monkeypatch, 2)
    assert hotpot_qa._select_paragraphs(ctx, titles, ["T4", "T6"]) == [4, 6]


def test_select_truncates_gold_only_when_gold_exceeds_limit(monkeypatch) -> None:
    ctx = [f"p{i}" for i in range(6)]
    titles = [f"T{i}" for i in range(6)]

    _set_n(monkeypatch, 2)
    keep = hotpot_qa._select_paragraphs(ctx, titles, ["T0", "T1", "T2", "T3"])
    assert len(keep) == 2
    assert set(keep).issubset({0, 1, 2, 3}), "distractors must not displace gold"


def test_select_tolerates_duplicate_supporting_titles(monkeypatch) -> None:
    """HotpotQA lists a supporting title once per sentence, so repeats are normal."""
    ctx = [f"p{i}" for i in range(8)]
    titles = [f"T{i}" for i in range(8)]

    _set_n(monkeypatch, 4)
    keep = hotpot_qa._select_paragraphs(ctx, titles, ["T2", "T2", "T2", "T5", "T5"])
    assert len(keep) == 4
    assert 2 in keep and 5 in keep
