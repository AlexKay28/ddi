"""Tests for the KILT-NQ loader (issue #45)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ddi.data import kilt_nq, list_datasets, load, manifest


@pytest.fixture(autouse=True)
def manifest_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the split manifest root off the real results/ tree."""
    root = tmp_path / "splits"
    monkeypatch.setattr(manifest, "DEFAULT_ROOT", root)
    return root


@pytest.fixture(autouse=True)
def reset_n_passages_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure per-test isolation for the module override + env var."""
    monkeypatch.setattr(kilt_nq, "_n_passages_override", None)
    monkeypatch.delenv(kilt_nq._ENV_N_PASSAGES, raising=False)


_INTEGRATION = pytest.mark.skipif(
    os.environ.get("DDI_INTEGRATION") != "1",
    reason="KILT-NQ loader hits HuggingFace; opt in with DDI_INTEGRATION=1",
)


# ---------------------------------------------------------------------------
# offline-safe: registry + config-mechanism sanity
# ---------------------------------------------------------------------------


def test_registered() -> None:
    assert "kilt_nq" in list_datasets()


def test_n_passages_default() -> None:
    assert kilt_nq._current_n_passages() == 5


def test_set_n_passages_overrides_default() -> None:
    kilt_nq.set_n_passages(3)
    assert kilt_nq._current_n_passages() == 3


def test_env_var_overrides_module_default(monkeypatch: pytest.MonkeyPatch) -> None:
    kilt_nq.set_n_passages(3)
    monkeypatch.setenv(kilt_nq._ENV_N_PASSAGES, "7")
    assert kilt_nq._current_n_passages() == 7


def test_set_n_passages_rejects_zero() -> None:
    with pytest.raises(ValueError):
        kilt_nq.set_n_passages(0)


def test_env_var_rejects_non_int(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(kilt_nq._ENV_N_PASSAGES, "not-a-number")
    with pytest.raises(ValueError):
        kilt_nq._current_n_passages()


# ---------------------------------------------------------------------------
# offline-safe: normalize() from a synthesized KILT-shaped row
# ---------------------------------------------------------------------------


def _fake_kilt_row(n_prov: int = 5) -> dict:
    provs = [
        {"title": f"Title {i}", "text": f"passage text {i}", "wikipedia_id": str(i)}
        for i in range(n_prov)
    ]
    return {
        "id": "row-0001",
        "input": "What is the answer?",
        "output": [
            {"answer": "the answer", "provenance": provs},
            {"answer": "the answer variant", "provenance": []},
        ],
    }


def test_normalize_shape_matches_spec() -> None:
    row = _fake_kilt_row(n_prov=5)
    rec = kilt_nq._normalize(row, n_passages=5)

    assert rec["id"] == "row-0001"
    assert len(rec["info_contexts"]) == 5
    assert all(isinstance(p, str) for p in rec["info_contexts"])
    assert rec["dialog"] == [
        {"role": "user", "content": "What is the answer?"},
        {"role": "assistant", "content": "the answer"},
    ]
    assert rec["possible_answers"] == ["the answer", "the answer variant"]
    assert rec["hop_type"] == "single"
    assert rec["gold_position"] is None  # KILT lacks a stable gold flag; #55
    assert rec["meta"] == {
        "retriever": "kilt_provenance",
        "n_passages": 5,
        "gold_title": "Title 0",
    }


def test_normalize_shrinks_when_provenance_thin() -> None:
    row = _fake_kilt_row(n_prov=2)
    rec = kilt_nq._normalize(row, n_passages=5)
    # Only 2 real passages available; no empty-placeholder padding (build_mask rejects
    # zero-length documents), so info_contexts and meta["n_passages"] both shrink to 2.
    assert len(rec["info_contexts"]) == 2
    assert rec["info_contexts"][0].startswith("passage text 0")
    assert rec["info_contexts"][1].startswith("passage text 1")
    assert rec["meta"]["n_passages"] == 2


def test_normalize_gold_position_always_none() -> None:
    # KILT has no rotation-stable gold identity, so gold_position is None regardless of
    # provenance order or truncation. See #55 and _normalize's inline comment for context.
    row = _fake_kilt_row(n_prov=5)
    row["output"][0]["provenance"] = row["output"][0]["provenance"][2:] + row["output"][0][
        "provenance"
    ][:2]
    rec = kilt_nq._normalize(row, n_passages=2)
    assert rec["gold_position"] is None
    assert rec["meta"]["gold_title"] == "Title 2"  # advisory only — first-listed passage


def test_normalize_dedups_provenance_by_title() -> None:
    row = {
        "id": "dup",
        "input": "q",
        "output": [
            {
                "answer": "a",
                "provenance": [
                    {"title": "T1", "text": "first"},
                    {"title": "T2", "text": "second"},
                ],
            },
            {
                "answer": "a2",
                "provenance": [
                    {"title": "T1", "text": "duplicate"},
                    {"title": "T3", "text": "third"},
                ],
            },
        ],
    }
    rec = kilt_nq._normalize(row, n_passages=3)
    # Order preserved from KILT stream, T1 not repeated.
    assert rec["info_contexts"] == ["first", "second", "third"]


# ---------------------------------------------------------------------------
# integration — real HF fetch, gated on DDI_INTEGRATION=1
# ---------------------------------------------------------------------------


@_INTEGRATION
def test_common_record_shape() -> None:
    stream = load("kilt_nq", "val", seed=0)
    rec = next(iter(stream))

    assert isinstance(rec["id"], str) and rec["id"]
    assert isinstance(rec["info_contexts"], list)
    assert all(isinstance(p, str) for p in rec["info_contexts"])
    assert len(rec["dialog"]) == 2
    assert rec["dialog"][0]["role"] == "user"
    assert rec["dialog"][1]["role"] == "assistant"
    assert isinstance(rec["possible_answers"], list)
    assert all(isinstance(a, str) for a in rec["possible_answers"])
    assert isinstance(rec["meta"], dict)
    assert rec["meta"]["retriever"] == "kilt_provenance"
    assert rec["meta"]["n_passages"] == len(rec["info_contexts"])


@_INTEGRATION
def test_hop_type_is_single() -> None:
    stream = load("kilt_nq", "val", seed=0)
    rec = next(iter(stream))
    assert rec["hop_type"] == "single"


@_INTEGRATION
def test_gold_position_is_int_or_none() -> None:
    stream = load("kilt_nq", "val", seed=0)
    rec = next(iter(stream))
    assert rec["gold_position"] is None or isinstance(rec["gold_position"], int)
    if isinstance(rec["gold_position"], int):
        assert 0 <= rec["gold_position"] < len(rec["info_contexts"])


@_INTEGRATION
def test_n_passages_configurable() -> None:
    kilt_nq.set_n_passages(3)
    stream = load("kilt_nq", "val", seed=0)
    rec = next(iter(stream))
    assert len(rec["info_contexts"]) == 3
    assert rec["meta"]["n_passages"] == 3


@_INTEGRATION
def test_deterministic() -> None:
    ids_a = [r["id"] for r in load("kilt_nq", "val", seed=0)]
    ids_b = [r["id"] for r in load("kilt_nq", "val", seed=0)]
    assert ids_a == ids_b
    assert len(ids_a) == len(set(ids_a))  # no dupes within a split
