"""Tests for the Phase 5.3 composition loader (issue #28)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ddi.data import list_datasets, load, manifest
from ddi.data.composition import _CITY_COUNTRY_PAIRS, _DEFAULT_N_DOCS


@pytest.fixture(autouse=True)
def manifest_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the manifest root away from the real results/splits/ for every test."""
    root = tmp_path / "splits"
    monkeypatch.setattr(manifest, "DEFAULT_ROOT", root)
    return root


def test_registered() -> None:
    assert "composition" in list_datasets()


def test_deterministic() -> None:
    a = list(load("composition", "train", 42))
    b = list(load("composition", "train", 42))
    assert a == b


def test_split_sizes() -> None:
    train = list(load("composition", "train", 0))
    val = list(load("composition", "val", 0))
    test = list(load("composition", "test", 0))
    assert len(train) == 60
    assert len(val) == 20
    assert len(test) == 20
    ids = {r["id"] for r in train} | {r["id"] for r in val} | {r["id"] for r in test}
    assert len(ids) == 100  # disjoint splits


def test_record_structure() -> None:
    records = list(load("composition", "val", 7))
    assert len(records) > 0
    known_countries = {c for _, c in _CITY_COUNTRY_PAIRS}
    for rec in records:
        assert isinstance(rec["id"], str) and rec["id"].startswith("comp_")
        assert isinstance(rec["info_contexts"], list)
        assert len(rec["info_contexts"]) == _DEFAULT_N_DOCS
        assert all(isinstance(d, str) and d for d in rec["info_contexts"])

        assert rec["hop_type"] == "multi"
        assert rec["gold_position"] is None

        meta = rec["meta"]
        assert set(
            ["distance", "pair_position", "doc_A_idx", "doc_B_idx", "entity", "city", "country", "n_docs"]
        ).issubset(meta)
        assert meta["n_docs"] == _DEFAULT_N_DOCS
        assert 0 <= meta["distance"] <= _DEFAULT_N_DOCS - 2
        assert meta["pair_position"] in {"start", "middle", "end"}
        a_idx = meta["doc_A_idx"]
        b_idx = meta["doc_B_idx"]
        assert 0 <= a_idx < _DEFAULT_N_DOCS
        assert 0 <= b_idx < _DEFAULT_N_DOCS
        assert b_idx - a_idx == meta["distance"] + 1

        # The recorded country is the record's answer and comes from the fixed fact pool.
        assert rec["possible_answers"] == [meta["country"]]
        assert meta["country"] in known_countries
        assert rec["dialog"][-1]["role"] == "assistant"
        assert rec["dialog"][-1]["content"] == meta["country"]

        # doc_A and doc_B are the gold pair.
        assert meta["city"] in rec["info_contexts"][a_idx]
        assert meta["entity"] in rec["info_contexts"][a_idx]
        assert meta["city"] in rec["info_contexts"][b_idx]
        assert meta["country"] in rec["info_contexts"][b_idx]


def test_answer_appears_only_in_doc_B_or_composition() -> None:
    """The composition property: the country answer must never appear in doc_A or any distractor.

    WHY this test exists: the whole point of Phase 5.3 is that the answer is REACHABLE only by
    composing doc_A and doc_B. If a distractor incidentally names the country, ddi_qf could
    recover it by attending to that distractor alone and the boundary story is a lie.
    """
    for split in ("train", "val", "test"):
        for rec in load("composition", split, seed=3):
            meta = rec["meta"]
            country = meta["country"]
            a_idx = meta["doc_A_idx"]
            b_idx = meta["doc_B_idx"]
            for i, doc in enumerate(rec["info_contexts"]):
                if i == b_idx:
                    continue  # doc_B is *supposed* to contain the country.
                assert country not in doc, (
                    f"record {rec['id']}: country {country!r} leaks into doc {i} "
                    f"(a_idx={a_idx}, b_idx={b_idx}): {doc!r}"
                )
