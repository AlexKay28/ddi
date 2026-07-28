"""Tests for the DDI-stress synthetic loader (issue #51)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from ddi.data import ddi_stress, list_datasets, load, manifest

_FAMILIES = ("fact_chain", "fact_aggregation", "contradiction", "coreference", "needle")


@pytest.fixture(autouse=True)
def manifest_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the manifest root away from the real results/splits/ for every test."""
    root = tmp_path / "splits"
    monkeypatch.setattr(manifest, "DEFAULT_ROOT", root)
    return root


@pytest.fixture(autouse=True)
def _reset_overrides() -> Iterator[None]:
    """Every test starts with no task-family or knob overrides."""
    ddi_stress.set_task_family(None)
    ddi_stress.set_knob_overrides()
    yield
    ddi_stress.set_task_family(None)
    ddi_stress.set_knob_overrides()


def _load_family(family: str, split: str, seed: int) -> list[dict]:
    ddi_stress.set_task_family(family)
    return list(load("ddi_stress", split, seed))


# ---------------------------------------------------------------------------


def test_registered() -> None:
    assert "ddi_stress" in list_datasets()


def test_split_sizes() -> None:
    train = _load_family("fact_chain", "train", 0)
    val = _load_family("fact_chain", "val", 0)
    test = _load_family("fact_chain", "test", 0)
    assert len(train) == 600
    assert len(val) == 200
    assert len(test) == 200
    ids = {r["id"] for r in train} | {r["id"] for r in val} | {r["id"] for r in test}
    assert len(ids) == 1000  # disjoint splits


def test_deterministic_per_family() -> None:
    for family in _FAMILIES:
        a = _load_family(family, "val", 42)
        b = _load_family(family, "val", 42)
        assert a == b, f"family {family!r} produced non-deterministic stream"


def test_all_families_produce_records() -> None:
    for family in _FAMILIES:
        records = _load_family(family, "val", 0)
        assert len(records) > 0
        for rec in records[:20]:
            assert isinstance(rec["id"], str) and rec["id"]
            assert isinstance(rec["info_contexts"], list)
            assert len(rec["info_contexts"]) > 0
            assert all(isinstance(d, str) and d for d in rec["info_contexts"])
            assert isinstance(rec["dialog"], list) and len(rec["dialog"]) == 2
            assert rec["dialog"][0]["role"] == "user"
            assert rec["dialog"][1]["role"] == "assistant"
            assert isinstance(rec["possible_answers"], list) and rec["possible_answers"]
            assert rec["hop_type"] in {"single", "multi"}
            meta = rec["meta"]
            assert meta["task_family"] == family
            # Every knob echoed:
            for knob in (
                "n_docs",
                "n_gold",
                "gold_positions",
                "distractor_strength",
                "hop_depth",
                "n_facts_per_doc",
            ):
                assert knob in meta, f"family {family!r} missing meta[{knob!r}]"


def test_answer_only_in_gold() -> None:
    """The non-leakage guard: the answer string must not appear verbatim in any distractor."""
    for family in _FAMILIES:
        ddi_stress.set_task_family(family)
        seen = 0
        for rec in load("ddi_stress", "train", seed=7):
            gold_positions = set(rec["meta"]["gold_positions"])
            answer = rec["possible_answers"][0]
            low_ans = answer.lower()
            for i, doc in enumerate(rec["info_contexts"]):
                if i in gold_positions:
                    continue
                assert low_ans not in doc.lower(), (
                    f"{family}/{rec['id']}: answer {answer!r} leaks "
                    f"into distractor doc {i}: {doc!r}"
                )
            seen += 1
            if seen >= 200:
                break
        assert seen >= 200, f"family {family!r}: only {seen} records available"


def test_gold_position_respected() -> None:
    """Forcing gold_positions must place gold docs at exactly those indices."""
    ddi_stress.set_task_family("fact_chain")
    ddi_stress.set_knob_overrides(gold_positions=[0, 3], hop_depth=2, n_docs=5)
    records = list(load("ddi_stress", "val", 0))
    assert records
    for rec in records[:5]:
        assert rec["meta"]["gold_positions"] == [0, 3]
        # Chain relation words appear in gold docs, distractors are plain prose.
        for pos in [0, 3]:
            doc = rec["info_contexts"][pos]
            assert any(rel in doc for rel in ("lives in", "born in", "founded", "leads")), (
                f"gold doc {pos} does not look like a chain fact: {doc!r}"
            )


def test_hop_depth_matches_config() -> None:
    """For fact_chain at hop_depth=4, the answer chain must require exactly 4 facts."""
    ddi_stress.set_task_family("fact_chain")
    ddi_stress.set_knob_overrides(hop_depth=4, n_docs=6)
    records = list(load("ddi_stress", "val", 0))
    assert records
    for rec in records[:10]:
        meta = rec["meta"]
        assert meta["hop_depth"] == 4
        assert len(meta["chain_relations"]) == 4
        assert len(meta["chain_entities"]) == 5  # hop_depth + 1 endpoints
        assert meta["n_gold"] == 4
        assert len(meta["gold_positions"]) == 4
        # The final entity is the answer.
        assert rec["possible_answers"][0] == meta["chain_entities"][-1]


def test_manifest_stable(manifest_root: Path) -> None:
    """Two loads with the same seed produce the same manifest hash."""
    ddi_stress.set_task_family("fact_chain")
    list(load("ddi_stress", "train", 0))
    manifest_path = manifest_root / "ddi_stress__fact_chain__seed0.json"
    assert manifest_path.exists()
    first = manifest_path.read_text()
    # Second load validates against the manifest — if hashes differed it'd raise.
    list(load("ddi_stress", "val", 0))
    second = manifest_path.read_text()
    assert first == second


def test_env_var_selects_family(monkeypatch: pytest.MonkeyPatch) -> None:
    """Absent programmatic override, DDI_STRESS_TASK env var picks the family."""
    monkeypatch.setenv("DDI_STRESS_TASK", "needle")
    records = list(load("ddi_stress", "val", 0))
    assert records
    for rec in records[:5]:
        assert rec["meta"]["task_family"] == "needle"
        assert rec["hop_type"] == "single"
