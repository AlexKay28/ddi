"""CPU tests for the V65 combined-final helpers: mixed-hit split determinism, schedule
plan building, combined-win gate logic, expectations, store pruning and the frozen
preregistration constants — no GPU, no model."""

from __future__ import annotations

import hashlib

import pytest

from scripts.run_v65_combined import (
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    COMBINED_WIN_MIN_RATIO,
    DECISION_RULE,
    FIXED_BUDGET,
    FLAT_THRESHOLD,
    MARGIN,
    NATURAL_BUDGET,
    QA_ROLE_LABEL,
    REPEATS,
    WARMUPS,
    _prune_store,
    _store_snapshot,
    build_schedule_plan,
    combined_win_verdict,
    expectation_check,
    mixed_hit_indices,
    paired_ratio_bootstrap,
)


def _hash_rank(record_id: str, index: int, seed: int) -> int:
    payload = f"{seed}:{record_id}:{index}".encode()
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def test_mixed_hit_indices_is_deterministic() -> None:
    first = mixed_hit_indices("5a72f0e25542991f9a20c5cf", 10)
    second = mixed_hit_indices("5a72f0e25542991f9a20c5cf", 10)
    assert first == second
    expected = sorted(
        sorted(
            range(10),
            key=lambda index: (_hash_rank("5a72f0e25542991f9a20c5cf", index, 47), index),
        )[:5]
    )
    assert first == expected


def test_mixed_hit_indices_floor_half_and_partition() -> None:
    for doc_count in range(0, 13):
        hits = mixed_hit_indices("record", doc_count)
        assert len(hits) == doc_count // 2
        assert hits == sorted(hits)
        assert len(set(hits)) == len(hits)
        assert all(0 <= index < doc_count for index in hits)
    assert mixed_hit_indices("record", 0) == []
    assert mixed_hit_indices("record", 1) == []
    assert len(mixed_hit_indices("record", 2)) == 1
    assert len(mixed_hit_indices("record", 10)) == 5
    with pytest.raises(ValueError):
        mixed_hit_indices("record", -1)


def test_mixed_hit_indices_seed_changes_the_split() -> None:
    record_ids = ["5a72f0e25542991f9a20c5cf", "5a7300b85542994cef4bc41a"]
    for record_id in record_ids:
        seed47 = mixed_hit_indices(record_id, 10, seed=47)
        seed48 = mixed_hit_indices(record_id, 10, seed=48)
        assert len(seed47) == len(seed48) == 5
        assert seed47 != seed48


def test_build_schedule_plan_aligns_and_partitions() -> None:
    cells = [
        {"record_id": "r1", "doc_lengths": [1, 2, 3, 4]},
        {"record_id": "r2", "doc_lengths": [5, 6, 7]},
        {"record_id": "r3", "doc_lengths": [8]},
    ]
    plan = build_schedule_plan(cells)
    assert [entry["record_id"] for entry in plan] == ["r1", "r2", "r3"]
    assert plan[0]["mixed_hit_indices"] == mixed_hit_indices("r1", 4)
    assert plan[1]["mixed_hit_indices"] == mixed_hit_indices("r2", 3)
    assert plan[2]["mixed_hit_indices"] == []
    for entry in plan:
        assert sorted(entry["mixed_hit_indices"] + entry["mixed_fresh_indices"]) == list(
            range(entry["doc_count"])
        )
        assert not set(entry["mixed_hit_indices"]) & set(entry["mixed_fresh_indices"])
    assert [len(entry["mixed_hit_indices"]) for entry in plan] == [2, 1, 0]


def test_combined_win_verdict_strict_lower_bound() -> None:
    def ratio(point: float, ci_low: float, ci_high: float) -> dict:
        return {"point": point, "ci_low": ci_low, "ci_high": ci_high}

    verdict = combined_win_verdict(ratio(1.2, 1.0001, 1.5))
    assert verdict["combined_win"] is True
    assert verdict["min_ci_low"] == COMBINED_WIN_MIN_RATIO
    assert combined_win_verdict(ratio(1.2, 1.0, 1.5))["combined_win"] is False
    assert combined_win_verdict(ratio(0.9, 0.8, 1.1))["combined_win"] is False


def test_paired_ratio_bootstrap_feeds_the_gate() -> None:
    monolithic = [100.0 + index for index in range(24)]
    mixed = [value * 0.7 for value in monolithic]
    result = paired_ratio_bootstrap(monolithic, mixed, seed=BOOTSTRAP_SEED, iterations=500)
    verdict = combined_win_verdict(result)
    assert verdict["ratio_point"] == pytest.approx(1 / 0.7, rel=1e-9)
    assert verdict["combined_win"] is True


def test_expectation_check_is_informational() -> None:
    walls = {"A": 100.0, "cold": 300.0, "all_hit": 50.0, "mixed_hit": 80.0}
    check = expectation_check(walls)
    assert check["reuse_beats_monolithic"] is True
    assert check["mixed_lands_between_reuse_and_monolithic"] is True
    assert check["gate"] == "informational only; the combined-win gate is separate"
    above_mono = dict(walls, mixed_hit=120.0)
    assert expectation_check(above_mono)["mixed_lands_between_reuse_and_monolithic"] is False
    below_reuse = dict(walls, mixed_hit=40.0)
    assert expectation_check(below_reuse)["mixed_lands_between_reuse_and_monolithic"] is False
    assert expectation_check(dict(walls, all_hit=150.0))["reuse_beats_monolithic"] is False


def test_prune_store_restores_the_hit_only_snapshot(tmp_path) -> None:
    keep = {"hit-a.pt", "hit-a.lock"}
    for name in ("hit-a.pt", "hit-a.lock", "fresh-b.pt", "fresh-b.lock"):
        (tmp_path / name).write_bytes(b"x")
    assert _store_snapshot(tmp_path) == {"hit-a.pt", "hit-a.lock", "fresh-b.pt", "fresh-b.lock"}
    removed = _prune_store(tmp_path, keep)
    assert removed == 2
    assert _store_snapshot(tmp_path) == keep
    assert _prune_store(tmp_path, keep) == 0


def test_default_protocol_matches_the_preregistration() -> None:
    assert BOOTSTRAP_SEED == 47
    assert BOOTSTRAP_ITERATIONS == 10_000
    assert COMBINED_WIN_MIN_RATIO == 1.0
    assert NATURAL_BUDGET == 48
    assert FIXED_BUDGET == 16
    assert WARMUPS == 2
    assert REPEATS == 10
    assert FLAT_THRESHOLD == 0.02
    assert MARGIN == -0.02
    assert DECISION_RULE == "non-inferior iff CI LB >= -0.02 for BOTH EM and F1"
    assert QA_ROLE_LABEL == "single-seed confirmation, n=100"
