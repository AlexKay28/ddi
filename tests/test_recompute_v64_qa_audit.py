"""Unit tests for the V64 step-3 preregistered QA recomputation (CPU-only)."""

import json

import numpy as np
import pytest

from scripts.recompute_v64_qa_audit import (
    AGG_TOLERANCE,
    AuditAbort,
    bootstrap_ci,
    draw_index_matrix,
    load_records,
    reproduce_aggregates,
)


def test_bootstrap_ci_matches_numpy_reference():
    rng = np.random.default_rng(7)
    deltas = np.array([0.10, -0.25, 0.30, 0.00, 0.15, -0.05, 0.20, 0.05])
    index = rng.integers(0, 8, size=(10000, 8))
    lo, hi = bootstrap_ci(deltas, index)
    resampled = np.array([deltas[row].mean() for row in index])
    ref_lo, ref_hi = np.quantile(resampled, [0.025, 0.975])
    assert lo == pytest.approx(ref_lo, abs=1e-12)
    assert hi == pytest.approx(ref_hi, abs=1e-12)


def test_bootstrap_ci_constant_vector_known_bounds():
    deltas = np.full(24, 0.25)
    index = np.zeros((50, 24), dtype=np.int64)
    assert bootstrap_ci(deltas, index) == (0.25, 0.25)


def test_shared_index_matrix_same_seed_identical_intervals():
    rng = np.random.default_rng(1)
    deltas_em = rng.integers(0, 2, size=24).astype(float)
    deltas_f1 = rng.random(24)
    matrix_1 = draw_index_matrix()
    matrix_2 = draw_index_matrix()
    assert matrix_1.shape == (10000, 24)
    assert np.array_equal(matrix_1, matrix_2)
    assert np.array_equal(bootstrap_ci(deltas_em, matrix_1), bootstrap_ci(deltas_em, matrix_2))
    assert np.array_equal(bootstrap_ci(deltas_f1, matrix_1), bootstrap_ci(deltas_f1, matrix_2))


def _tiny_fixture_records():
    values = {
        "A": [(0.0, 0.4), (1.0, 0.6), (1.0, 0.5)],
        "B": [(1.0, 0.5), (1.0, 0.6), (0.0, 0.5)],
        "C_cold": [(1.0, 0.5), (1.0, 0.6), (0.0, 0.5)],
        "C_all_hit": [(1.0, 0.5), (1.0, 0.6), (0.0, 0.5)],
        "D": [(0.0, 0.3), (0.0, 0.7), (1.0, 0.5)],
    }
    records = []
    for i in range(3):
        arms = {}
        for arm, per_record in values.items():
            em, f1 = per_record[i]
            entry = {"quality": {"em": em, "f1": f1}}
            if arm.startswith("C_"):
                arms.setdefault("C", {})[arm.split("_", 1)[1]] = entry
            else:
                arms[arm] = entry
        records.append({"record_id": f"id{i}", "arms": arms})
    return records


def test_aggregate_reproduction_on_tiny_synthetic_fixture():
    records = {record["record_id"]: record for record in _tiny_fixture_records()}
    frozen_ids = sorted(records)
    stored = {}
    for arm, per_record in {
        "A": [(0.0, 0.4), (1.0, 0.6), (1.0, 0.5)],
        "B": [(1.0, 0.5), (1.0, 0.6), (0.0, 0.5)],
        "C_cold": [(1.0, 0.5), (1.0, 0.6), (0.0, 0.5)],
        "C_all_hit": [(1.0, 0.5), (1.0, 0.6), (0.0, 0.5)],
        "D": [(0.0, 0.3), (0.0, 0.7), (1.0, 0.5)],
    }.items():
        stored[arm] = {
            "em": float(np.mean([v[0] for v in per_record])),
            "f1": float(np.mean([v[1] for v in per_record])),
        }
    report = reproduce_aggregates(records, frozen_ids, stored)
    assert report["A"]["em"]["match"] and report["A"]["em"]["abs_diff"] <= AGG_TOLERANCE
    assert report["D"]["f1"]["match"] and report["D"]["f1"]["abs_diff"] <= AGG_TOLERANCE
    drifted = json.loads(json.dumps(stored))
    drifted["B"]["em"] += 0.01
    with pytest.raises(AuditAbort, match="stored aggregates do not reproduce"):
        reproduce_aggregates(records, frozen_ids, drifted)


def test_load_records_aborts_on_missing_frozen_id(tmp_path):
    records = _tiny_fixture_records()
    path = tmp_path / "requests.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    frozen_ids = ["id0", "id1", "id2"]
    loaded = load_records(path, frozen_ids)
    assert sorted(loaded) == frozen_ids
    with pytest.raises(AuditAbort, match="count reconciliation"):
        load_records(path, frozen_ids + ["id3"])
