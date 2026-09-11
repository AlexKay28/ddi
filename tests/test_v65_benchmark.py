"""CPU tests for the V65 step-3 benchmark helpers: synthetic bootstrap statistics, ratio
arithmetic, equivalence/gate logic and frozen-cell loading — no GPU, no model."""

from __future__ import annotations

import json
import statistics

import pytest

from scripts.run_v65_benchmark import (
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    GATE_C_OVER_A_MAX,
    _agg,
    bootstrap_statistic,
    compare_generation,
    gate_verdict,
    load_frozen_cells,
    paired_ratio_bootstrap,
    ratio_of_medians,
    retained_medians_ms,
)


def _cell(record_id: str, a_ms: float, b_ms: float, c_ms: float) -> dict:
    def arm(value: float) -> dict:
        return {"ttft_median_seconds": value / 1000.0, "prediction": "x"}

    return {
        "record_id": record_id,
        "arms": {"A": arm(a_ms), "B": arm(b_ms), "C": {"all_hit": arm(c_ms), "cold": arm(c_ms)}},
    }


def test_ratio_of_medians_basic() -> None:
    assert ratio_of_medians([10.0, 20.0, 30.0], [1.0, 2.0, 3.0]) == 10.0
    assert ratio_of_medians([10.0, 20.0], [1.0, 3.0]) == 7.5
    with pytest.raises(ValueError):
        ratio_of_medians([], [])
    with pytest.raises(ValueError):
        ratio_of_medians([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        ratio_of_medians([1.0, -2.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        ratio_of_medians([1.0, 2.0], [0.0, 2.0])


def test_paired_ratio_bootstrap_constant_arms() -> None:
    result = paired_ratio_bootstrap(
        [50.0] * 24, [100.0] * 24, seed=BOOTSTRAP_SEED, iterations=200
    )
    assert result["point"] == 0.5
    assert result["ci_low"] == 0.5
    assert result["ci_high"] == 0.5
    assert result["seed"] == BOOTSTRAP_SEED
    assert result["iterations"] == 200
    assert result["pairs"] == 24


def test_paired_ratio_bootstrap_shifted_pairs() -> None:
    denominator = [100.0 + (index % 7) * 3.0 for index in range(24)]
    numerator = [value * 1.2 for value in denominator]
    result = paired_ratio_bootstrap(numerator, denominator, seed=7, iterations=500)
    assert result["point"] == pytest.approx(1.2, rel=1e-9)
    assert result["ci_low"] <= result["point"] <= result["ci_high"]
    assert result["ci_low"] == pytest.approx(1.2, abs=0.05)
    assert result["ci_high"] == pytest.approx(1.2, abs=0.05)


def test_paired_ratio_bootstrap_is_deterministic_per_seed() -> None:
    rng_values = [100.0 + index * index % 13 for index in range(24)]
    numerator = [value * 0.9 + (index % 5) for index, value in enumerate(rng_values)]
    first = paired_ratio_bootstrap(numerator, rng_values, seed=47, iterations=300)
    second = paired_ratio_bootstrap(numerator, rng_values, seed=47, iterations=300)
    assert first == second


def test_paired_ratio_bootstrap_outlier_widens_ci() -> None:
    denominator = [100.0 + (index % 7) for index in range(24)]
    numerator = [value * 1.1 for value in denominator]
    with_outlier = list(numerator)
    with_outlier[3] *= 10.0
    base = paired_ratio_bootstrap(numerator, denominator, seed=11, iterations=500)
    widened = paired_ratio_bootstrap(with_outlier, denominator, seed=11, iterations=500)
    assert (widened["ci_high"] - widened["ci_low"]) > (base["ci_high"] - base["ci_low"])


def test_paired_ratio_bootstrap_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        paired_ratio_bootstrap([1.0], [1.0, 2.0], seed=1, iterations=10)
    with pytest.raises(ValueError):
        paired_ratio_bootstrap([0.0, 1.0], [1.0, 1.0], seed=1, iterations=10)
    with pytest.raises(ValueError):
        paired_ratio_bootstrap([1.0, 1.0], [1.0, 1.0], seed=1, iterations=0)


def test_bootstrap_statistic_median_matches_point_and_contains_ci() -> None:
    values = [10.0, 12.0, 11.0, 40.0, 9.0, 15.0, 14.0, 13.0, 10.5, 12.5] * 3
    result = bootstrap_statistic(values, seed=47, iterations=1000)
    assert result["point"] == statistics.median(values)
    assert result["ci_low"] <= result["point"] <= result["ci_high"]
    assert result["iterations"] == 1000
    constant = bootstrap_statistic([5.0] * 8, seed=1, iterations=100)
    assert constant["ci_low"] == constant["ci_high"] == 5.0


def test_default_bootstrap_protocol_matches_manifest_declaration() -> None:
    assert BOOTSTRAP_SEED == 47
    assert BOOTSTRAP_ITERATIONS == 10_000


def test_compare_generation_all_equal_and_diffs() -> None:
    retained = {
        "prediction": "Paris",
        "natural_generated_ids": [5, 6, 7],
        "fixed_generated_ids": [5, 6],
        "fixed_ids_sha256": "abc",
    }
    assert compare_generation(retained, retained)["byte_identical"] is True
    for key, value in (
        ("prediction", "Lyon"),
        ("natural_generated_ids", [5, 6, 8]),
        ("fixed_generated_ids", [5, 7]),
        ("fixed_ids_sha256", "zzz"),
    ):
        drifted = dict(retained)
        drifted[key] = value
        report = compare_generation(drifted, retained)
        assert report["byte_identical"] is False


def test_gate_verdict_requires_both_conditions() -> None:
    passed = gate_verdict(equivalence_all_equal=True, c_over_a_point=1.05)
    assert passed["pass"] is True
    assert passed["c_over_a_max"] == GATE_C_OVER_A_MAX
    assert gate_verdict(equivalence_all_equal=True, c_over_a_point=1.0501)["pass"] is False
    assert gate_verdict(equivalence_all_equal=False, c_over_a_point=0.5)["pass"] is False


def test_load_frozen_cells_verifies_sha_and_parses(tmp_path) -> None:
    cells = [
        {"record_id": "r1", "doc_lengths": [3], "query_length": 2},
        {"record_id": "r2", "doc_lengths": [4, 5], "query_length": 1},
    ]
    path = tmp_path / "requests.jsonl"
    path.write_text("".join(json.dumps(cell) + "\n" for cell in cells))
    import hashlib

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = load_frozen_cells(path, expected_sha256=digest)
    assert [cell["record_id"] for cell in loaded] == ["r1", "r2"]
    with pytest.raises(ValueError, match="sha mismatch"):
        load_frozen_cells(path, expected_sha256="0" * 64)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(ValueError, match="no frozen request cells"):
        load_frozen_cells(empty, expected_sha256=hashlib.sha256(b"").hexdigest())


def test_retained_medians_ms_converts_seconds_to_ms() -> None:
    cells = [_cell("r1", 100.0, 400.0, 200.0), _cell("r2", 150.0, 440.0, 240.0)]
    assert retained_medians_ms(cells, "A") == [100.0, 150.0]
    assert retained_medians_ms(cells, "B") == [400.0, 440.0]
    assert retained_medians_ms(cells, "C_all_hit") == [200.0, 240.0]


def test_agg_summarizes_ordered_values() -> None:
    summary = _agg([4.0, 1.0, 3.0, 2.0])
    assert summary["min"] == 1.0
    assert summary["max"] == 4.0
    assert summary["median"] == 2.5
    assert summary["mean"] == 2.5
    assert summary["p25"] <= summary["median"] <= summary["p75"]
