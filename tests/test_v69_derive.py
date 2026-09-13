"""V69 derivation tests: pure helpers of scripts/v69_derive_attribution.py.

No GPU, no model: the artifact-backed integration test is skipped when the
pinned input artifacts are not present (e.g. on a fresh checkout).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/v69_derive_attribution.py"

spec = importlib.util.spec_from_file_location("v69_derive_attribution", SCRIPT)
v69 = importlib.util.module_from_spec(spec)
sys.modules.setdefault("v69_derive_attribution", v69)
spec.loader.exec_module(v69)


def synthetic_leaves() -> dict[str, float]:
    return {
        "validate_entry": 85.0,
        "disk_load": 60.0,
        "cache_assemble": 23.0,
        "check_model": 11.0,
        "attention_modules_cpu": 17.0,
        "query_forward_glue": 10.0,
        "mlp_modules_cpu": 4.0,
        "wrapper_glue_residual": 2.0,
    }


def synthetic_attribution(
    *,
    pre: float = 221.0,
    post: float = 62.0,
    leaves: dict[str, float] | None = None,
    residual_frac: float | None = None,
) -> dict:
    leaves = leaves if leaves is not None else synthetic_leaves()
    attr = v69.attribute(
        pre_total=pre,
        post_total=post,
        leaves_scaled=leaves,
        opt1_phases=("validate_entry",),
        opt2_phases=("disk_load",),
        opt3_budget_phases=("cache_assemble",),
    )
    if residual_frac is not None:
        # Force the coverage quantity to test T1 independently of the leaves.
        attr["residual_in_pre_frac"] = residual_frac
    return attr


class TestMedian:
    def test_odd_and_even(self):
        assert v69.median([1.0, 2.0, 3.0]) == 2.0
        assert v69.median([1.0, 2.0, 3.0, 4.0]) == 2.5

    def test_empty_raises(self):
        with pytest.raises(Exception):
            v69.median([])


class TestScaleLeaves:
    def test_uniform_scaling(self):
        contributors = [{"phase": "a", "ms": 10.0}, {"phase": "b", "ms": 5.0}]
        scaled = v69.scale_leaves(contributors, 0.5)
        assert scaled == {"a": 5.0, "b": 2.5}

    def test_scaling_preserves_all_phases(self):
        contributors = [{"phase": f"p{i}", "ms": float(i)} for i in range(7)]
        scaled = v69.scale_leaves(contributors, 1.25)
        assert set(scaled) == {f"p{i}" for i in range(7)}
        assert scaled["p3"] == 3.75


class TestAttribute:
    def test_closure_by_construction(self):
        attr = synthetic_attribution()
        # reduction = direct (85 + 60) + envelope; envelope = 99 - 145 < 0 here?
        # pre 221 - post 62 = 159; direct 145; envelope 14.
        assert attr["reduction_ms"] == pytest.approx(159.0)
        assert attr["direct_removed_ms"] == pytest.approx(145.0)
        assert attr["opt3_envelope_ms"] == pytest.approx(14.0)
        # Implied post cache_assemble = 23 - 14 = 9.
        assert attr["opt3_implied_post_cache_assemble_ms"] == pytest.approx(9.0)
        # Post-side reconstruction must close on the measured post total.
        recon = sum(attr["post_side_reconstruction_ms"].values())
        assert recon == pytest.approx(62.0)

    def test_zero_envelope_when_direct_equals_reduction(self):
        leaves = synthetic_leaves()
        leaves["cache_assemble"] = 0.0
        attr = synthetic_attribution(pre=221.0, post=76.0, leaves=leaves)
        assert attr["opt3_envelope_ms"] == pytest.approx(0.0)
        assert attr["opt3_implied_post_cache_assemble_ms"] == pytest.approx(0.0)

    def test_envelope_exceeding_budget_is_flagged(self):
        # Post so low that the remainder overshoots the cache_assemble budget.
        attr = synthetic_attribution(pre=221.0, post=40.0)
        assert attr["opt3_envelope_ms"] > attr["opt3_budget_ms"]
        assert attr["opt3_envelope_within_budget"] is False

    def test_negative_envelope_is_flagged(self):
        # Post so high that the two removed phases already over-explain.
        attr = synthetic_attribution(pre=221.0, post=100.0)
        assert attr["opt3_envelope_ms"] < 0.0
        assert attr["opt3_envelope_within_budget"] is False

    def test_missing_phase_raises(self):
        leaves = synthetic_leaves()
        del leaves["disk_load"]
        with pytest.raises(KeyError):
            synthetic_attribution(leaves=leaves)


class TestEvaluatePrereg:
    def test_passing_attribution_is_derivation_only(self):
        attr = synthetic_attribution()
        verdict = v69.evaluate_prereg(attr)
        assert verdict["verdict"] == "derivation_only"
        assert verdict["T1_phase_coverage_pass"] is True
        assert verdict["T2_direct_attribution_pass"] is True
        assert verdict["T3_opt3_envelope_pass"] is True

    def test_t1_failure_requires_toggles(self):
        attr = synthetic_attribution(residual_frac=0.10)  # tolerance is 0.05
        assert v69.evaluate_prereg(attr)["verdict"] == "toggles_needed"

    def test_t2_failure_requires_toggles(self):
        # Shrink the removed phases so direct attribution falls below 85%.
        leaves = synthetic_leaves()
        leaves["validate_entry"] = 50.0
        leaves["disk_load"] = 30.0
        attr = synthetic_attribution(leaves=leaves)
        assert attr["direct_frac_of_reduction"] < 0.85
        assert v69.evaluate_prereg(attr)["verdict"] == "toggles_needed"

    def test_t3_failure_requires_toggles(self):
        attr = synthetic_attribution(pre=221.0, post=40.0)
        assert v69.evaluate_prereg(attr)["T3_opt3_envelope_pass"] is False
        assert v69.evaluate_prereg(attr)["verdict"] == "toggles_needed"

    def test_prereg_thresholds_are_declared(self):
        assert v69.PREREG["tolerances"]["T1_phase_coverage"]["threshold"] == 0.05
        assert v69.PREREG["tolerances"]["T2_direct_attribution"]["threshold"] == 0.85
        assert "toggle_decision_rule" in v69.PREREG


class TestSha256File:
    def test_known_digest(self, tmp_path: Path):
        f = tmp_path / "x.txt"
        f.write_bytes(b"")
        assert v69.sha256_file(f) == (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )


class TestRealArtifacts:
    V60 = ROOT / "results/v60_confirmation24_20260910T093500Z/result.json"
    PROFILE = ROOT / "results/v65_profile_20260910T094553Z/result.json"
    BENCH = ROOT / "results/v65_benchmark_20260910T101710Z/result.json"
    COMBINED = ROOT / "results/v65_combined_20260910T102918Z/result.json"
    EQUIV = ROOT / "results/v65_equiv_20260910T100522Z/result.json"

    @pytest.mark.skipif(
        not all(p.exists() for p in (V60, PROFILE, BENCH, COMBINED, EQUIV)),
        reason="pinned V60/V65 artifacts not present",
    )
    def test_derivation_passes_declared_tolerances(self):
        data, attr, verdict = v69.derive()
        assert verdict["verdict"] == "derivation_only"
        # Headline numbers must reproduce the artifact-pinned values.
        assert attr["pre_total_ms"] == pytest.approx(221.2411, abs=1e-3)
        assert attr["post_total_ms"] == pytest.approx(62.2965, abs=1e-3)
        assert data["speedup_vs_pre"] == pytest.approx(3.5512, abs=1e-3)
        assert data["context"]["monolithic"]["a_over_c"] == pytest.approx(2.0473, abs=1e-3)
        # The three attribution lines must sum to the reduction.
        total = (
            attr["opt1_removed_ms"] + attr["opt2_removed_ms"] + attr["opt3_envelope_ms"]
        )
        assert total == pytest.approx(attr["reduction_ms"], abs=1e-9)
        # Post-side reconstruction closes on the measured post total.
        recon = sum(attr["post_side_reconstruction_ms"].values())
        assert recon == pytest.approx(attr["post_total_ms"], abs=0.05)
