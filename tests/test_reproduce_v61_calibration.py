"""CPU-only tests for scripts/reproduce_v61_calibration.py (bootstrap shapes, no GPU).

Covers the RMS metric reused from the original calibration, the preregistered tolerance
band + criterion logic, determinism/shape of the fresh input generation, a tiny CPU
attention smoke (fp64 dense vs fp32/bf16 SDPA — flash-attn needs CUDA and is excluded),
and the manifest preregistration contract.
"""

from __future__ import annotations

import json

import pytest
import torch

import scripts.calibrate_fa2_fp64 as cal
import scripts.reproduce_v61_calibration as repro
from scripts.reproduce_v61_calibration import (
    band_ratio,
    build_manifest,
    evaluate_criterion,
    evaluate_reproduction,
    in_band,
    make_cell_inputs,
)


def _fake_cell(
    cell: str, fp32: float, bf16: float, fa2: float, dis: float
) -> dict[str, object]:
    return {
        "cell": cell,
        "rms_rel": {"fp32_sdpa": fp32, "bf16_sdpa": bf16, "bf16_fa2_segmented": fa2},
        "argmax_dis": {
            "fp64_dense_vs_fp32_sdpa": 0.0,
            "fp64_dense_vs_bf16_sdpa": 0.01,
            "fp64_dense_vs_bf16_fa2_segmented": 0.01,
            "fp32_sdpa_vs_bf16_sdpa": 0.01,
            "fp32_sdpa_vs_bf16_fa2_segmented": 0.01,
            "bf16_sdpa_vs_bf16_fa2_segmented": dis,
        },
    }


def _fake_cells(bf16_multiplier: float = 1.0) -> list[dict[str, object]]:
    """9 synthetic cell rows carrying the stored values (bf16 cells optionally scaled)."""
    stored = repro.STORED
    cells = []
    for n_docs, doc_len, query_len in repro.REPRO_SHAPES:
        for seed in repro.REPRO_SEEDS:
            cells.append(
                _fake_cell(
                    f"seed{seed}_N{n_docs}_L{doc_len}_Q{query_len}",
                    fp32=stored["fp32_sdpa_rms_rel"],
                    bf16=stored["bf16_sdpa_rms_rel"] * bf16_multiplier,
                    fa2=stored["bf16_fa2_segmented_rms_rel"] * bf16_multiplier,
                    dis=stored["argmax_dis_bf16_sdpa_vs_bf16_fa2_segmented"],
                )
            )
    return cells


# ---- RMS metric (reused verbatim from the original calibration) -------------------------


def test_rms_relative_error_zero_for_identical() -> None:
    ref = torch.randn(2, 3, 8, 16, dtype=torch.float64)
    assert repro.rms_relative_error(ref, ref) == 0.0


def test_rms_relative_error_known_offset() -> None:
    ref = torch.ones(1, 2, 4, 8, dtype=torch.float64)
    out = ref + 0.25
    assert repro.rms_relative_error(out, ref) == pytest.approx(0.25)


def test_rms_relative_error_scale_invariant() -> None:
    ref = torch.randn(1, 4, 16, 32, dtype=torch.float64)
    out = ref * 1.01
    assert repro.rms_relative_error(out, ref) == pytest.approx(0.01, rel=1e-12)


def test_rms_relative_error_zero_reference_raises() -> None:
    ref = torch.zeros(1, 2, 4, 8, dtype=torch.float64)
    out = torch.randn(1, 2, 4, 8, dtype=torch.float64)
    with pytest.raises(ValueError, match="all zeros"):
        repro.rms_relative_error(out, ref)


def test_repro_reuses_original_measurement_functions() -> None:
    """Same function objects as the original script — no re-implementation drift."""
    assert repro.rms_relative_error is cal.rms_relative_error
    assert repro.dense_attention is cal.dense_attention
    assert repro.sdpa_attention is cal.sdpa_attention


# ---- preregistered tolerance band --------------------------------------------------------


def test_band_ratio_boundaries_inclusive() -> None:
    stored = repro.STORED["bf16_sdpa_rms_rel"]
    assert in_band(band_ratio(stored * 0.5, stored))  # lower edge, inclusive
    assert in_band(band_ratio(stored * 2.0, stored))  # upper edge, inclusive
    assert in_band(band_ratio(stored, stored))
    assert not in_band(band_ratio(stored * 0.49, stored))
    assert not in_band(band_ratio(stored * 2.01, stored))


def test_band_ratio_rejects_nonpositive_stored() -> None:
    with pytest.raises(ValueError, match="> 0"):
        band_ratio(1.0, 0.0)


def test_evaluate_reproduction_all_in_band() -> None:
    verdict = evaluate_reproduction(_fake_cells())
    assert verdict["verdict"] == "REPRODUCED"
    assert verdict["band"] == [0.5, 2.0]
    for name, row in verdict["methods"].items():
        assert row["all_in_band"], name
        assert row["stored"] == repro.STORED[name]
        for ratio in row["ratios"].values():
            assert ratio == pytest.approx(1.0)


def test_evaluate_reproduction_out_of_band_bf16_cell() -> None:
    verdict = evaluate_reproduction(_fake_cells(bf16_multiplier=3.0))
    assert verdict["verdict"] == "NOT_REPRODUCED"
    assert not verdict["methods"]["bf16_sdpa_rms_rel"]["all_in_band"]
    assert not verdict["methods"]["bf16_fa2_segmented_rms_rel"]["all_in_band"]
    assert verdict["methods"]["fp32_sdpa_rms_rel"]["all_in_band"]


# ---- criterion validation (RMS <= 0.02 with >= 10x observed margin) ----------------------


def test_criterion_validated_at_stored_scale() -> None:
    crit = evaluate_criterion(_fake_cells())
    expected_margin = 0.02 / repro.STORED["bf16_fa2_segmented_rms_rel"]
    assert crit["all_within_threshold"]
    assert crit["observed_margin"] == pytest.approx(expected_margin)
    assert crit["observed_margin"] > repro.CRITERION_REQUIRED_MARGIN
    assert crit["validated"]


def test_criterion_margin_below_10x_not_validated() -> None:
    # Worst bf16 rms 0.0021: within the 0.02 threshold but the margin is only ~9.5x.
    mult = 0.0021 / repro.STORED["bf16_fa2_segmented_rms_rel"]
    crit = evaluate_criterion(_fake_cells(bf16_multiplier=mult))
    assert crit["all_within_threshold"]
    assert not crit["validated"]


def test_criterion_margin_just_above_10x_validated() -> None:
    mult = 0.0019 / repro.STORED["bf16_fa2_segmented_rms_rel"]
    crit = evaluate_criterion(_fake_cells(bf16_multiplier=mult))
    assert crit["all_within_threshold"]
    assert crit["observed_margin"] > repro.CRITERION_REQUIRED_MARGIN
    assert crit["validated"]


def test_criterion_fails_threshold() -> None:
    mult = 0.5 / repro.STORED["bf16_fa2_segmented_rms_rel"]
    crit = evaluate_criterion(_fake_cells(bf16_multiplier=mult))
    assert not crit["all_within_threshold"]
    assert not crit["validated"]


# ---- fresh input generation (bootstrap shapes, CPU only) ---------------------------------


def test_make_cell_inputs_deterministic_and_shaped() -> None:
    n_docs, doc_len, query_len = 2, 8, 4
    q1, k1, v1, allow1 = make_cell_inputs(4047, n_docs, doc_len, query_len)
    q2, k2, v2, allow2 = make_cell_inputs(4047, n_docs, doc_len, query_len)
    assert torch.equal(q1, q2) and torch.equal(k1, k2) and torch.equal(v1, v2)
    assert torch.equal(allow1, allow2)

    q3, k3, v3, _ = make_cell_inputs(4048, n_docs, doc_len, query_len)
    assert not torch.equal(q1, q3)
    assert not torch.equal(k1, k3)
    assert not torch.equal(v1, v3)

    total = n_docs * doc_len + query_len
    assert q1.shape == (1, repro.NUM_Q_HEADS, total, repro.HEAD_DIM)
    assert k1.shape == (1, repro.NUM_KV_HEADS, total, repro.HEAD_DIM)
    assert v1.shape == (1, repro.NUM_KV_HEADS, total, repro.HEAD_DIM)
    assert q1.dtype == torch.bfloat16
    assert allow1.shape == (total, total) and allow1.dtype == torch.bool
    assert allow1.any(dim=1).all()  # every row attends something (dense_attention guard)


def test_make_cell_inputs_mask_matches_mask_builder() -> None:
    from ddi.masks import build_mask

    _, _, _, allow = make_cell_inputs(4047, 2, 8, 4)
    expected = build_mask([8, 8], 4, 0, repro.VARIANT)
    assert torch.equal(allow, expected)


def test_bootstrap_attention_cpu_smoke() -> None:
    """Tiny CPU end-to-end: fp64 dense vs fp32/bf16 SDPA on the same bf16 inputs."""
    q, k, v, allow = make_cell_inputs(4047, 2, 8, 4)
    scale = repro.SOFTMAX_SCALE
    fp64 = repro.dense_attention(q, k, v, allow, scale, torch.float64)
    fp32 = repro.sdpa_attention(q, k, v, allow, scale, torch.float32)
    bf16 = repro.sdpa_attention(q, k, v, allow, scale, torch.bfloat16)
    for out in (fp64, fp32, bf16):
        assert torch.isfinite(out).all()
    assert repro.rms_relative_error(fp32, fp64) < 1e-4
    assert repro.rms_relative_error(bf16, fp64) < 0.05
    assert cal.argmax_disagreement(fp32, fp64) < 0.5  # fp32 vs fp64 disagree on few rows


def test_argmax_disagreement_identical_and_opposite() -> None:
    x = torch.randn(1, 2, 5, 8, dtype=torch.float32)
    assert cal.argmax_disagreement(x, x) == 0.0
    y = -x  # argmax flips every row
    assert cal.argmax_disagreement(x, y) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="shape mismatch"):
        cal.argmax_disagreement(x, x[:, :, :3])


# ---- preregistration contract ------------------------------------------------------------


def test_manifest_is_preregistration_without_results() -> None:
    git = {"rev": None, "dirty": True}
    manifest = build_manifest("20260101T000000Z", "NVIDIA A100 80GB PCIe", git)
    pre = manifest["preregistration"]
    assert pre["seeds"] == [4047, 4048, 4049]
    assert pre["shapes_n_l_q"] == [[2, 128, 64], [5, 512, 64], [10, 512, 64]]
    assert pre["cells"] == 9
    assert pre["tolerance_band"]["band"] == [0.5, 2.0]
    assert pre["criterion"]["max_rms"] == 0.02
    assert pre["criterion"]["required_margin"] == 10.0
    assert manifest["runtime"]["device"] == "NVIDIA A100 80GB PCIe"
    # No measured values may be present at manifest time.
    for key in ("cells", "reproduction", "criterion_verdict", "results"):
        assert key not in manifest
    json.dumps(manifest, allow_nan=False)  # artifact-serializable by construction


def test_stored_values_match_original_artifact() -> None:
    """Guards the preregistered stored numbers against transcription errors."""
    summary_path = repro.ORIGINAL_ARTIFACT / "summary.json"
    if not summary_path.exists():
        pytest.skip("original calibration artifact not present")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    errors = summary["errors_vs_fp64_dense"]
    pairwise = summary["pairwise_argmax_disagreement"]
    assert repro.STORED["fp32_sdpa_rms_rel"] == errors["fp32_sdpa"]["rms_rel"]
    assert repro.STORED["bf16_sdpa_rms_rel"] == errors["bf16_sdpa"]["rms_rel"]
    assert repro.STORED["bf16_fa2_segmented_rms_rel"] == errors["bf16_fa2_segmented"]["rms_rel"]
    pair = repro.ARGMAX_PAIR
    assert repro.STORED[f"argmax_dis_{pair}"] == pairwise[pair]
