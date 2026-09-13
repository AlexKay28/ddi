"""CPU tests for scripts/calibrate_fa2_fp64.py — math helpers on tiny tensors.

GPU parts (model load, capture, SDPA/FA2 kernels) are skipped unless CUDA and
flash-attn are both available; everything else runs on CPU against hand-derived
expectations.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

import scripts.calibrate_fa2_fp64 as cal


def test_dense_attention_matches_manual_softmax_fp64() -> None:
    torch.manual_seed(0)
    q = torch.randn(1, 2, 4, 3, dtype=torch.float64)
    k = torch.randn(1, 2, 4, 3, dtype=torch.float64)
    v = torch.randn(1, 2, 4, 3, dtype=torch.float64)
    allow = torch.ones(4, 4, dtype=torch.bool)
    scale = 0.4
    out = cal.dense_attention(q, k, v, allow, scale, torch.float64)
    scores = q @ k.transpose(-1, -2) * scale
    expected = torch.softmax(scores, dim=-1) @ v
    assert torch.allclose(out, expected, rtol=0, atol=1e-12)


def test_dense_attention_respects_mask() -> None:
    # One query, two keys, only the first allowed: softmax over a single logit -> v[0].
    q = torch.randn(1, 1, 1, 2, dtype=torch.float64)
    k = torch.randn(1, 1, 2, 2, dtype=torch.float64)
    v = torch.randn(1, 1, 2, 2, dtype=torch.float64)
    allow = torch.tensor([[True, False]])
    out = cal.dense_attention(q, k, v, allow, 1.0, torch.float64)
    assert torch.allclose(out[:, :, :, :], v[:, :, :1, :].expand_as(out))


def test_dense_attention_rejects_fully_masked_row() -> None:
    q = torch.randn(1, 1, 2, 2)
    k = torch.randn(1, 1, 2, 2)
    v = torch.randn(1, 1, 2, 2)
    allow = torch.tensor([[True, True], [False, False]])
    with pytest.raises(ValueError, match="no allowed key"):
        cal.dense_attention(q, k, v, allow, 1.0, torch.float64)


def test_dense_attention_expands_gqa_like_manual() -> None:
    torch.manual_seed(1)
    q = torch.randn(1, 4, 3, 2, dtype=torch.float64)
    k = torch.randn(1, 1, 3, 2, dtype=torch.float64)
    v = torch.randn(1, 1, 3, 2, dtype=torch.float64)
    allow = torch.ones(3, 3, dtype=torch.bool)
    out = cal.dense_attention(q, k, v, allow, 0.25, torch.float64)
    k_exp = k.repeat_interleave(4, dim=1)
    v_exp = v.repeat_interleave(4, dim=1)
    expected = torch.softmax(q @ k_exp.transpose(-1, -2) * 0.25, dim=-1) @ v_exp
    assert torch.allclose(out, expected, rtol=0, atol=1e-12)


def test_rms_relative_error_known_value() -> None:
    ref = torch.linspace(-1.0, 1.0, 11, dtype=torch.float64)
    out = ref + 0.1
    # RMS(out - ref) / RMS(ref) = 0.1 / RMS(ref); linspace(-1,1,11) has mean square 0.4.
    assert cal.rms_relative_error(out, ref) == pytest.approx(0.1 / 0.4**0.5, rel=1e-9)
    assert cal.rms_relative_error(ref, ref) == 0.0


def test_rms_relative_error_rejects_zero_reference() -> None:
    with pytest.raises(ValueError, match="all zeros"):
        cal.rms_relative_error(torch.zeros(3), torch.zeros(3))


def test_max_abs_error() -> None:
    ref = torch.tensor([1.0, -2.0, 3.0])
    out = torch.tensor([1.5, -2.0, 2.0])
    assert cal.max_abs_error(out, ref) == pytest.approx(1.0)


def test_worst_head_rms_relative_error_picks_worst_head() -> None:
    # Two heads: head 0 perfect, head 1 perturbed by eps.
    ref = torch.ones(1, 2, 2, 2, dtype=torch.float64)
    out = ref.clone()
    out[:, 1] += 0.1
    assert cal.worst_head_rms_relative_error(out, ref) == pytest.approx(0.1, rel=1e-9)


def test_argmax_disagreement_rates() -> None:
    a = torch.tensor([[1.0, 0.0], [0.0, 1.0], [5.0, 1.0]])
    b = torch.tensor([[0.0, 1.0], [0.0, 1.0], [1.0, 5.0]])
    assert cal.argmax_disagreement(a, a) == 0.0
    assert cal.argmax_disagreement(a, b) == pytest.approx(2.0 / 3.0)
    with pytest.raises(ValueError, match="shape mismatch"):
        cal.argmax_disagreement(a, torch.zeros(1, 2))


def test_pairwise_table_is_complete_and_keyed() -> None:
    outputs = {
        "m1": torch.tensor([[[[1.0, 0.0]]]]),
        "m2": torch.tensor([[[[0.0, 1.0]]]]),
        "m3": torch.tensor([[[[1.0, 0.0]]]]),
    }
    table = cal.pairwise_argmax_disagreement(outputs)
    assert set(table) == {"m1_vs_m2", "m1_vs_m3", "m2_vs_m3"}
    assert table["m1_vs_m2"] == 1.0
    assert table["m1_vs_m3"] == 0.0


def test_error_table_skips_reference_and_reports_three_metrics() -> None:
    ref = torch.ones(1, 1, 2, 2, dtype=torch.float64)
    outputs = {
        "ref_method": ref,
        "other": ref + 0.5,
    }
    table = cal.error_table(outputs, ref, skip="ref_method")
    assert set(table) == {"other"}
    assert set(table["other"]) == {"rms_rel", "max_abs", "worst_head_rms_rel"}
    assert table["other"]["max_abs"] == pytest.approx(0.5)
    assert table["other"]["rms_rel"] == pytest.approx(0.5)


def test_pad_or_truncate() -> None:
    assert cal.pad_or_truncate([1, 2, 3, 4, 5], 3, 9) == [1, 2, 3]
    assert cal.pad_or_truncate([1, 2], 4, 9) == [1, 2, 9, 9]
    assert cal.pad_or_truncate([], 2, 7) == [7, 7]
    assert cal.pad_or_truncate([1, 2], 2, 9) == [1, 2]
    with pytest.raises(ValueError, match="length must be >= 1"):
        cal.pad_or_truncate([1], 0, 9)


def test_round_up_to_grid_125() -> None:
    assert cal.round_up_to_grid(0.003) == pytest.approx(0.005)
    assert cal.round_up_to_grid(0.01) == pytest.approx(0.01)
    assert cal.round_up_to_grid(0.019) == pytest.approx(0.02)
    assert cal.round_up_to_grid(0.03) == pytest.approx(0.05)
    assert cal.round_up_to_grid(0.6) == pytest.approx(1.0)
    assert cal.round_up_to_grid(1.0) == pytest.approx(1.0)
    assert cal.round_up_to_grid(12.0) == pytest.approx(20.0)
    with pytest.raises(ValueError):
        cal.round_up_to_grid(0.0)


def test_derive_gate_is_rounded_and_labeled_proposal() -> None:
    gate = cal.derive_gate(0.03)
    assert gate["proposed_max"] == pytest.approx(0.5)  # 10 * 0.03 = 0.3 -> round up to 0.5
    assert gate["safety_factor"] == cal.SAFETY_FACTOR
    assert "PROPOSAL" in gate["status"]
    assert "ratif" in gate["status"]
    with pytest.raises(ValueError):
        cal.derive_gate(0.0)


def test_derive_gate_monotone_in_worst_measured() -> None:
    low = cal.derive_gate(0.005)["proposed_max"]
    high = cal.derive_gate(0.05)["proposed_max"]
    assert high >= low


def test_render_calibration_md_marks_proposal_and_includes_numbers() -> None:
    summary = {
        "checkpoint": {"path": "ckpt"},
        "input": {"doc_lengths": [2, 2], "query_len": 2},
        "errors_vs_fp64_dense": {
            "bf16_fa2_segmented": {
                "rms_rel": 1.5e-2,
                "max_abs": 2.0e-1,
                "worst_head_rms_rel": 3.0e-2,
            }
        },
        "pairwise_argmax_disagreement": {"bf16_sdpa_vs_bf16_fa2_segmented": 0.01},
        "gate_proposal": {
            "proposed_max": 0.2,
            "derivation": "10x the worst measured candidate-vs-fp64 rms_rel (1.500e-02), "
            "rounded up on the 1-2-5 grid",
        },
    }
    md = cal.render_calibration_md(summary)
    assert "PROPOSAL" in md
    assert "NOT ratified" in md
    assert "0.2" in md
    assert "1.500e-02" in md
    assert "token_argmax_agreement == 1.0" in md
    assert "Limitations" in md


_GPU_AVAILABLE = torch.cuda.is_available()
_FA2_AVAILABLE = importlib.util.find_spec("flash_attn") is not None


@pytest.mark.skipif(not (_GPU_AVAILABLE and _FA2_AVAILABLE), reason="needs CUDA + flash-attn")
def test_fa2_segmented_close_to_dense_on_tiny_random_block() -> None:
    """Tiny GPU smoke: the production varlen path must track the dense mask semantics."""
    torch.manual_seed(2)
    doc_lengths, query_len = [4, 4], 2
    from ddi.masks import build_mask

    allow_mask = build_mask(doc_lengths, query_len, 0, "ddi_qf")
    q = torch.randn(1, 4, 10, 8, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, 2, 10, 8, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(1, 2, 10, 8, dtype=torch.bfloat16, device="cuda")
    dense = cal.dense_attention(q, k, v, allow_mask, 0.25, torch.float32)
    from ddi.segmented_fa2 import _segmented_prefill

    fa2 = _segmented_prefill(
        q, k, v, {"doc_lengths": doc_lengths, "query_len": query_len, "variant": "ddi_qf"}, 0.25
    ).to(q.dtype)
    rms_rel = cal.rms_relative_error(fa2, dense)
    assert torch.isfinite(fa2).all()
    assert rms_rel < 0.2  # bf16-vs-fp32 kernel noise on random data; loose on purpose
