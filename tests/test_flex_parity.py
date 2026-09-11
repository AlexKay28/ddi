import pytest
import torch

from ddi.flex_parity import ParityMetrics, compare_logits, first_divergence, passes_prefill_gate


def test_compare_logits_reports_exact_match() -> None:
    logits = torch.tensor([[[1.0, 2.0], [3.0, 1.0]]])
    metrics = compare_logits(logits, logits.clone())
    assert metrics.max_abs_error == 0.0
    assert metrics.mean_abs_error == 0.0
    assert metrics.cosine_similarity == pytest.approx(1.0)
    assert metrics.token_argmax_agreement == 1.0


def test_compare_logits_reports_token_argmax_movement() -> None:
    reference = torch.tensor([[[2.0, 1.0], [3.0, 1.0]]])
    candidate = torch.tensor([[[1.0, 2.0], [3.0, 1.0]]])
    metrics = compare_logits(reference, candidate)
    assert metrics.max_abs_error == 1.0
    assert metrics.token_argmax_agreement == 0.5


def test_compare_logits_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        compare_logits(torch.tensor([float("nan")]), torch.tensor([0.0]))


def test_prefill_gate_requires_cosine_and_exact_argmax() -> None:
    assert passes_prefill_gate(ParityMetrics(0.0, 0.0, 0.999, 1.0))
    assert not passes_prefill_gate(ParityMetrics(0.0, 0.0, 0.998, 1.0))
    assert not passes_prefill_gate(ParityMetrics(0.0, 0.0, 1.0, 0.99))


def test_first_divergence_retains_qkv_equality() -> None:
    same = torch.ones(1)
    reference = [
        {
            "layer": 0,
            "scale": 0.5,
            "query": same,
            "key": same,
            "value": same,
            "attention_output": torch.tensor([1.0]),
        }
    ]
    candidate = [
        {
            "layer": 0,
            "scale": 0.5,
            "query": same.clone(),
            "key": same.clone(),
            "value": same.clone(),
            "attention_output": torch.tensor([2.0]),
        }
    ]
    result = first_divergence(reference, candidate)
    assert result is not None
    assert result["layer"] == 0
    assert result["qkv_equal"] == {"query": True, "key": True, "value": True}
    assert result["max_attention_abs_error"] == 1.0
