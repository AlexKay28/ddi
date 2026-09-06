import pytest
import torch

from ddi.flex_parity import compare_logits


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
