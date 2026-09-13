import pytest

from scripts.validate_matched_qa import _audit, bootstrap_ratio


def _row(request_id, variant, digest="same", repeats=5):
    return {
        "request_id": request_id,
        "variant": variant,
        "input_ids_sha256": digest,
        "fixed": {"raw_request_ms": [1.0] * repeats},
    }


def test_matched_audit_rejects_mismatched_inputs_and_incomplete_arms():
    with pytest.raises(AssertionError, match="input IDs"):
        _audit([_row("r", "full", "a"), _row("r", "ddi_qf", "b")], ["full", "ddi_qf"])
    with pytest.raises(AssertionError, match="incomplete"):
        _audit([_row("r", "full")], ["full", "ddi_qf"])


def test_matched_audit_requires_five_repeats():
    with pytest.raises(AssertionError, match="five"):
        _audit([_row("r", "full", repeats=4)], ["full"])


def test_bootstrap_ratio_is_paired_and_deterministic():
    result = bootstrap_ratio([2.0, 4.0], [1.0, 2.0])
    assert result["per_request"] == [2.0, 2.0]
    assert result["mean"] == 2.0
    assert result["ci95_paired_bootstrap"] == [2.0, 2.0]
