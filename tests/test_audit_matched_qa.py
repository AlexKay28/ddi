from scripts.audit_matched_qa import paired_interval


def test_paired_interval_reports_constant_contrast():
    result = paired_interval([0.25, 0.25, 0.25])
    assert result["mean"] == 0.25
    assert result["ci95_paired_bootstrap"] == [0.25, 0.25]
