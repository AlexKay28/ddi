import json

import pytest

from scripts.run_v61_matched_qa import (
    StructuralMismatch,
    build_paired_rows,
    check_paired_ids,
    count_deltas,
    evaluate_decision,
    extract_development_ids,
    extract_sdpa_reference,
    load_base_records,
    paired_bootstrap,
    parse_budgets,
    sha_file,
    verify_structural_match,
)


def _base_record(record_id, prediction="Paris", em=1.0, f1=1.0, presence=1.0):
    return {
        "record_id": record_id,
        "request_id": f"hotpot_qa:0:{record_id}",
        "doc_lengths": [3, 4],
        "query_length": 5,
        "gold_answers": ["Paris"],
        "arms": {
            "A": {
                "prediction": prediction,
                "quality": {"em": em, "f1": f1, "presence": presence},
                "natural_generated_ids": [11, 22],
            }
        },
    }


def _candidate_row(record_id, em, f1, presence):
    return {
        "record_id": record_id,
        "quality": {"em": em, "f1": f1, "presence": presence},
    }


def test_sha_file_pins_exact_bytes(tmp_path):
    path = tmp_path / "artifact.json"
    path.write_bytes(b'{"a": 1}\n')
    assert (
        sha_file(path)
        == "e8c628edc9968ef0c668f54e0ba2636b35503357eb1aca0ddc828aeace432f67"
    )
    other = tmp_path / "other.json"
    other.write_bytes(b'{"a": 2}\n')
    assert sha_file(path) != sha_file(other)


def test_parse_budgets_reads_retained_decoding_params():
    manifest = {"budgets": {"natural_max_new_tokens": 48, "fixed_new_tokens": 16}}
    assert parse_budgets(manifest) == {
        "natural_max_new_tokens": 48,
        "fixed_new_tokens": 16,
    }


def test_extract_development_ids_validates_count():
    split = {"splits": {"development": {"count": 2, "record_ids": ["a", "b"]}}}
    assert extract_development_ids(split) == ["a", "b"]
    with pytest.raises(ValueError, match="count"):
        extract_development_ids({"splits": {"development": {"count": 3, "record_ids": ["a"]}}})


def test_extract_sdpa_reference_and_abort_on_missing_arm():
    records = [_base_record("a"), _base_record("b")]
    reference = extract_sdpa_reference(records)
    assert set(reference) == {"a", "b"}
    assert reference["a"]["quality"]["em"] == 1.0
    assert reference["a"]["gold_answers"] == ["Paris"]
    broken = [{"record_id": "c", "arms": {}}]
    with pytest.raises(ValueError, match="no arm A"):
        extract_sdpa_reference(broken)


def test_check_paired_ids_aborts_on_any_id_mismatch():
    records = [_base_record("a"), _base_record("b")]
    check_paired_ids(records, ["a", "b"])
    with pytest.raises(ValueError, match="development ids"):
        check_paired_ids(records, ["a", "c"])
    with pytest.raises(ValueError, match="development ids"):
        check_paired_ids(records, ["a"])
    with pytest.raises(ValueError, match="development ids"):
        check_paired_ids(records, ["b", "a"])


def test_verify_structural_match_aborts_on_geometry_change():
    retained = {"record_id": "a", "doc_lengths": [3, 4], "query_length": 5}
    verify_structural_match(retained, [3, 4], 5)
    with pytest.raises(StructuralMismatch, match="not identical"):
        verify_structural_match(retained, [3, 5], 4)


def test_paired_bootstrap_is_deterministic_percentile_ci():
    deltas = [0.1, -0.1, 0.0, 0.2]
    first = paired_bootstrap(deltas, seed=47, resamples=1000)
    second = paired_bootstrap(deltas, seed=47, resamples=1000)
    assert first == second
    assert first["mean"] == pytest.approx(0.05)
    assert first["ci95"][0] <= first["mean"] <= first["ci95"][1]
    constant = paired_bootstrap([0.3] * 10, seed=47, resamples=100)
    assert constant["ci95"] == [pytest.approx(0.3), pytest.approx(0.3)]


def test_count_deltas_uses_strict_threshold_bands():
    counts = count_deltas([0.05, -0.05, 0.0, 0.02, -0.02], threshold=0.02)
    assert counts == {"improved": 1, "flat": 3, "degraded": 1}


def _analysis(em_ci, f1_ci):
    def entry(ci):
        return {"mean": sum(ci) / 2, "ci95": ci}

    return {"em": entry(em_ci), "f1": entry(f1_ci), "presence": entry([-0.5, 0.5])}


def test_evaluate_decision_requires_both_lower_bounds():
    assert evaluate_decision(_analysis([-0.01, 0.05], [-0.02, 0.1]))["non_inferior"] is True
    verdict = evaluate_decision(_analysis([-0.01, 0.05], [-0.0201, 0.1]))
    assert verdict["non_inferior"] is False
    assert verdict["rule"] == "non-inferior iff CI LB >= -0.02 for BOTH EM and F1"
    assert set(verdict["ci_lower_bounds"]) == {"em", "f1"}


def test_build_paired_rows_pairs_by_record_id_and_aborts_on_unknown_id():
    reference = extract_sdpa_reference([_base_record("a"), _base_record("b")])
    paired = build_paired_rows(
        reference, [_candidate_row("b", 0.0, 0.5, 1.0), _candidate_row("a", 1.0, 1.0, 1.0)]
    )
    assert [entry["record_id"] for entry in paired] == ["b", "a"]
    with pytest.raises(ValueError, match="no retained SDPA reference"):
        build_paired_rows(reference, [_candidate_row("zzz", 0.0, 0.0, 0.0)])


def test_load_base_records_reads_synthetic_base_dir(tmp_path):
    records = [_base_record("a"), _base_record("b")]
    with (tmp_path / "requests.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    loaded = load_base_records(tmp_path)
    assert [record["record_id"] for record in loaded] == ["a", "b"]
    manifest = {
        "checkpoint_identity": {"mode": "directory_tree", "sha256": "x"},
        "budgets": {"natural_max_new_tokens": 48, "fixed_new_tokens": 16},
        "arms": {"A": {"backend": "sdpa", "execution": "original_monolithic_ddi_qf"}},
        "source_manifest": {"record_ids": ["a", "b"]},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    parsed = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert parse_budgets(parsed)["natural_max_new_tokens"] == 48
    assert parsed["arms"]["A"]["backend"] == "sdpa"
