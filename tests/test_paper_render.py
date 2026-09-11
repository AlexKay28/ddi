from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from paper import expanded
from paper.render import (
    DEFAULT_EVIDENCE,
    DEFAULT_OUTPUT,
    GENERATED_FILES,
    EvidenceError,
    load_evidence,
    render,
    verify_sources,
)


def test_checked_in_artifacts_match_renderer() -> None:
    evidence = load_evidence()
    rendered = render(evidence)

    assert tuple(rendered) == GENERATED_FILES
    for name, content in rendered.items():
        assert (DEFAULT_OUTPUT / name).read_text(encoding="utf-8") == content


def test_all_available_sources_match_preserved_hashes() -> None:
    assert verify_sources(load_evidence()) == []


def test_missing_required_source_fails(tmp_path: Path) -> None:
    evidence = load_evidence()
    evidence["sources"] = {
        "missing": {
            "path": "missing.jsonl",
            "sha256": "0" * 64,
            "bytes": 1,
            "required": True,
        }
    }

    with pytest.raises(EvidenceError, match="source is missing"):
        verify_sources(evidence, tmp_path)


def test_available_historical_source_hash_mismatch_fails(tmp_path: Path) -> None:
    (tmp_path / "raw.jsonl").write_text("changed\n", encoding="utf-8")
    evidence = {
        "sources": {
            "raw": {
                "path": "raw.jsonl",
                "sha256": "0" * 64,
                "bytes": 8,
                "required": False,
            }
        }
    }

    with pytest.raises(EvidenceError, match="SHA-256 mismatch"):
        verify_sources(evidence, tmp_path)


def test_malformed_evidence_fails_loudly(tmp_path: Path) -> None:
    evidence = json.loads(DEFAULT_EVIDENCE.read_text(encoding="utf-8"))
    del evidence["artifacts"]["T5"]["rows"]
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")

    with pytest.raises(EvidenceError, match=r"artifacts.T5.rows"):
        load_evidence(path)


def test_incomplete_seed_evidence_fails_loudly() -> None:
    evidence = copy.deepcopy(load_evidence())
    evidence["artifacts"]["F5"]["rows"][0][1] = [1.0, 2.0]

    with pytest.raises(EvidenceError, match="needs three seed values"):
        render(evidence)


def test_withdrawn_cache_evidence_is_not_plotted() -> None:
    evidence = load_evidence()
    assert evidence["artifacts"]["F5"]["status"] == "withdrawn"
    rendered = render(evidence)["F5.tex"]
    assert "Throughput claim withdrawn" in rendered
    assert "\\rule" not in rendered
    assert "cache_v51_throughput.jsonl" in rendered


def test_historical_cache_rows_remain_renderable_for_audit() -> None:
    evidence = copy.deepcopy(load_evidence())
    del evidence["artifacts"]["F5"]["status"]
    assert render(evidence)["F5.tex"].count("\\rule") == 3


def test_transfer_estimands_table_preserves_the_legacy_statistic_label() -> None:
    rendered = render(load_evidence())["T8_transfer_estimands.tex"]

    assert "normalized remaining tuned gap" in rendered
    assert "$\\Delta$ DDI" in rendered


def test_visualizations_are_rendered_from_audited_artifacts() -> None:
    rendered = render(load_evidence())

    assert "(25,20.160332)" in rendered["V2_efficiency_intuition.tex"]
    assert "Measured SDPA prefill" in rendered["V2_efficiency_intuition.tex"]
    assert "paired change" in rendered["V3_transfer_estimands.tex"]
    assert "COLD MISS" in rendered["V1_mask_geometry.tex"]
    assert "WARM HIT" in rendered["V1_mask_geometry.tex"]
    assert "Skip document prefill" in rendered["V1_mask_geometry.tex"]
    assert "V53 does not measure a sparse kernel" in rendered["V1_mask_geometry.tex"]
    assert "diagnostic only" in rendered["V4_production_opportunity.tex"]
    assert "(2,1.903480)" in rendered["V4_production_opportunity.tex"]
    assert "Behavioral fidelity is not QA quality" in rendered["V7_hit_diagnostics.tex"]
    assert "ANALYTICAL ONLY" in rendered["V9_analytical_surface.tex"]
    assert "77.96" in rendered["V9_analytical_surface.tex"]
    assert "Measured (validation program)" in rendered["V10_coverage.tex"]
    assert "Still missing" in rendered["V10_coverage.tex"]
    assert "Non-inferiority established" in rendered["V10_coverage.tex"]
    for name, content in rendered.items():
        if name.startswith("V"):
            assert "resizebox" not in content
            assert "group size=1 by 2" not in content


def test_exact_geometry_is_not_the_cache_hit_rate() -> None:
    full, ddi, remaining = expanded.pair_cost(5, 500, 32, 709 / 750)
    assert (full, ddi) == (3206778, 706778)
    assert 1 - ddi / full == pytest.approx(0.7795987124771344)
    assert remaining < ddi
    assert expanded.pair_cost(25, 512, 64)[:2] == (82747680, 4104480)


def test_cache_bins_preserve_pairing_denominators_and_failed_gates() -> None:
    stats = expanded.cache_analysis(load_evidence())
    assert [(b["hits"], b["n"], b["equal"]) for b in stats["bins"]] == [
        (3, 4, 4),
        (4, 33, 28),
        (5, 113, 99),
    ]
    assert stats["hits"] == 709
    assert stats["lookups"] == 750
    assert stats["validation_failures"] == 25
    assert stats["numerical_failures"] == 192
    assert stats["bins"][0]["median_paired_ratio"] == pytest.approx(0.8651759621814985)
    assert stats["bins"][2]["cached_median_ms"] == pytest.approx(498.2918053865433)


def test_t5_common_ids_and_means_are_verified() -> None:
    assert len(expanded.quality_analysis(load_evidence())["common_record_ids"]) == 100
    evidence = copy.deepcopy(load_evidence())
    evidence["artifacts"]["T5"]["rows"][0][1] = 0.99
    with pytest.raises(ValueError, match="aggregate mismatch"):
        expanded.quality_analysis(evidence)


@pytest.mark.parametrize("mutation", ["duplicate", "geometry", "work", "pair", "gate"])
def test_cache_trace_corruption_is_rejected(monkeypatch, mutation) -> None:
    evidence = load_evidence()
    real_source = expanded.source

    def altered(evidence, name, root):
        rows = real_source(evidence, name, root)
        if name == "cache_v53_timing":
            if mutation == "duplicate":
                rows[-1] = rows[0]
            elif mutation == "geometry":
                rows[0]["doc_lengths"][0] = 501
            elif mutation == "work":
                rows[0]["decode_forwards"] = 16
            elif mutation == "pair":
                rows[0]["generated_ids"][0] = -1
        if name == "cache_v53_validation" and mutation == "gate":
            rows[0]["comparison"]["numerical_pass"] = True
        return rows

    monkeypatch.setattr(expanded, "source", altered)
    with pytest.raises(ValueError):
        expanded.cache_analysis(evidence)


def test_new_cache_sources_are_hash_pinned_and_required() -> None:
    evidence = load_evidence()
    for suffix in ("summary", "timing", "validation"):
        name = f"cache_v53_{suffix}"
        assert evidence["sources"][name]["required"] is True
        modified = copy.deepcopy(evidence)
        modified["sources"][name]["sha256"] = "0" * 64
        with pytest.raises(ValueError, match="Source hash mismatch"):
            expanded.source(modified, name)


def test_flex_history_uses_detailed_latencies_in_independent_scopes() -> None:
    history = expanded.flex_history(load_evidence())
    expected = {
        "kernel": [(0.512, 0.532), (2.276, 1.583), (10.293, 3.181), (142.292, 17.263)],
        "model_prefill": [
            (140.86, 140.48),
            (405.57, 385.90),
            (1268.09, 1049.49),
            (8059.75, 4541.25),
        ],
    }
    for scope, timings in expected.items():
        rows = history[scope]
        assert [(r["n"], r["length"]) for r in rows] == [(3, 512), (9, 512), (25, 512), (25, 2048)]
        assert [(r["full_ms"], r["ddi_ms"]) for r in rows] == timings
        assert [r["ratio"] for r in rows] == pytest.approx([f / d for f, d in timings])
    assert history["kernel"][-1]["ratio"] == pytest.approx(8.2426, abs=5e-5)
    assert history["model_prefill"][-1]["ratio"] == pytest.approx(1.7748, abs=5e-5)


@pytest.mark.parametrize("name", ["flex_kernel_history", "flex_model_history"])
@pytest.mark.parametrize("drift", ["hash", "size", "missing"])
def test_flex_retained_source_drift_fails(tmp_path, name, drift) -> None:
    evidence = load_evidence()
    spec = evidence["sources"][name]
    assert spec["required"] is True
    path = tmp_path / spec["path"]
    path.parent.mkdir(parents=True)
    data = (expanded.ROOT / spec["path"]).read_bytes()
    if drift != "missing":
        path.write_bytes(b"X" + data[1:] if drift == "hash" else data + b"\n")
    evidence["sources"] = {name: spec}
    with pytest.raises(EvidenceError, match="source (SHA-256 mismatch|size mismatch|is missing)"):
        verify_sources(evidence, tmp_path)
    with pytest.raises((ValueError, FileNotFoundError)):
        expanded.source(evidence, name, tmp_path)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "shape", "zero"])
def test_flex_parser_rejects_incomplete_or_invalid_details(monkeypatch, mutation) -> None:
    real_source = expanded.source

    def altered(evidence, name, root):
        text = real_source(evidence, name, root)
        if name == "flex_model_history":
            line = "  flex full = 8059.75ms"
            replacement = {
                "missing": "",
                "duplicate": line + "\n" + line,
                "shape": line,
                "zero": "  flex full = 0.00ms",
            }[mutation]
            text = text.replace(line, replacement)
            if mutation == "shape":
                text = text.replace("N=25 L=2048 T=51264", "N=24 L=2048 T=51264")
        return text

    monkeypatch.setattr(expanded, "source", altered)
    with pytest.raises(ValueError, match="Flex model_prefill"):
        expanded.flex_history(load_evidence())


def test_flex_and_amdahl_figures_keep_evidence_boundaries() -> None:
    rendered = render(load_evidence())
    flex = rendered["V11_flex_history.tex"]
    assert "group size=2 by 1" in flex
    assert "Raw Flex kernel" in flex
    assert "Whole-model Flex prefill" in flex
    assert flex.count("HISTORICAL / UNVALIDATED") == 2
    # Commas inside a tick label must not split the pgfplots label list.
    assert flex.count(r"xticklabels={{$(3,512)$},{$(9,512)$},{$(25,512)$},{$(25,2048)$}}") == 2
    assert "Independent experiments; not a matched timing decomposition" in flex
    assert "3.98" not in flex
    for scope in expanded.flex_history(load_evidence()).values():
        for index, row in enumerate(scope):
            assert f"({index},{row['ratio']:.6f})" in flex
    amdahl = rendered["V12_amdahl.tex"]
    assert "ILLUSTRATIVE ONLY" in amdahl
    assert "not measured attention fractions" in amdahl
    assert "xmin=1,xmax=10" in amdahl
    assert amdahl.count("\\addplot[") == 4
    for style in ("ddiBlue,solid", "ddiGold,dashed", "ddiGreen,dotted", "ddiPurple,dashdotted"):
        assert style in amdahl
    assert "(2.0,1.052632)" in amdahl
    assert "(3.0,1.071429)" in amdahl


@pytest.mark.parametrize("fraction", [0.1, 0.3, 0.5, 0.8])
def test_amdahl_is_illustrative_formula(fraction) -> None:
    assert expanded.amdahl_speedup(fraction, 1) == 1
    assert expanded.amdahl_speedup(fraction, 10) == pytest.approx(
        1 / (1 - fraction + fraction / 10)
    )
    assert expanded.amdahl_speedup(0.1, 2) == pytest.approx(1.0526316)
    assert expanded.amdahl_speedup(0.1, 3) == pytest.approx(1.0714286)


NEW_PINNED_SOURCES = (
    "v57_matched_summary",
    "v57_matched_post_audit",
    "v58_cache_qa_summary",
    "v59_grid_summary",
    "v59_grid_analysis",
    "v60_confirmation_result",
    "v60_confirmation_requests",
    "v64_qa_recomputation",
    "v64_timing_recomputation",
    "v65_benchmark_result",
    "v65_combined_result",
    "v61_timing_result",
)


@pytest.mark.parametrize("name", NEW_PINNED_SOURCES)
def test_new_result_sources_are_hash_pinned_and_required(name) -> None:
    evidence = load_evidence()
    spec = evidence["sources"][name]
    assert spec["required"] is True
    modified = copy.deepcopy(evidence)
    modified["sources"][name]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Source hash mismatch"):
        expanded.source(modified, name)


@pytest.mark.parametrize("name", NEW_PINNED_SOURCES)
@pytest.mark.parametrize("drift", ["hash", "size", "missing"])
def test_new_result_source_drift_fails(tmp_path, name, drift) -> None:
    evidence = load_evidence()
    spec = evidence["sources"][name]
    assert spec["required"] is True
    path = tmp_path / spec["path"]
    path.parent.mkdir(parents=True)
    data = (expanded.ROOT / spec["path"]).read_bytes()
    if drift != "missing":
        path.write_bytes(b"X" + data[1:] if drift == "hash" else data + b"\n")
    evidence["sources"] = {name: spec}
    with pytest.raises(EvidenceError, match="source (SHA-256 mismatch|size mismatch|is missing)"):
        verify_sources(evidence, tmp_path)


def test_new_analysis_functions_verify_pinned_numbers() -> None:
    evidence = load_evidence()
    serving = expanded.v65_serving_analysis(evidence)
    assert serving["byte_identical_ac"] == [24, 24]
    qa = expanded.qa_confirmation_analysis(evidence)
    assert qa["v64_verdict"] == "inconclusive"
    assert qa["v65_non_inferior"] is False
    assert qa["floor"] == -0.02
    grid = expanded.ttft_grid_analysis(evidence)
    assert grid["cells"] == 9
    assert grid["crossover_cells"] == 5
    assert grid["summary_crossover_cells"] == 17
    matched = expanded.v57_matched_analysis(evidence)
    assert matched["n_requests"] == 100
    reference = expanded.reference_artifact_analysis(evidence)
    assert reference == {
        "v58": "diagnostic_not_admissible",
        "v61_cells": 8,
        "v64_timing_recheck": "pass",
    }


def test_reference_artifacts_are_pinned_without_rendered_tables() -> None:
    artifacts = load_evidence()["artifacts"]
    assert artifacts["V58_cache_qa"]["status"] == "diagnostic_not_admissible"
    assert artifacts["V61_timing_grid"]["status"] == "cross_backend_timing_reference"
    assert artifacts["V64_timing_recheck"]["status"] == "descriptive_reconstruction"


def test_v65_serving_table_is_rendered_from_pinned_results() -> None:
    t9 = render(load_evidence())["T9_v65_serving.tex"]
    for needle in (
        "127.52",
        "113.45",
        "145.11",
        "417.42",
        "62.30",
        "0.4885",
        "3.2735",
        "24/24",
        "58.78",
        "2.1713",
        "386.36",
        "0.3303",
        "7.479",
        "5.216",
    ):
        assert needle in t9
    assert "sha256=f0be727746f7788e668d39c57d29d56df047fad8cc64641a5574ea9bdcfac3a5" in t9


def test_qa_confirmation_table_is_rendered_from_pinned_audits() -> None:
    t10 = render(load_evidence())["T10_qa_confirmation.tex"]
    for needle in (
        "+0.041667",
        "-0.083333",
        "+0.021982",
        "inconclusive",
        "-0.0700",
        "-0.0095",
        "not non-inferior",
        "-0.02",
        "78/100",
        "91/100",
    ):
        assert needle in t10


def test_v59_crossover_table_is_rendered_from_the_pinned_grid() -> None:
    t11 = render(load_evidence())["T11_v59_crossover.tex"]
    for needle in ("0.716", "1.850", "2.254", "2.635", "0.616", "none"):
        assert needle in t11


def test_v57_matched_table_is_rendered_from_pinned_contrasts() -> None:
    t12 = render(load_evidence())["T12_v57_matched.tex"]
    for needle in ("0.5600", "0.7034", "0.3000", "0.4153", "0.2600", "0.2881", "1.0013", "0.9955"):
        assert needle in t12


@pytest.mark.parametrize("artifact,width", [
    ("T9_v65_serving", 4),
    ("T10_qa_confirmation", 6),
    ("T11_v59_crossover", 8),
    ("T12_v57_matched", 6),
])
def test_new_tables_reject_wrong_row_width(tmp_path: Path, artifact: str, width: int) -> None:
    evidence = json.loads(DEFAULT_EVIDENCE.read_text(encoding="utf-8"))
    evidence["artifacts"][artifact]["rows"][0].append("extra")
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")

    with pytest.raises(EvidenceError, match=artifact):
        load_evidence(path)


@pytest.mark.parametrize(
    ("artifact", "field", "index", "position", "value", "match"),
    [
        ("T9_v65_serving", "rows", 0, 3, [113.45], "does not match artifact shape"),
        ("T9_v65_serving", "rows", 6, 2, 0.9, "does not match artifact"),
        ("T9_v65_serving", "rows", 0, 2, float("inf"), "does not match artifact"),
        ("T10_qa_confirmation", "rows", 2, 5, "non-inferior", "verdict mismatch"),
        ("T10_qa_confirmation", "rows", 0, 2, 0.5, "does not match artifact"),
        ("T11_v59_crossover", "rows", 8, 4, 9.9, "does not match artifact"),
        ("T11_v59_crossover", "rows", 2, 7, 0.9, "does not match artifact"),
        ("T12_v57_matched", "rows", 0, 3, 0.99, "does not match artifact"),
        ("T12_v57_matched", "timing_ratios", 0, 1, 2.0, "does not match artifact"),
    ],
)
def test_new_table_mutations_fail_against_pinned_artifacts(
    artifact, field, index, position, value, match
) -> None:
    evidence = copy.deepcopy(load_evidence())
    evidence["artifacts"][artifact][field][index][position] = value

    with pytest.raises(ValueError, match=match):
        render(evidence)
