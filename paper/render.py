#!/usr/bin/env python3
"""Deterministically render paper artifacts from preserved audited evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

if __package__:
    from .expanded import render_visuals
else:
    from expanded import render_visuals

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE = ROOT / "paper" / "evidence.json"
DEFAULT_OUTPUT = ROOT / "paper" / "generated"
EXPECTED_ARTIFACTS = {
    "T1_mask_taxonomy",
    "T2",
    "T3",
    "T4",
    "T4_16L",
    "T4_restart",
    "T5",
    "T6_delta",
    "T7_transfer",
    "T8_transfer_estimands",
    "T9_v65_serving",
    "T10_qa_confirmation",
    "T11_v59_crossover",
    "T12_v57_matched",
    "T13_v66_boundary",
    "T14_v69_attribution",
    "T15_v70_gates",
    "T16_v68_anatomy",
    "T17_v71_boundary",
    "F5",
    "V53_cache_diagnostic",
    "V11_flex_history",
    "V58_cache_qa",
    "V61_timing_grid",
    "V64_timing_recheck",
    "V71_sweep",
    "V72_replication",
    "V14_v71_crossover_L512",
    "V15_v71_crossover_L128",
    "V16_gate_deltas",
    "V17_program_timeline",
}
GENERATED_FILES = (
    "T1_mask_taxonomy.md",
    "T1_mask_taxonomy.tex",
    "V1_mask_geometry.tex",
    "T2.tex",
    "T3.tex",
    "V2_efficiency_intuition.tex",
    "V4_production_opportunity.tex",
    "T4.tex",
    "T4_16L.tex",
    "T4_restart.tex",
    "T5.tex",
    "T6_delta.tex",
    "T7_transfer.tex",
    "T8_transfer_estimands.tex",
    "T9_v65_serving.tex",
    "T10_qa_confirmation.tex",
    "T11_v59_crossover.tex",
    "T12_v57_matched.tex",
    "T13_v66_crossover.tex",
    "T14_v69_attribution.tex",
    "T15_v70_gates.tex",
    "T16_v68_anatomy.tex",
    "T17_v71_boundary.tex",
    "V3_transfer_estimands.tex",
    "F5.tex",
    "V5_doc_quality.tex",
    "V6_mask_quality.tex",
    "V7_hit_diagnostics.tex",
    "V8_cache_requests.tex",
    "V9_analytical_surface.tex",
    "V10_coverage.tex",
    "V11_flex_history.tex",
    "V12_amdahl.tex",
    "V14_crossover_chart.tex",
    "V15_crossover_L128.tex",
    "V16_gate_deltas.tex",
    "V17_program_timeline.tex",
    "V18_anatomy.tex",
    "A1_cache_bins.tex",
    "analysis.json",
)
T_CRITICAL_95_DF2 = 4.302652729911275


class EvidenceError(ValueError):
    """The preserved evidence or its provenance is invalid."""


def _reject_json_constant(value: str) -> None:
    raise EvidenceError(f"non-finite JSON number is not allowed: {value}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def _rows(artifacts: dict[str, Any], name: str, width: int) -> list[list[Any]]:
    artifact = artifacts[name]
    _require(isinstance(artifact, dict), f"artifacts.{name} must be an object")
    rows = artifact.get("rows")
    _require(isinstance(rows, list) and rows, f"artifacts.{name}.rows must be non-empty")
    for index, row in enumerate(rows):
        _require(
            isinstance(row, list) and len(row) == width,
            f"artifacts.{name}.rows[{index}] must contain {width} values",
        )
    return rows


def load_evidence(path: Path = DEFAULT_EVIDENCE) -> dict[str, Any]:
    try:
        evidence = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant
        )
    except FileNotFoundError as error:
        raise EvidenceError(f"evidence file is missing: {path}") from error
    except json.JSONDecodeError as error:
        raise EvidenceError(f"evidence file is malformed: {path}: {error}") from error

    _require(isinstance(evidence, dict), "evidence root must be an object")
    _require(evidence.get("schema_version") == 1, "unsupported evidence schema_version")
    sources = evidence.get("sources")
    artifacts = evidence.get("artifacts")
    _require(isinstance(sources, dict) and sources, "sources must be a non-empty object")
    _require(isinstance(artifacts, dict), "artifacts must be an object")
    _require(
        set(artifacts) == EXPECTED_ARTIFACTS,
        f"artifact set mismatch: expected {sorted(EXPECTED_ARTIFACTS)}, got {sorted(artifacts)}",
    )

    for name, source in sources.items():
        _require(isinstance(source, dict), f"sources.{name} must be an object")
        _require(
            set(source) == {"path", "sha256", "bytes", "required"},
            f"sources.{name} has invalid keys",
        )
        _require(
            isinstance(source["path"], str) and source["path"], f"sources.{name}.path is invalid"
        )
        _require(
            isinstance(source["sha256"], str)
            and len(source["sha256"]) == 64
            and all(char in "0123456789abcdef" for char in source["sha256"]),
            f"sources.{name}.sha256 is invalid",
        )
        _require(
            type(source["bytes"]) is int and source["bytes"] > 0,
            f"sources.{name}.bytes is invalid",
        )
        _require(isinstance(source["required"], bool), f"sources.{name}.required must be boolean")

    for name, artifact in artifacts.items():
        _require(isinstance(artifact, dict), f"artifacts.{name} must be an object")
        source_names = artifact.get("sources")
        _require(
            isinstance(source_names, list) and source_names, f"artifacts.{name}.sources is empty"
        )
        unknown = set(source_names) - set(sources)
        _require(not unknown, f"artifacts.{name} names unknown sources: {sorted(unknown)}")

    for name, width in {
        "T1_mask_taxonomy": 5,
        "T2": 8,
        "T3": 8,
        "T4": 5,
        "T4_16L": 5,
        "T4_restart": 3,
        "T5": 5,
        "T6_delta": 3,
        "T7_transfer": 5,
        "T8_transfer_estimands": 7,
        "T9_v65_serving": 4,
        "T10_qa_confirmation": 6,
        "T11_v59_crossover": 8,
        "T12_v57_matched": 6,
        "T13_v66_boundary": 8,
        "T14_v69_attribution": 3,
        "T15_v70_gates": 5,
        "T16_v68_anatomy": 2,
        "T17_v71_boundary": 6,
        "F5": 2,
        "V53_cache_diagnostic": 2,
    }.items():
        _rows(artifacts, name, width)
    return evidence


def verify_sources(
    evidence: dict[str, Any], repo_root: Path = ROOT, *, require_all: bool = False
) -> list[str]:
    missing_historical: list[str] = []
    for name, source in evidence["sources"].items():
        relative = Path(source["path"])
        _require(
            not relative.is_absolute() and ".." not in relative.parts,
            f"sources.{name}.path escapes the repository",
        )
        path = repo_root / relative
        if not path.is_file():
            if source["required"] or require_all:
                raise EvidenceError(f"source is missing: {source['path']}")
            missing_historical.append(source["path"])
            continue
        data = path.read_bytes()
        actual_hash = hashlib.sha256(data).hexdigest()
        _require(len(data) == source["bytes"], f"source size mismatch: {source['path']}")
        _require(actual_hash == source["sha256"], f"source SHA-256 mismatch: {source['path']}")
    return missing_historical


def _source_header(evidence: dict[str, Any], name: str) -> str:
    parts = []
    for source_name in evidence["artifacts"][name]["sources"]:
        source = evidence["sources"][source_name]
        parts.append(f"{source['path']} sha256={source['sha256']}")
    return (
        "% Generated by paper/render.py from paper/evidence.json.\n% Sources: "
        + "; ".join(parts)
        + ".\n"
    )


def _tex_name(value: str) -> str:
    return value.replace("_", r"\_")


def _number(value: float | None, digits: int = 2) -> str:
    return "---" if value is None else f"{value:.{digits}f}"


def _signed(value: float, digits: int = 2) -> str:
    return f"{value:+.{digits}f}"


def _table(lines: list[str]) -> str:
    return "\n".join(lines) + "\n"


def _render_t1(evidence: dict[str, Any]) -> dict[str, str]:
    rows = _rows(evidence["artifacts"], "T1_mask_taxonomy", 5)
    tex = [
        _source_header(evidence, "T1_mask_taxonomy").rstrip(),
        r"\begin{tabular}{lcccc}",
        r"\toprule",
    ]
    tex.append(
        r"Variant & Doc$\leftrightarrow$Doc & Query$\leftrightarrow$Doc & "
        r"Answer$\leftrightarrow$Doc & Causal \\"
    )
    tex.append(r"\midrule")
    for variant, doc, query, answer, causal in rows:
        tex.append(
            f"\\texttt{{{_tex_name(variant)}}} & {doc} & {query} & {answer} & {causal} " + r"\\"
        )
    tex.extend([r"\bottomrule", r"\end{tabular}"])

    md = [
        "<!-- Generated by paper/render.py from paper/evidence.json. -->",
        "",
        "| variant | document isolation | query to documents | answer to documents | causality |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        md.append(f"| `{row[0]}` | {row[1]} | {row[2]} | {row[3]} | {row[4]} |")
    return {"T1_mask_taxonomy.md": _table(md), "T1_mask_taxonomy.tex": _table(tex)}


def _render_t2(evidence: dict[str, Any]) -> str:
    rows = _rows(evidence["artifacts"], "T2", 8)
    lines = [
        _source_header(evidence, "T2").rstrip(),
        r"\begin{tabular}{llccc ccc}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{EM} & \multicolumn{3}{c}{F1} \\",
        r"model & mask & hotpot & musique & synth & hotpot & musique & synth \\",
        r"\midrule",
    ]
    previous_model = None
    counts = {model: sum(row[0] == model for row in rows) for model in {row[0] for row in rows}}
    for model, mask, *values in rows:
        if previous_model is not None and model != previous_model:
            lines.append(r"\midrule")
        model_cell = (
            rf"\multirow{{{counts[model]}}}{{*}}{{{model}}}" if model != previous_model else ""
        )
        cells = " & ".join(_number(value) for value in values)
        lines.append(f"{model_cell} & {_tex_name(mask)} & {cells} " + r"\\")
        previous_model = model
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return _table(lines)


def _render_t3(evidence: dict[str, Any]) -> str:
    lines = [
        _source_header(evidence, "T3").rstrip(),
        r"\begin{tabular}{rccccccr}",
        r"\toprule",
        r" & \multicolumn{3}{c}{TTFT (ms)} & \multicolumn{3}{c}{TPOT (ms)} & \\",
        r"$N$ & full & ddi\_qf & blind & full & ddi\_qf & blind & sparsity \\",
        r"\midrule",
    ]
    for n, *values in _rows(evidence["artifacts"], "T3", 8):
        lines.append(f"{n} & " + " & ".join(f"{value:.2f}" for value in values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return _table(lines)


def _render_t4(evidence: dict[str, Any]) -> str:
    lines = [
        _source_header(evidence, "T4").rstrip(),
        r"\begin{tabular}{llcc cc}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{hotpot} & \multicolumn{2}{c}{musique} \\",
        r"checkpoint & mask & EM & $\Delta$ & EM & $\Delta$ \\",
        r"\midrule",
    ]
    rows = _rows(evidence["artifacts"], "T4", 5)
    for index, (checkpoint, hotpot_full, hotpot_ddi, musique_full, musique_ddi) in enumerate(rows):
        if index:
            lines.append(r"\midrule")
        hotpot_delta = hotpot_full - hotpot_ddi
        musique_delta = musique_full - musique_ddi
        lines.append(
            rf"\multirow{{2}}{{*}}{{{checkpoint}}} & full & {hotpot_full:.2f} & "
            rf"{_signed(hotpot_delta)} & {musique_full:.2f} & "
            rf"{_signed(musique_delta)} \\"
        )
        lines.append(
            rf" & ddi\_qf & {hotpot_ddi:.2f} & {_signed(hotpot_delta)} & "
            rf"{musique_ddi:.2f} & {_signed(musique_delta)} \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return _table(lines)


def _render_t4_16l(evidence: dict[str, Any]) -> str:
    lines = [
        _source_header(evidence, "T4_16L").rstrip(),
        r"\begin{tabular}{llcc cc}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{100 records} & \multicolumn{2}{c}{500 records} \\",
        r"dataset & mask & EM & $\Delta$ & EM & gap \\",
        r"\midrule",
    ]
    for index, (dataset, full_100, ddi_100, full_500, ddi_500) in enumerate(
        _rows(evidence["artifacts"], "T4_16L", 5)
    ):
        if index:
            lines.append(r"\midrule")
        delta_100 = full_100 - ddi_100
        delta_500 = full_500 - ddi_500
        lines.append(
            rf"\multirow{{2}}{{*}}{{{dataset}}} & full & {full_100:.2f} & "
            rf"\multirow{{2}}{{*}}{{{delta_100:+.2f}}} & {full_500:.3f} & "
            rf"\multirow{{2}}{{*}}{{{delta_500:.3f}}} \\"
        )
        lines.append(rf" & ddi\_qf & {ddi_100:.2f} & & {ddi_500:.3f} & \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return _table(lines)


def _render_t4_restart(evidence: dict[str, Any]) -> str:
    lines = [
        _source_header(evidence, "T4_restart").rstrip(),
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{hotpot EM} & \\",
        r"evaluator & sequential & restart & restart - sequential \\",
        r"\midrule",
    ]
    for evaluator, sequential, restart in _rows(evidence["artifacts"], "T4_restart", 3):
        lines.append(
            f"{evaluator} & {sequential:.2f} & {restart:.2f} & {_signed(restart - sequential)} "
            + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return _table(lines)


def _render_t5(evidence: dict[str, Any]) -> str:
    lines = [
        _source_header(evidence, "T5").rstrip(),
        r"\begin{tabular}{rccccc}",
        r"\toprule",
        r"& \multicolumn{2}{c}{EM} & & \multicolumn{2}{c}{F1} \\",
        r"$N$ & full & ddi\_qf & $\Delta$ & full & ddi\_qf \\",
        r"\midrule",
    ]
    for n, full_em, ddi_em, full_f1, ddi_f1 in _rows(evidence["artifacts"], "T5", 5):
        delta = full_em - ddi_em
        delta_text = r"\textbf{0.00}" if math.isclose(delta, 0.0, abs_tol=1e-12) else _signed(delta)
        lines.append(
            f"{n} & {full_em:.2f} & {ddi_em:.2f} & {delta_text} & {full_f1:.2f} & {ddi_f1:.2f} "
            + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return _table(lines)


def _render_t6(evidence: dict[str, Any]) -> str:
    header = (
        r"dataset & metric & seed 0 & seed 1 & seed 2 & $3\sigma$ spread & "
        "descriptive 95\\% t-interval " + r"\\"
    )
    lines = [
        _source_header(evidence, "T6_delta").rstrip(),
        "% Intervals and spreads are computed from the preserved unrounded seed means.",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for dataset, metric, seeds in _rows(evidence["artifacts"], "T6_delta", 3):
        _require(
            isinstance(seeds, list) and len(seeds) == 3,
            f"T6_delta {dataset}/{metric} needs three seeds",
        )
        mean = statistics.mean(seeds)
        stdev = statistics.stdev(seeds)
        half_width = T_CRITICAL_95_DF2 * stdev / math.sqrt(3)
        seed_digits = 2 if metric == "EM" else 3
        seed_text = " & ".join(f"{value:.{seed_digits}f}" for value in seeds)
        lines.append(
            f"{_tex_name(dataset)} & {metric} & {seed_text} & {3 * stdev:.3f} & "
            f"[{mean - half_width:.3f}, {mean + half_width:.3f}] " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return _table(lines)


def _render_t7(evidence: dict[str, Any]) -> str:
    artifact = evidence["artifacts"]["T7_transfer"]
    _require(
        artifact.get("bootstrap_repeats") == 10000, "T7_transfer bootstrap_repeats must be 10000"
    )
    _require(artifact.get("bootstrap_seed") == 46, "T7_transfer bootstrap_seed must be 46")
    lines = [
        _source_header(evidence, "T7_transfer").rstrip(),
        "% Bootstrap: 10,000 equal-seed draws, seed 46; practical threshold: 0.10.",
        r"\begin{tabular}{@{}lllll@{}}",
        r"\toprule",
        r"Dataset & Metric & Qwen verdict & Llama verdict & Contrast [95\% CI] \\",
        r"\midrule",
    ]
    for dataset, metric, qwen, llama, contrast in _rows(evidence["artifacts"], "T7_transfer", 5):
        if contrast is None:
            contrast_text = "unavailable"
        else:
            _require(
                isinstance(contrast, list) and len(contrast) == 3,
                f"T7 contrast is invalid for {dataset}/{metric}",
            )
            contrast_text = rf"${contrast[0]:.3f}\;[{contrast[1]:.3f}, {contrast[2]:.3f}]$"
        lines.append(
            f"{_tex_name(dataset)} & {metric} & {qwen} & {llama} & {contrast_text} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return _table(lines)


def _render_t8(evidence: dict[str, Any]) -> str:
    lines = [
        _source_header(evidence, "T8_transfer_estimands").rstrip(),
        "% C is the retained historical normalized remaining tuned gap, not training recovery.",
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        r"Architecture & Dataset & Metric & $C$ & $\Delta$ DDI & $\Delta$ full & $\Delta$ gap \\",
        r"\midrule",
    ]
    for architecture, dataset, metric, closure, ddi_change, full_change, gap_change in _rows(
        evidence["artifacts"], "T8_transfer_estimands", 7
    ):
        lines.append(
            f"{architecture} & {_tex_name(dataset)} & {metric} & {closure:.3f} & "
            f"{_signed(ddi_change, 3)} & {_signed(full_change, 3)} & {_signed(gap_change, 3)} "
            + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return _table(lines)


def _finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def _render_t9(evidence: dict[str, Any]) -> str:
    artifact = evidence["artifacts"]["T9_v65_serving"]
    rows = _rows(evidence["artifacts"], "T9_v65_serving", 4)
    byte_ac = artifact["byte_identical_ac"]
    _require(
        isinstance(byte_ac, list) and len(byte_ac) == 2,
        "T9_v65_serving.byte_identical_ac must hold two counts",
    )
    walls = artifact["schedule_walls_s"]
    _require(
        isinstance(walls, dict) and set(walls) == {"A", "all_hit"},
        "T9_v65_serving.schedule_walls_s is invalid",
    )
    lines = [
        _source_header(evidence, "T9_v65_serving").rstrip(),
        r"\begin{tabular}{llr}",
        r"\toprule",
        r"Stage & Arm / contrast & TTFT median (ms) [95\% CI] \\",
        r"\midrule",
    ]
    previous_stage = None
    for stage, label, point, ci in rows:
        _require(
            stage in ("benchmark", "combined", "ratio"),
            f"T9_v65_serving stage is invalid: {stage}",
        )
        _require(
            isinstance(ci, list) and len(ci) == 2 and _finite(point, *ci),
            f"T9_v65_serving interval is invalid for {label}",
        )
        if previous_stage is not None and stage != previous_stage:
            lines.append(r"\midrule")
        digits = 4 if stage == "ratio" else 2
        lines.append(
            f"{stage} & {_tex_name(label)} & "
            rf"${point:.{digits}f}\;[{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]$ " + r"\\"
        )
        previous_stage = stage
    lines.append(r"\midrule")
    lines.append(
        r"\multicolumn{3}{l}{Arm outputs byte-identical to the V60 retained generations: "
        rf"A {byte_ac[0]}/{byte_ac[0]}, C {byte_ac[1]}/{byte_ac[1]}.}} \\"
    )
    lines.append(
        r"\multicolumn{3}{l}{End-to-end schedule walls: "
        rf"A {walls['A']:.3f} s, all-hit {walls['all_hit']:.3f} s.}} \\"
    )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return _table(lines)


def _render_t10(evidence: dict[str, Any]) -> str:
    artifact = evidence["artifacts"]["T10_qa_confirmation"]
    rows = _rows(evidence["artifacts"], "T10_qa_confirmation", 6)
    precision = artifact["precision"]
    _require(isinstance(precision, dict), "T10_qa_confirmation.precision must be an object")
    prediction = artifact["prediction_equal"]
    first_token = artifact["first_token_equal"]
    _require(
        isinstance(prediction, list) and len(prediction) == 2,
        "T10_qa_confirmation.prediction_equal is invalid",
    )
    _require(
        isinstance(first_token, list) and len(first_token) == 2,
        "T10_qa_confirmation.first_token_equal is invalid",
    )
    lines = [
        _source_header(evidence, "T10_qa_confirmation").rstrip(),
        r"\begin{tabular}{llrrl}",
        r"\toprule",
        r"Cohort & Metric & $\Delta$ mean & 95\% CI & Verdict \\",
        r"\midrule",
    ]
    previous_cohort = None
    for cohort, metric, mean, lo, hi, verdict in rows:
        _require(cohort in precision, f"T10_qa_confirmation cohort is invalid: {cohort}")
        _require(_finite(mean, lo, hi), f"T10_qa_confirmation {cohort} {metric} is not finite")
        digits = precision[cohort]
        if previous_cohort is not None and cohort != previous_cohort:
            lines.append(r"\midrule")
        lines.append(
            rf"{_tex_name(cohort)} & {metric} & {_signed(mean, digits)} & "
            rf"$[{lo:.{digits}f}, {hi:.{digits}f}]$ & {verdict} " + r"\\"
        )
        previous_cohort = cohort
    lines.append(r"\midrule")
    lines.append(
        r"\multicolumn{5}{l}{Non-inferiority floor: 95\% CI lower bound "
        rf"$\geq {artifact['floor']:.2f}$ for both EM and F1.}} \\"
    )
    lines.append(
        r"\multicolumn{5}{l}{V65 untouched: prediction-equal "
        rf"{prediction[0]}/{prediction[1]}, "
        rf"first-token-equal {first_token[0]}/{first_token[1]}.}} \\"
    )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return _table(lines)


def _render_t11(evidence: dict[str, Any]) -> str:
    rows = _rows(evidence["artifacts"], "T11_v59_crossover", 8)
    lines = [
        _source_header(evidence, "T11_v59_crossover").rstrip(),
        r"\begin{tabular}{rrcccc}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{TTFT median (ms)} & warm speedup & hit-rate \\",
        r"$N$ & $L$ & fresh & warm & [95\% CI] & crossover \\",
        r"\midrule",
    ]
    for n, length, fresh, warm, speedup, lo, hi, crossover in rows:
        _require(
            _finite(fresh, warm, speedup, lo, hi),
            f"T11_v59_crossover cell ({n},{length}) is not finite",
        )
        _require(
            crossover is None or math.isfinite(crossover),
            "T11_v59_crossover crossover is invalid",
        )
        crossover_text = "none" if crossover is None else f"{crossover:.3f}"
        lines.append(
            rf"{n} & {length} & {fresh:.1f} & {warm:.1f} & "
            rf"${speedup:.3f}\;[{lo:.3f}, {hi:.3f}]$ & {crossover_text} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return _table(lines)


def _render_t12(evidence: dict[str, Any]) -> str:
    artifact = evidence["artifacts"]["T12_v57_matched"]
    rows = _rows(evidence["artifacts"], "T12_v57_matched", 6)
    ratios = artifact["timing_ratios"]
    _require(
        isinstance(ratios, list) and len(ratios) == 2,
        "T12_v57_matched.timing_ratios must hold two latency ratios",
    )
    lines = [
        _source_header(evidence, "T12_v57_matched").rstrip(),
        "% Latency ratios are means of paired per-request full/ddi_qf ratios.",
        r"\begin{tabular}{lrrl}",
        r"\toprule",
        r" & \multicolumn{2}{c}{arm mean} & paired full $-$ ddi\_qf \\",
        r"metric & full & ddi\_qf & $\Delta$ [95\% CI] \\",
        r"\midrule",
    ]
    for metric, full, ddi, delta, lo, hi in rows:
        _require(_finite(full, ddi, delta, lo, hi), f"T12_v57_matched {metric} is not finite")
        lines.append(
            rf"{metric} & {full:.4f} & {ddi:.4f} & "
            rf"${delta:.4f}\;[{lo:.4f}, {hi:.4f}]$ " + r"\\"
        )
    lines.append(r"\midrule")
    for label, point, lo, hi in ratios:
        _require(_finite(point, lo, hi), f"T12_v57_matched {label} is not finite")
        lines.append(
            rf"{_tex_name(label)} & \multicolumn{{3}}{{l}}{{${point:.4f}\;[{lo:.4f}, {hi:.4f}]$}} "
            + r"\\"
        )
    lines.append(r"\midrule")
    lines.append(
        rf"\multicolumn{{4}}{{l}}{{{artifact['n_requests']} matched HotpotQA requests, seed 0.}} \\"
    )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return _table(lines)


def _render_t13(evidence: dict[str, Any]) -> str:
    artifact = evidence["artifacts"]["T13_v66_boundary"]
    rows = _rows(evidence["artifacts"], "T13_v66_boundary", 8)
    identical = artifact["byte_identical_repeats"]
    _require(
        isinstance(identical, list) and len(identical) == 2,
        "T13_v66_boundary.byte_identical_repeats must hold two counts",
    )
    headline = artifact["headline_ttft_speedup"]
    _require(
        isinstance(headline, dict)
        and set(headline) == {"context", "point", "ci"}
        and isinstance(headline["ci"], list)
        and len(headline["ci"]) == 2
        and _finite(headline["point"], *headline["ci"]),
        "T13_v66_boundary.headline_ttft_speedup is invalid",
    )
    dilution = artifact["all_hit_wall_dilution"]
    _require(
        isinstance(dilution, dict)
        and set(dilution) == {"d1", "d64"}
        and _finite(dilution["d1"], dilution["d64"]),
        "T13_v66_boundary.all_hit_wall_dilution is invalid",
    )
    divergent = identical[1] - identical[0]
    budgets = ("d=1", "d=8", "d=16", "d=32", "d=64", "TTFT")
    lines = [
        _source_header(evidence, "T13_v66_boundary").rstrip(),
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"& & \multicolumn{5}{l}{smallest winning $h_{\mathrm{eff}}$ (wall)} & "
        r"smallest winning \\",
        r"$L$ & $N$ & d=1 & d=8 & d=16 & d=32 & d=64 & TTFT $h_{\mathrm{eff}}$ \\",
        r"\midrule",
    ]
    prev_l = None
    for row in rows:
        length, n_docs = row[0], row[1]
        _require(isinstance(length, int) and length in (32, 128, 512), f"T13 L invalid: {length}")
        _require(isinstance(n_docs, int) and n_docs in (2, 5, 10), f"T13 N invalid: {n_docs}")
        cells = row[2:]
        rendered = []
        for value in cells:
            _require(value is None or _finite(value), f"T13 invalid h_eff: {value}")
            rendered.append("---" if value is None else rf"${value:g}$")
        if prev_l is not None and length != prev_l:
            lines.append(r"\midrule")
        prev_l = length
        lines.append(f"L={length} & {n_docs} & " + " & ".join(rendered) + r" \\")
    lines.append(r"\midrule")
    lines.append(
        r"\multicolumn{8}{p{0.97\linewidth}}{\footnotesize Cells are $h_{\mathrm{eff}}$"
        r"{=}hit\_count/$N$ of the \emph{smallest tested winning configuration} for that "
        r"shape under the frozen materiality rule (win iff bootstrap CI low${}>1.05$); "
        r"---{} = none in the frozen grid. These are grid-specific boundaries, not "
        r"universal thresholds.} \\"
    )
    lines.append(
        r"\multicolumn{8}{p{0.97\linewidth}}{\footnotesize Development data (frozen V62 "
        rf"split, seed 47, single seed): outputs byte-identical on {identical[0]}/{identical[1]}"
        r" timed repeats.} \\"
    )
    lines.append(
        r"\multicolumn{8}{p{0.97\linewidth}}{\footnotesize Headline all-hit TTFT speedup "
        rf"({_tex_name(headline['context'])}): ${headline['point']:.2f}\,"
        rf"[{headline['ci'][0]:.2f}, {headline['ci'][1]:.2f}]$; the all-hit wall speedup "
        rf"dilutes from ${dilution['d1']:.2f}\times$ (d=1) to ${dilution['d64']:.2f}\times$"
        r" (d=64).} \\"
    )
    lines.append(
        r"\multicolumn{8}{p{0.97\linewidth}}{\footnotesize The "
        rf"{divergent} divergent repeats are bf16 "
        r"reduction-order flips deeper in the trajectory; the first decoded token is "
        r"always identical.} \\"
    )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return _table([lines[0], r"\resizebox{\linewidth}{!}{%"] + lines[1:] + [r"}"])


def _render_t14(evidence: dict[str, Any]) -> str:
    artifact = evidence["artifacts"]["T14_v69_attribution"]
    rows = _rows(evidence["artifacts"], "T14_v69_attribution", 3)
    reduction = artifact["reduction_ms"]
    _require(_finite(reduction) and reduction > 0, "T14_v69_attribution reduction is invalid")
    total = math.fsum(row[2] for row in rows)
    _require(
        _finite(total) and math.isclose(total, reduction, rel_tol=1e-12, abs_tol=1e-12),
        "T14_v69_attribution removed phases must sum to the pinned reduction",
    )
    pre, post = artifact["pre_total_ms"], artifact["post_total_ms"]
    budget = artifact["opt3_budget_ms"]
    implied = artifact["opt3_implied_post_ms"]
    residual = artifact["residual_ms"]
    residual_frac = artifact["residual_frac_of_pre"]
    speedup = artifact["speedup_vs_pre"]
    reconstruction = artifact["post_reconstruction_ms"]
    delta = artifact["post_reconstruction_delta_ms"]
    ca_post = artifact["ca_post"]
    ca_pre = artifact["ca_pre_v60"]
    cold_b = artifact["cold_b_ms"]
    b_over_oldb = artifact["cold_b_over_oldb"]
    _require(
        _finite(reconstruction, delta, ca_pre, cold_b, b_over_oldb)
        and isinstance(ca_post, list)
        and len(ca_post) == 3
        and _finite(*ca_post),
        "T14_v69_attribution context values are invalid",
    )
    _require(
        math.isclose(reconstruction, post, rel_tol=1e-9, abs_tol=1e-9),
        "T14_v69_attribution post-optimization reconstruction must close at the pinned post-total",
    )
    lines = [
        _source_header(evidence, "T14_v69_attribution").rstrip(),
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Optimization (V65) & Pre-optimization phase removed/reduced & ms removed & "
        r"\% of reduction \\",
        r"\midrule",
    ]
    for label, phase, removed in rows:
        _require(_finite(removed) and removed > 0, f"T14_v69_attribution {label} is invalid")
        dagger = r"$^\dagger$" if "bounded envelope" in label else ""
        lines.append(
            f"{_tex_name(label)} & {_tex_name(phase)} & {removed:.2f} & "
            f"{100 * removed / reduction:.1f}{dagger} " + r"\\"
        )
    lines.append(r"\midrule")
    lines.append(rf"Total removed & & {reduction:.2f} & 100.0 " + r"\\")
    lines.append(r"\midrule")
    lines.append(
        r"\multicolumn{4}{l}{Pre-optimization all-hit reuse "
        rf"{pre:.4f} ms $\to$ post-optimization {post:.4f} ms ({speedup:.2f}$\times$).}} \\"
    )
    lines.append(
        r"\multicolumn{4}{l}{OPT-3 is a bounded envelope, not a direct measurement: it fits "
        rf"within the {budget:.2f} ms pre-optimization}} \\"
    )
    lines.append(
        r"\multicolumn{4}{l}{cache\_assemble budget; the implied post-optimization assemble of "
        rf"{implied:.2f} ms is inferred, not measured.}} \\"
    )
    lines.append(
        r"\multicolumn{4}{l}{Unattributed wall remainder "
        rf"{residual:.2f} ms ({100 * residual_frac:.1f}\% of the pre-total, within the declared "
        r"5\% tolerance).} \\"
    )
    lines.append(
        r"\multicolumn{4}{l}{Post-optimization phase reconstruction closes at "
        rf"{reconstruction:.2f} ms (delta {delta:.2f} vs the pinned post-total).}} \\"
    )
    lines.append(
        r"\multicolumn{4}{l}{Post-optimization C/A $"
        rf"{ca_post[0]:.4f}\;[{ca_post[1]:.4f}, {ca_post[2]:.4f}]$; pre-optimization C/A "
        rf"{ca_pre:.4f} ({ca_pre:.2f}$\times$ slower).}} \\"
    )
    lines.append(
        r"\multicolumn{4}{l}{Cold arm B unchanged: "
        rf"{cold_b:.2f} ms (B/old-B {b_over_oldb:.4f}).}} \\"
    )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return _table([lines[0], r"\resizebox{\linewidth}{!}{%"] + lines[1:] + [r"}"])


def _render_t15(evidence: dict[str, Any]) -> str:
    artifact = evidence["artifacts"]["T15_v70_gates"]
    gates = _rows(evidence["artifacts"], "T15_v70_gates", 5)
    arms = artifact["arm_means"]
    _require(
        isinstance(arms, dict) and set(arms) == {"A", "B", "C", "D"},
        "T15_v70_gates.arm_means must hold arms A-D",
    )
    _require(
        artifact["n_records"] == 300 and artifact["margin"] == -0.02 and artifact["k"] == 5,
        "T15_v70_gates confirmation shape is invalid",
    )
    bootstrap = artifact["bootstrap"]
    _require(
        isinstance(bootstrap, dict)
        and bootstrap.get("seed") == 47
        and bootstrap.get("resamples") == 10000,
        "T15_v70_gates.bootstrap is invalid",
    )
    cache = artifact["cache_validation"]
    _require(
        isinstance(cache, dict)
        and cache.get("hit_only") is True
        and cache.get("reopened_store") is True
        and cache.get("keys_unchanged") is True
        and cache.get("encode_on_miss") == "fail_closed"
        and type(cache.get("entries")) is int,
        "T15_v70_gates.cache_validation is invalid",
    )
    lines = [
        _source_header(evidence, "T15_v70_gates").rstrip(),
        r"\begin{tabular}{llrrl}",
        r"\toprule",
        r"Arm & Role & EM & F1 & \\",
        r"\midrule",
    ]
    for arm in ("A", "B", "C", "D"):
        means = arms[arm]
        _require(
            isinstance(means, dict)
            and set(means) == {"em", "f1", "role"}
            and _finite(means["em"], means["f1"]),
            f"T15_v70_gates arm {arm} is invalid",
        )
        lines.append(
            f"{arm} & {_tex_name(means['role'])} & {means['em']:.6f} & {means['f1']:.6f} "
            + r"\\"
        )
    lines.append(r"\midrule")
    lines.append(r"Gate & Metric & $\Delta$ mean & 95\% CI & Verdict \\")
    for contrast, metric, mean, ci, verdict in gates:
        _require(
            contrast in ("C-B", "C-D") and metric in ("EM", "F1") and verdict == "PASS",
            f"T15_v70_gates gate row is invalid: {contrast}/{metric}/{verdict}",
        )
        _require(
            isinstance(ci, list)
            and len(ci) == 2
            and _finite(mean, *ci)
            and ci[0] <= mean <= ci[1],
            f"T15_v70_gates {contrast} {metric} mean is inconsistent with its CI",
        )
        lines.append(
            rf"{contrast} & {metric} & {_signed(mean, 6)} & "
            rf"$[{ci[0]:.6f}, {ci[1]:.6f}]$ & {verdict} " + r"\\"
        )
    lines.append(r"\midrule")
    lines.append(
        rf"\multicolumn{{5}}{{l}}{{{artifact['n_records']} frozen confirmation300 records; "
        rf"non-inferiority margin {artifact['margin']:.2f}; "
        rf"{bootstrap['resamples']} paired bootstrap resamples, seed {bootstrap['seed']}.}} \\"
    )
    lines.append(
        rf"\multicolumn{{5}}{{l}}{{k={artifact['k']} selected checkpoint; arms A and B are "
        r"diagnostic only per the frozen verdict.} \\"
    )
    lines.append(
        r"\multicolumn{5}{l}{Arm C read hit-only from a reopened persisted store: "
        rf"{cache['entries']} entries before and after, keys unchanged, encode-on-miss "
        r"fails closed.} \\"
    )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return _table(lines)


def _render_t16(evidence: dict[str, Any]) -> str:
    artifact = evidence["artifacts"]["T16_v68_anatomy"]
    rows = _rows(evidence["artifacts"], "T16_v68_anatomy", 2)
    position = artifact["position"]
    direction = artifact["direction"]
    _require(
        isinstance(position, dict) and set(position) == {"first_token", "mid_generation"},
        "T16_v68_anatomy.position is invalid",
    )
    _require(
        isinstance(direction, dict)
        and set(direction) == {"cached_strictly_loses_em", "cached_strictly_gains_em"},
        "T16_v68_anatomy.direction is invalid",
    )
    _require(
        artifact["n_divergent"]
        == sum(count for _, count in rows)
        == position["first_token"] + position["mid_generation"],
        "T16_v68_anatomy label counts must sum to the pinned divergent count",
    )
    lines = [
        _source_header(evidence, "T16_v68_anatomy").rstrip(),
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Divergence label (frozen rubric) & Records \\",
        r"\midrule",
    ]
    for label, count in rows:
        _require(type(count) is int and count >= 0, f"T16_v68_anatomy {label} count is invalid")
        lines.append(f"{_tex_name(label)} & {count} " + r"\\")
    lines.append(r"\midrule")
    lines.append(
        rf"\multicolumn{{2}}{{l}}{{Divergence position: {position['first_token']} first-token, "
        rf"{position['mid_generation']} mid-generation (of {artifact['n_divergent']}).}} \\"
    )
    lines.append(
        rf"\multicolumn{{2}}{{l}}{{Direction is two-sided: cached strictly loses EM on "
        rf"{direction['cached_strictly_loses_em']} records and strictly gains on "
        rf"{direction['cached_strictly_gains_em']}, consistent with the V65 paired "
        r"aggregate (EM $-0.01$).} \\"
    )
    lines.append(
        rf"\multicolumn{{2}}{{l}}{{Diagnostic only: single-seed labeled sample of "
        rf"{artifact['n_divergent']}/{artifact['n_confirmation_records']} confirmation "
        r"records; no rate or causal claims.} \\"
    )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return _table([lines[0], r"\resizebox{\linewidth}{!}{%"] + lines[1:] + [r"}"])


def _render_t17(evidence: dict[str, Any]) -> str:
    rows = _rows(evidence["artifacts"], "T17_v71_boundary", 6)
    lines = [
        _source_header(evidence, "T17_v71_boundary").rstrip(),
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"& & \multicolumn{2}{l}{smallest winning $h_{\mathrm{eff}}$ (wall d=1)}"
        r" & \multicolumn{2}{l}{smallest winning $h_{\mathrm{eff}}$ (TTFT)} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"$L$ & $N$ & V66 & V71 & V66 & V71 \\",
        r"\midrule",
    ]

    def fmt(value: Any) -> str:
        _require(value is None or _finite(value), f"T17 invalid h_eff: {value}")
        return "---" if value is None else rf"${value:g}$"

    prev_l = None
    for i, row in enumerate(rows):
        L, n_docs, w66, w71, t66, t71 = row
        _require(isinstance(L, int) and L in (32, 128, 512), f"T17 L invalid: {L}")
        _require(isinstance(n_docs, int) and n_docs in (2, 5, 10), f"T17 N invalid: {n_docs}")
        if prev_l is not None and L != prev_l:
            lines.append(r"\midrule")
        prev_l = L
        lines.append(
            f"L={L} & {n_docs} & {fmt(w66)} & {fmt(w71)} & {fmt(t66)} & {fmt(t71)} \\\\"
        )
    lines.append(r"\midrule")
    lines.append(
        r"\multicolumn{6}{p{0.97\linewidth}}{\footnotesize Every value is "
        r"$h_{\mathrm{eff}}$=hit\_count/$N$ of the smallest tested winning configuration "
        r"on the frozen development grid (win iff bootstrap CI low${}>1.05$); ---{} = none "
        r"in the grid. Grid-specific boundaries, not universal thresholds. V66 columns "
        r"from the pinned V66 crossover table; V71 columns from the pinned V71 sweep "
        r"verdicts (A/CP, untrained-restart reference A).} \\"
    )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return _table([lines[0], r"\resizebox{\linewidth}{!}{%"] + lines[1:] + [r"}"])


def _render_f5(evidence: dict[str, Any]) -> str:
    artifact = evidence["artifacts"]["F5"]
    _require(
        artifact.get("shape") == "zipf(a=1.5),top=32,N=5,L=500,Q=32,A=16", "F5 shape is invalid"
    )
    rows = _rows(evidence["artifacts"], "F5", 2)
    for label, values in rows:
        _require(
            isinstance(values, list) and len(values) == 3, f"F5 {label} needs three seed values"
        )
    if artifact.get("status") == "withdrawn":
        return _table(
            [
                _source_header(evidence, "F5").rstrip(),
                r"\begin{minipage}{0.82\linewidth}",
                r"\small\textbf{Throughput claim withdrawn (V53).} "
                r"Historical V51 rows do not establish exact token geometry or equal decode work. "
                r"The source rows are retained for audit, not plotted as validated throughput.",
                r"\end{minipage}",
            ]
        )
    lines = [
        _source_header(evidence, "F5").rstrip(),
        r"\begin{minipage}{0.82\linewidth}",
        r"\small\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrll}",
        r"\toprule",
        r"Mode & mean QPS & \multicolumn{2}{l}{relative bar (20 mm per QPS)} \\",
        r"\midrule",
    ]
    for label, values in rows:
        mean = statistics.mean(values)
        lines.append(rf"{label} & {mean:.3f} & \rule{{{mean * 20:.1f}mm}}{{1.5ex}} & \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{minipage}"])
    return _table(lines)


_SERIES_COLORS = {2: "ddiBlue", 5: "ddiGreen", 10: "ddiGold"}
_N_LABEL = {2: r"$N{=}2$", 5: r"$N{=}5$", 10: r"$N{=}10$"}


def _crossover_panel(
    evidence: dict[str, Any],
    artifact_name: str,
    *,
    ymax: float,
) -> str:
    """Two-panel crossover chart (wall d=1 | TTFT) from pinned per-cell ratios."""
    rows = _rows(evidence["artifacts"], artifact_name, 7)
    series: dict[tuple[str, int], list[tuple[float, float, float]]] = {}
    for contract, n_docs, h_eff, hit_count, nominal, wall, ttft in rows:
        _require(
            contract in ("V66", "V71") and n_docs in (2, 5, 10),
            f"{artifact_name} series invalid: {contract}, {n_docs}",
        )
        _require(
            _finite(h_eff, wall, ttft) and 0.0 <= h_eff <= 1.0,
            f"{artifact_name} values invalid for {contract} N={n_docs}",
        )
        series.setdefault((contract, n_docs), []).append((h_eff, wall, ttft))
    for key, points in series.items():
        points.sort()
        h_effs = [p[0] for p in points]
        _require(len(set(h_effs)) == len(h_effs), f"{artifact_name} duplicate x for {key}")

    def coords(contract: str, n_docs: int, column: int) -> str:
        return " ".join(
            f"({p[0]:g},{p[column]:g})" for p in series[(contract, n_docs)]
        )

    lines = [_source_header(evidence, artifact_name).rstrip(), r"\begin{tikzpicture}", r"\begin{groupplot}["]
    lines.append(
        r"  group style={group size=2 by 1, horizontal sep=1.5cm},"
        r"  width=0.46\linewidth, height=5.8cm,"
        r"  xlabel={$h_{\mathrm{eff}}$ = hit\_count/$N$ (actual cached fraction)},"
        r"  xmin=0, xmax=1.05, ymin=0,"
        rf"  ymax={ymax:g},"
        r"  grid=both, grid style={gray!20},"
        r"  title style={font=\small},"
        r"  tick label style={font=\scriptsize}, label style={font=\small},"
        r"]"
    )
    titles = ("A/CP wall speedup, decode $d{=}1$", "A/CP TTFT speedup")
    for panel, (title, idx) in enumerate(zip(titles, (1, 2))):
        head = r"\nextgroupplot[title={" + title + r"}]"
        if panel == 0:
            head += (
                r"\legend{"
                + ",".join(
                    f"{{V66 {_N_LABEL[n]}}}" for n in (2, 5, 10)
                )
                + "}"
            )
        lines.append(head)
        lines.append(r"\addplot[gray, dashed, no marks, domain=0:1.05] {1};")
        for n_docs in (2, 5, 10):
            color = _SERIES_COLORS[n_docs]
            lines.append(
                rf"\addplot[{color}, dashed, mark=*, mark size=1.5pt,"
                rf" mark options={{solid}}, forget plot] coordinates"
                rf" {{{coords('V66', n_docs, idx)}}};"
            )
            lines.append(
                rf"\addplot[{color}, solid, mark=square*, mark size=1.5pt,"
                rf" mark options={{solid}}, forget plot] coordinates"
                rf" {{{coords('V71', n_docs, idx)}}};"
            )
    lines.append(r"\end{groupplot}")
    # one shared legend matrix under the panels
    lines.append(
        r"\matrix [draw=none, anchor=north, font=\scriptsize,"
        r" matrix of nodes, nodes={anchor=west}, row sep=1pt,"
        r" column sep=10pt] at (current bounding box.south) {"
    )
    legend_cells = []
    for n_docs in (2, 5, 10):
        color = _SERIES_COLORS[n_docs]
        legend_cells.append(
            rf"\tikz\draw[{color},dashed] (0,0) -- (0.35,0); V66 {_N_LABEL[n_docs]} & "
            rf"\tikz\draw[{color},solid] (0,0) -- (0.35,0); V71 {_N_LABEL[n_docs]}"
        )
    lines.append(" \\\\\n".join(legend_cells) + r" \\};")
    lines.append(r"\end{tikzpicture}")
    return "\n".join(lines) + "\n"


def _render_v14(evidence: dict[str, Any]) -> str:
    return _crossover_panel(evidence, "V14_v71_crossover_L512", ymax=7.0)


def _render_v15(evidence: dict[str, Any]) -> str:
    return _crossover_panel(evidence, "V15_v71_crossover_L128", ymax=3.0)


def _render_v16(evidence: dict[str, Any]) -> str:
    rows = _rows(evidence["artifacts"], "V16_gate_deltas", 11)
    stations = []
    for row in rows:
        station, family, seed, k = row[0], row[1], row[2], row[3]
        cb, cb_lo, cb_hi, cd, cd_lo, cd_hi, n = row[4:11]
        _require(
            _finite(cb, cb_lo, cb_hi, cd, cd_lo, cd_hi) and n == 300,
            f"V16_gate_deltas values invalid for {station}",
        )
        _require(
            cb == 0.0 and cb_lo == 0.0 and cb_hi == 0.0,
            f"V16_gate_deltas: C-B is not exact zero for {station}",
        )
        stations.append(
            {
                "coord": station,
                "label": f"{station} ({family}, seed {seed}, k={k})",
                "cb": cb,
                "cb_lo": cb_lo,
                "cb_hi": cb_hi,
                "cd": cd,
                "cd_lo": cd_lo,
                "cd_hi": cd_hi,
            }
        )
    names = [s["coord"] for s in stations]

    def ci(s: dict, prefix: str) -> str:
        return (
            f"({s['coord']},{s[prefix]:g}) += (0,{s[prefix + '_hi'] - s[prefix]:g})"
            f" -= (0,{s[prefix] - s[prefix + '_lo']:g})"
        )

    lines = [
        _source_header(evidence, "V16_gate_deltas").rstrip(),
        r"% V16: trained-contract quality gates across the three n=300 confirmations.",
        r"% C-B is exactly 0.000 with degenerate CI [0,0] at every station: the",
        r"% all-zero paired differences make every bootstrap resample zero, so the",
        r"% interval carries no information beyond the point mass (VIBEQUEUE-94).",
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"  width=0.95\linewidth, height=6.4cm,",
        r"  ybar, bar width=16pt,",
        rf"  symbolic x coords={{{','.join(names)}}},",
        r"  xtick=data,",
        r"  xticklabels={" + ",".join("{" + s["label"] + "}" for s in stations) + "},",
        r"  xticklabel style={font=\scriptsize},",
        r"  ylabel={EM delta vs.\ reference},",
        r"  ymin=-0.03, ymax=0.42, enlarge x limits=0.25,",
        r"  legend style={at={(0.02,0.98)}, anchor=north west, font=\scriptsize},",
        r"  grid=both, grid style={gray!20},",
        r"]",
        r"\addplot[ddiGreen, fill=ddiGreen!50, mark=diamond*, mark size=3pt,"
        r"         error bars/.cd, y dir=both, y explicit]",
        "  coordinates {" + " ".join(ci(s, "cb") for s in stations) + "};",
        r"\addlegendentry{C$-$B identity gate (exact $0$, degenerate CI)}",
        r"\addplot[ddiBlue, fill=ddiBlue!55,",
        r"         error bars/.cd, y dir=both, y explicit]",
        "  coordinates {" + " ".join(ci(s, "cd") for s in stations) + "};",
        r"\addlegendentry{C$-$D vs no-cache reference (95\% CI)}",
    ]
    for s in stations:
        lines.append(
            rf"\node[font=\scriptsize, ddiGreen, anchor=south] at (axis cs:{s['coord']},0.012)"
            rf" {{$0.000\;[0,0]$}};"
        )
        lines.append(
            rf"\node[font=\scriptsize, ddiBlue, anchor=south] at (axis cs:{s['coord']},"
            rf"{s['cd_hi'] + 0.055:g}) {{$+{s['cd']:.3f}$}};"
        )
    lines.extend([r"\end{axis}", r"\end{tikzpicture}"])
    return "\n".join(lines) + "\n"


def _render_v17(evidence: dict[str, Any]) -> str:
    rows = _rows(evidence["artifacts"], "V17_program_timeline", 3)
    values = {row[1]: row[2] for row in rows}
    expected = {
        "hit_rate",
        "cache_over_fresh_wall",
        "divergent_streams",
        "measured_streams",
        "allhit_speedup_vs_monolithic",
        "byte_identical_pairs",
        "byte_identical_total",
        "cb_em",
        "n_records",
        "allhit_ttft_speedup_L512_N10",
        "identity_violations",
        "identity_blocks_checked",
        "grid_cells",
        "repeats_per_cell",
    }
    _require(set(values) == expected, f"V17_program_timeline fields mismatch: {sorted(values)}")
    _require(
        values["cb_em"] == 0.0
        and values["byte_identical_pairs"] == values["byte_identical_total"]
        and values["identity_violations"] == 0
        and values["grid_cells"] == 240,
        "V17_program_timeline anchor values drifted",
    )
    lines = [
        _source_header(evidence, "V17_program_timeline").rstrip(),
        r"% V17: program arc from diagnostic cache to production trained contract.",
        r"% All displayed values come from the pinned V17_program_timeline ledger rows.",
        r"\begin{tikzpicture}[font=\scriptsize]",
        r"  \draw[gray, thick, ->] (0,0) -- (13.4,0);",
        r"  \node[circle, fill=ddiRed,   inner sep=2.4pt] (v53) at (1.6,0) {};",
        r"  \node[circle, fill=ddiGold,  inner sep=2.4pt] (v65) at (5.2,0) {};",
        r"  \node[circle, fill=ddiGreen, inner sep=2.4pt] (v70) at (8.8,0) {};",
        r"  \node[circle, fill=ddiBlue,  inner sep=2.4pt] (v71) at (12.0,0) {};",
        r"  \node[font=\tiny, gray, anchor=south] at (1.6,0.10) {Sep 7};",
        r"  \node[font=\tiny, gray, anchor=south] at (5.2,0.10) {Sep 10};",
        r"  \node[font=\tiny, gray, anchor=south] at (8.8,0.10) {Sep 11};",
        r"  \node[font=\tiny, gray, anchor=south] at (12.0,0.10) {Sep 12};",
        r"  \node[draw=gray!40, fill=gray!5, rounded corners, align=left, anchor=north, text width=3.9cm]",
        r"    at (1.6,-0.35)",
        rf"    {{\textbf{{V53}} diagnostic cache:\\ {values['hit_rate'] * 100:.1f}\% hits but"
        rf" ${values['cache_over_fresh_wall']:.2f}\times$",
        rf"     \emph{{slower}} than fresh\\ [1pt] and greedy decodes diverge on"
        rf" {values['divergent_streams']}/{values['measured_streams']} measured streams}};",
        r"  \node[draw=gray!40, fill=gray!5, rounded corners, align=left, anchor=south, text width=3.9cm]",
        r"    at (5.2,0.35)",
        r"    {\textbf{V65} serving contract fixed:\\ byte-identical"
        rf" {values['byte_identical_pairs']:.0f}/{values['byte_identical_total']:.0f};\\",
        rf"     all-hit ${values['allhit_speedup_vs_monolithic']:.2f}\times$ faster than"
        r" monolithic};",
        r"  \node[draw=gray!40, fill=gray!5, rounded corners, align=left, anchor=north, text width=4.1cm]",
        r"    at (8.8,-0.35)",
        r"    {\textbf{V70} admissible confirmation ($n{=}$"
        rf"{values['n_records']:.0f}$)$:\\",
        rf"     C$-$B $={values['cb_em']:.3f}$ $[0,0]$ on EM and F1;\\ both preregistered"
        r" gates PASS};",
        r"  \node[draw=gray!40, fill=gray!5, rounded corners, align=left, anchor=south, text width=3.4cm]",
        r"    at (12.0,0.35)",
        r"    {\textbf{V71} trained contract:\\ all-hit TTFT up to"
        rf" ${values['allhit_ttft_speedup_L512_N10']:.2f}\times$;\\",
        r"     zero identity violations in"
        rf" {values['grid_cells']:.0f} configurations"
        rf" ({values['repeats_per_cell']:.0f} repeats)}};",
        r"\end{tikzpicture}",
    ]
    return "\n".join(lines) + "\n"


def _render_v18(evidence: dict[str, Any]) -> str:
    artifact = evidence["artifacts"]["T16_v68_anatomy"]
    rows = _rows(evidence["artifacts"], "T16_v68_anatomy", 2)
    labels = {
        "near_tie_greedy_flip": "near-tie flip",
        "clear_semantic_change": "semantic change",
        "truncation_stopping_artifact": "truncation artifact",
        "ambiguous": "ambiguous",
    }
    counts = {row[0]: row[1] for row in rows}
    _require(set(counts) == set(labels), f"T16_v68_anatomy categories mismatch: {sorted(counts)}")
    total = sum(counts.values())
    _require(
        total == artifact["n_divergent"] == 22,
        f"T16_v68_anatomy total {total} != n_divergent {artifact['n_divergent']}",
    )
    order = ["ambiguous", "truncation_stopping_artifact", "near_tie_greedy_flip",
             "clear_semantic_change"]
    lines = [
        _source_header(evidence, "T16_v68_anatomy").rstrip(),
        r"% V18: anatomy of the 22 divergent cached/fresh confirmation records (V68).",
        r"% Counts from the pinned T16_v68_anatomy ledger artifact.",
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"  width=0.92\linewidth, height=5.4cm,",
        r"  xbar, bar width=13pt,",
        rf"  symbolic y coords={{{','.join(labels[c] for c in order)}}},",
        r"  ytick=data,",
        r"  xtick={0,2,4,6,8,10,12},",
        r"  xlabel={{Divergent records (of 22)}},",
        r"  xmin=0, xmax=13,",
        r"  yticklabel style={font=\scriptsize},",
        r"  tick label style={font=\scriptsize},",
        r"  label style={font=\small},",
        r"  nodes near coords, nodes near coords align={horizontal},",
        r"  every node near coord/.append style={font=\scriptsize},",
        r"  grid=both, grid style={gray!20},",
        r"]",
        r"\addplot[ddiBlue!60, draw=ddiBlue]",
        "  coordinates {"
        + " ".join(f"({counts[c]},{labels[c]})" for c in order)
        + "};",
        r"\end{axis}",
        r"\end{tikzpicture}",
    ]
    return "\n".join(lines) + "\n"


def render(evidence: dict[str, Any]) -> dict[str, str]:
    rendered = _render_t1(evidence)
    visuals = render_visuals(evidence)
    rendered["V1_mask_geometry.tex"] = visuals["V1_mask_geometry.tex"]
    rendered.update(
        {
            "T2.tex": _render_t2(evidence),
            "T3.tex": _render_t3(evidence),
            "V2_efficiency_intuition.tex": visuals["V2_efficiency_intuition.tex"],
            "V4_production_opportunity.tex": visuals["V4_production_opportunity.tex"],
            "T4.tex": _render_t4(evidence),
            "T4_16L.tex": _render_t4_16l(evidence),
            "T4_restart.tex": _render_t4_restart(evidence),
            "T5.tex": _render_t5(evidence),
            "T6_delta.tex": _render_t6(evidence),
            "T7_transfer.tex": _render_t7(evidence),
            "T8_transfer_estimands.tex": _render_t8(evidence),
            "T9_v65_serving.tex": _render_t9(evidence),
            "T10_qa_confirmation.tex": _render_t10(evidence),
            "T11_v59_crossover.tex": _render_t11(evidence),
            "T12_v57_matched.tex": _render_t12(evidence),
            "T13_v66_crossover.tex": _render_t13(evidence),
            "T14_v69_attribution.tex": _render_t14(evidence),
            "T15_v70_gates.tex": _render_t15(evidence),
            "T16_v68_anatomy.tex": _render_t16(evidence),
            "T17_v71_boundary.tex": _render_t17(evidence),
            "V3_transfer_estimands.tex": visuals["V3_transfer_estimands.tex"],
            "F5.tex": _render_f5(evidence),
        }
    )
    rendered.update(visuals)
    rendered.update(
        {
            "V14_crossover_chart.tex": _render_v14(evidence),
            "V15_crossover_L128.tex": _render_v15(evidence),
            "V16_gate_deltas.tex": _render_v16(evidence),
            "V17_program_timeline.tex": _render_v17(evidence),
            "V18_anatomy.tex": _render_v18(evidence),
        }
    )
    _require(set(rendered) == set(GENERATED_FILES), "renderer output set is incomplete")
    return rendered


def write_or_check(rendered: dict[str, str], output: Path, *, check: bool) -> None:
    if check:
        drift = [
            name
            for name, content in rendered.items()
            if not (output / name).is_file()
            or (output / name).read_text(encoding="utf-8") != content
        ]
        if drift:
            raise EvidenceError(f"generated artifacts are stale: {', '.join(sorted(drift))}")
        return
    output.mkdir(parents=True, exist_ok=True)
    for name, content in rendered.items():
        (output / name).write_text(content, encoding="utf-8", newline="\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="fail if checked-in output differs")
    parser.add_argument(
        "--require-all-sources",
        action="store_true",
        help="also require ignored historical raw stores (for a provenance audit)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        evidence = load_evidence(args.evidence)
        missing = verify_sources(evidence, require_all=args.require_all_sources)
        write_or_check(render(evidence), args.output, check=args.check)
    except (ValueError, KeyError, OSError, TypeError) as error:
        print(f"paper render failed: {error}", file=sys.stderr)
        return 1
    action = "validated" if args.check else "rendered"
    print(f"paper artifacts {action} from {args.evidence.relative_to(ROOT)}")
    if missing:
        print("historical raw sources not present (hash provenance retained):")
        for path in missing:
            print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
