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
    "F5",
}
GENERATED_FILES = (
    "T1_mask_taxonomy.md",
    "T1_mask_taxonomy.tex",
    "T2.tex",
    "T3.tex",
    "T4.tex",
    "T4_16L.tex",
    "T4_restart.tex",
    "T5.tex",
    "T6_delta.tex",
    "T7_transfer.tex",
    "F5.tex",
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
        "F5": 2,
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


def _render_f5(evidence: dict[str, Any]) -> str:
    artifact = evidence["artifacts"]["F5"]
    _require(
        artifact.get("shape") == "zipf(a=1.5),top=32,N=5,L=500,Q=32,A=16", "F5 shape is invalid"
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
    for label, values in _rows(evidence["artifacts"], "F5", 2):
        _require(
            isinstance(values, list) and len(values) == 3, f"F5 {label} needs three seed values"
        )
        mean = statistics.mean(values)
        lines.append(rf"{label} & {mean:.3f} & \rule{{{mean * 20:.1f}mm}}{{1.5ex}} & \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{minipage}"])
    return _table(lines)


def render(evidence: dict[str, Any]) -> dict[str, str]:
    rendered = _render_t1(evidence)
    rendered.update(
        {
            "T2.tex": _render_t2(evidence),
            "T3.tex": _render_t3(evidence),
            "T4.tex": _render_t4(evidence),
            "T4_16L.tex": _render_t4_16l(evidence),
            "T4_restart.tex": _render_t4_restart(evidence),
            "T5.tex": _render_t5(evidence),
            "T6_delta.tex": _render_t6(evidence),
            "T7_transfer.tex": _render_t7(evidence),
            "F5.tex": _render_f5(evidence),
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
    except (EvidenceError, KeyError, OSError, TypeError) as error:
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
