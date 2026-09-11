"""V61 step — independent reproduction of the FP64-reference calibration numbers.

Re-derives the calibration evidence behind the PROPOSED task-level RMS gate
(``scripts/calibrate_fa2_fp64.py``: attention-output RMS relative error <= 0.02 vs the
fp64 dense reference) with NEW random inputs — not by re-running the original script's
stored numbers. 3 seeds x 3 shapes (N docs x L tokens + Q query tokens) = 9 cells.

Reused measurement definitions (imported verbatim from ``scripts/calibrate_fa2_fp64.py``):
fp64 dense reference, fp32+SDPA, bf16+SDPA, bf16 segmented-FA2 (the production
``ddi.segmented_fa2._segmented_prefill`` varlen path, variant ``ddi_qf``), RMS relative
error, and pairwise argmax disagreement. New here: freshly drawn standard-normal q/k/v
cast to bf16 once per cell (mirroring the original's bf16 captured-state contract — all
four methods see identical inputs), per-cell CPU generator, head geometry and softmax
scale matched to the original study (28 q-heads / 4 kv-heads, head_dim 128).

PREREGISTERED before any measurement (module constants below, also written into
``manifest.json`` first):
- reproduction band: new/stored ratio in [0.5, 2.0] (same order of magnitude), applied
  per cell to fp32_sdpa / bf16_sdpa / bf16_fa2_segmented ``rms_rel`` and to the
  bf16_sdpa-vs-bf16_fa2_segmented argmax disagreement rate;
- criterion validation: RMS <= 0.02 is validated iff ALL bf16 cells' ``rms_rel`` are
  <= 0.02 AND the observed margin (0.02 / worst) is >= 10x.

Stored comparison values are quoted from
``results/v61_fp64_calibration_20260910T062211Z/summary.json`` (read-only reference;
no stored states or outputs are re-run).

Run: ``<python> scripts/reproduce_v61_calibration.py`` (CUDA + flash-attn required).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ddi.masks import build_mask  # noqa: E402
from scripts.calibrate_fa2_fp64 import (  # noqa: E402
    dense_attention,
    error_table,
    pairwise_argmax_disagreement,
    rms_relative_error,
    sdpa_attention,
)

# rms_relative_error is reused verbatim from the original calibration; listing it in
# __all__ re-exports it so tests can pin the identity against scripts.calibrate_fa2_fp64.
__all__ = ["rms_relative_error"]

SCHEMA = "v61_calibration_repro_v1"
ORIGINAL_ARTIFACT = ROOT / "results/v61_fp64_calibration_20260910T062211Z"

REPRO_SEEDS = (4047, 4048, 4049)
REPRO_SHAPES = ((2, 128, 64), (5, 512, 64), (10, 512, 64))  # (N docs, L doc tokens, Q query)

# Head geometry of the original study (its summary.json: 28/4 heads, head_dim 128,
# softmax_scale = 128**-0.5). Only seeds/shapes vary vs the original.
NUM_Q_HEADS = 28
NUM_KV_HEADS = 4
HEAD_DIM = 128
SOFTMAX_SCALE = HEAD_DIM**-0.5
VARIANT = "ddi_qf"
ANSWER_LEN = 0

# ---- preregistration (stated here BEFORE the run; manifest.json is written first) ------
TOLERANCE_BAND = (0.5, 2.0)  # new/stored ratio band: same order of magnitude, inclusive
CRITERION_MAX_RMS = 0.02  # PROPOSED gate: rms_rel <= 0.02 vs the fp64 dense reference
CRITERION_REQUIRED_MARGIN = 10.0  # ...and the observed margin 0.02/worst must be >= 10x

# Stored values quoted from ORIGINAL_ARTIFACT/summary.json (errors_vs_fp64_dense and the
# bf16_sdpa_vs_bf16_fa2_segmented entry of pairwise_argmax_disagreement — the "0.0003").
STORED = {
    "fp32_sdpa_rms_rel": 2.172389073829691e-06,
    "bf16_sdpa_rms_rel": 1.3906336469263947e-03,
    "bf16_fa2_segmented_rms_rel": 1.390924348669742e-03,
    "argmax_dis_bf16_sdpa_vs_bf16_fa2_segmented": 0.0003348214285714286,
}
REPRO_METHODS = ("fp32_sdpa", "bf16_sdpa", "bf16_fa2_segmented")
ARGMAX_PAIR = "bf16_sdpa_vs_bf16_fa2_segmented"


# ---- preregistration evaluation helpers (CPU-tested in tests/) -------------------------


def band_ratio(value: float, stored: float) -> float:
    """new/stored ratio; the preregistered band is TOLERANCE_BAND = [0.5, 2.0]."""
    if stored <= 0.0:
        raise ValueError(f"stored value must be > 0, got {stored}")
    return value / stored


def in_band(ratio: float, band: tuple[float, float] = TOLERANCE_BAND) -> bool:
    """Inclusive band membership."""
    return band[0] <= ratio <= band[1]


def evaluate_reproduction(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-metric preregistered-band verdicts across all cells + an overall verdict."""
    methods: dict[str, Any] = {}
    for method in REPRO_METHODS:
        stored_key = f"{method}_rms_rel"
        ratios = {
            cell["cell"]: band_ratio(cell["rms_rel"][method], STORED[stored_key])
            for cell in cells
        }
        methods[stored_key] = {
            "stored": STORED[stored_key],
            "ratios": ratios,
            "ratio_min": min(ratios.values()),
            "ratio_max": max(ratios.values()),
            "all_in_band": all(in_band(ratio) for ratio in ratios.values()),
        }
    stored_key = f"argmax_dis_{ARGMAX_PAIR}"
    ratios = {
        cell["cell"]: band_ratio(cell["argmax_dis"][ARGMAX_PAIR], STORED[stored_key])
        for cell in cells
    }
    methods[stored_key] = {
        "stored": STORED[stored_key],
        "ratios": ratios,
        "ratio_min": min(ratios.values()),
        "ratio_max": max(ratios.values()),
        "all_in_band": all(in_band(ratio) for ratio in ratios.values()),
    }
    verdict = "REPRODUCED" if all(m["all_in_band"] for m in methods.values()) else "NOT_REPRODUCED"
    return {"band": list(TOLERANCE_BAND), "methods": methods, "verdict": verdict}


def evaluate_criterion(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Criterion RMS<=0.02 validated iff ALL bf16 cells pass with a >=10x observed margin."""
    bf16_rms = {
        cell["cell"]: max(
            cell["rms_rel"]["bf16_sdpa"], cell["rms_rel"]["bf16_fa2_segmented"]
        )
        for cell in cells
    }
    worst = max(bf16_rms.values())
    all_within = worst <= CRITERION_MAX_RMS
    margin: float | None
    if worst == 0.0:
        margin = None  # unlimited margin (json-safe); exact match of the fp64 reference
        validated = all_within
    else:
        margin = CRITERION_MAX_RMS / worst
        validated = all_within and margin >= CRITERION_REQUIRED_MARGIN
    return {
        "criterion": f"attention-output rms_rel vs fp64 reference <= {CRITERION_MAX_RMS:g}",
        "required_margin": CRITERION_REQUIRED_MARGIN,
        "per_cell_worst_bf16_rms_rel": bf16_rms,
        "worst_bf16_rms_rel": worst,
        "all_within_threshold": all_within,
        "observed_margin": margin,
        "validated": validated,
    }


# ---- inputs and measurement (GPU) -------------------------------------------------------


def make_cell_inputs(
    seed: int, n_docs: int, doc_len: int, query_len: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fresh random inputs for one cell: bf16 q/k/v + the ddi_qf allow mask (CPU).

    Mirrors the original study's input contract: all four methods receive the SAME bf16
    states (there the captured checkpoint states; here standard-normal tensors cast to
    bf16 once, drawn from a per-cell CPU generator so the cell is device-independent).
    """
    doc_lengths = [doc_len] * n_docs
    total = sum(doc_lengths) + query_len
    gen = torch.Generator("cpu").manual_seed(seed)
    q = torch.randn(
        (1, NUM_Q_HEADS, total, HEAD_DIM), generator=gen, dtype=torch.float32
    ).to(torch.bfloat16)
    k = torch.randn(
        (1, NUM_KV_HEADS, total, HEAD_DIM), generator=gen, dtype=torch.float32
    ).to(torch.bfloat16)
    v = torch.randn(
        (1, NUM_KV_HEADS, total, HEAD_DIM), generator=gen, dtype=torch.float32
    ).to(torch.bfloat16)
    allow = build_mask(doc_lengths, query_len, ANSWER_LEN, VARIANT)
    return q, k, v, allow


def _sha256_tensor(t: torch.Tensor) -> str:
    raw = t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def run_cell(
    seed: int, n_docs: int, doc_len: int, query_len: int, device: torch.device
) -> dict[str, Any]:
    """One cell: four attention methods on the same bf16 inputs, metrics vs fp64 dense."""
    from ddi.segmented_fa2 import _segmented_prefill

    doc_lengths = [doc_len] * n_docs
    q, k, v, allow = make_cell_inputs(seed, n_docs, doc_len, query_len)
    input_sha256 = {"q": _sha256_tensor(q), "k": _sha256_tensor(k), "v": _sha256_tensor(v)}
    q, k, v, allow = q.to(device), k.to(device), v.to(device), allow.to(device)
    state = {"doc_lengths": doc_lengths, "query_len": query_len, "variant": VARIANT}
    outputs = {
        "fp64_dense": dense_attention(q, k, v, allow, SOFTMAX_SCALE, torch.float64),
        "fp32_sdpa": sdpa_attention(q, k, v, allow, SOFTMAX_SCALE, torch.float32),
        "bf16_sdpa": sdpa_attention(q, k, v, allow, SOFTMAX_SCALE, torch.bfloat16),
        "bf16_fa2_segmented": _segmented_prefill(q, k, v, state, SOFTMAX_SCALE).to(q.dtype),
    }
    for name, out in outputs.items():
        if not torch.isfinite(out).all():
            raise RuntimeError(f"{name} produced non-finite attention output")
    errors = error_table(outputs, outputs["fp64_dense"])
    return {
        "cell": f"seed{seed}_N{n_docs}_L{doc_len}_Q{query_len}",
        "seed": seed,
        "n_docs": n_docs,
        "doc_len": doc_len,
        "query_len": query_len,
        "tokens": int(q.shape[2]),
        "input_sha256": input_sha256,
        "rms_rel": {m: errors[m]["rms_rel"] for m in REPRO_METHODS},
        "max_abs": {m: errors[m]["max_abs"] for m in REPRO_METHODS},
        "worst_head_rms_rel": {m: errors[m]["worst_head_rms_rel"] for m in REPRO_METHODS},
        "argmax_dis": pairwise_argmax_disagreement(outputs),
    }


# ---- provenance and artifact rendering --------------------------------------------------


def _git_info() -> dict[str, Any]:
    def _run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()

    try:
        return {"rev": _run("rev-parse", "HEAD"), "dirty": bool(_run("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"rev": None, "dirty": None}


def build_manifest(timestamp: str, device_name: str, git: dict[str, Any]) -> dict[str, Any]:
    """Preregistration record — written to disk BEFORE any cell is measured."""
    return {
        "schema": SCHEMA,
        "generated_utc": timestamp,
        "purpose": (
            "independent re-derivation of the V61 FP64-calibration numbers with new "
            "seeds/shapes; stored values are quoted for comparison only, never re-run"
        ),
        "original_artifact": {
            "path": str(ORIGINAL_ARTIFACT),
            "summary": str(ORIGINAL_ARTIFACT / "summary.json"),
        },
        "preregistration": {
            "seeds": list(REPRO_SEEDS),
            "shapes_n_l_q": [list(shape) for shape in REPRO_SHAPES],
            "cells": len(REPRO_SEEDS) * len(REPRO_SHAPES),
            "tolerance_band": {
                "rule": "new/stored ratio, inclusive",
                "band": list(TOLERANCE_BAND),
                "applies_to": [f"{m}_rms_rel" for m in REPRO_METHODS]
                + [f"argmax_dis_{ARGMAX_PAIR}"],
            },
            "criterion": {
                "max_rms": CRITERION_MAX_RMS,
                "required_margin": CRITERION_REQUIRED_MARGIN,
                "rule": (
                    "validated iff ALL bf16 cells' rms_rel <= 0.02 AND the observed margin "
                    "0.02/worst is >= 10x"
                ),
            },
            "stored_values": STORED,
            "stored_values_source": str(ORIGINAL_ARTIFACT / "summary.json"),
        },
        "method_definitions": {
            "reference": "fp64 dense masked softmax (scripts.calibrate_fa2_fp64.dense_attention)",
            "fp32_sdpa": "scripts.calibrate_fa2_fp64.sdpa_attention, float32, additive mask",
            "bf16_sdpa": "scripts.calibrate_fa2_fp64.sdpa_attention, bfloat16",
            "bf16_fa2_segmented": "ddi.segmented_fa2._segmented_prefill, variant ddi_qf",
            "metrics": (
                "rms_relative_error / pairwise argmax disagreement reused verbatim from "
                "scripts/calibrate_fa2_fp64.py"
            ),
            "inputs": (
                "per-cell CPU generator, q/k/v ~ N(0,1) cast to bf16; 28 q-heads / 4 kv-heads / "
                "head_dim 128; softmax scale 128**-0.5; mask ddi.masks.build_mask(ddi_qf)"
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda or "none",
            "device": device_name,
            "git": git,
        },
    }


def render_summary_md(result: dict[str, Any]) -> str:
    """Human-readable SUMMARY.md kept in sync with result.json by construction."""
    rep = result["reproduction"]
    crit = result["criterion_verdict"]
    methods = rep["methods"]
    lines = [
        "# V61 FP64-calibration reproduction (new seeds/shapes)",
        "",
        f"Preregistered band: new/stored ratio in [{TOLERANCE_BAND[0]:g}, {TOLERANCE_BAND[1]:g}]"
        f"; criterion rms_rel <= {CRITERION_MAX_RMS:g} with a required "
        f"{CRITERION_REQUIRED_MARGIN:g}x observed margin.",
        f"Stored source: `{ORIGINAL_ARTIFACT.name}/summary.json` (values quoted, not re-run).",
        "",
        "## Per-cell RMS relative error vs the fp64 dense reference",
        "",
        "| cell | T | fp32 SDPA | bf16 SDPA | bf16 seg-FA2 | argmax dis (bf16 pair) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for cell in result["cells"]:
        lines.append(
            f"| {cell['cell']} | {cell['tokens']} "
            f"| {cell['rms_rel']['fp32_sdpa']:.3e} | {cell['rms_rel']['bf16_sdpa']:.3e} "
            f"| {cell['rms_rel']['bf16_fa2_segmented']:.3e} "
            f"| {cell['argmax_dis'][ARGMAX_PAIR]:.4f} |"
        )
    lines += [
        "",
        f"## Ratios vs stored (band [{TOLERANCE_BAND[0]:g}, {TOLERANCE_BAND[1]:g}])",
        "",
        "| cell | fp32 SDPA | bf16 SDPA | bf16 seg-FA2 | argmax dis |",
        "| --- | --- | --- | --- | --- |",
    ]
    for cell in result["cells"]:
        key = cell["cell"]
        lines.append(
            f"| {key} "
            f"| {methods['fp32_sdpa_rms_rel']['ratios'][key]:.2f} "
            f"| {methods['bf16_sdpa_rms_rel']['ratios'][key]:.2f} "
            f"| {methods['bf16_fa2_segmented_rms_rel']['ratios'][key]:.2f} "
            f"| {methods[f'argmax_dis_{ARGMAX_PAIR}']['ratios'][key]:.2f} |"
        )
    in_band_count = sum(m["all_in_band"] for m in methods.values())
    margin = crit["observed_margin"]
    margin_text = "unlimited" if margin is None else f"{margin:.1f}x"
    lines += [
        "",
        "## Verdicts",
        "",
        f"- Reproduction vs stored values: **{rep['verdict']}** "
        f"({in_band_count}/{len(methods)} tracked metrics fully in band across all 9 cells)",
        f"- Criterion rms_rel <= {CRITERION_MAX_RMS:g}: worst bf16 cell "
        f"{crit['worst_bf16_rms_rel']:.3e}, observed margin {margin_text} "
        f"(required {CRITERION_REQUIRED_MARGIN:g}x) "
        f"-> **{'VALIDATED' if crit['validated'] else 'NOT VALIDATED'}**",
        "",
        "## Stored comparison values (quoted, not re-run)",
        "",
    ]
    for key, value in STORED.items():
        lines.append(f"- `{key}`: {value:.6e}")
    lines += ["", "## Limitations", ""]
    lines += [f"- {item}" for item in result["limitations"]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        parser.error("CUDA is required (the fp64 dense and FA2 cells cannot run on CPU)")

    device = torch.device(args.device)
    device_name = torch.cuda.get_device_name(device)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = args.output_root / f"v61_calibration_repro_{timestamp}"
    artifact_dir.mkdir(parents=True, exist_ok=False)

    manifest = build_manifest(timestamp, device_name, _git_info())
    with (artifact_dir / "manifest.json").open("x", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, allow_nan=False)
        f.write("\n")
    print(f"[repro] preregistered seeds={list(REPRO_SEEDS)} shapes={list(REPRO_SHAPES)}")
    print(
        f"[repro] band new/stored in [{TOLERANCE_BAND[0]:g}, {TOLERANCE_BAND[1]:g}]; "
        f"criterion rms_rel <= {CRITERION_MAX_RMS:g} with >= {CRITERION_REQUIRED_MARGIN:g}x margin"
    )
    print(f"[repro] manifest written before measurement: {artifact_dir / 'manifest.json'}")

    cells = []
    for n_docs, doc_len, query_len in REPRO_SHAPES:
        for seed in REPRO_SEEDS:
            row = run_cell(seed, n_docs, doc_len, query_len, device)
            cells.append(row)
            rms = row["rms_rel"]
            print(
                f"[repro] {row['cell']:>22}: T={row['tokens']} "
                f"fp32={rms['fp32_sdpa']:.3e} bf16_sdpa={rms['bf16_sdpa']:.3e} "
                f"segFA2={rms['bf16_fa2_segmented']:.3e} "
                f"argmax_dis={row['argmax_dis'][ARGMAX_PAIR]:.4f}",
                flush=True,
            )

    reproduction = evaluate_reproduction(cells)
    criterion = evaluate_criterion(cells)
    result = {
        "schema": SCHEMA,
        "generated_utc": timestamp,
        "runtime": manifest["runtime"],
        "preregistration": manifest["preregistration"],
        "cells": cells,
        "reproduction": reproduction,
        "criterion_verdict": criterion,
        "limitations": [
            "synthetic N(0,1)->bf16 q/k/v inputs, not real checkpoint states; this reproduces "
            "the kernel-error scale, not the original captured-state tensors",
            "argmax disagreement is over the attention-output feature dim, not the vocabulary",
            "band and criterion were preregistered in manifest.json before the first measurement",
        ],
    }
    with (artifact_dir / "result.json").open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    (artifact_dir / "SUMMARY.md").write_text(render_summary_md(result), encoding="utf-8")

    print(f"\n[repro] reproduction vs stored: {reproduction['verdict']}")
    for name, row in reproduction["methods"].items():
        print(
            f"[repro]   {name}: ratio [{row['ratio_min']:.2f}, {row['ratio_max']:.2f}] -> "
            f"{'in band' if row['all_in_band'] else 'OUT OF BAND'}"
        )
    margin = criterion["observed_margin"]
    margin_text = "unlimited" if margin is None else f"{margin:.1f}x"
    print(
        f"[repro] criterion rms_rel <= {CRITERION_MAX_RMS:g}: worst bf16 "
        f"{criterion['worst_bf16_rms_rel']:.3e}, margin {margin_text} -> "
        f"{'VALIDATED' if criterion['validated'] else 'NOT VALIDATED'}"
    )
    print(f"[repro] artifacts: {artifact_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
