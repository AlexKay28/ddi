"""V65 step 3: benchmark the optimized canonical path against the monolithic SDPA baseline.

Three arms over the same 24 frozen V60 confirmation records (real re-tokenized
documents, shapes verified against the retained ``requests.jsonl``), one shared
SDPA model (the V60 ``--canonical-on-sdpa`` execution, so A-vs-B/C is a
same-backend execution comparison per the V64 audit qualifier (c)):

- ``A`` monolithic-SDPA — arm-A-like plain SDPA additive-mask ``ddi_qf`` prefill;
  the timed wall covers the single ``model()`` prefill call between
  synchronizes (mask build untimed), mirroring the ``generate_ids(...)
  ["prefill_ms"]`` semantics that produced the V64-corrected A target.
- ``B`` fresh-canonical — full synchronized wall of
  ``CanonicalCache(store=None).request(docs, query_ids)`` (id validation,
  metadata digest, per-document encode, cache assembly, mask, query forward).
- ``C`` all-hit-reuse — full synchronized wall of
  ``CanonicalCache(DocCache store).request(docs, query_ids)`` with the store
  pre-populated; two warmups absorb the cold disk tier into the V65 LRU, so the
  timed repeats run in the same LRU-warm steady state the V60 runner's timed
  TTFT passes ran in. The disk payload stays authoritative.

Protocol: seed 47, 2 warmups + 10 synchronized repeats per (record, arm); CUDA
peak memory is reset per (record, arm) timed section. The success criterion is
declared in ``manifest.json`` before any timing runs: C/A (ratio of medians
over the 24 per-record repeat medians) <= 1.05, with arm-A and arm-C outputs
byte-identical to the V60 retained generations — the run aborts on any
normalized-prediction or generation diff before the timing phase starts.

Outputs ``manifest.json`` / ``result.json`` / ``summary.md`` plus the arm-C
canonical store under the output directory; nothing else is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
import time
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from ddi import inference
from ddi.cache import DocCache
from ddi.canonical_cache import CanonicalCache
from ddi.data import load as load_dataset
from ddi.inference import load
from ddi.masks import build_mask, to_additive
from ddi.prompt import build_query_text
from scripts.run_canonical_matrix import (
    _arm_identity,
    _checkpoint_identity,
    _contract,
    _git_bytes,
    _git_text,
    _peak_bytes,
    _prepare_output,
    _reset_peak,
    _sha_file,
    _source_hashes,
    _store_bytes,
    _sync,
    _write_json,
)
from scripts.run_canonical_qa_dev import _default_tokenizer, _split_docs_query
from scripts.validate_canonical_matrix import ARMS
from scripts.validate_matched_qa import digest, generate_ids

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 65
SEED = 47
WARMUPS = 2
REPEATS = 10
BOOTSTRAP_SEED = 47
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_LEVEL = 0.95
BENCHMARK_ARMS = ("A", "B", "C")
GATE_ARMS = ("A", "C")
GATE_C_OVER_A_MAX = 1.05
NATURAL_BUDGET = 48
FIXED_BUDGET = 16
MIN_FREE_GPU_FRACTION = 0.7
DEFAULT_CHECKPOINT = ROOT / "results/checkpoints/Qwen__Qwen2.5-7B-Instruct__ddi_qf__8L_restart"
DEFAULT_REQUESTS = ROOT / "results/v60_confirmation24_20260910T093500Z/requests.jsonl"
DEFAULT_SOURCE_MANIFEST = ROOT / "experiments/v60_confirmation_manifest.json"
REQUESTS_SHA256 = "ec08d056d6ebe9208cf831e8a63e2d527cc8afec90d377ad9356635767410a5c"
SOURCE_MANIFEST_SHA256 = "2dffebf5d04065021dfb2b9e694651b34f0ebfae4f3d4b9aabd672908aa6cb07"
CHECKPOINT_SHA256 = "00df4c47d67660cab8292da313ea133290b0bb320437290d725cd149c3d31de9"
V64_TARGETS = {
    "A_monolithic_median_ms": 124.9597,
    "B_fresh_median_ms_v60": 417.6208,
    "C_all_hit_median_ms_v60": 221.2411,
    "B_fresh_median_ms_v65_profile": 497.44,
    "C_over_A_v60": 1.7705,
    "source": (
        "results/v64_audit_20260910T091755Z_TIMING_CLAIMS/CLAIMS_TABLE.md; the B profile "
        "figure from results/v65_profile_20260910T094553Z/summary.md"
    ),
}


def _percentile(ordered: list[float], q: float) -> float:
    return ordered[max(0, int(q * (len(ordered) - 1)))]


def _agg(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "min": ordered[0],
        "p25": _percentile(ordered, 0.25),
        "p75": _percentile(ordered, 0.75),
        "max": ordered[-1],
    }


def ratio_of_medians(numerator: list[float], denominator: list[float]) -> float:
    """Point estimate: median(numerator) / median(denominator) over paired records (pure)."""
    if not numerator or len(numerator) != len(denominator):
        raise ValueError("numerator and denominator must be non-empty equal-length lists")
    if any(value <= 0 for value in numerator) or any(value <= 0 for value in denominator):
        raise ValueError("ratio of medians requires strictly positive paired values")
    return statistics.median(numerator) / statistics.median(denominator)


def _resampled_ratio(
    numerator: list[float], denominator: list[float], indices: list[int]
) -> float:
    num_sample = [numerator[index] for index in indices]
    den_sample = [denominator[index] for index in indices]
    return statistics.median(num_sample) / statistics.median(den_sample)


def paired_ratio_bootstrap(
    numerator: list[float],
    denominator: list[float],
    *,
    seed: int,
    iterations: int,
    ci_level: float = BOOTSTRAP_LEVEL,
) -> dict[str, Any]:
    """Paired percentile bootstrap of the ratio of medians over record pairs (pure).

    Each iteration resamples record indices with replacement and recomputes
    median(numerator) / median(denominator) on the resampled pairs; the CI is the
    percentile interval of that resampled ratio distribution.
    """
    point = ratio_of_medians(numerator, denominator)
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if not 0.0 < ci_level < 1.0:
        raise ValueError("ci_level must be in (0, 1)")
    count = len(numerator)
    rng = random.Random(seed)
    samples = sorted(
        _resampled_ratio(numerator, denominator,
                         [rng.randrange(count) for _ in range(count)])
        for _ in range(iterations)
    )
    low_q = (1.0 - ci_level) / 2.0
    return {
        "point": point,
        "ci_low": _percentile(samples, low_q),
        "ci_high": _percentile(samples, 1.0 - low_q),
        "ci_level": ci_level,
        "iterations": iterations,
        "seed": seed,
        "pairs": count,
    }


def bootstrap_statistic(
    values: list[float],
    *,
    seed: int,
    iterations: int,
    ci_level: float = BOOTSTRAP_LEVEL,
    statistic: Any = statistics.median,
) -> dict[str, Any]:
    """Percentile bootstrap of one statistic over resampled records (pure)."""
    if not values:
        raise ValueError("values must be non-empty")
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if not 0.0 < ci_level < 1.0:
        raise ValueError("ci_level must be in (0, 1)")
    count = len(values)
    rng = random.Random(seed)
    samples = sorted(
        statistic([values[rng.randrange(count)] for _ in range(count)])
        for _ in range(iterations)
    )
    low_q = (1.0 - ci_level) / 2.0
    return {
        "point": statistic(values),
        "ci_low": _percentile(samples, low_q),
        "ci_high": _percentile(samples, 1.0 - low_q),
        "ci_level": ci_level,
        "iterations": iterations,
        "seed": seed,
        "pairs": count,
    }


def compare_generation(new: Mapping[str, Any], retained: Mapping[str, Any]) -> dict[str, Any]:
    """Byte-identity flags of one arm's outputs against its V60 retained generation (pure)."""
    natural_equal = list(new["natural_generated_ids"]) == list(retained["natural_generated_ids"])
    fixed_equal = list(new["fixed_generated_ids"]) == list(retained["fixed_generated_ids"])
    prediction_equal = new["prediction"] == retained["prediction"]
    sha_equal = new["fixed_ids_sha256"] == retained["fixed_ids_sha256"]
    return {
        "prediction_equal": prediction_equal,
        "natural_ids_equal": natural_equal,
        "fixed_ids_equal": fixed_equal,
        "fixed_ids_sha256_equal": sha_equal,
        "byte_identical": prediction_equal and natural_equal and fixed_equal and sha_equal,
    }


def gate_verdict(
    *,
    equivalence_all_equal: bool,
    c_over_a_point: float,
    threshold: float = GATE_C_OVER_A_MAX,
) -> dict[str, Any]:
    """Evaluate the manifest-declared success criterion (pure)."""
    return {
        "criterion": (
            "C/A ratio of medians <= "
            f"{threshold} with arm-A and arm-C outputs byte-identical to the V60 "
            "retained generations"
        ),
        "c_over_a_max": threshold,
        "c_over_a_point": c_over_a_point,
        "equivalence_all_equal": equivalence_all_equal,
        "pass": bool(equivalence_all_equal and c_over_a_point <= threshold),
    }


def load_frozen_cells(path: Path, *, expected_sha256: str) -> list[dict[str, Any]]:
    """Load the frozen request cells after verifying the pinned file digest."""
    actual = _sha_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"frozen requests sha mismatch for {path}: expected {expected_sha256}, got {actual}"
        )
    cells = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not cells:
        raise ValueError("no frozen request cells found")
    return cells


def retained_arm(cell: dict[str, Any], key: str) -> dict[str, Any]:
    """Retained per-record arm payload for a summary key (A, B, C_cold, C_all_hit)."""
    if key.startswith("C_"):
        return cell["arms"]["C"][key.split("_", 1)[1]]
    return cell["arms"][key]


def retained_medians_ms(cells: list[dict[str, Any]], key: str) -> list[float]:
    """Per-record retained ttft medians in milliseconds, in frozen-cell order (pure)."""
    return [float(retained_arm(cell, key)["ttft_median_seconds"]) * 1000.0 for cell in cells]


def _generation_payload(
    tokenizer: Any, natural_ids: list[int], fixed_ids: list[int]
) -> dict[str, Any]:
    return {
        "prediction": tokenizer.decode(natural_ids, skip_special_tokens=True).strip(),
        "natural_generated_ids": list(natural_ids),
        "fixed_generated_ids": list(fixed_ids),
        "fixed_ids_sha256": digest(json.dumps(list(fixed_ids), separators=(",", ":")).encode()),
    }


def _no_prepare() -> None:
    return None


def _canonical_pass_fn(cache: Any, docs: list[Any], query_ids: Any) -> Any:
    return lambda: cache.request(docs, query_ids)


def _monolithic_pass_fns(model: Any, batch: Any, device: torch.device) -> tuple[Any, Any]:
    """Untimed mask prep + timed prefill body, mirroring generate_ids prefill semantics."""
    prefix_ids = batch.input_ids.to(device)
    positions = batch.position_ids.to(device)
    prefix_len = int(prefix_ids.shape[-1])
    dtype = model.dtype
    state: dict[str, Any] = {}

    def prepare() -> None:
        allow = build_mask(batch.doc_lengths, batch.query_len, 1, "ddi_qf")
        additive = to_additive(allow, dtype=dtype).to(device)
        state["prefill_mask"] = additive[:prefix_len, :prefix_len].clone()[None, None]

    def body() -> Any:
        return model(
            input_ids=prefix_ids,
            attention_mask=state["prefill_mask"],
            position_ids=positions,
            use_cache=True,
        )

    return prepare, body


def _timed_arm(
    prepare: Any, body: Any, *, warmups: int, repeats: int, device: torch.device
) -> dict[str, Any]:
    """Warm up, then run synchronized timed repeats with a per-arm peak-memory window."""
    for _ in range(warmups):
        prepare()
        body()
        _sync(device)
    _reset_peak(device)
    walls: list[float] = []
    first_tokens: set[int] = set()
    for _ in range(repeats):
        prepare()
        _sync(device)
        started = time.perf_counter()
        result = body()
        _sync(device)
        walls.append((time.perf_counter() - started) * 1e3)
        first_tokens.add(int(result.logits[0, -1].argmax()))
    allocated, reserved = _peak_bytes(device)
    return {
        "wall_ms_repeats": walls,
        "wall_ms_median": statistics.median(walls),
        "first_token": sorted(first_tokens)[0],
        "first_token_stable": len(first_tokens) == 1,
        "peak_allocated_bytes": allocated,
        "peak_reserved_bytes": reserved,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--requests", type=Path, default=DEFAULT_REQUESTS)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--record-limit", type=int, default=None)
    parser.add_argument("--warmups", type=int, default=WARMUPS)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--bootstrap-iterations", type=int, default=BOOTSTRAP_ITERATIONS)
    args = parser.parse_args(argv)
    if args.warmups < 0 or args.repeats < 1:
        parser.error("--warmups must be >= 0 and --repeats must be >= 1")
    if args.bootstrap_iterations < 1:
        parser.error("--bootstrap-iterations must be >= 1")
    return args


def _build_manifest(
    args: argparse.Namespace,
    *,
    cells: list[dict[str, Any]],
    checkpoint_identity: tuple[str, str],
    source_config: dict[str, Any],
    gpu: dict[str, Any] | None,
) -> dict[str, Any]:
    dtype_str = "torch.bfloat16" if args.device == "cuda" else "torch.float32"
    bootstrap = {
        "seed": BOOTSTRAP_SEED,
        "iterations": args.bootstrap_iterations,
        "ci_level": BOOTSTRAP_LEVEL,
        "method": (
            "paired percentile resample of the frozen records; each resample recomputes "
            "the ratio of medians (or the median) on the resampled records"
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "data_role": "confirmation_frozen",
        "data_role_note": (
            "the 24 frozen V60 confirmation ids are reused for timing and byte-equivalence "
            "only, per V64 audit qualifier (d); no new quality claims are made on them here"
        ),
        "runner": "scripts/run_v65_benchmark.py",
        "created_utc": datetime.now(UTC).isoformat(),
        "manifest_written_before_run": True,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_identity": {
            "mode": checkpoint_identity[0],
            "sha256": checkpoint_identity[1],
        },
        "requests": {
            "path": str(args.requests.resolve()),
            "sha256": _sha_file(args.requests),
            "record_ids": [cell["record_id"] for cell in cells],
            "source_manifest": {
                "path": str(args.source_manifest.resolve()),
                "sha256": _sha_file(args.source_manifest),
                "dataset": source_config["dataset"],
                "seed": source_config["seed"],
            },
        },
        "protocol": {
            "seed": args.seed,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "synchronize": True,
            "canonical_on_sdpa": True,
            "device": args.device,
            "gpu": gpu,
            "bootstrap": bootstrap,
            "budgets": {
                "natural_max_new_tokens": NATURAL_BUDGET,
                "fixed_new_tokens": FIXED_BUDGET,
            },
        },
        "wall_definitions": {
            "A_monolithic_sdpa_ms": (
                "synchronized wall of the single ddi_qf prefill model() call only (mask "
                "build untimed), mirroring the generate_ids(...)[\"prefill_ms\"] semantics "
                "behind the V64-corrected A target"
            ),
            "B_fresh_canonical_ms": (
                "synchronized wall of the full CanonicalCache(store=None).request(docs, "
                "query_ids) call, mirroring the V60/V64 canonical ttft _timed semantics"
            ),
            "C_all_hit_reuse_ms": (
                "synchronized wall of the full CanonicalCache(store=DocCache).request(docs, "
                "query_ids) call with the store pre-populated; warmups absorb the cold disk "
                "tier into the V65 LRU so timed repeats run in the runner's LRU-warm steady "
                "state; the disk payload stays authoritative"
            ),
            "aggregation": (
                "per record: median of the synchronized repeat walls; per arm: median over "
                "the records (the V64 ratio-of-medians estimand)"
            ),
        },
        "memory_definitions": {
            "peak_per_arm": (
                "torch.cuda max_memory_allocated/reserved within the per-(record, arm) "
                "timed window (reset before the repeats, after warmups); reported as the "
                "max and median over the 24 records; includes the shared model weights"
            ),
        },
        "targets": V64_TARGETS,
        "success_criterion": {
            "name": "cold_path_competitiveness_v65",
            "declared_before_run": True,
            "c_over_a_ratio_max": GATE_C_OVER_A_MAX,
            "c_over_a_estimator": (
                "ratio of medians over the frozen records of per-record median walls "
                "(numerator arm C all-hit-reuse, denominator arm A monolithic SDPA)"
            ),
            "equivalence": (
                "per record, arm A and arm C normalized predictions and natural/fixed "
                "generations byte-identical to the V60 retained generations; the run "
                "aborts before timing on any diff"
            ),
        },
        "cache_store_plan": {
            "arm_C_dir": "canonical_cache/all_hit",
            "growth_measurement": (
                "store file count and bytes snapshot before and after the timed phase; "
                "all-hit timed passes must not grow the store"
            ),
        },
        "arms": {
            arm: {
                **ARMS[arm],
                "dtype": dtype_str,
                "arm_identity": _arm_identity(arm, dtype_str),
                "executed_backend_note": (
                    "canonical arms execute on the same SDPA model "
                    "(--canonical-on-sdpa execution, V64 audit qualifier (c)); the "
                    "ARMS-declared 'eager' backend label is retained verbatim for identity"
                ),
            }
            for arm in BENCHMARK_ARMS
        },
        "sources": {
            "ddi/cache.py": _sha_file(ROOT / "ddi" / "cache.py"),
            "ddi/canonical_cache.py": _sha_file(ROOT / "ddi" / "canonical_cache.py"),
            "scripts/run_v65_benchmark.py": _sha_file(Path(__file__).resolve()),
            "ddi_tree": _source_hashes(),
        },
        "git": {
            "head": _git_text("rev-parse", "HEAD"),
            "diff_head_sha256": hashlib.sha256(_git_bytes("diff", "HEAD")).hexdigest(),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
        },
    }


def _equivalence_summary(prepared: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for arm in BENCHMARK_ARMS:
        matches = sum(int(item["equivalence"][arm]["byte_identical"]) for item in prepared)
        failing = [
            item["cell"]["record_id"]
            for item in prepared
            if not item["equivalence"][arm]["byte_identical"]
        ]
        summary[arm] = {
            "byte_identical_records": matches,
            "records": len(prepared),
            "gating": arm in GATE_ARMS,
            "failing_record_ids": failing,
        }
    return summary


def _summary_markdown(payload: dict[str, Any]) -> str:
    manifest = payload["manifest"]
    arms = payload["arms"]
    ratios = payload["ratios"]
    gate = payload["gate"]
    store = payload["cache_store"]
    equivalence = payload["equivalence_summary"]
    prepopulate = payload["prepopulate"]
    lines = [
        "# V65 step 3 — optimized canonical path benchmark vs monolithic SDPA",
        "",
        f"- Created (UTC): {manifest['created_utc']}",
        f"- Checkpoint: `{manifest['checkpoint']}`",
        f"- Workload: V60 confirmation 24 frozen ids, real tokens "
        f"(`{manifest['requests']['path']}`), shapes verified per record",
        f"- Protocol: seed {manifest['protocol']['seed']}, "
        f"warmups {manifest['protocol']['warmups']}, "
        f"repeats {manifest['protocol']['repeats']}, synchronized; one SDPA model "
        "(V60 --canonical-on-sdpa execution)",
        f"- Targets: A ~{V64_TARGETS['A_monolithic_median_ms']} ms, old-C "
        f"~{V64_TARGETS['C_all_hit_median_ms_v60']} ms (old C/A "
        f"~{V64_TARGETS['C_over_A_v60']}x), B ~{V64_TARGETS['B_fresh_median_ms_v60']} ms "
        f"(V60) / ~{V64_TARGETS['B_fresh_median_ms_v65_profile']} ms (V65 step-1 profile)",
        "- Success criterion (declared in the manifest before the run): C/A <= 1.05 with "
        "byte-identical A/C outputs; abort on any diff",
        "",
        "## Per-arm table (24 records x median of 10 synchronized repeats; "
        "CI = percentile bootstrap of the median, seed 47, 10k)",
        "",
        "| arm | like | wall_ms median [95% CI] | peak alloc MiB (max) | peak reserved MiB (max) |",
        "|---|---|---|---|---|",
    ]
    likes = {"A": "A monolithic", "B": "B fresh-canonical", "C": "C all-hit-reuse"}
    for arm in BENCHMARK_ARMS:
        wall = arms[arm]["wall_ms"]
        alloc_mib = arms[arm]["peak_allocated_bytes"]["max"] / 2**20
        reserved_mib = arms[arm]["peak_reserved_bytes"]["max"] / 2**20
        lines.append(
            f"| {arm} | {likes[arm]} | {wall['median']:.2f} "
            f"[{arms[arm]['wall_ms_ci']['ci_low']:.2f}, {arms[arm]['wall_ms_ci']['ci_high']:.2f}] "
            f"| {alloc_mib:.0f} | {reserved_mib:.0f} |"
        )
    lines += [
        "",
        "## Ratios (ratio of medians; paired percentile bootstrap, 24 records, seed 47, 10k)",
        "",
        "| contrast | point | 95% CI | reference |",
        "|---|---|---|---|",
        f"| C/A | {ratios['C_over_A']['point']:.4f} | "
        f"[{ratios['C_over_A']['ci_low']:.4f}, {ratios['C_over_A']['ci_high']:.4f}] | "
        f"gate <= {GATE_C_OVER_A_MAX} (old C/A {V64_TARGETS['C_over_A_v60']}) |",
        f"| B/A | {ratios['B_over_A']['point']:.4f} | "
        f"[{ratios['B_over_A']['ci_low']:.4f}, {ratios['B_over_A']['ci_high']:.4f}] | "
        "informational, no gate |",
        f"| C/oldC | {ratios['C_over_oldC']['point']:.4f} | "
        f"[{ratios['C_over_oldC']['ci_low']:.4f}, {ratios['C_over_oldC']['ci_high']:.4f}] | "
        "old-C = V60 retained per-record medians "
        f"({V64_TARGETS['C_all_hit_median_ms_v60']} ms) |",
        f"| B/oldB | {ratios['B_over_oldB']['point']:.4f} | "
        f"[{ratios['B_over_oldB']['ci_low']:.4f}, {ratios['B_over_oldB']['ci_high']:.4f}] | "
        f"old-B = V60 retained per-record medians ({V64_TARGETS['B_fresh_median_ms_v60']} ms) |",
        "",
        "## Equivalence verdict (normalized predictions + natural/fixed generations, "
        "byte-identical vs the V60 retained generations)",
        "",
    ]
    for arm in BENCHMARK_ARMS:
        item = equivalence[arm]
        lines.append(
            f"- {arm}: {item['byte_identical_records']}/{item['records']} records "
            f"({'GATING' if item['gating'] else 'informational'})"
        )
    lines += [
        "",
        f"- GATE: {'PASS' if gate['pass'] else 'FAIL'} — {gate['criterion']}; "
        f"C/A point {gate['c_over_a_point']:.4f} <= {gate['c_over_a_max']} "
        f"and equivalence {gate['equivalence_all_equal']}",
        "",
        "## Cache store (arm C)",
        "",
        f"- After pre-population: {store['prepopulate']['files']} files, "
        f"{store['prepopulate']['bytes']} bytes",
        f"- After the timed phase: {store['after_timed']['files']} files, "
        f"{store['after_timed']['bytes']} bytes",
        f"- Growth during the timed phase: {store['growth_during_timed_bytes']} bytes / "
        f"{store['growth_during_timed_files']} files",
        f"- Cold per-record encode (pre-population) wall: median "
        f"{prepopulate['wall_ms']['median']:.2f} ms, total "
        f"{prepopulate['wall_ms']['mean'] * len(payload['record_ids']):.0f} ms "
        "(excluded from arm-C walls by design: all-hit reuse amortizes it)",
        "",
        "## Notes",
        "",
        "- Arm A's timed wall is the prefill model() call only (mask build untimed), matching "
        "the V64-corrected target semantics; arms B/C time the full canonical request call.",
        "- Arm C timed passes run with the V65 LRU tier warm (runner steady state); a freshly "
        "started process pays one disk load per document (profiled at ~62.6 ms/pass in "
        "results/v65_profile_20260910T094553Z) before the LRU warms.",
        "- Peak memory rows include the shared model weights; the arm deltas are the "
        "canonical-path overhead (assembled KV + pinned staging buffer).",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")

    if _sha_file(args.source_manifest) != SOURCE_MANIFEST_SHA256:
        raise ValueError(f"source manifest sha mismatch: {args.source_manifest}")
    source_manifest = json.loads(args.source_manifest.read_text())
    source_config = source_manifest["config"]
    cells = load_frozen_cells(args.requests, expected_sha256=REQUESTS_SHA256)
    if args.record_limit is not None:
        cells = cells[: args.record_limit]
    checkpoint_identity = _checkpoint_identity(args.checkpoint)
    if checkpoint_identity[1] != CHECKPOINT_SHA256:
        raise ValueError(
            f"checkpoint identity mismatch: expected {CHECKPOINT_SHA256}, "
            f"got {checkpoint_identity[1]}"
        )

    gpu: dict[str, Any] | None = None
    if device.type == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        gpu = {
            "name": torch.cuda.get_device_name(device),
            "free_bytes_before": int(free_bytes),
            "total_bytes": int(total_bytes),
        }
        if free_bytes < MIN_FREE_GPU_FRACTION * total_bytes:
            raise RuntimeError(
                f"GPU is not idle: only {free_bytes}/{total_bytes} bytes free "
                f"(requires >= {MIN_FREE_GPU_FRACTION:.0%} free)"
            )

    output = args.output or (
        ROOT / "results" / f"v65_benchmark_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    output = _prepare_output(Path(output))
    manifest = _build_manifest(
        args,
        cells=cells,
        checkpoint_identity=checkpoint_identity,
        source_config=source_config,
        gpu=gpu,
    )
    _write_json(output / "manifest.json", manifest)
    print(json.dumps({"stage": "manifest_written", "output": str(output)}))

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model, _ = load(
        str(args.checkpoint), dtype="bf16" if device.type == "cuda" else "fp32", backend="sdpa"
    )
    model = model.to(device).eval()
    contract = replace(_contract(model, args.checkpoint), backend="sdpa")
    tokenizer = _default_tokenizer(str(args.checkpoint))

    records = {
        record["id"]: record
        for record in load_dataset(source_config["dataset"], "val", source_config["seed"])
    }
    prepared: list[dict[str, Any]] = []
    for cell in cells:
        record = records[cell["record_id"]]
        batch = inference.encode(
            tokenizer, list(record["info_contexts"]), build_query_text(tokenizer, record)
        )
        if batch.doc_lengths != cell["doc_lengths"] or batch.query_len != cell["query_length"]:
            raise ValueError(
                f"re-tokenized shape drift for {cell['record_id']}: "
                f"{batch.doc_lengths}/{batch.query_len} vs frozen "
                f"{cell['doc_lengths']}/{cell['query_length']}"
            )
        docs, query_ids = _split_docs_query(batch)
        prepared.append({"cell": cell, "batch": batch, "docs": docs, "query_ids": query_ids})

    hit_store = DocCache(output / "canonical_cache" / "all_hit")
    for index, item in enumerate(prepared):
        cell, batch, docs, query_ids = (
            item["cell"], item["batch"], item["docs"], item["query_ids"],
        )
        fresh = CanonicalCache(model, contract)
        cached_hit = CanonicalCache(model, contract, hit_store)
        encode_started = time.perf_counter()
        cached_hit.documents([doc.cpu() for doc in docs])
        item["prepopulate_wall_ms"] = (time.perf_counter() - encode_started) * 1e3

        a_natural = generate_ids(
            model, tokenizer, batch, "ddi_qf", NATURAL_BUDGET, stop_on_eos=True
        )
        a_fixed = generate_ids(
            model, tokenizer, batch, "ddi_qf", FIXED_BUDGET, stop_on_eos=False
        )
        c_natural = cached_hit.request(
            docs, query_ids, max_new_tokens=NATURAL_BUDGET, eos_token_id=tokenizer.eos_token_id
        )
        c_fixed = cached_hit.request(
            docs, query_ids, max_new_tokens=FIXED_BUDGET, eos_token_id=None
        )
        b_natural = fresh.request(
            docs, query_ids, max_new_tokens=NATURAL_BUDGET, eos_token_id=tokenizer.eos_token_id
        )
        b_fixed = fresh.request(docs, query_ids, max_new_tokens=FIXED_BUDGET, eos_token_id=None)
        item["equivalence"] = {
            "A": compare_generation(
                _generation_payload(
                    tokenizer, a_natural["generated_ids"], a_fixed["generated_ids"]
                ),
                cell["arms"]["A"],
            ),
            "B": compare_generation(
                _generation_payload(tokenizer, b_natural, b_fixed), cell["arms"]["B"]
            ),
            "C": compare_generation(
                _generation_payload(tokenizer, c_natural, c_fixed), cell["arms"]["C"]["all_hit"]
            ),
        }
        for arm in GATE_ARMS:
            failed = [key for key, ok in item["equivalence"][arm].items() if not ok]
            if failed:
                raise RuntimeError(
                    "V65 step 3 aborted before timing: arm "
                    f"{arm} outputs differ from the V60 retained generations for record "
                    f"{cell['record_id']}: {sorted(failed)}"
                )
        print(json.dumps({
            "stage": "equivalence",
            "request_index": index,
            "record_id": cell["record_id"],
            "a_c_byte_identical": True,
            "b_byte_identical": item["equivalence"]["B"]["byte_identical"],
            "prepopulate_wall_ms": round(item["prepopulate_wall_ms"], 2),
        }))

    equivalence_summary = _equivalence_summary(prepared)
    equivalence_all_equal = all(
        item["equivalence"][arm]["byte_identical"] for item in prepared for arm in GATE_ARMS
    )

    store_bytes_before = _store_bytes(hit_store.cache_dir)
    store_files_before = hit_store.size()
    for index, item in enumerate(prepared):
        docs, query_ids = item["docs"], item["query_ids"]
        fresh = CanonicalCache(model, contract)
        cached_hit = CanonicalCache(model, contract, hit_store)
        mono_prepare, mono_body = _monolithic_pass_fns(model, item["batch"], device)
        timed = {
            "A": _timed_arm(
                mono_prepare, mono_body,
                warmups=args.warmups, repeats=args.repeats, device=device,
            ),
            "B": _timed_arm(
                _no_prepare, _canonical_pass_fn(fresh, docs, query_ids),
                warmups=args.warmups, repeats=args.repeats, device=device,
            ),
            "C": _timed_arm(
                _no_prepare, _canonical_pass_fn(cached_hit, docs, query_ids),
                warmups=args.warmups, repeats=args.repeats, device=device,
            ),
        }
        item["timed"] = timed
        print(json.dumps({
            "stage": "timing",
            "request_index": index,
            "record_id": item["cell"]["record_id"],
            "wall_ms": {arm: round(timed[arm]["wall_ms_median"], 2) for arm in BENCHMARK_ARMS},
        }))

    store_bytes_after = _store_bytes(hit_store.cache_dir)
    store_files_after = hit_store.size()

    per_record_walls = {
        arm: [item["timed"][arm]["wall_ms_median"] for item in prepared]
        for arm in BENCHMARK_ARMS
    }
    arms_summary: dict[str, Any] = {}
    for arm in BENCHMARK_ARMS:
        allocated = [item["timed"][arm]["peak_allocated_bytes"] for item in prepared]
        reserved = [item["timed"][arm]["peak_reserved_bytes"] for item in prepared]
        arms_summary[arm] = {
            "wall_ms": _agg(per_record_walls[arm]),
            "wall_ms_ci": bootstrap_statistic(
                per_record_walls[arm],
                seed=BOOTSTRAP_SEED,
                iterations=args.bootstrap_iterations,
            ),
            "peak_allocated_bytes": {"max": max(allocated), "median": statistics.median(allocated)},
            "peak_reserved_bytes": {"max": max(reserved), "median": statistics.median(reserved)},
        }

    common = {"seed": BOOTSTRAP_SEED, "iterations": args.bootstrap_iterations}
    ratios = {
        "C_over_A": paired_ratio_bootstrap(
            per_record_walls["C"], per_record_walls["A"], **common
        ),
        "B_over_A": paired_ratio_bootstrap(
            per_record_walls["B"], per_record_walls["A"], **common
        ),
        "C_over_oldC": paired_ratio_bootstrap(
            per_record_walls["C"], retained_medians_ms(cells, "C_all_hit"), **common
        ),
        "B_over_oldB": paired_ratio_bootstrap(
            per_record_walls["B"], retained_medians_ms(cells, "B"), **common
        ),
    }
    gate = gate_verdict(
        equivalence_all_equal=equivalence_all_equal,
        c_over_a_point=ratios["C_over_A"]["point"],
    )
    cache_store = {
        "path": str(hit_store.cache_dir),
        "prepopulate": {"files": store_files_before, "bytes": store_bytes_before},
        "after_timed": {"files": store_files_after, "bytes": store_bytes_after},
        "growth_during_timed_bytes": store_bytes_after - store_bytes_before,
        "growth_during_timed_files": store_files_after - store_files_before,
    }
    prepopulate = {"wall_ms": _agg([item["prepopulate_wall_ms"] for item in prepared])}

    result = {
        "schema_version": SCHEMA_VERSION,
        "data_role": "confirmation_frozen",
        "manifest_sha256": _sha_file(output / "manifest.json"),
        "aggregate": {
            "arms": arms_summary,
            "ratios": ratios,
            "gate": gate,
            "cache_store": cache_store,
            "prepopulate": prepopulate,
            "equivalence": equivalence_summary,
            "equivalence_all_equal_A_C": equivalence_all_equal,
        },
        "records": [
            {
                "request_index": index,
                "record_id": item["cell"]["record_id"],
                "doc_lengths": item["cell"]["doc_lengths"],
                "query_length": item["cell"]["query_length"],
                "prepopulate_wall_ms": item["prepopulate_wall_ms"],
                "arms": {
                    arm: {
                        "wall_ms_repeats": item["timed"][arm]["wall_ms_repeats"],
                        "wall_ms_median": item["timed"][arm]["wall_ms_median"],
                        "first_token": item["timed"][arm]["first_token"],
                        "first_token_stable": item["timed"][arm]["first_token_stable"],
                        "peak_allocated_bytes": item["timed"][arm]["peak_allocated_bytes"],
                        "peak_reserved_bytes": item["timed"][arm]["peak_reserved_bytes"],
                    }
                    for arm in BENCHMARK_ARMS
                },
                "equivalence": item["equivalence"],
            }
            for index, item in enumerate(prepared)
        ],
    }
    _write_json(output / "result.json", result)
    payload = {
        "manifest": manifest,
        "arms": arms_summary,
        "ratios": ratios,
        "gate": gate,
        "cache_store": cache_store,
        "equivalence_summary": equivalence_summary,
        "prepopulate": prepopulate,
        "record_ids": [item["cell"]["record_id"] for item in prepared],
    }
    (output / "summary.md").write_text(_summary_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "records": len(prepared),
        "arms_ms": {arm: arms_summary[arm]["wall_ms"]["median"] for arm in BENCHMARK_ARMS},
        "c_over_a": ratios["C_over_A"]["point"],
        "equivalence_all_equal": equivalence_all_equal,
        "gate_pass": gate["pass"],
    }))
    if not gate["pass"]:
        raise SystemExit(
            f"V65 step 3 gate FAILED: C/A {ratios['C_over_A']['point']:.4f} vs max "
            f"{GATE_C_OVER_A_MAX}, equivalence_all_equal={equivalence_all_equal}"
        )
    return result


if __name__ == "__main__":
    main()
