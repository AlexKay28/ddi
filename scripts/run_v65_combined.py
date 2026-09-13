"""V65 final step: combined serving validation + untouched-record QA confirmation.

Two preregistered validations in one process on one idle GPU; both rule sets are
written to ``manifest.json`` before the model loads or anything is measured:

(1) COMBINED serving validation on the 24 frozen V60 confirmation requests.
    One shared SDPA model (the V60 ``--canonical-on-sdpa`` execution), four
    serving arms over identical tokens:
    - ``A`` monolithic-SDPA baseline — ddi_qf prefill wall only (mask build
      untimed), the step-3 semantics behind the 0.4885 C/A target;
    - ``cold`` — every request fresh-canonical (``CanonicalCache(store=None)``);
    - ``all_hit`` — every request served from the prebuilt DocCache store in the
      LRU-warm steady state (disk payload stays authoritative);
    - ``mixed_hit`` — each request's docs split 50% hit / 50% fresh canonical by
      a deterministic record-id hash (seed 47, ``hit_count = doc_count // 2``);
      fresh write-through entries are pruned after every pass so each pass is a
      genuine 50/50 request.
    Per-request TTFT medians (10 synchronized repeats after 2 warmups per
    record/arm) with paired percentile-bootstrap CIs (seed 47, 10k) of the
    A/schedule ratio of medians; end-to-end per-schedule wall (1 warmup + 3
    timed synchronized passes over the 24 natural-generation requests); peak
    CUDA memory per timed window. Equivalence gate before any timing: every
    arm's natural/fixed generations and normalized predictions must be
    byte-identical to the V60 retained generations (fresh docs re-encode
    identically); the run aborts on any diff. Preregistered combined-win gate:
    the CI lower bound of the paired A/mixed_hit ratio of medians must exceed
    1.0 — reported honestly whichever way it falls (a FAIL is a completed
    result, not an abort).

(2) UNTOUCHED-RECORD QA confirmation: 100 ids from the frozen V62 confirmation
    arm (disjoint from every prior eval set via the split manifest's exclusion
    union), hotpot_qa train split seed 0. Arms: monolithic-SDPA (A) vs
    optimized all-hit-canonical-reuse (C) — the canonical store for those
    records' docs is prebuilt first, then requests are served with reuse.
    Greedy 48/16 budgets, the same scorer as the prior runners. Paired deltas
    C−A, percentile bootstrap CI (seed 47, 10k); rule: non-inferior iff
    CI LB >= -0.02 for BOTH EM and F1. Label: single-seed confirmation, n=100.

Outputs ``manifest.json`` / ``serving_equivalence.jsonl`` /
``serving_timing.jsonl`` / ``qa_requests.jsonl`` / ``result.json`` /
``summary.md`` plus the canonical stores under the output directory; nothing
else is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ddi import inference  # noqa: E402
from ddi.cache import DocCache  # noqa: E402
from ddi.canonical_cache import CanonicalCache  # noqa: E402
from ddi.data import load as load_dataset  # noqa: E402
from ddi.inference import load  # noqa: E402
from ddi.prompt import build_query_text  # noqa: E402
from scripts.run_canonical_matrix import (  # noqa: E402
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
from scripts.run_canonical_qa_dev import (  # noqa: E402
    _default_tokenizer,
    _quality,
    _split_docs_query,
)
from scripts.run_v61_matched_qa import (  # noqa: E402
    DECISION_METRICS,
    DECISION_RULE,
    MARGIN,
    METRICS,
    count_deltas,
    evaluate_decision,
    paired_bootstrap,
)
from scripts.run_v65_benchmark import (  # noqa: E402
    BOOTSTRAP_LEVEL,
    FIXED_BUDGET,
    NATURAL_BUDGET,
    _agg,
    _canonical_pass_fn,
    _generation_payload,
    _monolithic_pass_fns,
    _no_prepare,
    bootstrap_statistic,
    compare_generation,
    load_frozen_cells,
    paired_ratio_bootstrap,
    retained_arm,
)
from scripts.validate_matched_qa import digest, generate_ids  # noqa: E402

SCHEMA_VERSION = 65
SEED = 47
WARMUPS = 2
REPEATS = 10
SCHEDULE_WARMUPS = 1
SCHEDULE_REPEATS = 3
BOOTSTRAP_SEED = 47
BOOTSTRAP_ITERATIONS = 10_000
MIXED_SEED = 47
FLAT_THRESHOLD = 0.02
COMBINED_WIN_MIN_RATIO = 1.0
MIN_FREE_GPU_FRACTION = 0.7
QA_ROLE_LABEL = "single-seed confirmation, n=100"
SERVING_ARMS = ("A", "cold", "all_hit", "mixed_hit")
CANONICAL_SCHEDULES = ("cold", "all_hit", "mixed_hit")

DEFAULT_CHECKPOINT = ROOT / "results/checkpoints/Qwen__Qwen2.5-7B-Instruct__ddi_qf__8L_restart"
DEFAULT_REQUESTS = ROOT / "results/v60_confirmation24_20260910T093500Z/requests.jsonl"
DEFAULT_SOURCE_MANIFEST = ROOT / "experiments/v60_confirmation_manifest.json"
DEFAULT_SPLIT_MANIFEST = ROOT / "experiments/v62_split_manifest.json"
REQUESTS_SHA256 = "ec08d056d6ebe9208cf831e8a63e2d527cc8afec90d377ad9356635767410a5c"
SOURCE_MANIFEST_SHA256 = "2dffebf5d04065021dfb2b9e694651b34f0ebfae4f3d4b9aabd672908aa6cb07"
CHECKPOINT_SHA256 = "00df4c47d67660cab8292da313ea133290b0bb320437290d725cd149c3d31de9"
SPLIT_MANIFEST_SHA256 = "5ad3761106ecaed967b28d21c3919fa1869ebf70bf0e178c1e773a14bf8ee098"
CONFIRMATION_SHA256 = "529f0109eefbf2ef42ede00be146d95cf19876e35859bc1358fbceb1b1bf04d7"
DEVELOPMENT_SHA256 = "aa03e6892b9a05454e254567d6865e56bfc24e2958a62806447f836cb80760df"
STEP3_C_OVER_A = 0.4885
STEP3_SOURCE = "results/v65_benchmark_20260910T101710Z/summary.md (V65 step 3)"


def mixed_hit_indices(
    record_id: str, doc_count: int, *, seed: int = MIXED_SEED
) -> list[int]:
    """Deterministic 50%-hit subset of document indices for one record (pure).

    ``hit_count = doc_count // 2`` (floor, so a request never serves more than
    half its docs from cache); documents are ranked by the first 16 hex digits
    of ``sha256("{seed}:{record_id}:{doc_index}")`` with ties broken by index,
    and the first ``hit_count`` ranks are hits. Same inputs always give the
    same subset; changing the seed changes the subset.
    """
    if doc_count < 0:
        raise ValueError("doc_count must be non-negative")
    hit_count = doc_count // 2
    if hit_count == 0:
        return []

    def rank(index: int) -> tuple[int, int]:
        payload = f"{seed}:{record_id}:{index}".encode()
        return int(hashlib.sha256(payload).hexdigest()[:16], 16), index

    ranked = sorted(range(doc_count), key=rank)
    return sorted(ranked[:hit_count])


def build_schedule_plan(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-record mixed-hit serving plan for the frozen workload (pure)."""
    plan = []
    for cell in cells:
        record_id = cell["record_id"]
        doc_count = len(cell["doc_lengths"])
        hits = mixed_hit_indices(record_id, doc_count)
        hit_set = set(hits)
        plan.append(
            {
                "record_id": record_id,
                "doc_count": doc_count,
                "mixed_hit_indices": hits,
                "mixed_fresh_indices": [
                    index for index in range(doc_count) if index not in hit_set
                ],
            }
        )
    return plan


def combined_win_verdict(
    ratio: dict[str, Any], *, min_ci_low: float = COMBINED_WIN_MIN_RATIO
) -> dict[str, Any]:
    """Evaluate the preregistered combined-win gate (pure)."""
    return {
        "criterion": (
            "combined-win iff the paired A/mixed_hit ratio-of-medians bootstrap "
            f"CI lower bound > {min_ci_low} (mixed-hit faster than monolithic "
            "with 95% confidence)"
        ),
        "min_ci_low": min_ci_low,
        "ratio_point": ratio["point"],
        "ratio_ci_low": ratio["ci_low"],
        "ratio_ci_high": ratio["ci_high"],
        "combined_win": bool(ratio["ci_low"] > min_ci_low),
    }


def expectation_check(wall_medians_ms: dict[str, float]) -> dict[str, Any]:
    """Informational check of the preregistered expectations (pure, no gate)."""
    mono = wall_medians_ms["A"]
    reuse = wall_medians_ms["all_hit"]
    mixed = wall_medians_ms["mixed_hit"]
    return {
        "declared": (
            "reuse beats monolithic (step-3 C/A 0.4885) and mixed-hit lands "
            "between reuse and monolithic"
        ),
        "observed_wall_ms": dict(wall_medians_ms),
        "reuse_beats_monolithic": bool(reuse < mono),
        "mixed_lands_between_reuse_and_monolithic": bool(reuse < mixed < mono),
        "gate": "informational only; the combined-win gate is separate",
    }


def _store_snapshot(store_dir: Path) -> set[str]:
    return {path.name for path in Path(store_dir).iterdir()}


def _prune_store(store_dir: Path, keep: set[str]) -> int:
    """Remove every store file outside ``keep`` (mixed-hit write-through reset)."""
    removed = 0
    for path in sorted(Path(store_dir).iterdir()):
        if path.name not in keep:
            path.unlink()
            removed += 1
    return removed


def _timed_request(
    prepare: Any,
    body: Any,
    *,
    warmups: int,
    repeats: int,
    device: torch.device,
    after_each: Any = None,
) -> dict[str, Any]:
    """Warm up, then run synchronized timed repeats with a peak-memory window.

    Mirrors the step-3 ``_timed_arm`` semantics plus an untimed ``after_each``
    hook (the mixed-hit store pruning between passes).
    """
    for _ in range(warmups):
        prepare()
        body()
        _sync(device)
        if after_each is not None:
            after_each()
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
        if after_each is not None:
            after_each()
    allocated, reserved = _peak_bytes(device)
    return {
        "wall_ms_repeats": walls,
        "wall_ms_median": statistics.median(walls),
        "first_token": sorted(first_tokens)[0],
        "first_token_stable": len(first_tokens) == 1,
        "peak_allocated_bytes": allocated,
        "peak_reserved_bytes": reserved,
    }


def _schedule_passes(
    serve_request: Any,
    items: list[dict[str, Any]],
    *,
    device: torch.device,
    warmups: int,
    repeats: int,
    after_pass: Any = None,
) -> tuple[list[float], list[dict[str, Any]]]:
    """Synchronized end-to-end walls of the full 24-request schedule (pure timing glue)."""
    for _ in range(warmups):
        for item in items:
            serve_request(item)
        _sync(device)
        if after_pass is not None:
            after_pass()
    walls: list[float] = []
    peaks: list[dict[str, Any]] = []
    for _ in range(repeats):
        _reset_peak(device)
        _sync(device)
        started = time.perf_counter()
        for item in items:
            serve_request(item)
        _sync(device)
        walls.append(time.perf_counter() - started)
        allocated, reserved = _peak_bytes(device)
        peaks.append({"peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved})
        if after_pass is not None:
            after_pass()
    return walls, peaks


def _mono_generations(model: Any, tokenizer: Any, batch: Any) -> dict[str, Any]:
    natural = generate_ids(model, tokenizer, batch, "ddi_qf", NATURAL_BUDGET, stop_on_eos=True)
    fixed = generate_ids(model, tokenizer, batch, "ddi_qf", FIXED_BUDGET, stop_on_eos=False)
    return _generation_payload(tokenizer, natural["generated_ids"], fixed["generated_ids"])


def _canonical_generations(
    cache: CanonicalCache, tokenizer: Any, docs: list[Any], query_ids: Any
) -> dict[str, Any]:
    natural = cache.request(
        docs, query_ids, max_new_tokens=NATURAL_BUDGET, eos_token_id=tokenizer.eos_token_id
    )
    fixed = cache.request(docs, query_ids, max_new_tokens=FIXED_BUDGET, eos_token_id=None)
    return _generation_payload(tokenizer, natural, fixed)


def _retained_for(arm: str, cell: dict[str, Any]) -> dict[str, Any]:
    if arm == "A":
        return retained_arm(cell, "A")
    if arm == "cold":
        return retained_arm(cell, "B")
    return retained_arm(cell, "C_all_hit")


def _extract_confirmation_ids(split_manifest: dict[str, Any]) -> list[str]:
    confirmation = split_manifest["splits"]["confirmation"]
    ids = list(confirmation["record_ids"])
    if len(ids) != int(confirmation["count"]):
        raise ValueError(
            f"split manifest confirmation count {confirmation['count']} != {len(ids)} ids"
        )
    if confirmation["sha256"] != CONFIRMATION_SHA256:
        raise ValueError("confirmation arm sha256 does not match the pinned value")
    return ids


def _check_untouchedness(
    confirmation_ids: list[str], split_manifest: dict[str, Any], v60_ids: list[str]
) -> dict[str, Any]:
    """Verify the confirmation arm is disjoint from every prior eval id set."""
    development = list(split_manifest["splits"]["development"]["record_ids"])
    train = list(split_manifest["splits"]["train"]["record_ids"])
    checks = {
        "vs_v60_confirmation_24": len(set(confirmation_ids) & set(v60_ids)),
        "vs_v62_development_100": len(set(confirmation_ids) & set(development)),
        "vs_v62_train_800": len(set(confirmation_ids) & set(train)),
    }
    if any(count for count in checks.values()):
        raise ValueError(f"confirmation ids overlap a prior eval set: {checks}")
    exclusions = split_manifest["exclusions"]
    return {
        "checks_zero_overlap": checks,
        "exclusion_union_count": exclusions["union_count"],
        "exclusion_sources": [
            source["path"] for source in exclusions["sources"]
        ],
        "split_manifest_disjointness": split_manifest["disjointness"],
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--requests", type=Path, default=DEFAULT_REQUESTS)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--record-limit", type=int, default=None)
    parser.add_argument("--qa-record-limit", type=int, default=None)
    parser.add_argument("--warmups", type=int, default=WARMUPS)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--schedule-warmups", type=int, default=SCHEDULE_WARMUPS)
    parser.add_argument("--schedule-repeats", type=int, default=SCHEDULE_REPEATS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--bootstrap-iterations", type=int, default=BOOTSTRAP_ITERATIONS)
    args = parser.parse_args(argv)
    if args.warmups < 0 or args.repeats < 1:
        parser.error("--warmups must be >= 0 and --repeats must be >= 1")
    if args.schedule_warmups < 0 or args.schedule_repeats < 1:
        parser.error("--schedule-warmups must be >= 0 and --schedule-repeats must be >= 1")
    if args.bootstrap_iterations < 1:
        parser.error("--bootstrap-iterations must be >= 1")
    return args


def _build_manifest(
    args: argparse.Namespace,
    *,
    cells: list[dict[str, Any]],
    plan: list[dict[str, Any]],
    confirmation_ids: list[str],
    untouchedness: dict[str, Any],
    split_manifest: dict[str, Any],
    source_config: dict[str, Any],
    checkpoint_identity: tuple[str, str],
    gpu: dict[str, Any] | None,
) -> dict[str, Any]:
    dtype_str = "torch.bfloat16" if args.device == "cuda" else "torch.float32"
    bootstrap = {
        "seed": BOOTSTRAP_SEED,
        "iterations": args.bootstrap_iterations,
        "ci_level": BOOTSTRAP_LEVEL,
        "method": (
            "paired percentile resample of the frozen records; ratios recomputed as "
            "ratio of medians on the resampled records"
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "runner": "scripts/run_v65_combined.py",
        "created_utc": datetime.now(UTC).isoformat(),
        "manifest_written_before_run": True,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_identity": {
            "mode": checkpoint_identity[0],
            "sha256": checkpoint_identity[1],
        },
        "pins": {
            "v60_requests": {
                "path": str(args.requests.resolve()),
                "sha256": _sha_file(args.requests),
            },
            "v60_source_manifest": {
                "path": str(args.source_manifest.resolve()),
                "sha256": _sha_file(args.source_manifest),
                "dataset": source_config["dataset"],
                "seed": source_config["seed"],
            },
            "v62_split_manifest": {
                "path": str(args.split_manifest.resolve()),
                "sha256": _sha_file(args.split_manifest),
                "confirmation_sha256": split_manifest["splits"]["confirmation"]["sha256"],
                "development_sha256": split_manifest["splits"]["development"]["sha256"],
            },
        },
        "preregistered": {
            "validation_1_combined_serving": {
                "name": "combined_serving_v65",
                "data_role": "confirmation_frozen",
                "data_role_note": (
                    "the 24 frozen V60 confirmation ids are reused for timing and "
                    "byte-equivalence only, per V64 audit qualifier (d); no new quality "
                    "claims are made on them here"
                ),
                "workload": {
                    "requests": [cell["record_id"] for cell in cells],
                    "record_count": len(cells),
                },
                "protocol": {
                    "one_process": True,
                    "seed": args.seed,
                    "ttft_warmups": args.warmups,
                    "ttft_repeats": args.repeats,
                    "schedule_warmups": args.schedule_warmups,
                    "schedule_timed_passes": args.schedule_repeats,
                    "synchronize": True,
                    "canonical_on_sdpa": True,
                    "device": args.device,
                    "gpu": gpu,
                    "budgets": {
                        "natural_max_new_tokens": NATURAL_BUDGET,
                        "fixed_new_tokens": FIXED_BUDGET,
                    },
                    "bootstrap": bootstrap,
                },
                "schedules": {
                    "A_monolithic_sdpa": (
                        "monolithic SDPA ddi_qf baseline; TTFT wall = the single prefill "
                        "model() call only (mask build untimed), mirroring the "
                        "generate_ids(...)[\"prefill_ms\"] semantics behind the step-3 "
                        "targets"
                    ),
                    "cold": (
                        "all 24 requests fresh-canonical: synchronized wall of the full "
                        "CanonicalCache(store=None).request(docs, query_ids) call"
                    ),
                    "all_hit": (
                        "all 24 requests reuse: synchronized wall of the full "
                        "CanonicalCache(store=DocCache).request call, store pre-populated; "
                        "warmups absorb the disk tier into the V65 LRU; the disk payload "
                        "stays authoritative"
                    ),
                    "mixed_hit": {
                        "rule": (
                            "each record's docs split 50% hit / 50% fresh canonical; "
                            "hit_count = doc_count // 2 (floor); hit docs selected by "
                            "ascending sha256('{seed}:{record_id}:{doc_index}') first "
                            "16 hex digits, ties by index; fresh docs re-encode on every "
                            "pass and their write-through entries are pruned after every "
                            "pass so each pass is a genuine 50/50 request"
                        ),
                        "seed": MIXED_SEED,
                        "plan": plan,
                    },
                },
                "ttft_definition": (
                    "per (record, arm): median of the synchronized repeat walls; A = "
                    "prefill model() call only; canonical arms = full "
                    "CanonicalCache.request wall (prefill through query logits)"
                ),
                "schedule_wall_definition": (
                    "end-to-end synchronized wall around serving all 24 natural-generation "
                    "requests (budget 48, stop on EOS) sequentially; 1 warmup pass then "
                    "3 timed passes per arm; median reported; mixed store pruned after "
                    "every pass"
                ),
                "memory_definition": (
                    "torch.cuda max_memory_allocated/reserved within each timed window "
                    "(per (record, arm) TTFT repeats and per schedule pass); reported as "
                    "max over records/passes; includes the shared model weights"
                ),
                "gate": {
                    "name": "combined_win",
                    "criterion": (
                        "the CI lower bound of the paired A/mixed_hit ratio-of-medians "
                        "bootstrap > 1.0 for a combined-win claim"
                    ),
                    "min_ci_low": COMBINED_WIN_MIN_RATIO,
                    "declared_before_run": True,
                    "on_fail": (
                        "report honestly whichever way it falls; a FAIL is a completed "
                        "preregistered result, not an abort"
                    ),
                },
                "equivalence": {
                    "rule": (
                        "before any timing, every arm's normalized predictions and "
                        "natural/fixed generations must be byte-identical to the V60 "
                        "retained generations for every request (fresh docs must "
                        "re-encode identically); the run aborts on any diff"
                    ),
                    "retained_mapping": {
                        "A": "V60 arm A",
                        "cold": "V60 arm B (fresh canonical)",
                        "all_hit": "V60 arm C all_hit",
                        "mixed_hit": "V60 arm C all_hit (V60 verified B == C on all 24)",
                    },
                    "informational": "mixed_hit is additionally compared against V60 arm B",
                },
                "expectations_informational": {
                    "declared": (
                        "reuse beats monolithic (step-3 C/A 0.4885) and mixed-hit lands "
                        "between reuse and monolithic"
                    ),
                    "step3_reference": {"c_over_a": STEP3_C_OVER_A, "source": STEP3_SOURCE},
                },
                "arms": {
                    "A": {
                        "backend": "sdpa",
                        "execution": "original_monolithic_ddi_qf",
                        "dtype": dtype_str,
                        "arm_identity": _arm_identity("A", dtype_str),
                    },
                    **{
                        arm: {
                            "backend": "sdpa",
                            "execution": "canonical_per_document_ddi_qf",
                            "cache": {
                                "cold": "none",
                                "all_hit": "persisted",
                                "mixed_hit": "persisted_partial",
                            }[arm],
                            "dtype": dtype_str,
                            "arm_identity": _arm_identity(
                                {"cold": "B", "all_hit": "C", "mixed_hit": "C"}[arm], dtype_str
                            ),
                        }
                        for arm in CANONICAL_SCHEDULES
                    },
                },
                "canonical_on_sdpa_note": (
                    "all canonical schedules execute on the same SDPA model "
                    "(--canonical-on-sdpa execution, V64 audit qualifier (c))"
                ),
                "cache_store_plan": {
                    "all_hit_dir": "canonical_cache/all_hit",
                    "mixed_hit_dir": "canonical_cache/mixed_hit",
                    "growth_check": (
                        "store file/byte snapshots around the timed phases; the all-hit "
                        "store must not grow, the mixed store must end at its hit-only "
                        "prepopulation snapshot"
                    ),
                },
            },
            "validation_2_qa_confirmation": {
                "name": "untouched_record_qa_confirmation",
                "data_role": "confirmation",
                "role_label": QA_ROLE_LABEL,
                "ids": {
                    "split_manifest": str(args.split_manifest.resolve()),
                    "count": len(confirmation_ids),
                    "record_ids": confirmation_ids,
                    "dataset": "hotpot_qa",
                    "split": "train",
                    "seed": 0,
                    "selection_rule": split_manifest["selection_rule"],
                },
                "untouchedness": untouchedness,
                "arms": {
                    "A": {
                        "name": "monolithic_sdpa",
                        "execution": "original_monolithic_ddi_qf greedy on the SDPA model",
                        "backend": "sdpa",
                        "dtype": dtype_str,
                    },
                    "C": {
                        "name": "optimized_all_hit_canonical_reuse",
                        "execution": (
                            "CanonicalCache(store=DocCache).request on the same SDPA "
                            "model (canonical-on-sdpa); the store for these records' "
                            "docs is prebuilt first, then requests are served with reuse"
                        ),
                        "backend": "sdpa",
                        "dtype": dtype_str,
                    },
                },
                "budgets": {
                    "natural_max_new_tokens": NATURAL_BUDGET,
                    "fixed_new_tokens": FIXED_BUDGET,
                    "decoding": "greedy",
                },
                "scorer": (
                    "ddi.metrics exact_match/token_f1/answer_presence via "
                    "scripts.run_canonical_qa_dev._quality (the same scorer as the "
                    "prior V60/V61/V62 runners)"
                ),
                "analysis": {
                    "paired_deltas": "C - A per record",
                    "bootstrap": {
                        "seed": BOOTSTRAP_SEED,
                        "resamples": args.bootstrap_iterations,
                        "method": "percentile bootstrap of the mean paired delta",
                    },
                    "flat_threshold": FLAT_THRESHOLD,
                },
                "decision_rule": {
                    "verbatim": DECISION_RULE,
                    "margin": MARGIN,
                    "metrics": list(DECISION_METRICS),
                    "declared": "before the GPU run",
                },
                "label": QA_ROLE_LABEL,
            },
        },
        "arms_identity_note": (
            "arm_identity values reuse the V60 A/B/C identities; the mixed_hit schedule "
            "reuses the C identity because it is the same canonical execution contract "
            "with a partially populated store"
        ),
        "sources": {
            "scripts/run_v65_combined.py": _sha_file(Path(__file__).resolve()),
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
            "device": args.device,
            "gpu_name": gpu["name"] if gpu else None,
        },
    }


def _equivalence_rows(
    items: list[dict[str, Any]], *, model: Any, tokenizer: Any, contract: Any,
    hit_store: DocCache, mixed_store: DocCache, mixed_keep: set[str], device: torch.device,
) -> list[dict[str, Any]]:
    """Serve every request on every arm and verify byte-identity (abort on diff)."""
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        cell, batch, docs, query_ids = (
            item["cell"], item["batch"], item["docs"], item["query_ids"],
        )
        fresh = CanonicalCache(model, contract)
        cached_hit = CanonicalCache(model, contract, hit_store)
        cached_mixed = CanonicalCache(model, contract, mixed_store)
        generations = {
            "A": _mono_generations(model, tokenizer, batch),
            "cold": _canonical_generations(fresh, tokenizer, docs, query_ids),
            "all_hit": _canonical_generations(cached_hit, tokenizer, docs, query_ids),
            "mixed_hit": _canonical_generations(cached_mixed, tokenizer, docs, query_ids),
        }
        # Fresh write-through entries of the mixed store must not leak into the
        # next record's mixed request.
        _prune_store(mixed_store.cache_dir, mixed_keep)
        equivalence = {
            arm: compare_generation(generations[arm], _retained_for(arm, cell))
            for arm in SERVING_ARMS
        }
        mixed_vs_b = compare_generation(generations["mixed_hit"], retained_arm(cell, "B"))
        for arm in SERVING_ARMS:
            failed = [key for key, ok in equivalence[arm].items() if not ok]
            if failed:
                raise RuntimeError(
                    "V65 combined aborted before timing: arm "
                    f"{arm} outputs differ from the V60 retained generations for record "
                    f"{cell['record_id']}: {sorted(failed)}"
                )
        row = {
            "request_index": index,
            "record_id": cell["record_id"],
            "doc_lengths": cell["doc_lengths"],
            "query_length": cell["query_length"],
            "mixed_hit_indices": item["plan"]["mixed_hit_indices"],
            "mixed_fresh_indices": item["plan"]["mixed_fresh_indices"],
            "generations": generations,
            "equivalence": equivalence,
            "mixed_hit_vs_retained_B": mixed_vs_b,
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "stage": "equivalence",
                    "request_index": index,
                    "record_id": cell["record_id"],
                    "all_arms_byte_identical": True,
                    "mixed_vs_B_identical": mixed_vs_b["byte_identical"],
                }
            )
        )
    return rows


def _timing_rows(
    items: list[dict[str, Any]], *, model: Any, tokenizer: Any, contract: Any,
    hit_store: DocCache, mixed_store: DocCache, mixed_keep: set[str],
    device: torch.device, warmups: int, repeats: int,
) -> list[dict[str, Any]]:
    """Per-(record, arm) synchronized TTFT repeats with peak-memory windows."""
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        batch, docs, query_ids = item["batch"], item["docs"], item["query_ids"]
        fresh = CanonicalCache(model, contract)
        cached_hit = CanonicalCache(model, contract, hit_store)
        cached_mixed = CanonicalCache(model, contract, mixed_store)
        mono_prepare, mono_body = _monolithic_pass_fns(model, batch, device)
        timed = {
            "A": _timed_request(
                mono_prepare, mono_body, warmups=warmups, repeats=repeats, device=device
            ),
            "cold": _timed_request(
                _no_prepare,
                _canonical_pass_fn(fresh, docs, query_ids),
                warmups=warmups,
                repeats=repeats,
                device=device,
            ),
            "all_hit": _timed_request(
                _no_prepare,
                _canonical_pass_fn(cached_hit, docs, query_ids),
                warmups=warmups,
                repeats=repeats,
                device=device,
            ),
            "mixed_hit": _timed_request(
                _no_prepare,
                _canonical_pass_fn(cached_mixed, docs, query_ids),
                warmups=warmups,
                repeats=repeats,
                device=device,
                after_each=lambda keep=mixed_keep: _prune_store(mixed_store.cache_dir, keep),
            ),
        }
        row = {
            "request_index": index,
            "record_id": item["cell"]["record_id"],
            "timed": timed,
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "stage": "ttft",
                    "request_index": index,
                    "record_id": row["record_id"],
                    "wall_ms": {
                        arm: round(timed[arm]["wall_ms_median"], 2) for arm in SERVING_ARMS
                    },
                }
            )
        )
    return rows


def _schedule_server(
    arm: str,
    *,
    model: Any,
    tokenizer: Any,
    contract: Any,
    hit_store: DocCache,
    mixed_store: DocCache,
    mixed_keep: set[str],
    eos_token_id: int | None,
) -> tuple[Any, Any]:
    """Per-schedule request server plus its after-pass hook (mixed store pruning)."""
    if arm == "A":

        def serve_monolithic(item: dict[str, Any]) -> Any:
            return generate_ids(
                model, tokenizer, item["batch"], "ddi_qf", NATURAL_BUDGET, stop_on_eos=True
            )

        return serve_monolithic, None
    cache = {
        "cold": CanonicalCache(model, contract),
        "all_hit": CanonicalCache(model, contract, hit_store),
        "mixed_hit": CanonicalCache(model, contract, mixed_store),
    }[arm]

    def serve_canonical(item: dict[str, Any], cache: Any = cache) -> Any:
        return cache.request(
            item["docs"],
            item["query_ids"],
            max_new_tokens=NATURAL_BUDGET,
            eos_token_id=eos_token_id,
        )

    after_pass = (
        (lambda keep=mixed_keep: _prune_store(mixed_store.cache_dir, keep))
        if arm == "mixed_hit"
        else None
    )
    return serve_canonical, after_pass


def _schedule_walls(
    items: list[dict[str, Any]], *, model: Any, tokenizer: Any, contract: Any,
    hit_store: DocCache, mixed_store: DocCache, mixed_keep: set[str],
    device: torch.device, warmups: int, repeats: int,
) -> dict[str, Any]:
    """End-to-end per-schedule walls over the 24-request natural-generation workload."""
    schedules: dict[str, Any] = {}
    for arm in SERVING_ARMS:
        serve, after_pass = _schedule_server(
            arm,
            model=model,
            tokenizer=tokenizer,
            contract=contract,
            hit_store=hit_store,
            mixed_store=mixed_store,
            mixed_keep=mixed_keep,
            eos_token_id=tokenizer.eos_token_id,
        )
        walls, peaks = _schedule_passes(
            serve,
            items,
            device=device,
            warmups=warmups,
            repeats=repeats,
            after_pass=after_pass,
        )
        schedules[arm] = {
            "wall_seconds_repeats": walls,
            "wall_seconds_median": statistics.median(walls),
            "peak_allocated_bytes": max(
                peak["peak_allocated_bytes"] for peak in peaks
            ),
            "peak_reserved_bytes": max(peak["peak_reserved_bytes"] for peak in peaks),
        }
        print(
            json.dumps(
                {
                    "stage": "schedule_wall",
                    "arm": arm,
                    "wall_seconds": [round(wall, 3) for wall in walls],
                }
            )
        )
    return schedules


def _prebuild_qa_store(
    records: dict[str, dict[str, Any]],
    confirmation_ids: list[str],
    *,
    model: Any,
    contract: Any,
    store: DocCache,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    """Encode every confirmation record's docs into the store before any serving."""
    cache = CanonicalCache(model, contract, store)
    rows: list[dict[str, Any]] = []
    for index, record_id in enumerate(confirmation_ids):
        record = records[record_id]
        batch = inference.encode(
            tokenizer, list(record["info_contexts"]), build_query_text(tokenizer, record)
        )
        docs, _query_ids = _split_docs_query(batch)
        started = time.perf_counter()
        cache.documents([doc.cpu() for doc in docs])
        wall = time.perf_counter() - started
        rows.append(
            {
                "request_index": index,
                "record_id": record_id,
                "doc_count": len(docs),
                "prebuild_wall_seconds": wall,
            }
        )
        print(
            json.dumps(
                {
                    "stage": "qa_prebuild",
                    "request_index": index,
                    "record_id": record_id,
                    "prebuild_wall_seconds": round(wall, 3),
                }
            )
        )
    return rows


def _qa_rows(
    records: dict[str, dict[str, Any]],
    confirmation_ids: list[str],
    *,
    model: Any,
    tokenizer: Any,
    cache: CanonicalCache,
) -> list[dict[str, Any]]:
    """Serve both QA arms per record with the shared scorer (same as prior runners)."""
    rows: list[dict[str, Any]] = []
    for index, record_id in enumerate(confirmation_ids):
        record = records[record_id]
        batch = inference.encode(
            tokenizer, list(record["info_contexts"]), build_query_text(tokenizer, record)
        )
        docs, query_ids = _split_docs_query(batch)
        gold = list(record["possible_answers"])
        natural = generate_ids(
            model, tokenizer, batch, "ddi_qf", NATURAL_BUDGET, stop_on_eos=True
        )
        fixed = generate_ids(model, tokenizer, batch, "ddi_qf", FIXED_BUDGET, stop_on_eos=False)
        a_prediction = tokenizer.decode(
            natural["generated_ids"], skip_special_tokens=True
        ).strip()
        c_natural = cache.request(
            docs, query_ids, max_new_tokens=NATURAL_BUDGET, eos_token_id=tokenizer.eos_token_id
        )
        c_fixed = cache.request(docs, query_ids, max_new_tokens=FIXED_BUDGET, eos_token_id=None)
        c_prediction = tokenizer.decode(c_natural, skip_special_tokens=True).strip()
        row = {
            "request_index": index,
            "record_id": record_id,
            "request_id": f"hotpot_qa:0:{record_id}",
            "doc_lengths": batch.doc_lengths,
            "query_length": batch.query_len,
            "gold_answers": gold,
            "A": {
                "prediction": a_prediction,
                "natural_generated_ids": natural["generated_ids"],
                "stop_reason": natural["stop_reason"],
                "fixed_generated_ids": fixed["generated_ids"],
                "fixed_ids_sha256": digest(
                    json.dumps(fixed["generated_ids"], separators=(",", ":")).encode()
                ),
                "quality": _quality(a_prediction, gold),
            },
            "C": {
                "prediction": c_prediction,
                "natural_generated_ids": list(c_natural),
                "stop_reason": "eos" if tokenizer.eos_token_id in c_natural else "length",
                "fixed_generated_ids": list(c_fixed),
                "fixed_ids_sha256": digest(json.dumps(c_fixed, separators=(",", ":")).encode()),
                "quality": _quality(c_prediction, gold),
            },
            "prediction_equal": a_prediction == c_prediction,
            "first_token_equal": bool(
                natural["generated_ids"][0] == c_natural[0]
            ),
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "stage": "qa",
                    "request_index": index,
                    "record_id": record_id,
                    "quality": {"A": row["A"]["quality"], "C": row["C"]["quality"]},
                }
            )
        )
    return rows


def _analyze_qa(rows: list[dict[str, Any]], *, seed: int, resamples: int) -> dict[str, Any]:
    """Paired C−A analysis with the v61 bootstrap/decision machinery."""
    analysis: dict[str, Any] = {}
    for metric in METRICS:
        deltas = [row["C"]["quality"][metric] - row["A"]["quality"][metric] for row in rows]
        bootstrap = paired_bootstrap(deltas, seed=seed, resamples=resamples)
        counts = count_deltas(deltas, threshold=FLAT_THRESHOLD)
        analysis[metric] = {
            **bootstrap,
            **counts,
            "reference_mean": statistics.mean(row["A"]["quality"][metric] for row in rows),
            "candidate_mean": statistics.mean(row["C"]["quality"][metric] for row in rows),
        }
    decision = evaluate_decision(analysis)
    return {"analysis": analysis, "decision": decision}


def _summary_markdown(payload: dict[str, Any]) -> str:
    manifest = payload["manifest"]
    serving = payload["serving"]
    qa = payload["qa"]
    lines = [
        "# V65 final step — combined serving validation + untouched-record QA confirmation",
        "",
        f"- Created (UTC): {manifest['created_utc']}",
        f"- Checkpoint: `{manifest['checkpoint']}`",
        "- Both rule sets were written to `manifest.json` before the model loaded "
        "(`manifest_written_before_run: true`).",
        "",
        "## Validation 1 — combined serving on the 24 frozen V60 confirmation requests",
        "",
        "- Protocol: one process, seed "
        f"{manifest['preregistered']['validation_1_combined_serving']['protocol']['seed']}, "
        "TTFT = median of 10 synchronized repeats after 2 warmups per (record, arm); "
        "schedule wall = 1 warmup + 3 timed synchronized passes over the 24 "
        "natural-generation requests; one SDPA model (canonical-on-sdpa execution).",
        "- Equivalence: every arm byte-identical to the V60 retained generations on all "
        f"24 requests: {serving['equivalence_all_equal']}",
        "",
        "### Per-schedule table (per-request TTFT wall ms; CI = paired percentile "
        "bootstrap, seed 47, 10k; ratio vs monolithic A on the same tokens)",
        "",
        "| schedule | TTFT ms median [95% CI] | A/schedule ratio [95% CI] | "
        "peak alloc MiB (max) | peak reserved MiB (max) |",
        "|---|---|---|---|---|",
    ]
    arms = serving["arms"]
    likes = {
        "A": "A monolithic-SDPA",
        "cold": "cold (fresh canonical)",
        "all_hit": "all-hit (reuse)",
        "mixed_hit": "mixed-hit (50/50 combined)",
    }
    for arm in SERVING_ARMS:
        entry = arms[arm]
        ratio = serving["ratios"][f"A_over_{arm}"] if arm != "A" else None
        ratio_text = (
            "1.0 (reference)"
            if ratio is None
            else f"{ratio['point']:.4f} [{ratio['ci_low']:.4f}, {ratio['ci_high']:.4f}]"
        )
        lines.append(
            f"| {likes[arm]} | {entry['wall_ms']['median']:.2f} "
            f"[{entry['wall_ms_ci']['ci_low']:.2f}, {entry['wall_ms_ci']['ci_high']:.2f}] "
            f"| {ratio_text} | {entry['peak_allocated_bytes']['max'] / 2**20:.0f} "
            f"| {entry['peak_reserved_bytes']['max'] / 2**20:.0f} |"
        )
    gate = serving["combined_win_gate"]
    lines += [
        "",
        "### End-to-end schedule wall (24 natural-generation requests, "
        "median of 3 timed passes)",
        "",
        "| schedule | wall s (median) | repeats s |",
        "|---|---|---|",
    ]
    for arm in SERVING_ARMS:
        schedule = serving["schedule_walls"][arm]
        lines.append(
            f"| {likes[arm]} | {schedule['wall_seconds_median']:.3f} | "
            f"[{', '.join(f'{w:.3f}' for w in schedule['wall_seconds_repeats'])}] |"
        )
    expectations = serving["expectations"]
    lines += [
        "",
        f"- Combined-win gate: {'PASS' if gate['combined_win'] else 'FAIL'} — "
        f"{gate['criterion']}; A/mixed point {gate['ratio_point']:.4f}, "
        f"CI [{gate['ratio_ci_low']:.4f}, {gate['ratio_ci_high']:.4f}] vs min LB "
        f"{gate['min_ci_low']}.",
        f"- Expectations (informational): reuse beats monolithic "
        f"({expectations['reuse_beats_monolithic']}); mixed-hit lands between reuse and "
        f"monolithic ({expectations['mixed_lands_between_reuse_and_monolithic']}).",
        "",
        "## Validation 2 — untouched-record QA confirmation (single-seed confirmation, n=100)",
        "",
        "- 100 ids from the frozen V62 confirmation arm (disjoint from every prior eval "
        "set; checks recorded in the manifest), hotpot_qa train split seed 0; arms: "
        "monolithic-SDPA (A) vs optimized all-hit reuse (C, store prebuilt first); "
        "greedy 48/16 budgets; paired deltas C−A, bootstrap CI (seed 47, 10k).",
        "",
        "| metric | A mean | C mean | mean delta (C−A) | 95% CI | improved/flat/degraded |",
        "|---|---|---|---|---|---|",
    ]
    analysis = qa["analysis"]
    for metric in METRICS:
        entry = analysis[metric]
        lines.append(
            f"| {metric} | {entry['reference_mean']:.4f} | {entry['candidate_mean']:.4f} "
            f"| {entry['mean']:+.4f} | [{entry['ci95'][0]:.4f}, {entry['ci95'][1]:.4f}] "
            f"| {entry['improved']}/{entry['flat']}/{entry['degraded']} |"
        )
    decision = qa["decision"]
    lines += [
        "",
        f"- Decision: {'NON-INFERIOR' if decision['non_inferior'] else 'NOT NON-INFERIOR'} "
        f"— {decision['rule']}; CI lower bounds EM "
        f"{decision['ci_lower_bounds']['em']:.4f}, F1 {decision['ci_lower_bounds']['f1']:.4f} "
        f"(margin {decision['margin']}).",
        f"- Label: {QA_ROLE_LABEL}. Paired records {qa['paired_records']}, errors "
        f"{qa['errors']}, prediction-equal {qa['prediction_equal']}, first-token-equal "
        f"{qa['first_token_equal']}.",
        "",
        "## Cache stores",
        "",
    ]
    for name, store in serving["cache_stores"].items():
        lines.append(
            f"- {name}: {store['files_after_prebuild']} files / "
            f"{store['bytes_after_prebuild']} bytes after prebuild; "
            f"{store['files_final']} files / {store['bytes_final']} bytes final "
            f"(growth {store['growth_bytes']} bytes)"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    started = time.perf_counter()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")

    if _sha_file(args.source_manifest) != SOURCE_MANIFEST_SHA256:
        raise ValueError(f"source manifest sha mismatch: {args.source_manifest}")
    if _sha_file(args.split_manifest) != SPLIT_MANIFEST_SHA256:
        raise ValueError(f"split manifest sha mismatch: {args.split_manifest}")
    source_manifest = json.loads(args.source_manifest.read_text())
    source_config = source_manifest["config"]
    split_manifest = json.loads(args.split_manifest.read_text())
    cells = load_frozen_cells(args.requests, expected_sha256=REQUESTS_SHA256)
    if args.record_limit is not None:
        cells = cells[: args.record_limit]
    confirmation_ids = _extract_confirmation_ids(split_manifest)
    if args.qa_record_limit is not None:
        confirmation_ids = confirmation_ids[: args.qa_record_limit]
    untouchedness = _check_untouchedness(
        confirmation_ids, split_manifest, source_manifest["config"]["record_ids"]
    )
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

    plan = build_schedule_plan(cells)
    output = args.output or (
        ROOT / "results" / f"v65_combined_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    output = _prepare_output(Path(output))
    manifest = _build_manifest(
        args,
        cells=cells,
        plan=plan,
        confirmation_ids=confirmation_ids,
        untouchedness=untouchedness,
        split_manifest=split_manifest,
        source_config=source_config,
        checkpoint_identity=checkpoint_identity,
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

    v60_records = {
        record["id"]: record
        for record in load_dataset(source_config["dataset"], "val", source_config["seed"])
    }
    items: list[dict[str, Any]] = []
    for cell, plan_cell in zip(cells, plan, strict=True):
        if plan_cell["record_id"] != cell["record_id"]:
            raise ValueError("schedule plan does not align with the frozen cells")
        record = v60_records[cell["record_id"]]
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
        items.append(
            {
                "cell": cell,
                "batch": batch,
                "docs": docs,
                "query_ids": query_ids,
                "plan": plan_cell,
            }
        )

    hit_store = DocCache(output / "canonical_cache" / "all_hit")
    mixed_store = DocCache(output / "canonical_cache" / "mixed_hit")
    prebuilder = CanonicalCache(model, contract, hit_store)
    mixed_prebuilder = CanonicalCache(model, contract, mixed_store)
    for item in items:
        started_prebuild = time.perf_counter()
        prebuilder.documents([doc.cpu() for doc in item["docs"]])
        item["prepopulate_wall_ms"] = (time.perf_counter() - started_prebuild) * 1e3
        hit_docs = [item["docs"][i].cpu() for i in item["plan"]["mixed_hit_indices"]]
        if hit_docs:
            mixed_prebuilder.documents(hit_docs)
    mixed_keep = _store_snapshot(mixed_store.cache_dir)

    with (output / "serving_equivalence.jsonl").open("x", encoding="utf-8") as handle:
        equivalence_rows = _equivalence_rows(
            items,
            model=model,
            tokenizer=tokenizer,
            contract=contract,
            hit_store=hit_store,
            mixed_store=mixed_store,
            mixed_keep=mixed_keep,
            device=device,
        )
        for row in equivalence_rows:
            handle.write(
                json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            )

    with (output / "serving_timing.jsonl").open("x", encoding="utf-8") as handle:
        timing_rows = _timing_rows(
            items,
            model=model,
            tokenizer=tokenizer,
            contract=contract,
            hit_store=hit_store,
            mixed_store=mixed_store,
            mixed_keep=mixed_keep,
            device=device,
            warmups=args.warmups,
            repeats=args.repeats,
        )
        for row in timing_rows:
            handle.write(
                json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            )

    store_snapshots_before = {
        "all_hit": (hit_store.size(), _store_bytes(hit_store.cache_dir)),
        "mixed_hit": (mixed_store.size(), _store_bytes(mixed_store.cache_dir)),
    }
    schedule_walls = _schedule_walls(
        items,
        model=model,
        tokenizer=tokenizer,
        contract=contract,
        hit_store=hit_store,
        mixed_store=mixed_store,
        mixed_keep=mixed_keep,
        device=device,
        warmups=args.schedule_warmups,
        repeats=args.schedule_repeats,
    )

    per_record_walls = {
        arm: [row["timed"][arm]["wall_ms_median"] for row in timing_rows]
        for arm in SERVING_ARMS
    }
    arms_summary: dict[str, Any] = {}
    for arm in SERVING_ARMS:
        allocated = [row["timed"][arm]["peak_allocated_bytes"] for row in timing_rows]
        reserved = [row["timed"][arm]["peak_reserved_bytes"] for row in timing_rows]
        arms_summary[arm] = {
            "wall_ms": _agg(per_record_walls[arm]),
            "wall_ms_ci": bootstrap_statistic(
                per_record_walls[arm],
                seed=BOOTSTRAP_SEED,
                iterations=args.bootstrap_iterations,
            ),
            "peak_allocated_bytes": {
                "max": max(allocated),
                "median": statistics.median(allocated),
            },
            "peak_reserved_bytes": {
                "max": max(reserved),
                "median": statistics.median(reserved),
            },
        }
    common = {"seed": BOOTSTRAP_SEED, "iterations": args.bootstrap_iterations}
    ratios = {
        f"A_over_{arm}": paired_ratio_bootstrap(
            per_record_walls["A"], per_record_walls[arm], **common
        )
        for arm in CANONICAL_SCHEDULES
    }
    combined_win = combined_win_verdict(ratios["A_over_mixed_hit"])
    expectations = expectation_check(
        {arm: arms_summary[arm]["wall_ms"]["median"] for arm in SERVING_ARMS}
    )
    equivalence_all_equal = all(
        row["equivalence"][arm]["byte_identical"]
        for row in equivalence_rows
        for arm in SERVING_ARMS
    )
    cache_stores = {
        "all_hit": {
            "path": str(hit_store.cache_dir),
            "files_after_prebuild": store_snapshots_before["all_hit"][0],
            "bytes_after_prebuild": store_snapshots_before["all_hit"][1],
            "files_final": hit_store.size(),
            "bytes_final": _store_bytes(hit_store.cache_dir),
            "growth_bytes": (
                _store_bytes(hit_store.cache_dir) - store_snapshots_before["all_hit"][1]
            ),
        },
        "mixed_hit": {
            "path": str(mixed_store.cache_dir),
            "files_after_prebuild": store_snapshots_before["mixed_hit"][0],
            "bytes_after_prebuild": store_snapshots_before["mixed_hit"][1],
            "files_final": mixed_store.size(),
            "bytes_final": _store_bytes(mixed_store.cache_dir),
            "growth_bytes": (
                _store_bytes(mixed_store.cache_dir) - store_snapshots_before["mixed_hit"][1]
            ),
            "hit_only_snapshot_files": len(mixed_keep),
        },
    }
    serving = {
        "equivalence_all_equal": equivalence_all_equal,
        "arms": arms_summary,
        "ratios": ratios,
        "combined_win_gate": combined_win,
        "expectations": expectations,
        "schedule_walls": schedule_walls,
        "cache_stores": cache_stores,
        "prepopulate_wall_ms": _agg([item["prepopulate_wall_ms"] for item in items]),
    }

    train_records = {
        record["id"]: record for record in load_dataset("hotpot_qa", "train", 0)
    }
    missing = [record_id for record_id in confirmation_ids if record_id not in train_records]
    if missing:
        raise ValueError(f"confirmation ids absent from the hotpot_qa train stream: {missing[:5]}")

    qa_store = DocCache(output / "canonical_cache" / "qa_confirmation")
    prebuild_rows = _prebuild_qa_store(
        train_records,
        confirmation_ids,
        model=model,
        contract=contract,
        store=qa_store,
        tokenizer=tokenizer,
    )
    qa_store_stats_before = (qa_store.size(), _store_bytes(qa_store.cache_dir))
    qa_cache = CanonicalCache(model, contract, qa_store)
    with (output / "qa_requests.jsonl").open("x", encoding="utf-8") as handle:
        qa_row_values = _qa_rows(
            train_records,
            confirmation_ids,
            model=model,
            tokenizer=tokenizer,
            cache=qa_cache,
        )
        for row in qa_row_values:
            handle.write(
                json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            )
    qa_analyzed = _analyze_qa(
        qa_row_values, seed=BOOTSTRAP_SEED, resamples=args.bootstrap_iterations
    )
    qa = {
        "paired_records": len(qa_row_values),
        "errors": 0,
        "delta_direction": "C - A",
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "resamples": args.bootstrap_iterations,
            "method": "percentile",
        },
        "analysis": qa_analyzed["analysis"],
        "decision": qa_analyzed["decision"],
        "role_label": QA_ROLE_LABEL,
        "prediction_equal": sum(int(row["prediction_equal"]) for row in qa_row_values),
        "first_token_equal": sum(int(row["first_token_equal"]) for row in qa_row_values),
        "store": {
            "path": str(qa_store.cache_dir),
            "files_after_prebuild": qa_store_stats_before[0],
            "bytes_after_prebuild": qa_store_stats_before[1],
            "files_final": qa_store.size(),
            "bytes_final": _store_bytes(qa_store.cache_dir),
            "prebuild_wall_seconds": _agg(
                [row["prebuild_wall_seconds"] for row in prebuild_rows]
            ),
        },
    }

    result = {
        "schema_version": SCHEMA_VERSION,
        "manifest_sha256": _sha_file(output / "manifest.json"),
        "validation_1_combined_serving": serving,
        "validation_2_qa_confirmation": qa,
        "wall_seconds": time.perf_counter() - started,
        "finished_utc": datetime.now(UTC).isoformat(),
    }
    _write_json(output / "result.json", result)
    (output / "summary.md").write_text(
        _summary_markdown({"manifest": manifest, "serving": serving, "qa": qa}),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "validation_1": {
                    "equivalence_all_equal": equivalence_all_equal,
                    "ttft_wall_ms_median": {
                        arm: arms_summary[arm]["wall_ms"]["median"] for arm in SERVING_ARMS
                    },
                    "a_over_mixed": ratios["A_over_mixed_hit"]["point"],
                    "a_over_mixed_ci": [
                        ratios["A_over_mixed_hit"]["ci_low"],
                        ratios["A_over_mixed_hit"]["ci_high"],
                    ],
                    "combined_win": combined_win["combined_win"],
                    "schedule_wall_s_median": {
                        arm: schedule_walls[arm]["wall_seconds_median"] for arm in SERVING_ARMS
                    },
                },
                "validation_2": {
                    "paired_records": qa["paired_records"],
                    "decision": qa["decision"],
                },
                "wall_seconds": result["wall_seconds"],
            }
        )
    )
    return result


if __name__ == "__main__":
    main()
