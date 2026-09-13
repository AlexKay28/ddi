"""V61 matched task-level QA: segmented-FA2 arm versus the retained SDPA reference.

Paired per-record comparison on the frozen V62 development arm (100 ids): the
SDPA arm-A per-record outputs of results/v62_dev_eval_20260910T073635Z/base are
REUSED (sha-pinned, never rerun) and only the segmented-FlashAttention-2 arm is
executed on the same ids, with the same doc_lengths/positions/masks (the batch
is re-encoded and verified against the retained doc_lengths/query_length, so
tokens and positions are identical) and the same decoding budgets read from the
retained manifest. Quality is scored with the same ddi.metrics functions the
base run used. Analysis is paired per-record deltas (segFA2 - SDPA) for
EM/F1/presence with a paired bootstrap 95% CI.

Development-arm diagnostic, single seed, NOT a confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ddi import inference  # noqa: E402
from ddi.data import load as load_dataset  # noqa: E402
from ddi.masks import build_mask, to_additive  # noqa: E402
from ddi.metrics import answer_presence, exact_match, token_f1  # noqa: E402
from ddi.prompt import build_query_text  # noqa: E402
from ddi.segmented_fa2 import (  # noqa: E402
    clear_segmented_state,
    patch_model_for_segmented_fa2,
    set_segmented_state,
)
from scripts.run_canonical_matrix import _checkpoint_identity  # noqa: E402

SCHEMA_VERSION = "v61_matched_qa_v1"
METRICS = ("em", "f1", "presence")
DECISION_METRICS = ("em", "f1")
MARGIN = -0.02
FLAT_THRESHOLD = 0.02
DECISION_RULE = "non-inferior iff CI LB >= -0.02 for BOTH EM and F1"
ROLE_LABEL = "development-arm diagnostic, single seed, NOT a confirmation"
REFERENCE_ARM = "A"
VARIANT = "ddi_qf"
DEFAULT_BASE_DIR = ROOT / "results/v62_dev_eval_20260910T073635Z/base"
DEFAULT_SOURCE_MANIFEST = ROOT / "results/v62_dev_eval_20260910T073635Z/source_manifest.json"
DEFAULT_SPLIT_MANIFEST = ROOT / "experiments/v62_split_manifest.json"
DEFAULT_CHECKPOINT = ROOT / "results/checkpoints/Qwen__Qwen2.5-7B-Instruct__ddi_qf__8L_restart"


class StructuralMismatch(RuntimeError):
    """The re-encoded batch disagrees with the retained request geometry."""


def sha_file(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _write_jsonl_row(handle: Any, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    handle.flush()


def _digest_ids(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def parse_budgets(base_manifest: dict[str, Any]) -> dict[str, int]:
    budgets = base_manifest["budgets"]
    return {
        "natural_max_new_tokens": int(budgets["natural_max_new_tokens"]),
        "fixed_new_tokens": int(budgets["fixed_new_tokens"]),
    }


def extract_development_ids(split_manifest: dict[str, Any]) -> list[str]:
    development = split_manifest["splits"]["development"]
    ids = list(development["record_ids"])
    if len(ids) != int(development["count"]):
        raise ValueError(
            f"split manifest development count {development['count']} != {len(ids)} ids"
        )
    return ids


def load_base_records(base_dir: Path) -> list[dict[str, Any]]:
    with (Path(base_dir) / "requests.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def extract_sdpa_reference(base_records: list[dict[str, Any]], *, arm: str = REFERENCE_ARM) -> dict:
    reference: dict[str, dict[str, Any]] = {}
    for record in base_records:
        arms = record.get("arms", {})
        if arm not in arms:
            raise ValueError(f"record {record.get('record_id')} has no arm {arm}")
        entry = arms[arm]
        reference[record["record_id"]] = {
            "prediction": entry["prediction"],
            "quality": entry["quality"],
            "natural_generated_ids": entry["natural_generated_ids"],
            "gold_answers": list(record["gold_answers"]),
        }
    return reference


def check_paired_ids(base_records: list[dict[str, Any]], development_ids: list[str]) -> None:
    base_ids = [record["record_id"] for record in base_records]
    if base_ids != list(development_ids):
        only_base = sorted(set(base_ids) - set(development_ids))
        only_dev = sorted(set(development_ids) - set(base_ids))
        raise ValueError(
            "retained base records do not match the frozen development ids: "
            f"missing={only_dev[:5]} unexpected={only_base[:5]} "
            f"counts base={len(base_ids)} development={len(development_ids)}"
        )


def verify_structural_match(
    retained: dict[str, Any], doc_lengths: list[int], query_len: int
) -> None:
    if list(retained["doc_lengths"]) != list(doc_lengths) or int(retained["query_length"]) != int(
        query_len
    ):
        raise StructuralMismatch(
            f"record {retained['record_id']}: re-encoded geometry "
            f"(doc_lengths={list(doc_lengths)}, query_len={query_len}) differs from the retained "
            f"request (doc_lengths={retained['doc_lengths']}, "
            f"query_length={retained['query_length']}); tokens/positions are not identical"
        )


def paired_bootstrap(deltas: list[float], *, seed: int, resamples: int) -> dict[str, Any]:
    values = np.asarray(deltas, dtype=float)
    if values.size == 0:
        raise ValueError("paired deltas must be non-empty")
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples)
    for i in range(resamples):
        indices = rng.integers(0, values.size, values.size)
        draws[i] = values[indices].mean()
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {
        "mean": float(values.mean()),
        "ci95": [float(lower), float(upper)],
        "n": int(values.size),
        "seed": int(seed),
        "resamples": int(resamples),
        "method": "percentile",
    }


def count_deltas(deltas: list[float], *, threshold: float = FLAT_THRESHOLD) -> dict[str, int]:
    return {
        "improved": sum(delta > threshold for delta in deltas),
        "flat": sum(abs(delta) <= threshold for delta in deltas),
        "degraded": sum(delta < -threshold for delta in deltas),
    }


def evaluate_decision(analysis: dict[str, dict[str, Any]]) -> dict[str, Any]:
    lower_bounds = {metric: analysis[metric]["ci95"][0] for metric in DECISION_METRICS}
    non_inferior = all(lower_bound >= MARGIN for lower_bound in lower_bounds.values())
    return {
        "rule": DECISION_RULE,
        "margin": MARGIN,
        "ci_lower_bounds": lower_bounds,
        "non_inferior": non_inferior,
    }


def build_paired_rows(
    reference: dict[str, dict[str, Any]], candidate_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    paired = []
    for row in candidate_rows:
        record_id = row["record_id"]
        if record_id not in reference:
            raise ValueError(f"segmented row {record_id} has no retained SDPA reference")
        paired.append(
            {
                "record_id": record_id,
                "reference": reference[record_id]["quality"],
                "candidate": row["quality"],
            }
        )
    return paired


def _analyze(
    reference: dict[str, dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    seed: int,
    resamples: int,
    flat_threshold: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    paired = build_paired_rows(reference, candidate_rows)
    analysis: dict[str, dict[str, Any]] = {}
    for metric in METRICS:
        deltas = [
            entry["candidate"][metric] - entry["reference"][metric] for entry in paired
        ]
        bootstrap = paired_bootstrap(deltas, seed=seed, resamples=resamples)
        counts = count_deltas(deltas, threshold=flat_threshold)
        analysis[metric] = {
            **bootstrap,
            **counts,
            "reference_mean": statistics.mean(entry["reference"][metric] for entry in paired),
            "candidate_mean": statistics.mean(entry["candidate"][metric] for entry in paired),
        }
    decision = evaluate_decision(analysis)
    return analysis, decision


@torch.no_grad()
def generate_segmented_ids(
    model: Any,
    tokenizer: Any,
    batch: inference.BatchEncoding,
    variant: str,
    budget: int,
    *,
    stop_on_eos: bool,
) -> dict[str, Any]:
    """Greedy generation identical to scripts.validate_matched_qa.generate_ids, except the
    prefill runs with segmented-FA2 state armed; decode is the patched layer's SDPA fallback
    over the assembled cache, driven by the same additive step masks and positions."""
    if budget < 1:
        raise ValueError("generation budget must be positive")
    device = model.device
    prefix_ids = batch.input_ids.to(device)
    positions = batch.position_ids.to(device)
    prefix_len = int(prefix_ids.shape[-1])
    allow = build_mask(batch.doc_lengths, batch.query_len, budget, variant)
    additive = to_additive(allow, dtype=model.dtype).to(device)
    prefill_mask = additive[:prefix_len, :prefix_len].clone()[None, None]
    set_segmented_state(model, batch.doc_lengths, batch.query_len, variant=variant)
    try:
        out = model(
            input_ids=prefix_ids,
            attention_mask=prefill_mask,
            position_ids=positions,
            use_cache=True,
        )
    finally:
        clear_segmented_state(model)
    past = out.past_key_values
    next_token = out.logits[:, -1].argmax(-1, keepdim=True)
    generated: list[int] = []
    stop_reason = "length"
    for i in range(budget):
        token = int(next_token.item())
        generated.append(token)
        if stop_on_eos and tokenizer.eos_token_id is not None and token == tokenizer.eos_token_id:
            stop_reason = "eos"
            break
        if i == budget - 1:
            break
        kv_len = prefix_len + i + 1
        step_mask = additive[prefix_len + i, :kv_len].clone().reshape(1, 1, 1, kv_len)
        step_position = positions[:, -1:] + i + 1
        out = model(
            input_ids=next_token,
            attention_mask=step_mask,
            position_ids=step_position,
            cache_position=torch.tensor([prefix_len + i], device=device),
            past_key_values=past,
            use_cache=True,
        )
        past = out.past_key_values
        next_token = out.logits[:, -1].argmax(-1, keepdim=True)
    return {"generated_ids": generated, "stop_reason": stop_reason}


def _score(prediction: str, gold: list[str]) -> dict[str, float]:
    return {
        "em": exact_match(prediction, gold),
        "f1": token_f1(prediction, gold),
        "presence": answer_presence(prediction, gold),
    }


def _source_hashes() -> dict[str, str]:
    paths = sorted((ROOT / "ddi").rglob("*.py"))
    paths += [Path(__file__).resolve(), ROOT / "scripts" / "run_canonical_matrix.py"]
    return {str(path.relative_to(ROOT)): sha_file(path) for path in paths}


def _git_text(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--bootstrap-seed", type=int, default=47)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--flat-threshold", type=float, default=FLAT_THRESHOLD)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"--output must not exist: {args.output}")
    if args.resamples < 1:
        parser.error("--resamples must be >= 1")
    return args


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    started = time.perf_counter()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")

    base_manifest = read_json(args.base_dir / "manifest.json")
    base_result = read_json(args.base_dir / "result.json")
    base_records = load_base_records(args.base_dir)
    source_manifest = read_json(args.source_manifest)
    split_manifest = read_json(args.split_manifest)
    development_ids = extract_development_ids(split_manifest)
    source_config = source_manifest["config"]
    dataset, seed = source_config["dataset"], int(source_config["seed"])

    check_paired_ids(base_records, development_ids)
    if list(source_config["record_ids"]) != development_ids:
        raise ValueError("source manifest record_ids do not match the frozen development ids")
    if list(base_manifest["source_manifest"]["record_ids"]) != development_ids:
        raise ValueError("base manifest source record_ids do not match the frozen development ids")

    reused_files = {
        "base/manifest.json": sha_file(args.base_dir / "manifest.json"),
        "base/requests.jsonl": sha_file(args.base_dir / "requests.jsonl"),
        "base/result.json": sha_file(args.base_dir / "result.json"),
        "source_manifest.json": sha_file(args.source_manifest),
        "split_manifest.json": sha_file(args.split_manifest),
    }
    base_manifest_sha = reused_files["base/manifest.json"]
    if base_result.get("manifest_sha256") != base_manifest_sha:
        raise ValueError(
            "base/manifest.json sha256 does not match base/result.json manifest_sha256; "
            "the retained base artifact is inconsistent"
        )
    checkpoint_mode, checkpoint_digest = _checkpoint_identity(args.checkpoint)
    if base_manifest["checkpoint_identity"]["sha256"] != checkpoint_digest:
        raise ValueError(
            "checkpoint identity does not match the base manifest's checkpoint_identity; "
            "the SDPA reference was produced by a different checkpoint"
        )

    reference = extract_sdpa_reference(base_records)
    budgets = parse_budgets(base_manifest)
    reference_mean = {
        metric: statistics.mean(
            record["arms"][REFERENCE_ARM]["quality"][metric] for record in base_records
        )
        for metric in METRICS
    }

    args.output.mkdir(parents=True)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "runner": "scripts/run_v61_matched_qa.py",
        "created_utc": datetime.now(UTC).isoformat(),
        "data_role": "development",
        "role_label": ROLE_LABEL,
        "decision_rule": {
            "verbatim": DECISION_RULE,
            "margin": MARGIN,
            "metrics": list(DECISION_METRICS),
            "declared": "before the GPU arm ran",
        },
        "bootstrap": {
            "seed": args.bootstrap_seed,
            "resamples": args.resamples,
            "method": "percentile",
            "delta_direction": "segmented_fa2 - sdpa",
        },
        "flat_threshold": args.flat_threshold,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_identity": {"mode": checkpoint_mode, "sha256": checkpoint_digest},
        "base": {
            "dir": str(args.base_dir.resolve()),
            "reference_arm": REFERENCE_ARM,
            "reference_backend": base_manifest["arms"][REFERENCE_ARM]["backend"],
            "reference_execution": base_manifest["arms"][REFERENCE_ARM]["execution"],
            "reused": True,
            "rerun": False,
            "files_sha256": reused_files,
            "budgets": budgets,
            "reference_quality_mean": reference_mean,
        },
        "split_manifest": {
            "path": str(args.split_manifest.resolve()),
            "sha256": reused_files["split_manifest.json"],
            "development_sha256": split_manifest["splits"]["development"]["sha256"],
            "development_count": len(development_ids),
        },
        "record_ids": development_ids,
        "candidate_arm": {
            "backend": "segmented_fa2",
            "execution": (
                "patch_model_for_segmented_fa2 + set_segmented_state per request on the same "
                "doc_lengths/positions/masks as the retained SDPA run; decode is the patched "
                "layer's SDPA fallback"
            ),
            "variant": VARIANT,
            "dtype": args.dtype,
            "layout": "sequential",
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "device": args.device,
            "gpu_name": torch.cuda.get_device_name() if device.type == "cuda" else None,
        },
        "provenance": {
            "source_files": _source_hashes(),
            "git_base_sha256": _git_text("rev-parse", "HEAD"),
            "git_diff_head_sha256": hashlib.sha256(
                subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)
            ).hexdigest(),
        },
    }
    write_json(args.output / "manifest.json", manifest)

    model, tokenizer = inference.load(str(args.checkpoint), dtype=args.dtype, backend="eager")
    model = patch_model_for_segmented_fa2(model.to(device).eval())

    records = {record["id"]: record for record in load_dataset(dataset, "train", seed)}
    missing = [record_id for record_id in development_ids if record_id not in records]
    if missing:
        raise ValueError(f"development ids absent from the {dataset} train stream: {missing[:5]}")

    candidate_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    with (args.output / "segmented_requests.jsonl").open("x", encoding="utf-8") as handle, (
        args.output / "errors.jsonl"
    ).open("x", encoding="utf-8") as error_handle:
        for index, retained in enumerate(base_records):
            record_id = retained["record_id"]
            try:
                record = records[record_id]
                query_text = build_query_text(tokenizer, record)
                batch = inference.encode(tokenizer, list(record["info_contexts"]), query_text)
                verify_structural_match(retained, batch.doc_lengths, batch.query_len)
                gold = list(retained["gold_answers"])
                natural = generate_segmented_ids(
                    model,
                    tokenizer,
                    batch,
                    VARIANT,
                    budgets["natural_max_new_tokens"],
                    stop_on_eos=True,
                )
                prediction = tokenizer.decode(
                    natural["generated_ids"], skip_special_tokens=True
                ).strip()
                fixed = generate_segmented_ids(
                    model,
                    tokenizer,
                    batch,
                    VARIANT,
                    budgets["fixed_new_tokens"],
                    stop_on_eos=False,
                )
                row = {
                    "record_id": record_id,
                    "request_id": f"{dataset}:{seed}:{record_id}",
                    "request_index": index,
                    "variant": VARIANT,
                    "doc_lengths": batch.doc_lengths,
                    "query_length": batch.query_len,
                    "gold_answers": gold,
                    "prediction": prediction,
                    "natural_generated_ids": natural["generated_ids"],
                    "stop_reason": natural["stop_reason"],
                    "fixed_generated_ids": fixed["generated_ids"],
                    "fixed_ids_sha256": _digest_ids(fixed["generated_ids"]),
                    "quality": _score(prediction, gold),
                }
            except StructuralMismatch:
                raise
            except Exception as error:  # noqa: BLE001 - per-record isolation is the protocol
                entry = {"record_id": record_id, "error": repr(error)}
                errors.append(entry)
                _write_jsonl_row(error_handle, entry)
                continue
            _write_jsonl_row(handle, row)
            candidate_rows.append(row)

    run_ids = [row["record_id"] for row in candidate_rows]
    if sorted(run_ids + [error["record_id"] for error in errors]) != sorted(development_ids):
        raise ValueError(
            "paired record ids mismatch the frozen development ids after the run; aborting"
        )

    analysis, decision = _analyze(
        reference,
        candidate_rows,
        seed=args.bootstrap_seed,
        resamples=args.resamples,
        flat_threshold=args.flat_threshold,
    )
    first_token_matches = sum(
        int(
            row["natural_generated_ids"][0]
            == reference[row["record_id"]]["natural_generated_ids"][0]
        )
        for row in candidate_rows
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "data_role": "development",
        "role_label": ROLE_LABEL,
        "manifest_sha256": sha_file(args.output / "manifest.json"),
        "base": manifest["base"],
        "paired": {
            "records": len(candidate_rows),
            "errors": len(errors),
            "analysis_scope": "paired records only; errored records are excluded and listed",
        },
        "delta_direction": "segmented_fa2 - sdpa",
        "bootstrap": manifest["bootstrap"],
        "reference_quality_mean": {
            metric: statistics.mean(
                reference[row["record_id"]]["quality"][metric] for row in candidate_rows
            )
            for metric in METRICS
        },
        "candidate_quality_mean": {
            metric: statistics.mean(row["quality"][metric] for row in candidate_rows)
            for metric in METRICS
        },
        "analysis": analysis,
        "decision": decision,
        "first_token_matches_vs_reference": {
            "matches": first_token_matches,
            "requests": len(candidate_rows),
        },
        "errors": errors,
        "wall_seconds": time.perf_counter() - started,
        "finished_utc": datetime.now(UTC).isoformat(),
    }
    write_json(args.output / "result.json", result)
    verdict = {
        "non_inferior": decision["non_inferior"],
        "decision_rule": DECISION_RULE,
        "role_label": ROLE_LABEL,
        "paired_records": len(candidate_rows),
        "errors": len(errors),
    }
    write_json(args.output / "verdict.json", verdict)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "paired": len(candidate_rows),
                "errors": len(errors),
                "decision": decision,
                "wall_seconds": result["wall_seconds"],
            }
        )
    )
    return result


if __name__ == "__main__":
    main()
