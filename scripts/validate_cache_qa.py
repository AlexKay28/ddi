"""V58 matched natural-QA validation for fresh and persisted document-KV reuse."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import transformers

from ddi import inference
from ddi.cache import DocCache
from ddi.data import load as load_dataset
from ddi.inference import load as load_model
from ddi.metrics import answer_presence, exact_match, token_f1
from ddi.prompt import build_query_text
from scripts.validate_cache_workload import compare, digest, json_bytes, request


def bootstrap_difference(
    fresh: list[float], cached: list[float], *, seed: int = 58
) -> dict:
    if len(fresh) != len(cached) or not fresh:
        raise ValueError("paired non-empty samples are required")
    differences = np.asarray(cached, dtype=float) - np.asarray(fresh, dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.empty(10_000)
    for i in range(len(draws)):
        indices = rng.integers(0, len(differences), len(differences))
        draws[i] = differences[indices].mean()
    return {
        "per_request": differences.tolist(),
        "mean": float(differences.mean()),
        "ci95_paired_bootstrap": np.quantile(draws, [0.025, 0.975]).tolist(),
    }


def bootstrap_ratio(
    fresh: list[float], cached: list[float], *, seed: int = 58
) -> dict:
    if len(fresh) != len(cached) or not fresh:
        raise ValueError("paired non-empty samples are required")
    ratios = np.asarray(fresh, dtype=float) / np.asarray(cached, dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.empty(10_000)
    for i in range(len(draws)):
        indices = rng.integers(0, len(ratios), len(ratios))
        draws[i] = ratios[indices].mean()
    return {
        "per_request": ratios.tolist(),
        "mean": float(ratios.mean()),
        "ci95_paired_bootstrap": np.quantile(draws, [0.025, 0.975]).tolist(),
    }


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def quality(prediction: str, answers: list[str]) -> dict:
    return {
        "em": exact_match(prediction, answers),
        "f1": token_f1(prediction, answers),
        "presence": answer_presence(prediction, answers),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--dataset", default="hotpot_qa")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-records", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--natural-max-new-tokens", type=int, default=16)
    parser.add_argument("--fixed-new-tokens", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 5:
        raise ValueError("V58 requires at least five timed repeats")
    args.output.mkdir(parents=True, exist_ok=False)

    root = Path(__file__).resolve().parents[1]
    manifest_payload = json.loads(args.heldout_manifest.read_text())
    selected_ids = manifest_payload["ids"][: args.max_records]
    records = {record["id"]: record for record in load_dataset(args.dataset, "val", args.seed)}
    selected = [records[record_id] for record_id in selected_ids]
    source_paths = [
        *sorted((root / "ddi").rglob("*.py")),
        root / "scripts" / "validate_cache_workload.py",
        Path(__file__).resolve(),
    ]
    source_hashes = {
        str(path.relative_to(root)): digest(path.read_bytes()) for path in source_paths
    }
    source_dir = args.output / "source"
    for path in source_paths:
        target = source_dir / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=root)
    (args.output / "working.patch").write_bytes(diff)
    config = {
        "dataset": args.dataset,
        "seed": args.seed,
        "record_ids": selected_ids,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "natural_max_new_tokens": args.natural_max_new_tokens,
        "fixed_new_tokens": args.fixed_new_tokens,
        "backend": "sdpa",
        "dtype": "bf16",
        "layout": "restart",
        "timing_scope": "mask construction, persisted cache IO, prefill, and fixed decode",
    }
    manifest = {
        "config": config,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_files": {
            path.name: file_sha256(path)
            for path in sorted(args.checkpoint.iterdir())
            if path.is_file()
        },
        "heldout_manifest_sha256": digest(args.heldout_manifest.read_bytes()),
        "source_hashes": source_hashes,
        "source_sha256": digest(json_bytes(source_hashes)),
        "diff_sha256": digest(diff),
        "base_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root)
        .decode()
        .strip(),
        "clean_revision": False,
        "admissible_paper_evidence": False,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "started_unix": time.time(),
    }
    identity = digest(json_bytes(manifest))
    manifest["cache_identity"] = identity
    (args.output / "manifest.json").write_bytes(json_bytes(manifest))

    model, tokenizer = load_model(str(args.checkpoint))
    model = model.to("cuda").eval()
    rows = []
    logits_dir = args.output / "logits"
    logits_dir.mkdir()
    with (args.output / "requests.jsonl").open("x") as output:
        for record in selected:
            docs = list(record["info_contexts"])
            query_text = build_query_text(tokenizer, record)
            batch = inference.encode(tokenizer, docs, query_text, layout="restart")
            split = np.cumsum([0, *batch.doc_lengths])
            all_ids = batch.input_ids.tolist()[0]
            doc_ids = [all_ids[split[i] : split[i + 1]] for i in range(len(docs))]
            query_ids = all_ids[split[-1] :]
            indices = list(range(len(doc_ids)))
            cache = DocCache(args.output / "cache" / str(record["id"]))

            fresh, _ = request(
                model,
                doc_ids,
                query_ids,
                indices,
                "ddi_recompute",
                None,
                identity,
                budget=args.natural_max_new_tokens,
                eos=tokenizer.eos_token_id,
            )
            cold, _ = request(
                model,
                doc_ids,
                query_ids,
                indices,
                "ddi_cache",
                cache,
                identity,
                budget=args.natural_max_new_tokens,
                eos=tokenizer.eos_token_id,
            )
            cached, _ = request(
                model,
                doc_ids,
                query_ids,
                indices,
                "ddi_cache",
                cache,
                identity,
                budget=args.natural_max_new_tokens,
                eos=tokenizer.eos_token_id,
            )
            fresh_fixed, fresh_logits = request(
                model,
                doc_ids,
                query_ids,
                indices,
                "ddi_recompute",
                None,
                identity,
                budget=args.fixed_new_tokens,
                diagnostic=True,
            )
            _, cached_logits = request(
                model,
                doc_ids,
                query_ids,
                indices,
                "ddi_cache",
                cache,
                identity,
                budget=args.fixed_new_tokens,
                diagnostic=True,
                forced=fresh_fixed["generated_ids"],
            )
            for _ in range(args.warmups):
                request(
                    model,
                    doc_ids,
                    query_ids,
                    indices,
                    "ddi_recompute",
                    None,
                    identity,
                    budget=args.fixed_new_tokens,
                )
                request(
                    model,
                    doc_ids,
                    query_ids,
                    indices,
                    "ddi_cache",
                    cache,
                    identity,
                    budget=args.fixed_new_tokens,
                )
            fresh_times = []
            cached_times = []
            for _ in range(args.repeats):
                fresh_time, _ = request(
                    model,
                    doc_ids,
                    query_ids,
                    indices,
                    "ddi_recompute",
                    None,
                    identity,
                    budget=args.fixed_new_tokens,
                )
                cached_time, _ = request(
                    model,
                    doc_ids,
                    query_ids,
                    indices,
                    "ddi_cache",
                    cache,
                    identity,
                    budget=args.fixed_new_tokens,
                )
                fresh_times.append(fresh_time["elapsed_ms"])
                cached_times.append(cached_time["elapsed_ms"])

            answers = list(record["possible_answers"])
            fresh_prediction = tokenizer.decode(
                fresh["generated_ids"], skip_special_tokens=True
            ).strip()
            cached_prediction = tokenizer.decode(
                cached["generated_ids"], skip_special_tokens=True
            ).strip()
            comparison = compare(fresh_logits, cached_logits)
            logits_path = logits_dir / f"{record['id']}.pt"
            torch.save(
                {"fresh": fresh_logits, "cached_forced": cached_logits}, logits_path
            )
            row = {
                "request_id": f"{args.dataset}:{args.seed}:{record['id']}",
                "record_id": record["id"],
                "input_ids_sha256": digest(np.asarray(all_ids, dtype=np.int64).tobytes()),
                "doc_lengths": batch.doc_lengths,
                "query_length": batch.query_len,
                "gold_answers": answers,
                "fresh": {
                    "prediction": fresh_prediction,
                    "generated_ids": fresh["generated_ids"],
                    "quality": quality(fresh_prediction, answers),
                },
                "cached": {
                    "prediction": cached_prediction,
                    "generated_ids": cached["generated_ids"],
                    "quality": quality(cached_prediction, answers),
                },
                "cold_hit_state": cold["hit_state"],
                "all_hit_state": cached["hit_state"],
                "cache_files": {
                    path.name: file_sha256(path)
                    for path in sorted(cache.cache_dir.glob("*.pt"))
                },
                "comparison": comparison,
                "logits_file": str(logits_path.relative_to(args.output)),
                "logits_sha256": file_sha256(logits_path),
                "greedy_equal": fresh["generated_ids"] == cached["generated_ids"],
                "timing": {
                    "fresh_ms": fresh_times,
                    "cached_ms": cached_times,
                    "fresh_median_ms": statistics.median(fresh_times),
                    "cached_median_ms": statistics.median(cached_times),
                },
            }
            output.write(json.dumps(row) + "\n")
            output.flush()
            rows.append(row)
            print(f"{record['id']}: complete", flush=True)

    quality_summary = {}
    for metric in ("em", "f1", "presence"):
        fresh_values = [row["fresh"]["quality"][metric] for row in rows]
        cached_values = [row["cached"]["quality"][metric] for row in rows]
        quality_summary[metric] = {
            "fresh_mean": statistics.mean(fresh_values),
            "cached_mean": statistics.mean(cached_values),
            "cached_minus_fresh": bootstrap_difference(fresh_values, cached_values),
        }
    summary = {
        "quality": quality_summary,
        "exact": {
            "finite": all(row["comparison"]["finite"] for row in rows),
            "numerical_failures": sum(
                not row["comparison"]["numerical_pass"] for row in rows
            ),
            "greedy_failures": sum(not row["greedy_equal"] for row in rows),
            "max_abs": max(row["comparison"]["max_abs"] for row in rows),
        },
        "hit_state_failures": sum(
            any(row["cold_hit_state"])
            or not all(row["all_hit_state"])
            for row in rows
        ),
        "timing_speedup": bootstrap_ratio(
            [row["timing"]["fresh_median_ms"] for row in rows],
            [row["timing"]["cached_median_ms"] for row in rows],
        ),
        "record_count": len(rows),
        "wall_seconds": time.time() - manifest["started_unix"],
        "admissible_paper_evidence": False,
    }
    (args.output / "summary.json").write_bytes(json_bytes(summary))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
