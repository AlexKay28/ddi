"""V57 matched natural-QA and fixed-work timing validation on identical requests."""

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
from ddi.data import load as load_dataset
from ddi.inference import load as load_model
from ddi.masks import build_mask, to_additive
from ddi.metrics import answer_presence, exact_match, token_f1
from ddi.prompt import build_query_text
from ddi.run import _sparsity


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def bootstrap_ratio(full: list[float], candidate: list[float], *, seed: int = 57) -> dict:
    if len(full) != len(candidate) or not full:
        raise ValueError("paired non-empty latency samples are required")
    ratios = np.asarray(full, dtype=float) / np.asarray(candidate, dtype=float)
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


@torch.no_grad()
def generate_ids(model, tokenizer, batch, variant: str, budget: int, *, stop_on_eos: bool):
    if budget < 1:
        raise ValueError("generation budget must be positive")
    device = model.device
    prefix_ids = batch.input_ids.to(device)
    positions = batch.position_ids.to(device)
    prefix_len = int(prefix_ids.shape[-1])
    if device.type == "cuda":
        torch.cuda.synchronize()
    request_started = time.perf_counter()
    allow = build_mask(batch.doc_lengths, batch.query_len, budget, variant)
    additive = to_additive(allow, dtype=model.dtype).to(device)
    prefill_mask = additive[:prefix_len, :prefix_len].clone()[None, None]
    if device.type == "cuda":
        torch.cuda.synchronize()
    prefill_started = time.perf_counter()
    out = model(
        input_ids=prefix_ids,
        attention_mask=prefill_mask,
        position_ids=positions,
        use_cache=True,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - prefill_started) * 1000
    past = out.past_key_values
    next_token = out.logits[:, -1].argmax(-1, keepdim=True)
    generated = []
    forwards = 0
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
        forwards += 1
    if device.type == "cuda":
        torch.cuda.synchronize()
    request_ms = (time.perf_counter() - request_started) * 1000
    return {
        "generated_ids": generated,
        "generated_count": len(generated),
        "decode_forwards": forwards,
        "stop_reason": stop_reason,
        "prefill_ms": prefill_ms,
        "request_ms": request_ms,
    }


def _hash_files(path: Path) -> dict[str, str]:
    hashes = {}
    for file_path in sorted(path.iterdir()):
        if not file_path.is_file():
            continue
        hasher = hashlib.sha256()
        with file_path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                hasher.update(chunk)
        hashes[file_path.name] = hasher.hexdigest()
    return hashes


def _audit(rows: list[dict], variants: list[str]) -> dict:
    grouped = {}
    for row in rows:
        key = row["request_id"]
        grouped.setdefault(key, []).append(row)
    for request_id, request_rows in grouped.items():
        if sorted(row["variant"] for row in request_rows) != sorted(variants):
            raise AssertionError(f"{request_id}: incomplete or duplicate arms")
        if len({row["input_ids_sha256"] for row in request_rows}) != 1:
            raise AssertionError(f"{request_id}: input IDs differ across arms")
        if any(len(row["fixed"]["raw_request_ms"]) < 5 for row in request_rows):
            raise AssertionError(f"{request_id}: fewer than five timed repeats")
    return {"artifact_integrity_pass": True, "request_count": len(grouped), "row_count": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", default="hotpot_qa")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=1)
    parser.add_argument("--variants", nargs="+", default=["full", "ddi_qf"])
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--natural-max-new-tokens", type=int, default=16)
    parser.add_argument("--fixed-new-tokens", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 5:
        raise ValueError("V57 requires at least five timed repeats")
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    manifest_payload = json.loads(args.heldout_manifest.read_text())
    selected_ids = manifest_payload["ids"][: args.max_records]
    records = {record["id"]: record for record in load_dataset(args.dataset, "val", args.seed)}
    selected = [records[record_id] for record_id in selected_ids]
    source_paths = [*sorted((root / "ddi").rglob("*.py")), Path(__file__).resolve()]
    source_hashes = {
        str(path.relative_to(root)): digest(path.read_bytes()) for path in source_paths
    }
    source_dir = args.output / "source"
    for path in source_paths:
        relative = path.relative_to(root)
        target = source_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=root)
    (args.output / "working.patch").write_bytes(diff)
    config = {
        "dataset": args.dataset,
        "seed": args.seed,
        "record_ids": selected_ids,
        "variants": args.variants,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "natural_max_new_tokens": args.natural_max_new_tokens,
        "fixed_new_tokens": args.fixed_new_tokens,
        "timing_scope": "pretokenized model request: mask construction, prefill, and fixed decode",
        "material_speedup_ci_lower_bound": 1.05,
    }
    manifest = {
        "config": config,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_files": _hash_files(args.checkpoint),
        "heldout_manifest_sha256": digest(args.heldout_manifest.read_bytes()),
        "source_hashes": source_hashes,
        "source_sha256": digest(json_bytes(source_hashes)),
        "diff_sha256": digest(diff),
        "base_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root
        ).decode().strip(),
        "clean_revision": False,
        "admissible_paper_evidence": False,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "started_unix": time.time(),
    }
    (args.output / "manifest.json").write_bytes(json_bytes(manifest))
    model, tokenizer = load_model(str(args.checkpoint))
    model = model.to("cuda").eval()
    rows = []
    with (args.output / "requests.jsonl").open("x") as output:
        for record in selected:
            query_text = build_query_text(tokenizer, record)
            batch = inference.encode(tokenizer, list(record["info_contexts"]), query_text)
            input_ids = batch.input_ids.tolist()[0]
            request_id = f"{args.dataset}:{args.seed}:{record['id']}"
            for variant in args.variants:
                natural = generate_ids(
                    model,
                    tokenizer,
                    batch,
                    variant,
                    args.natural_max_new_tokens,
                    stop_on_eos=True,
                )
                prediction = tokenizer.decode(
                    natural["generated_ids"], skip_special_tokens=True
                ).strip()
                for _ in range(args.warmups):
                    generate_ids(
                        model, tokenizer, batch, variant, args.fixed_new_tokens, stop_on_eos=False
                    )
                repeats = [
                    generate_ids(
                        model, tokenizer, batch, variant, args.fixed_new_tokens, stop_on_eos=False
                    )
                    for _ in range(args.repeats)
                ]
                allow = build_mask(batch.doc_lengths, batch.query_len, 0, variant)
                row = {
                    "request_id": request_id,
                    "dataset": args.dataset,
                    "seed": args.seed,
                    "record_id": record["id"],
                    "variant": variant,
                    "input_ids": input_ids,
                    "input_ids_sha256": digest(np.asarray(input_ids, dtype=np.int64).tobytes()),
                    "doc_lengths": batch.doc_lengths,
                    "query_length": batch.query_len,
                    "actual_sparsity": _sparsity(allow),
                    "gold_answers": list(record["possible_answers"]),
                    "natural": {**natural, "prediction": prediction},
                    "quality": {
                        "em": exact_match(prediction, record["possible_answers"]),
                        "f1": token_f1(prediction, record["possible_answers"]),
                        "presence": answer_presence(prediction, record["possible_answers"]),
                    },
                    "fixed": {
                        "generated_ids": repeats[0]["generated_ids"],
                        "generated_count": repeats[0]["generated_count"],
                        "decode_forwards": repeats[0]["decode_forwards"],
                        "raw_prefill_ms": [repeat["prefill_ms"] for repeat in repeats],
                        "raw_request_ms": [repeat["request_ms"] for repeat in repeats],
                        "median_prefill_ms": statistics.median(r["prefill_ms"] for r in repeats),
                        "median_request_ms": statistics.median(r["request_ms"] for r in repeats),
                    },
                    "cache": {"supported": False, "hits": None},
                    "unavailable_scopes": {
                        "attention_kernel": "Flex model parity gate failed",
                        "attention_module": "Flex model parity gate failed",
                    },
                }
                output.write(json.dumps(row) + "\n")
                output.flush()
                rows.append(row)
    audit = _audit(rows, args.variants)
    by_variant = {
        variant: [row for row in rows if row["variant"] == variant]
        for variant in args.variants
    }
    summary = {
        "quality": {
            variant: {
                metric: statistics.mean(row["quality"][metric] for row in variant_rows)
                for metric in ("em", "f1", "presence")
            }
            for variant, variant_rows in by_variant.items()
        },
        "median_prefill_ms": {
            variant: statistics.mean(row["fixed"]["median_prefill_ms"] for row in variant_rows)
            for variant, variant_rows in by_variant.items()
        },
        "median_request_ms": {
            variant: statistics.mean(row["fixed"]["median_request_ms"] for row in variant_rows)
            for variant, variant_rows in by_variant.items()
        },
        "admissible_paper_evidence": False,
        "scope": "bounded dirty-source local diagnostic",
    }
    if "full" in by_variant and "ddi_qf" in by_variant:
        summary["ddi_qf_speedup"] = {
            "prefill": bootstrap_ratio(
                [row["fixed"]["median_prefill_ms"] for row in by_variant["full"]],
                [row["fixed"]["median_prefill_ms"] for row in by_variant["ddi_qf"]],
            ),
            "request": bootstrap_ratio(
                [row["fixed"]["median_request_ms"] for row in by_variant["full"]],
                [row["fixed"]["median_request_ms"] for row in by_variant["ddi_qf"]],
            ),
        }
    (args.output / "summary.json").write_bytes(json_bytes(summary))
    (args.output / "audit.json").write_bytes(json_bytes(audit))
    print(json.dumps({"summary": summary, "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
