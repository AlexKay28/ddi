"""Executable V60 development-QA A/B/C/D runner (CPU or a single GPU).

Per HotpotQA record, run four arms over identical prompts and record per-arm
natural predictions, quality, and timing: A (original monolithic SDPA DDI-QF),
B (CanonicalCache fresh encode, nothing persisted), C (CanonicalCache backed by
a DocCache store, cold and all-hit scenarios), and D (full-causal reference).
Quality reuses the V57 matched-QA metric functions.

This is development selection only ("data_role": "development"): the source
records are development-observed V58 seed-0 requests. It supports no
confirmation and no cross-backend performance claim (eager-vs-SDPA timing is
inadmissible per scripts.validate_canonical_matrix.validate_comparison).
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from ddi import inference
from ddi.cache import DocCache
from ddi.canonical_cache import CanonicalCache
from ddi.data import load as load_dataset
from ddi.metrics import answer_presence, exact_match, token_f1
from ddi.prompt import build_query_text
from scripts.run_canonical_matrix import (
    _arm_identity,
    _check_models,
    _checkpoint_identity,
    _contract,
    _git_bytes,
    _git_text,
    _load_models,
    _prepare_output,
    _reset_peak,
    _sha_file,
    _source_hashes,
    _timed,
)
from scripts.validate_canonical_matrix import ARMS, SCHEMA_VERSION
from scripts.validate_matched_qa import digest, generate_ids

ROOT = Path(__file__).resolve().parents[1]

LoadRecords = Any


def _write_json(path: Path, value: Any) -> None:
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _load_records(dataset: str, seed: int) -> dict[str, Any]:
    return {record["id"]: record for record in load_dataset(dataset, "val", seed)}


def _split_docs_query(batch: Any) -> tuple[list[torch.Tensor], torch.Tensor]:
    ids = batch.input_ids[0]
    docs, offset = [], 0
    for length in batch.doc_lengths:
        docs.append(ids[offset : offset + length][None])
        offset += length
    return docs, ids[offset:][None]


def _monolithic_arm(
    model: Any, tokenizer: Any, batch: Any, variant: str, *, natural_budget: int, fixed_budget: int
) -> dict[str, Any]:
    natural = generate_ids(model, tokenizer, batch, variant, natural_budget, stop_on_eos=True)
    fixed = generate_ids(model, tokenizer, batch, variant, fixed_budget, stop_on_eos=False)
    ttft = [
        generate_ids(model, tokenizer, batch, variant, 1, stop_on_eos=False)["prefill_ms"]
        / 1000
        for _ in range(3)
    ]
    prediction = tokenizer.decode(natural["generated_ids"], skip_special_tokens=True).strip()
    return {
        "prediction": prediction,
        "natural_generated_ids": natural["generated_ids"],
        "stop_reason": natural["stop_reason"],
        "fixed_generated_ids": fixed["generated_ids"],
        "fixed_ids_sha256": digest(
            json.dumps(fixed["generated_ids"], separators=(",", ":")).encode()
        ),
        "ttft_seconds": ttft,
        "ttft_median_seconds": statistics.median(ttft),
    }


def _canonical_arm(
    cache: CanonicalCache,
    tokenizer: Any,
    docs: list[torch.Tensor],
    query_ids: torch.Tensor,
    device: torch.device,
    *,
    natural_budget: int,
    fixed_budget: int,
) -> dict[str, Any]:
    natural = cache.request(
        docs, query_ids, max_new_tokens=natural_budget, eos_token_id=tokenizer.eos_token_id
    )
    fixed = cache.request(docs, query_ids, max_new_tokens=fixed_budget, eos_token_id=None)
    _reset_peak(device)
    times, _ = _timed(lambda: cache.request(docs, query_ids), 3, device)
    prediction = tokenizer.decode(natural, skip_special_tokens=True).strip()
    return {
        "prediction": prediction,
        "natural_generated_ids": natural,
        "stop_reason": "eos" if tokenizer.eos_token_id in natural else "length",
        "fixed_generated_ids": fixed,
        "fixed_ids_sha256": digest(json.dumps(fixed, separators=(",", ":")).encode()),
        "ttft_seconds": times,
        "ttft_median_seconds": statistics.median(times),
    }


def _quality(prediction: str, gold: list[str]) -> dict[str, float]:
    return {
        "em": exact_match(prediction, gold),
        "f1": token_f1(prediction, gold),
        "presence": answer_presence(prediction, gold),
    }


def _arm_record(
    arm: str, scenario: str, data: dict[str, Any], gold: list[str], *, backend: str
) -> dict[str, Any]:
    return {
        "arm": arm,
        "scenario": scenario,
        "backend": backend,
        "execution": ARMS[arm]["execution"],
        "cache": ARMS[arm]["cache"],
        "prediction": data["prediction"],
        "natural_generated_ids": data["natural_generated_ids"],
        "stop_reason": data["stop_reason"],
        "fixed_generated_ids": data["fixed_generated_ids"],
        "fixed_ids_sha256": data["fixed_ids_sha256"],
        "quality": _quality(data["prediction"], gold),
        "ttft_seconds": data["ttft_seconds"],
        "ttft_median_seconds": data["ttft_median_seconds"],
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=ROOT / "results/v58_cache_qa_100_20260908T122202Z/manifest.json",
    )
    parser.add_argument("--record-count", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), required=True)
    parser.add_argument("--natural-max-new-tokens", type=int, default=48)
    parser.add_argument("--fixed-new-tokens", type=int, default=16)
    parser.add_argument(
        "--canonical-on-sdpa",
        action="store_true",
        help=(
            "run the canonical contract on the SDPA model instead of eager, isolating "
            "execution-contract effects from backend numerics (development diagnostic)"
        ),
    )
    args = parser.parse_args(argv)
    if args.record_count < 1:
        parser.error("--record-count must be >= 1")
    if args.natural_max_new_tokens < 1 or args.fixed_new_tokens < 1:
        parser.error("generation budgets must be >= 1")
    return args


def _default_tokenizer(checkpoint: str) -> Any:
    from transformers import AutoTokenizer

    from ddi.model import _resolve_offline_snapshot

    return AutoTokenizer.from_pretrained(
        _resolve_offline_snapshot(checkpoint), local_files_only=True
    )


def main(
    argv: list[str] | None = None,
    *,
    load_models: Any = None,
    record_loader: LoadRecords | None = None,
    tokenizer: Any = None,
) -> dict[str, Any]:
    args = _parse_args(argv)
    output = _prepare_output(args.output)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")
    source_manifest = json.loads(args.source_manifest.read_text())
    source_config = source_manifest["config"]
    dataset, seed = source_config["dataset"], source_config["seed"]
    record_ids = source_config["record_ids"][: args.record_count]
    loader = _load_records if record_loader is None else record_loader
    records = loader(dataset, seed)
    selected = [records[record_id] for record_id in record_ids]

    model_a, model_bc = (_load_models if load_models is None else load_models)(
        args.checkpoint, args.device
    )
    _check_models(model_a, model_bc)
    if tokenizer is None:
        tokenizer = _default_tokenizer(str(args.checkpoint))
    canonical_model = model_bc
    canonical_backend = "eager"
    if args.canonical_on_sdpa:
        canonical_model = model_a
        canonical_backend = "sdpa"
    contract = _contract(canonical_model, args.checkpoint)
    if args.canonical_on_sdpa:
        contract = replace(contract, backend="sdpa")
    dtype_str = str(model_bc.dtype)
    identities = {arm: _arm_identity(arm, dtype_str) for arm in ARMS}

    _mode, checkpoint_digest = _checkpoint_identity(args.checkpoint)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "data_role": "development",
        "runner": "scripts/run_canonical_qa_dev.py",
        "created_utc": datetime.now(UTC).isoformat(),
        "data_role_note": (
            "development selection only: source records are development-observed V58 "
            "seed-0 requests and must never serve as untouched confirmation"
        ),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_identity": {"mode": _mode, "sha256": checkpoint_digest},
        "source_manifest": {
            "path": str(args.source_manifest.resolve()),
            "sha256": _sha_file(args.source_manifest),
            "dataset": dataset,
            "seed": seed,
            "record_ids": record_ids,
        },
        "runtime": {
            "python": __import__("platform").python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "device": args.device,
            "gpu_name": torch.cuda.get_device_name() if device.type == "cuda" else None,
        },
        "provenance": {
            "source_files": _source_hashes(),
            "git_base_sha256": _git_text("rev-parse", "HEAD"),
            "git_diff_head_sha256": __import__("hashlib").sha256(
                _git_bytes("diff", "HEAD")
            ).hexdigest(),
        },
        "budgets": {
            "natural_max_new_tokens": args.natural_max_new_tokens,
            "fixed_new_tokens": args.fixed_new_tokens,
        },
        "canonical_on_sdpa": args.canonical_on_sdpa,
        "arms": {
            arm: {**ARMS[arm], "dtype": dtype_str, "arm_identity": identities[arm]}
            for arm in ARMS
        },
    }
    _write_json(output / "manifest.json", manifest)

    cells = []
    with (output / "requests.jsonl").open("x", encoding="utf-8") as handle:
        for index, record in enumerate(selected):
            query_text = build_query_text(tokenizer, record)
            batch = inference.encode(tokenizer, list(record["info_contexts"]), query_text)
            docs, query_ids = _split_docs_query(batch)
            gold = list(record["possible_answers"])

            fresh = CanonicalCache(canonical_model, contract)
            cold = CanonicalCache(
                canonical_model, contract, DocCache(output / "canonical_cache" / "cold")
            )
            hit_store = DocCache(output / "canonical_cache" / "all_hit")
            cached_hit = CanonicalCache(canonical_model, contract, hit_store)
            cached_hit.documents([doc.cpu() for doc in docs])
            hit_bytes = sum(path.stat().st_size for path in hit_store.cache_dir.glob("*.pt"))

            arms = {
                "A": _arm_record(
                    "A",
                    "standard",
                    _monolithic_arm(
                        model_a,
                        tokenizer,
                        batch,
                        "ddi_qf",
                        natural_budget=args.natural_max_new_tokens,
                        fixed_budget=args.fixed_new_tokens,
                    ),
                    gold,
                    backend="sdpa",
                ),
                "D": _arm_record(
                    "D",
                    "standard",
                    _monolithic_arm(
                        model_a,
                        tokenizer,
                        batch,
                        "full",
                        natural_budget=args.natural_max_new_tokens,
                        fixed_budget=args.fixed_new_tokens,
                    ),
                    gold,
                    backend="sdpa",
                ),
                "B": _arm_record(
                    "B",
                    "standard",
                    _canonical_arm(
                        fresh,
                        tokenizer,
                        docs,
                        query_ids,
                        device,
                        natural_budget=args.natural_max_new_tokens,
                        fixed_budget=args.fixed_new_tokens,
                    ),
                    gold,
                    backend=canonical_backend,
                ),
                "C": {
                    "cold": _arm_record(
                        "C",
                        "cold",
                        _canonical_arm(
                            cold,
                            tokenizer,
                            docs,
                            query_ids,
                            device,
                            natural_budget=args.natural_max_new_tokens,
                            fixed_budget=args.fixed_new_tokens,
                        ),
                        gold,
                        backend=canonical_backend,
                    ),
                    "all_hit": _arm_record(
                        "C",
                        "all_hit",
                        _canonical_arm(
                            cached_hit,
                            tokenizer,
                            docs,
                            query_ids,
                            device,
                            natural_budget=args.natural_max_new_tokens,
                            fixed_budget=args.fixed_new_tokens,
                        ),
                        gold,
                        backend=canonical_backend,
                    ),
                },
            }
            cell = {
                "request_index": index,
                "record_id": record["id"],
                "request_id": f"{dataset}:{seed}:{record['id']}",
                "gold_answers": gold,
                "doc_lengths": batch.doc_lengths,
                "query_length": batch.query_len,
                "arms": arms,
                "b_vs_c_equal": {
                    "cold_natural_prediction": arms["B"]["prediction"]
                    == arms["C"]["cold"]["prediction"],
                    "all_hit_natural_prediction": arms["B"]["prediction"]
                    == arms["C"]["all_hit"]["prediction"],
                    "cold_fixed_ids": arms["B"]["fixed_ids_sha256"]
                    == arms["C"]["cold"]["fixed_ids_sha256"],
                    "all_hit_fixed_ids": arms["B"]["fixed_ids_sha256"]
                    == arms["C"]["all_hit"]["fixed_ids_sha256"],
                },
                "hit_cache_bytes": hit_bytes,
            }
            handle.write(
                json.dumps(cell, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            )
            handle.flush()
            cells.append(cell)

    summary = _summarize(cells, _sha_file(output / "manifest.json"))
    _write_json(output / "result.json", summary)
    print(
        json.dumps(
            {
                "output": str(output),
                "requests": len(cells),
                "b_vs_c_equal_all": summary["aggregate"]["b_vs_c_equal_all"],
            }
        )
    )
    return summary


def _summarize(cells: list[dict[str, Any]], manifest_sha256: str) -> dict[str, Any]:
    keys = ("A", "B", "C_cold", "C_all_hit", "D")

    def record(cell: dict[str, Any], key: str) -> dict[str, Any]:
        if key.startswith("C_"):
            return cell["arms"]["C"][key.split("_", 1)[1]]
        return cell["arms"][key]

    quality = {
        key: {
            metric: statistics.mean(record(cell, key)["quality"][metric] for cell in cells)
            for metric in ("em", "f1", "presence")
        }
        for key in keys
    }
    ttft = {
        key: statistics.median(record(cell, key)["ttft_median_seconds"] for cell in cells)
        for key in keys
    }
    b_vs_c_all = all(all(cell["b_vs_c_equal"].values()) for cell in cells)
    anchor_agreement = {}
    for key in keys[1:]:
        matches = sum(
            int(
                record(cell, key)["natural_generated_ids"][0]
                == cell["arms"]["A"]["natural_generated_ids"][0]
            )
            for cell in cells
        )
        anchor_agreement[key] = {"first_token_matches": matches, "requests": len(cells)}
    return {
        "schema_version": SCHEMA_VERSION,
        "data_role": "development",
        "manifest_sha256": manifest_sha256,
        "aggregate": {
            "quality_mean": quality,
            "ttft_median_seconds": ttft,
            "b_vs_c_equal_all": b_vs_c_all,
            "first_token_agreement_vs_A": anchor_agreement,
        },
        "timing_caveat": (
            "TTFT medians are development observations; cross-backend A/B performance "
            "comparisons are inadmissible per scripts.validate_canonical_matrix.validate_comparison"
        ),
    }


if __name__ == "__main__":
    main()
