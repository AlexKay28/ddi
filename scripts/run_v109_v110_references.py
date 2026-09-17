#!/usr/bin/env python3
"""V109/V110 (VIBEQUEUE-109/110): original-model reference and D-arm decomposition.

Arms (preregistered in experiments/v109_preregistration.md and
experiments/v110_preregistration.md):

- base-std    (V109): hub Qwen2.5-7B-Instruct, plain stream (docs + chat query,
              no markers/BOS), sequential positions, full-causal mask.
- k5-std      (V110): trained k5 checkpoint, same plain-stream construction.
- k5-iso-seq  (V110): trained k5 checkpoint, plain stream, ddi_qf isolated mask,
              sequential positions.
- k5-d-replica(V110): the pinned V70 D construction via the V70 harness
              (full-causal mask swapped onto the restart/link layout).

All arms: frozen confirmation300 records, greedy 64, EOS stop, bf16/eager,
metrics from ddi.metrics, paired bootstrap (10k, seed 47) against the pinned
V70 C and B arms. Diagnostic; no gates.

The generation loop mirrors ddi.inference.generate (prefill mask + per-step
decode masks + sequential position continuation + EOS break) and additionally
returns generated token ids.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from ddi import inference  # noqa: E402
from ddi.data import load as load_dataset  # noqa: E402
from ddi.metrics import answer_presence, exact_match, token_f1  # noqa: E402
from ddi.prompt import build_query_text  # noqa: E402
from v70_harness import (  # noqa: E402
    BOOTSTRAP_SEED,
    DocumentPayload,  # noqa: F401  (re-exported for callers)
    _quality,
    generate_monolithic_ids,
    install_fixed_link_tokens,  # noqa: F401
    paired_bootstrap,
)
from v70_run import (  # noqa: E402
    MODEL_ID,
    _generation_request,
    encode_real_record,
    load_frozen_eval_ids,
    load_frozen_eval_records,
    load_v70_checkpoint,
)

CONFIRMATION_MANIFEST = ROOT / "experiments/v70_confirmation300.json"
PINNED_V70 = ROOT / "results/v70_confirmation300_k5.json"
BOOTSTRAP_RESAMPLES = 10_000
SMOKE_RECORDS = 2


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dir_aggregate_sha256(path: Path) -> str:
    entries = sorted(p for p in path.rglob("*") if p.is_file())
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(str(entry.relative_to(path)).encode())
        digest.update(_file_sha256(entry).encode())
    return digest.hexdigest()


def _runtime_identity() -> dict:
    return {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dtype": "bfloat16",
        "backend": "eager",
        "batch": 1,
        "decoding": "greedy",
        "max_new_tokens": 64,
    }


def generate_ids_with_variant(
    model, tokenizer, batch, variant: str, max_new_tokens: int
) -> list[int]:
    """ddi.inference.generate, returning ids (see module docstring)."""
    prefill_ids = batch.input_ids.to(model.device)
    positions = batch.position_ids.to(model.device)
    prefix_len = int(prefill_ids.shape[-1])

    expected_prefix = sum(batch.doc_lengths) + batch.query_len
    if prefix_len != expected_prefix:
        raise ValueError(
            f"BatchEncoding is inconsistent: input_ids has {prefix_len} tokens but "
            f"doc_lengths={batch.doc_lengths} + query_len={batch.query_len} = "
            f"{expected_prefix}"
        )

    from ddi.masks import build_mask, to_additive

    allow_full = build_mask(
        batch.doc_lengths,
        query_len=batch.query_len,
        answer_len=max_new_tokens,
        variant=variant,
    )
    additive_full = to_additive(allow_full, dtype=model.dtype).to(model.device)
    additive = additive_full[:prefix_len, :prefix_len].clone().unsqueeze(0).unsqueeze(0)

    decode_positions = torch.arange(
        int(positions[0, -1].item()) + 1,
        int(positions[0, -1].item()) + 1 + max_new_tokens,
        dtype=torch.long,
        device=model.device,
    ).unsqueeze(0)

    eos_id = tokenizer.eos_token_id
    generated: list[int] = []

    with torch.no_grad():
        out = model(
            input_ids=prefill_ids,
            attention_mask=additive,
            position_ids=positions,
            use_cache=True,
        )
        past = out.past_key_values
        next_token = out.logits[0, -1].argmax(dim=-1, keepdim=True).unsqueeze(0)

        for i in range(max_new_tokens):
            tok_id = int(next_token.item())
            generated.append(tok_id)
            if eos_id is not None and tok_id == eos_id:
                break
            kv_len = prefix_len + i + 1
            step_mask = (
                additive_full[prefix_len + i, :kv_len].clone().reshape(1, 1, 1, kv_len)
            )
            step = model(
                input_ids=next_token,
                attention_mask=step_mask,
                position_ids=decode_positions[:, i : i + 1].clone(),
                past_key_values=past,
                use_cache=True,
            )
            past = step.past_key_values
            next_token = step.logits[0, -1].argmax(dim=-1, keepdim=True).unsqueeze(0)

    return generated


def _plain_batch(tokenizer, record: dict):
    """Plain stream: docs + chat-templated query, sequential layout, no markers."""
    docs = list(record["info_contexts"])
    query_text = build_query_text(tokenizer, record)
    return inference.encode(tokenizer, docs, query_text, layout="sequential")


def _score(tokenizer, prediction: str, gold: list[str]) -> dict:
    return {
        "prediction": prediction,
        "em": exact_match(prediction, gold),
        "f1": token_f1(prediction, gold),
        "presence": answer_presence(prediction, gold),
    }


def _ids_sha256(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def run_plain_arm(model, tokenizer, records, variant: str, log) -> list[dict]:
    rows = []
    for index, record in enumerate(records):
        batch = _plain_batch(tokenizer, record)
        ids = generate_ids_with_variant(model, tokenizer, batch, variant, 64)
        prediction = tokenizer.decode(ids, skip_special_tokens=True).strip()
        scores = _score(tokenizer, prediction, record["possible_answers"])
        rows.append(
            {
                "record_id": record["id"],
                "generated_ids_sha256": _ids_sha256(ids),
                "n_generated": len(ids),
                **scores,
            }
        )
        if (index + 1) % 25 == 0 or index + 1 == len(records):
            em = statistics.mean(r["em"] for r in rows)
            log(f"  [{_utc()}] {index + 1}/{len(records)} running EM {em:.4f}")
    return rows


def run_k5_d_replica(model, tokenizer, state, records, log) -> list[dict]:
    rows = []
    for index, record in enumerate(records):
        request = _generation_request(encode_real_record(tokenizer, record))
        ids = generate_monolithic_ids(
            model,
            state.contract,
            request,
            k=state.contract.k if hasattr(state.contract, "k") else 5,
            max_new_tokens=64,
            eos_token_id=tokenizer.eos_token_id,
            full_causal=True,
        )
        prediction = tokenizer.decode(ids, skip_special_tokens=True).strip()
        scores = _score(tokenizer, prediction, record["possible_answers"])
        rows.append(
            {
                "record_id": record["id"],
                "generated_ids_sha256": _ids_sha256(list(ids)),
                "n_generated": len(ids),
                **scores,
            }
        )
        if (index + 1) % 25 == 0 or index + 1 == len(records):
            em = statistics.mean(r["em"] for r in rows)
            log(f"  [{_utc()}] {index + 1}/{len(records)} running EM {em:.4f}")
    return rows


def _paired_vs_pinned(arm_rows: list[dict], metric: str) -> dict:
    pinned = json.loads(PINNED_V70.read_text())
    pinned_rows = {row["record_id"]: row for row in pinned["rows"]}
    deltas = []
    for row in arm_rows:
        pinned_c = pinned_rows[row["record_id"]]["arms"]["C"][metric]
        deltas.append(row[metric] - pinned_c)
    boot = paired_bootstrap(deltas, seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES)
    return {
        "metric": metric,
        "arm_mean": statistics.mean(r[metric] for r in arm_rows),
        "pinned_c_mean": statistics.mean(
            pinned_rows[r["record_id"]]["arms"]["C"][metric] for r in arm_rows
        ),
        "c_minus_arm_mean": statistics.mean(deltas),
        "ci95": boot["ci95"],
        "n": len(deltas),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arms",
        nargs="+",
        required=True,
        choices=["base-std", "k5-std", "k5-iso-seq", "k5-d-replica"],
    )
    parser.add_argument("--smoke", action="store_true", help="2 train-split records only")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="results parent (default results/); a timestamped dir is created",
    )
    args = parser.parse_args()

    output_parent = args.output or (ROOT / "results")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    tag = "smoke" if args.smoke else f"v109_v110_{stamp}"
    output_dir = output_parent / tag
    output_dir.mkdir(parents=True, exist_ok=False)

    log_path = output_dir / "log.txt"

    def log(message: str) -> None:
        line = f"[{_utc()}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    partial_path = output_dir / "partial.json"

    def persist_partial() -> None:
        partial_path.write_text(
            json.dumps({"arms": arms, "pins": pins}, indent=2) + "\n"
        )

    log(f"start tag={tag} arms={args.arms} smoke={args.smoke}")
    log(f"runtime: {_runtime_identity()}")

    # --- frozen records ------------------------------------------------------
    split_info = load_frozen_eval_ids("confirmation300")
    records = load_frozen_eval_records(split_info)
    if args.smoke:
        train_ids = json.loads(
            (ROOT / "experiments/v62_split_manifest.json").read_text()
        )["splits"]["train"]["record_ids"][:SMOKE_RECORDS]
        stream = {r["id"]: r for r in load_dataset("hotpot_qa", "train", 0)}
        records = [stream[rid] for rid in train_ids]
    log(f"records: {len(records)} ({'train smoke' if args.smoke else 'confirmation300 frozen'})")

    needs_base = "base-std" in args.arms
    needs_k5 = any(a in args.arms for a in ("k5-std", "k5-iso-seq", "k5-d-replica"))

    arms: dict[str, dict] = {}
    pins: dict[str, dict] = {}

    # --- base model ----------------------------------------------------------
    if needs_base:
        log("loading base hub model (bf16, eager)...")
        model, tokenizer = inference.load(MODEL_ID, dtype="bf16", backend="eager")
        model = model.to("cuda").eval()
        from huggingface_hub import snapshot_download

        snapshot = Path(snapshot_download(MODEL_ID, local_files_only=True))
        pins["base_model"] = {
            "model_id": MODEL_ID,
            "resolved_path": str(snapshot),
            "config_sha256": _file_sha256(snapshot / "config.json"),
            "weights_sha256": _dir_aggregate_sha256(snapshot),
        }
        log(f"base pinned: {pins['base_model']['weights_sha256'][:16]}…")
        for arm, variant in (("base-std", "full"),):
            log(f"arm {arm} (variant={variant})")
            started = time.perf_counter()
            arms[arm] = {
                "rows": run_plain_arm(model, tokenizer, records, variant, log),
                "variant": variant,
                "layout": "sequential",
            }
            log(f"arm {arm} done in {(time.perf_counter() - started) / 60:.1f} min")
            persist_partial()
        del model
        torch.cuda.empty_cache()

    # --- trained k5 checkpoint ----------------------------------------------
    if needs_k5:
        checkpoint = ROOT / "results/v70_arms_valid_seed47/checkpoint_k5"
        log("loading trained k5 checkpoint (bf16, eager)...")
        model, tokenizer, state, metadata = load_v70_checkpoint(checkpoint)
        model = model.to("cuda").eval()
        pins["k5_checkpoint"] = {
            "path": str(checkpoint.relative_to(ROOT)),
            "aggregate_sha256": _dir_aggregate_sha256(checkpoint),
            "metadata_arm": metadata.get("arm"),
            "metadata_k": metadata.get("k"),
        }
        log(f"k5 pinned: {pins['k5_checkpoint']['aggregate_sha256'][:16]}…")

        if "k5-std" in args.arms:
            log("arm k5-std (plain stream, sequential, full-causal)")
            started = time.perf_counter()
            arms["k5-std"] = {
                "rows": run_plain_arm(model, tokenizer, records, "full", log),
                "variant": "full",
                "layout": "sequential",
                "note": "plain stream, no markers, no BOS",
            }
            log(f"arm k5-std done in {(time.perf_counter() - started) / 60:.1f} min")
            persist_partial()
        if "k5-iso-seq" in args.arms:
            log("arm k5-iso-seq (plain stream, sequential, ddi_qf)")
            started = time.perf_counter()
            arms["k5-iso-seq"] = {
                "rows": run_plain_arm(model, tokenizer, records, "ddi_qf", log),
                "variant": "ddi_qf",
                "layout": "sequential",
                "note": "isolated mask without restart/link machinery",
            }
            log(f"arm k5-iso-seq done in {(time.perf_counter() - started) / 60:.1f} min")
            persist_partial()
        if "k5-d-replica" in args.arms:
            log("arm k5-d-replica (V70 harness, full_causal=True on restart/link layout)")
            started = time.perf_counter()
            arms["k5-d-replica"] = {
                "rows": run_k5_d_replica(model, tokenizer, state, records, log),
                "variant": "full_causal_on_contract_layout",
                "layout": "restart-documents-request-tail-v1",
                "note": "pinned D construction; replication anchor",
            }
            log(f"arm k5-d-replica done in {(time.perf_counter() - started) / 60:.1f} min")
            persist_partial()
        del model
        torch.cuda.empty_cache()

    # --- paired analysis vs pinned C/B --------------------------------------
    analysis = {}
    if not args.smoke:
        for arm, payload in arms.items():
            analysis[arm] = {
                "em": _paired_vs_pinned(payload["rows"], "em"),
                "f1": _paired_vs_pinned(payload["rows"], "f1"),
            }

    result = {
        "schema": "v109-v110-references-v1",
        "created_utc": _utc(),
        "smoke": args.smoke,
        "arms": arms,
        "pins": pins,
        "runtime": _runtime_identity(),
        "records": {
            "manifest": str(CONFIRMATION_MANIFEST.relative_to(ROOT)),
            "count": len(records),
            "order": "confirmation manifest order" if not args.smoke else "train smoke",
        },
        "analysis": analysis,
        "preregs": ["experiments/v109_preregistration.md", "experiments/v110_preregistration.md"],
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    summary = {
        "schema": "v109-v110-summary-v1",
        "created_utc": _utc(),
        "smoke": args.smoke,
        "arms": {
            arm: {
                "em": statistics.mean(r["em"] for r in payload["rows"]),
                "f1": statistics.mean(r["f1"] for r in payload["rows"]),
                "presence": statistics.mean(r["presence"] for r in payload["rows"]),
                "n": len(payload["rows"]),
            }
            for arm, payload in arms.items()
        },
        "paired_vs_pinned_c": analysis,
        "pins": pins,
        "runtime": _runtime_identity(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log(f"done: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
