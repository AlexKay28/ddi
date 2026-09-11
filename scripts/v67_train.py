"""V67 training script: perturbation-consistent recipe with link-token arms.

Implements the revised H2 recipe (bf16 re-encode noise, cache dropout, document
order randomization) for 3 equal-budget arms (k0, k1, k5).

This is a NEW file; no existing file is modified. It imports from ddi.finetune
for the training loop internals but applies the perturbation recipe externally
by wrapping the record stream.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/v67_train.py --arm k0 --steps 500
    CUDA_VISIBLE_DEVICES=0 python scripts/v67_train.py --arm k1 --steps 500
    CUDA_VISIBLE_DEVICES=0 python scripts/v67_train.py --arm k5 --steps 500
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ddi import inference
from ddi.data import load as load_dataset
from ddi.finetune import (
    _answer_kl,
    _assert_finite_weights,
    _build_additive,
    _encode_record,
    _flush_optimizer_step,
    _freeze_trunk,
    _linear_warmup_lr,
    _maybe_grad_checkpoint,
    _prefilter_records,
    _teacher_first_token_ce,
)
from ddi.masks import build_mask, to_additive
from ddi.positions import build_positions
from ddi.provenance import capture
from ddi.prompt import encode_record


def _log(msg: str) -> None:
    print(f"[v67.train] {msg}", file=sys.stderr, flush=True)


def _load_manifest_ids(path: str) -> list[str]:
    """Load record IDs from a split manifest file."""
    data = json.loads(Path(path).read_text())
    if "splits" in data:
        return data["splits"]["train"]["record_ids"]
    if "development" in data:
        return data["development"]["record_ids"]
    raise ValueError(f"Unknown manifest format: {path}")


def _filter_records_by_ids(records: list[dict], ids: list[str]) -> list[dict]:
    """Filter records to only include those with IDs in the given list, preserving order."""
    id_set = set(ids)
    return [r for r in records if r["id"] in id_set]


def _bf16_reencode_perturbation(record: dict, tokenizer, noise_rate: float, rng: random.Random) -> dict:
    """With probability noise_rate, re-encode each document through bf16 round-trip.

    This simulates the independent re-encoding perturbation identified in V68:
    tokenizer.decode(tokenizer.encode(text)) introduces tiny numerical differences
    in the KV cache that can flip greedy near-tie decisions.
    """
    if rng.random() >= noise_rate:
        return record
    docs = list(record.get("info_contexts") or [])
    if not docs:
        return record
    perturbed_docs = []
    for doc in docs:
        if rng.random() < noise_rate:
            tokens = tokenizer.encode(doc, add_special_tokens=False)
            reencoded = tokenizer.decode(tokens, skip_special_tokens=True)
            perturbed_docs.append(reencoded)
        else:
            perturbed_docs.append(doc)
    new_record = dict(record)
    new_record["info_contexts"] = perturbed_docs
    return new_record


def _cache_dropout_perturbation(record: dict, dropout_rate: float, rng: random.Random) -> dict:
    """With probability dropout_rate, drop one random document.

    This teaches the model invariance to cache misses — a document that would
    normally be cached might not be present, and the model should still produce
    the same answer.
    """
    docs = list(record.get("info_contexts") or [])
    if len(docs) <= 2 or rng.random() >= dropout_rate:
        return record
    drop_idx = rng.randint(0, len(docs) - 1)
    new_docs = docs[:drop_idx] + docs[drop_idx + 1:]
    new_record = dict(record)
    new_record["info_contexts"] = new_docs
    return new_record


def _shuffle_document_order(record: dict, rng: random.Random) -> dict:
    """Shuffle document order to teach order-invariance."""
    docs = list(record.get("info_contexts") or [])
    if len(docs) <= 1:
        return record
    indices = list(range(len(docs)))
    rng.shuffle(indices)
    new_docs = [docs[i] for i in indices]
    new_record = dict(record)
    new_record["info_contexts"] = new_docs
    return new_record


def _apply_perturbation_recipe(
    record: dict,
    tokenizer,
    step: int,
    seed: int,
    bf16_noise_rate: float,
    cache_dropout_rate: float,
    shuffle_order: bool,
) -> dict:
    """Apply the full perturbation recipe to a record."""
    rng = random.Random(seed + step * 7919 + hash(record["id"]) % 1000)
    rec = _bf16_reencode_perturbation(record, tokenizer, bf16_noise_rate, rng)
    rec = _cache_dropout_perturbation(rec, cache_dropout_rate, rng)
    if shuffle_order:
        rec = _shuffle_document_order(rec, rng)
    return rec


def train_arm(
    arm: str,
    steps: int,
    seed: int,
    output_dir: Path,
    log_path: Path,
    bf16_noise_rate: float = 0.15,
    cache_dropout_rate: float = 0.10,
    shuffle_order: bool = True,
) -> dict:
    """Train one arm of the V67 experiment."""

    # GPU check
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for training")
    if torch.cuda.memory_allocated() > 0 or torch.cuda.memory_reserved() > 0:
        _log("WARNING: GPU has allocated memory at start — may not be idle")

    _log(f"GPU: {torch.cuda.get_device_name()}")
    free, total = torch.cuda.mem_get_info()
    _log(f"GPU memory: {free / 1e9:.1f} GB free / {total / 1e9:.1f} GB total")

    # Load split manifest
    manifest_ids = _load_manifest_ids("experiments/v62_split_manifest.json")
    _log(f"Loading {len(manifest_ids)} train ids from V62 manifest")

    # Load records
    all_records = list(load_dataset("hotpot_qa", "train", seed))
    records_by_id = {r["id"]: r for r in all_records}
    train_records = [records_by_id[rid] for rid in manifest_ids if rid in records_by_id]
    _log(f"Loaded {len(train_records)} train records")

    # Prefilter (same as ddi.finetune)
    train_records = _prefilter_records(train_records, min_docs=2)
    _log(f"After prefilter (min_docs=2): {len(train_records)} records")

    # Load model
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    base_checkpoint = str(ROOT / "results/checkpoints/Qwen__Qwen2.5-7B-Instruct__ddi_qf__8L_restart")

    _log(f"Loading teacher (frozen) + student from {base_checkpoint}")
    teacher, tokenizer = inference.load(model_name, checkpoint_dir=base_checkpoint)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student, _ = inference.load(model_name, checkpoint_dir=base_checkpoint)
    student.train()

    trainable_layers = 8
    _freeze_trunk(student, trainable_layers)
    _maybe_grad_checkpoint(student)

    device = torch.device("cuda")
    teacher = teacher.to(device)
    student = student.to(device)
    dtype = student.dtype

    # Optimizer
    try:
        import bitsandbytes as bnb
        trainable_params = [p for p in student.parameters() if p.requires_grad]
        optimizer = bnb.optim.AdamW8bit(trainable_params, lr=1e-5, weight_decay=0.0)
        _log("Using AdamW8bit (bitsandbytes)")
    except (ImportError, ModuleNotFoundError):
        trainable_params = [p for p in student.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-5, weight_decay=0.0)
        _log("Using torch.optim.AdamW (bitsandbytes not available)")

    # Training loop
    lr = 1e-5
    warmup_steps = 20
    grad_accum_steps = 4
    kl_temperature = 1.0
    teacher_first_token_weight = 1.0
    full_anchor_weight = 1.0
    blind_anchor_weight = 1.0
    max_grad_norm = 1.0
    max_skipped_steps = 10
    layout = "restart"

    global_step = 0
    skipped_steps = 0
    accum_since_step = 0
    optimizer.zero_grad(set_to_none=True)
    nan_count = 0
    loss_curve = []
    start_time = time.time()

    # Per-record state for the perturbation recipe
    epoch_rng = random.Random(seed)

    # Cycle through records
    record_queue = list(train_records)
    epoch_rng.shuffle(record_queue)

    stop_early = False
    rec_idx = 0
    while global_step < steps and not stop_early:
        if rec_idx >= len(record_queue):
            record_queue = list(train_records)
            epoch_rng.shuffle(record_queue)
            rec_idx = 0

        record = record_queue[rec_idx]
        rec_idx += 1

        # Apply perturbation recipe
        perturbed = _apply_perturbation_recipe(
            record, tokenizer, global_step, seed,
            bf16_noise_rate, cache_dropout_rate, shuffle_order,
        )

        if not perturbed.get("possible_answers"):
            continue

        try:
            full_ids, doc_lengths, query_len, answer_len = _encode_record(tokenizer, perturbed)
        except Exception as exc:
            _log(f"skip record id={perturbed.get('id', '?')}: encode failed: {exc!r}")
            continue

        if answer_len == 0:
            continue

        full_ids = full_ids.to(device)
        student_variant = "ddi_qf" if doc_lengths else "full"

        # Build masks
        additive_teacher = _build_additive(doc_lengths, query_len, answer_len, "full", dtype, device)
        additive_student = _build_additive(doc_lengths, query_len, answer_len, student_variant, dtype, device)
        additive_blind = _build_additive(doc_lengths, query_len, answer_len, "blind", dtype, device)

        # Positions
        teacher_positions = build_positions(doc_lengths, query_len, answer_len, "sequential").unsqueeze(0).to(device)
        student_positions = build_positions(doc_lengths, query_len, answer_len, layout).unsqueeze(0).to(device)

        with torch.no_grad():
            teacher_out = teacher(
                input_ids=full_ids,
                attention_mask=additive_teacher,
                position_ids=teacher_positions,
                use_cache=False,
            )
            teacher_blind_out = teacher(
                input_ids=full_ids,
                attention_mask=additive_blind,
                position_ids=teacher_positions,
                use_cache=False,
            )

        T = int(full_ids.shape[-1])
        lo = T - answer_len - 1
        hi = T - 1
        teacher_ans = teacher_out.logits[:, lo:hi, :].float()

        # Build loss components
        components = [
            ("ddi_kl_loss", 1.0, additive_student, student_positions, teacher_ans),
            ("full_anchor_loss", full_anchor_weight, additive_teacher, teacher_positions, teacher_ans),
            ("blind_anchor_loss", blind_anchor_weight, additive_blind, teacher_positions, teacher_blind_out.logits[:, lo:hi, :].float()),
        ]

        will_step = (accum_since_step + 1) >= grad_accum_steps
        component_values = {}
        objective_value = 0.0

        for component_i, (name, weight, mask, positions, target_logits) in enumerate(components):
            sync_this_backward = will_step and component_i == len(components) - 1
            sync_ctx = torch.cuda.streams.Stream() if False else __import__("contextlib", fromlist=["nullcontext"]).nullcontext()
            with sync_ctx:
                student_out = student(
                    input_ids=full_ids,
                    attention_mask=mask,
                    position_ids=positions,
                    use_cache=False,
                )
                student_ans = student_out.logits[:, lo:hi, :].float()
                component_loss = _answer_kl(student_ans, target_logits, kl_temperature)
                weighted_loss = weight * component_loss
                if name == "ddi_kl_loss" and teacher_first_token_weight > 0:
                    first_token_loss = _teacher_first_token_ce(student_ans, teacher_ans)
                    component_values["teacher_first_token_loss"] = float(first_token_loss.detach().item())
                    weighted_loss = weighted_loss + (teacher_first_token_weight * first_token_loss)
                (weighted_loss / grad_accum_steps).backward()
            value = float(component_loss.detach().item())
            component_values[name] = value
            objective_value += float(weighted_loss.detach().item())

        accum_since_step += 1

        if accum_since_step >= grad_accum_steps:
            current_lr, stepped = _flush_optimizer_step(
                optimizer, warmup_steps, lr, global_step,
                params=trainable_params, max_grad_norm=max_grad_norm,
            )
            global_step += 1
            accum_since_step = 0

            if not stepped:
                skipped_steps += 1
                nan_count += 1
                _log(f"step {global_step}: non-finite gradients, SKIPPED ({skipped_steps}/{max_skipped_steps})")
                if skipped_steps > max_skipped_steps:
                    _log(f"QUALITY COLLAPSE: {skipped_steps} non-finite steps exceeded max_skipped_steps={max_skipped_steps}")
                    stop_early = True
                    break

            should_log = global_step == 1 or global_step % 25 == 0 or global_step >= steps
            if should_log:
                loss_curve.append({"step": global_step, "loss": objective_value, "lr": current_lr})
                _log(f"step {global_step} loss={objective_value:.4f} lr={current_lr:.2e} (rec {rec_idx}/{len(record_queue)})")

            if global_step >= steps:
                _log(f"max_steps={steps} reached; stopping")
                stop_early = True

    elapsed = time.time() - start_time

    # Save checkpoint
    _log("Saving checkpoint")
    _assert_finite_weights(student, "before saving checkpoint")
    output_dir.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    result = {
        "arm": arm,
        "steps": global_step,
        "elapsed_s": elapsed,
        "nan_count": nan_count,
        "skipped_steps": skipped_steps,
        "quality_collapse": skipped_steps > max_skipped_steps,
        "loss_curve": loss_curve,
        "final_loss": loss_curve[-1]["loss"] if loss_curve else None,
        "checkpoint_path": str(output_dir),
        "perturbation_recipe": {
            "bf16_reencode_noise_rate": bf16_noise_rate,
            "cache_dropout_rate": cache_dropout_rate,
            "shuffle_order": shuffle_order,
        },
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }

    with open(log_path, "w") as f:
        json.dump(result, f, indent=2, allow_nan=False)

    _log(f"Done: {global_step} steps, {elapsed:.1f}s, {nan_count} NaN, loss={result['final_loss']}")

    # Cleanup GPU
    del teacher, student, optimizer
    torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(description="V67 training arm")
    parser.add_argument("--arm", required=True, choices=["k0", "k1", "k5"])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="20-step smoke run")
    args = parser.parse_args()

    if args.smoke:
        args.steps = 20

    train_arm(
        arm=args.arm,
        steps=args.steps,
        seed=args.seed,
        output_dir=args.output_dir,
        log_path=args.log_path,
    )


if __name__ == "__main__":
    main()
