"""From-scratch SFT loop for Phase 2.2 (issue #14).

Runs supervised fine-tuning with a DDI mask present throughout so we can distinguish
"model just needs a nudge to route through the query region" from "isolation destroyed
retrievable information" (RESEARCH_PLAN §8 Phase 2.2). No Trainer / trl / peft — every
number is a plain torch step so the code path is auditable end-to-end. Loss is answer-only
cross-entropy over the shifted logits, identical to `phase1_baseline`'s scoring path.

Full SFT rung only for now; LoRA / light-adaptation is a follow-up (marked TODO in the
example config).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ddi.data import load
from ddi.masks import build_mask, to_additive
from ddi.metrics import make_answer_labels
from ddi.model import forward_with_mask, load_model
from ddi.provenance import Provenance
from ddi.store import append


@dataclass(frozen=True)
class TrainConfig:
    model_name: str
    dataset: str
    mask_variant: str  # full | ddi_qf | ddi_strict | ddi_bridge_k
    bridge_k: int | None = None
    split: str = "train"
    lr: float = 1e-5
    warmup_steps: int = 50
    max_steps: int = 500
    grad_accum_steps: int = 8
    # RESEARCH_PLAN §5: batch_size pinned at 1 for Phase 0. Kept as a parameter so lifting
    # it later (varlen batched prefill with per-sample block-diagonal masks) is a config
    # change, not a refactor.
    batch_size: int = 1
    gradient_checkpointing: bool = True
    validate_every: int = 100
    output_dir: Path = field(default_factory=lambda: Path("checkpoints/phase2_finetune"))


def _shifted_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Answer-only cross-entropy over the standard next-token shift.

    Predict position `t` from logits at position `t-1`; `-100` labels are ignored, so
    only answer tokens contribute — identical scoring path to `phase1_baseline`.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def _run_tag(cfg: TrainConfig, seed: int) -> str:
    """Deterministic checkpoint subdirectory name for a given cfg + seed.

    Model / dataset slashes are illegal in most FSes → replace; the config-content hash
    covers every remaining field so two runs with different LR / mask land in different
    subdirs even if the model+dataset+seed match.
    """
    payload = json.dumps(
        {**asdict(cfg), "output_dir": str(cfg.output_dir), "seed": seed},
        sort_keys=True,
        default=str,
    )
    cfg_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    safe_model = cfg.model_name.replace("/", "_")
    safe_dataset = cfg.dataset.replace("/", "_")
    return f"{safe_model}__{safe_dataset}__{cfg.mask_variant}__seed{seed}__{cfg_hash}"


def _prepare_batch(
    record: dict,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
    dtype: torch.dtype,
    mask_variant: str,
    bridge_k: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Tokenize one record and build (`full_ids`, `additive_mask`, `labels`).

    Returns `None` for records that are unscoreable (empty answer / degenerate span);
    callers should skip loudly rather than sneak zero-loss micro-batches into the
    accumulator.
    """
    if not record["dialog"]:
        return None
    user_dialog = record["dialog"][:-1]
    answer = record["dialog"][-1]["content"]
    if not answer:
        return None

    prompt_text = tokenizer.apply_chat_template(
        user_dialog, tokenize=False, add_generation_prompt=True
    )
    if record.get("info_contexts"):
        prompt_text = "\n\n".join(record["info_contexts"]) + "\n\n" + prompt_text

    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
    answer_ids = tokenizer(
        answer, return_tensors="pt", add_special_tokens=False
    ).input_ids
    if answer_ids.shape[-1] == 0:
        return None

    full_ids = torch.cat([prompt_ids, answer_ids], dim=-1).to(device)
    prompt_len = prompt_ids.shape[-1]
    total_len = full_ids.shape[-1]

    labels = make_answer_labels(full_ids, prompt_len, total_len).to(device)

    # Docs / query / answer regions get length 0 for the doc/query parts: chat-templated
    # prompt is treated as one contiguous doc block from the mask builder's perspective,
    # because Phase 2.2 wants "mask present during SFT" using the same variant lookup as
    # eval — parameterizing per-doc lengths inside SFT is a follow-up that belongs with
    # the multi-doc training set, not the first SFT rung.
    allow = build_mask(
        [total_len],
        query_len=0,
        answer_len=0,
        variant=mask_variant,
        bridge_k=bridge_k,
    )
    additive = (
        to_additive(allow, dtype=dtype).to(device).unsqueeze(0).unsqueeze(0)
    )

    return full_ids, additive, labels


def _iter_records(dataset: str, split: str, seed: int) -> Iterator[dict]:
    """Cycle the dataset loader so we don't exhaust records before `max_steps` × accum.

    Note we re-invoke `load()` each pass rather than `itertools.cycle(load(...))`:
    cycle would buffer the entire split in memory, whereas `load` is guaranteed
    deterministic given `(split, seed)` so re-invoking it yields the same records in
    the same order without the memory hit.
    """
    def _stream() -> Iterator[dict]:
        while True:
            yield from load(dataset, split, seed)

    return _stream()


def _validate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    cfg: TrainConfig,
    seed: int,
    n_records: int = 32,
) -> float:
    """Compute mean answer-only loss on the first `n_records` of the val split."""
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for i, record in enumerate(load(cfg.dataset, "val", seed)):
            if i >= n_records:
                break
            batch = _prepare_batch(
                record,
                tokenizer,
                model.device,
                model.dtype,
                cfg.mask_variant,
                cfg.bridge_k,
            )
            if batch is None:
                continue
            full_ids, additive, labels = batch
            logits = forward_with_mask(model, full_ids, additive)
            loss = _shifted_cross_entropy(logits, labels)
            if torch.isfinite(loss):
                losses.append(loss.item())
    model.train()
    if not losses:
        return float("nan")
    return sum(losses) / len(losses)


def _lr_lambda(step: int, warmup_steps: int) -> float:
    """Linear warmup to 1.0 over `warmup_steps`, then constant."""
    if warmup_steps <= 0:
        return 1.0
    if step < warmup_steps:
        # step is 0-indexed; scale from 1/warmup on the first update to 1.0 at warmup end.
        return float(step + 1) / float(warmup_steps)
    return 1.0


def train(cfg: TrainConfig, seed: int, provenance: Provenance) -> Path:
    """Runs the SFT loop and saves the final checkpoint to `cfg.output_dir/{run_tag}/`.

    Returns the path to the saved checkpoint (loadable via load_model).
    """
    if cfg.batch_size != 1:
        # WHY: RESEARCH_PLAN §5 pins this at 1 for Phase 0; batched prefill with per-sample
        # block-diagonal masks is a follow-up. Fail loudly rather than silently accept a
        # value that the mask + collation path do not honour yet.
        raise ValueError(
            f"batch_size={cfg.batch_size} not supported yet; Phase 0 pins batch_size=1"
        )
    if cfg.max_steps <= 0:
        raise ValueError(f"max_steps must be > 0, got {cfg.max_steps}")
    if cfg.grad_accum_steps <= 0:
        raise ValueError(f"grad_accum_steps must be > 0, got {cfg.grad_accum_steps}")

    model, tokenizer = load_model(cfg.model_name)
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.train()
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # Gradient checkpointing needs `use_cache=False`; `forward_with_mask` already sets
        # this, but the model-level flag prevents HF from warning about the conflict.
        if hasattr(model, "config"):
            model.config.use_cache = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: _lr_lambda(step, cfg.warmup_steps)
    )

    run_tag = _run_tag(cfg, seed)
    run_id = f"phase2_finetune:{cfg.model_name}:{cfg.dataset}:{cfg.mask_variant}:seed={seed}:{provenance.git_sha[:8]}"

    stream = _iter_records(cfg.dataset, cfg.split, seed)

    for step in range(cfg.max_steps):
        optimizer.zero_grad()
        step_loss_sum = 0.0
        step_loss_count = 0
        skip_step = False

        for _ in range(cfg.grad_accum_steps):
            # Skip unscoreable records until we get a real micro-batch — a `None` here
            # would zero out this accumulator slot rather than delay the update.
            batch = None
            while batch is None:
                record = next(stream)
                batch = _prepare_batch(
                    record,
                    tokenizer,
                    model.device,
                    model.dtype,
                    cfg.mask_variant,
                    cfg.bridge_k,
                )
                if batch is None:
                    print(
                        f"[train] skip record id={record.get('id', '<no-id>')}: unscoreable"
                    )

            full_ids, additive, labels = batch
            logits = forward_with_mask(model, full_ids, additive)
            loss = _shifted_cross_entropy(logits, labels)

            # WHY non-finite skip: bf16 SFT occasionally produces inf/NaN on a single
            # micro-batch (long doc, extreme logit). Poisoning the accumulator would
            # corrupt an otherwise clean step. We zero grads and drop the whole step
            # rather than the offending micro-batch alone, because partial-accumulator
            # updates change the effective batch size unpredictably.
            if not torch.isfinite(loss):
                print(
                    f"[train] step={step}: non-finite loss detected; skipping optimizer step"
                )
                skip_step = True
                break

            (loss / cfg.grad_accum_steps).backward()
            step_loss_sum += loss.item()
            step_loss_count += 1

        if skip_step:
            optimizer.zero_grad()
            # WHY: do NOT advance the LR schedule on a skipped step — the schedule counts
            # applied optimizer steps, not attempted ones, so warmup stays honest.
            continue

        optimizer.step()
        scheduler.step()

        mean_loss = step_loss_sum / max(step_loss_count, 1)

        if (step + 1) % 100 == 0 or step == 0:
            append(run_id, "train_loss", mean_loss, provenance)
            print(f"[train] step={step + 1}/{cfg.max_steps} loss={mean_loss:.4f}")

        if cfg.validate_every > 0 and (step + 1) % cfg.validate_every == 0:
            val_loss = _validate(model, tokenizer, cfg, seed)
            append(run_id, "train_val_loss", val_loss, provenance)
            print(f"[train] step={step + 1} val_loss={val_loss:.4f}")

    ckpt_dir = cfg.output_dir / run_tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    print(f"[train] saved checkpoint to {ckpt_dir}")
    return ckpt_dir
