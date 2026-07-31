"""Self-distillation fine-tuner for the DDI ``ddi_qf`` student (issue #120).

Teacher = base model + ``full`` causal mask (frozen).
Student = base model + ``ddi_qf`` mask (same weights, only mask differs).
Loss    = temperature-scaled KL divergence on answer-position logits.

Goal is the "does training under ddi_qf close the C1 quality gap?" experiment.
This module is intentionally minimal — one function, one loop, no scheduler
zoo. Fancier ergonomics (LoRA, multi-seed, aggregator renderers) are follow-ups.

Store rows:

* Per-step training loss:
  ``run_id=finetune:{model}:ddi_qf:seed={seed}:{git_sha}:step={i}``
  metric ``kl_loss`` (value) and ``lr`` (value).

* Per-epoch perplexity (test set, held out from the mixed train stream):
  ``run_id=finetune:{model}:ddi_qf:seed={seed}:{git_sha}:epoch={e}``
  metrics ``perplexity_student_ddi_qf_pre``, ``perplexity_student_ddi_qf_post``,
  ``perplexity_teacher_full``.

The checkpoint is written to ``output_dir`` via `save_pretrained`, producing an
HF-loadable directory that ``ddi.inference.load(..., checkpoint_dir=...)``
picks up.
"""

from __future__ import annotations

import os
import sys
from itertools import zip_longest
from pathlib import Path

import bitsandbytes as bnb
import torch

from ddi import inference
from ddi.data import load as load_dataset
from ddi.masks import build_mask, to_additive
from ddi.positions import build_positions
from ddi.provenance import Provenance
from ddi.store import append


def _log(msg: str) -> None:
    print(f"[ddi.finetune] {msg}", file=sys.stderr, flush=True)


def _mix_records(datasets: list[str], seed: int, max_records: int | None) -> list[dict]:
    """Round-robin interleave each dataset's train split so every batch is mixed.

    ``max_records`` caps the TOTAL number of mixed records (not per-dataset), so
    a smoke test with ``max_records=8`` sees 2 records from each of 4 datasets
    (in interleaved order), not 8 from the first dataset.
    """
    streams: list[list[dict]] = []
    for name in datasets:
        streams.append(list(load_dataset(name, "train", seed)))

    mixed: list[dict] = []
    for tup in zip_longest(*streams, fillvalue=None):
        for rec in tup:
            if rec is None:
                continue
            mixed.append(rec)
            if max_records is not None and len(mixed) >= max_records:
                return mixed
    return mixed


def _holdout_split(records: list[dict], held_out_frac: float) -> tuple[list[dict], list[dict]]:
    """Deterministic tail split: last ``held_out_frac`` of ``records`` becomes the test set.

    Records already arrive in the interleaved order chosen by ``_mix_records``, so
    a tail split preserves the per-dataset mix in the holdout.
    """
    if not 0.0 <= held_out_frac < 1.0:
        raise ValueError(
            f"held_out_frac must be in [0, 1), got {held_out_frac}"
        )
    n = len(records)
    n_test = max(1, int(round(n * held_out_frac))) if held_out_frac > 0 else 0
    if n_test >= n:
        raise ValueError(
            f"held_out_frac={held_out_frac} leaves no training records (n={n})"
        )
    if n_test == 0:
        return records, []
    return records[:-n_test], records[-n_test:]


def _encode_record(tokenizer, record: dict) -> tuple[torch.Tensor, list[int], int, int]:
    """Return (input_ids [1,T], doc_lengths, query_len, answer_len) for one record.

    The chat template is applied to the ``(role, content)`` list up to (but not
    including) the assistant answer; the raw answer text is appended after.
    Matches ``ddi/run.py::_quality_cell``'s tokenization exactly.
    """
    doc_texts = list(record["info_contexts"])
    user_dialog = record["dialog"][:-1]
    answer_text = record["dialog"][-1]["content"]

    query_text = tokenizer.apply_chat_template(
        user_dialog, tokenize=False, add_generation_prompt=True
    )
    doc_id_tensors = [
        tokenizer(d, return_tensors="pt", add_special_tokens=False).input_ids
        for d in doc_texts
    ]
    doc_lengths = [int(t.shape[-1]) for t in doc_id_tensors]
    query_ids = tokenizer(
        query_text, return_tensors="pt", add_special_tokens=False
    ).input_ids
    answer_ids = tokenizer(
        answer_text, return_tensors="pt", add_special_tokens=False
    ).input_ids
    query_len = int(query_ids.shape[-1])
    answer_len = int(answer_ids.shape[-1])
    full_ids = torch.cat(doc_id_tensors + [query_ids, answer_ids], dim=-1)
    return full_ids, doc_lengths, query_len, answer_len


def _build_additive(
    doc_lengths: list[int],
    query_len: int,
    answer_len: int,
    variant: str,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """4D additive mask for a single record — matches _quality_cell's shape."""
    allow = build_mask(
        doc_lengths,
        query_len=query_len,
        answer_len=answer_len,
        variant=variant,  # type: ignore[arg-type]
    )
    return to_additive(allow, dtype=dtype).to(device).unsqueeze(0).unsqueeze(0)


def _maybe_grad_checkpoint(model) -> None:
    """Enable HF gradient checkpointing on the student if ``DDI_GRAD_CHECKPOINT=1``.

    Uses the model's built-in ``gradient_checkpointing_enable`` rather than
    ``torch.utils.checkpoint`` splicing so we don't have to know the block
    layout of an arbitrary decoder architecture. ``use_cache=False`` is
    already set on every forward in this loop, so no cache-vs-checkpointing
    conflict.
    """
    if os.environ.get("DDI_GRAD_CHECKPOINT") != "1":
        return
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        _log("gradient checkpointing enabled on student (DDI_GRAD_CHECKPOINT=1)")


def _linear_warmup_lr(step: int, warmup_steps: int, base_lr: float) -> float:
    """Return the LR for ``step`` under linear warmup then constant."""
    if warmup_steps <= 0 or step >= warmup_steps:
        return base_lr
    return base_lr * (step + 1) / warmup_steps


def _flush_optimizer_step(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    base_lr: float,
    step_before: int,
) -> float:
    """Compute LR for the current step, apply to every param group, step, zero grads.

    Returned LR is what was applied on THIS step. Callers use it for logging.
    """
    current_lr = _linear_warmup_lr(step_before, warmup_steps, base_lr)
    for group in optimizer.param_groups:
        group["lr"] = current_lr
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return current_lr


def finetune(
    model_name: str,
    datasets: list[str],
    output_dir: Path,
    *,
    seed: int,
    max_records: int | None = None,
    held_out_frac: float = 0.05,
    epochs: int = 1,
    lr: float = 1e-5,
    weight_decay: float = 0.0,
    kl_temperature: float = 1.0,
    grad_accum_steps: int = 1,
    warmup_steps: int = 100,
    log_every: int = 50,
    max_steps: int | None = None,
    provenance: Provenance,
    store_path: Path,
) -> None:
    """Fine-tune ``model_name`` under ``ddi_qf`` via self-distillation from the same
    model under ``full``. See module docstring for the store-row schema.

    ``max_steps``, if set, stops training after that many optimizer steps
    (across all epochs) — useful for a quick end-to-end run without waiting on
    the full dataset. The current epoch's post-training perplexity is still
    computed on whatever partial progress was made.
    """
    if not datasets:
        raise ValueError("finetune: `datasets` must be non-empty")
    if grad_accum_steps < 1:
        raise ValueError(f"grad_accum_steps must be >= 1, got {grad_accum_steps}")
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    if kl_temperature <= 0:
        raise ValueError(f"kl_temperature must be > 0, got {kl_temperature}")
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
    if max_steps is not None and max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _log(f"loading teacher (frozen) + student from {model_name!r}")
    teacher, tokenizer = inference.load(model_name)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student, _ = inference.load(model_name)
    student.train()
    _maybe_grad_checkpoint(student)

    if torch.cuda.is_available():
        teacher = teacher.to("cuda")
        student = student.to("cuda")

    device = student.device
    dtype = student.dtype

    _log(f"mixing records from {len(datasets)} datasets (seed={seed}, cap={max_records})")
    all_records = _mix_records(datasets, seed, max_records)
    train_records, test_records = _holdout_split(all_records, held_out_frac)
    _log(
        f"records: {len(train_records)} train / {len(test_records)} test "
        f"(held_out_frac={held_out_frac})"
    )
    if not train_records:
        raise ValueError("finetune: no training records after holdout split")

    trainable_params = [p for p in student.parameters() if p.requires_grad]
    # 8-bit optimizer states on CUDA: full AdamW states in bf16 already push a 7B
    # teacher+student pair close to an 80GB GPU's limit (issue #120 follow-up).
    # bitsandbytes needs CUDA tensors, so CPU-only runs (e.g. unit tests) keep
    # plain torch.optim.AdamW.
    if student.device.type == "cuda":
        optimizer = bnb.optim.AdamW8bit(trainable_params, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    run_prefix = f"finetune:{model_name}:ddi_qf:seed={seed}:{provenance.git_sha[:8]}"

    # ---- Baseline perplexities (epoch 0, before any update) ----
    if test_records:
        from ddi.perplexity import compute_perplexity

        _log("computing baseline perplexities on holdout")
        pplx_student_pre = compute_perplexity(
            student, tokenizer, test_records,
            mask_variant="ddi_qf", layout="sequential",
        )
        pplx_teacher = compute_perplexity(
            teacher, tokenizer, test_records,
            mask_variant="full", layout="sequential",
        )
        student.train()  # compute_perplexity flipped to eval; restore.
        epoch_run_id = f"{run_prefix}:epoch=0"
        append(epoch_run_id, "perplexity_student_ddi_qf_pre", pplx_student_pre,
               provenance, store=store_path)
        append(epoch_run_id, "perplexity_teacher_full", pplx_teacher,
               provenance, store=store_path)
        _log(
            f"epoch 0 baseline: student_ddi_qf={pplx_student_pre:.3f} "
            f"teacher_full={pplx_teacher:.3f}"
        )

    # ---- Training loop ----
    global_step = 0
    # Count backward() calls, not enumerate index — skipped records (empty answer,
    # encode failure) must NOT count toward the grad-accum boundary, otherwise a
    # run of consecutive skips can fire optimizer.step() with stale gradients.
    accum_since_step = 0
    optimizer.zero_grad(set_to_none=True)

    stop_early = False
    for epoch in range(epochs):
        _log(f"epoch {epoch + 1}/{epochs}: {len(train_records)} records")
        for rec_i, record in enumerate(train_records):
            is_last_record = (rec_i + 1) == len(train_records)
            if not record["possible_answers"]:
                if is_last_record and accum_since_step > 0:
                    _flush_optimizer_step(optimizer, warmup_steps, lr, global_step)
                    global_step += 1
                    accum_since_step = 0
                    if max_steps is not None and global_step >= max_steps:
                        stop_early = True
                continue
            try:
                full_ids, doc_lengths, query_len, answer_len = _encode_record(
                    tokenizer, record
                )
            except Exception as exc:
                _log(f"skip record id={record.get('id', '?')}: encode failed: {exc!r}")
                if is_last_record and accum_since_step > 0:
                    _flush_optimizer_step(optimizer, warmup_steps, lr, global_step)
                    global_step += 1
                    accum_since_step = 0
                    if max_steps is not None and global_step >= max_steps:
                        stop_early = True
                continue
            if answer_len == 0:
                if is_last_record and accum_since_step > 0:
                    _flush_optimizer_step(optimizer, warmup_steps, lr, global_step)
                    global_step += 1
                    accum_since_step = 0
                    if max_steps is not None and global_step >= max_steps:
                        stop_early = True
                continue

            full_ids = full_ids.to(device)
            # Coerce DDI variants to full when there are no docs — matches _quality_cell.
            student_variant = "ddi_qf" if doc_lengths else "full"
            additive_teacher = _build_additive(
                doc_lengths, query_len, answer_len, "full", dtype, device
            )
            additive_student = _build_additive(
                doc_lengths, query_len, answer_len, student_variant, dtype, device
            )
            positions = (
                build_positions(
                    doc_lengths, query_len, answer_len, "sequential"  # type: ignore[arg-type]
                )
                .unsqueeze(0)
                .to(device)
            )

            with torch.no_grad():
                teacher_out = teacher(
                    input_ids=full_ids,
                    attention_mask=additive_teacher,
                    position_ids=positions,
                    use_cache=False,
                )
            student_out = student(
                input_ids=full_ids,
                attention_mask=additive_student,
                position_ids=positions,
                use_cache=False,
            )

            # Next-token shift: logits at position t predict token t+1. The answer
            # region on the input axis is [T-answer_len, T); its predictive logits
            # therefore live at [T-answer_len-1, T-1) on the logits axis.
            T = int(full_ids.shape[-1])
            lo = T - answer_len - 1
            hi = T - 1
            teacher_ans = teacher_out.logits[:, lo:hi, :].float()
            student_ans = student_out.logits[:, lo:hi, :].float()

            T_temp = kl_temperature
            log_p_student = torch.nn.functional.log_softmax(student_ans / T_temp, dim=-1)
            p_teacher = torch.nn.functional.softmax(teacher_ans / T_temp, dim=-1)
            # Mean over answer tokens; KL^2 scale so LR is temperature-independent.
            loss = torch.nn.functional.kl_div(
                log_p_student, p_teacher, reduction="batchmean"
            ) * (T_temp ** 2)

            (loss / grad_accum_steps).backward()
            accum_since_step += 1

            if accum_since_step >= grad_accum_steps or is_last_record:
                current_lr = _flush_optimizer_step(
                    optimizer, warmup_steps, lr, global_step
                )
                global_step += 1
                accum_since_step = 0

                if global_step % log_every == 0:
                    loss_val = float(loss.detach().item())
                    step_rid = f"{run_prefix}:step={global_step}"
                    append(step_rid, "kl_loss", loss_val, provenance, store=store_path)
                    append(step_rid, "lr", current_lr, provenance, store=store_path)
                    _log(
                        f"step {global_step} loss={loss_val:.4f} lr={current_lr:.2e} "
                        f"(epoch {epoch + 1}, rec {rec_i + 1}/{len(train_records)})"
                    )

                if max_steps is not None and global_step >= max_steps:
                    _log(f"max_steps={max_steps} reached; stopping early")
                    stop_early = True
                    break

        # ---- End-of-epoch perplexities ----
        if test_records:
            from ddi.perplexity import compute_perplexity

            _log(f"epoch {epoch + 1}: computing post-epoch perplexity")
            pplx_post = compute_perplexity(
                student, tokenizer, test_records,
                mask_variant="ddi_qf", layout="sequential",
            )
            student.train()
            epoch_rid = f"{run_prefix}:epoch={epoch + 1}"
            append(epoch_rid, "perplexity_student_ddi_qf_post", pplx_post,
                   provenance, store=store_path)
            _log(f"epoch {epoch + 1} post: student_ddi_qf={pplx_post:.3f}")

        if stop_early:
            break

    # ---- Save the trained student as an HF-loadable directory ----
    _log(f"saving checkpoint -> {output_dir}")
    student.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    _log("finetune: done")
