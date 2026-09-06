"""Self-distillation fine-tuner for the DDI ``ddi_qf`` student (issue #120).

Teacher = base model + ``full`` causal mask (frozen).
Student = base model + ``ddi_qf`` mask (same weights, only mask differs).
Loss    = temperature-scaled KL divergence on answer-position logits, with
          optional first-token and control-path anchors.

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

import contextlib
import os
import shutil
import sys
from datetime import timedelta
from itertools import zip_longest
from pathlib import Path

import torch
import torch.distributed as dist

from ddi import inference
from ddi.data import load as load_dataset
from ddi.masks import build_mask, to_additive
from ddi.positions import build_positions
from ddi.prompt import encode_record
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

    Thin alias for `ddi.prompt.encode_record`, kept because `ddi/diagnose_grads.py`
    imports this name. The training path deliberately owns no tokenisation of its own:
    it has to match the eval and perplexity paths token for token, and the way that
    guarantee was previously stated — a docstring claiming to match
    ``ddi/run.py::_quality_cell`` "exactly" — is not a guarantee anyone can check.
    """
    return encode_record(tokenizer, record)


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


def _answer_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Mean temperature-scaled KL over answer positions."""
    log_p_student = torch.nn.functional.log_softmax(
        student_logits / temperature, dim=-1
    )
    p_teacher = torch.nn.functional.softmax(teacher_logits / temperature, dim=-1)
    return torch.nn.functional.kl_div(
        log_p_student.reshape(-1, log_p_student.shape[-1]),
        p_teacher.reshape(-1, p_teacher.shape[-1]),
        reduction="batchmean",
    ) * (temperature ** 2)


def _teacher_first_token_ce(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor
) -> torch.Tensor:
    """CE against the full teacher's greedy first answer token."""
    target = teacher_logits[:, 0, :].argmax(dim=-1)
    return torch.nn.functional.cross_entropy(student_logits[:, 0, :], target)


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
        # `use_reentrant=False` is not a style preference, it is required as soon as any
        # part of the model is frozen. Under the reentrant implementation a checkpointed
        # segment only rebuilds its autograd graph if one of its *inputs* requires grad.
        # With a frozen trunk the hidden state arriving at the first trainable layer does
        # not, so every trainable layer under checkpointing would recompute with no graph
        # and receive no gradient — training would run to completion, log a falling loss
        # from the frozen teacher/student KL, and save a checkpoint identical to the base
        # model. The non-reentrant implementation tracks the parameters themselves.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        _log("gradient checkpointing enabled on student (DDI_GRAD_CHECKPOINT=1, non-reentrant)")


def _decoder_layers(model) -> tuple[str, torch.nn.ModuleList]:
    """Locate the decoder's per-layer ``ModuleList``, whatever the architecture calls it.

    Qwen2 puts it at ``model.layers``; the ``qwen3_5`` checkpoints nest it one level
    deeper. Rather than hardcode either, look for a ``ModuleList`` whose attribute path
    ends in ``layers`` and that holds more than one module.

    There is usually more than one such list, which is the whole reason this picks rather
    than takes the first: Qwen3.6-27B-Instruct is multimodal and its vision tower has a
    ``layers`` list too. Freezing "all but the top 8" of a 27-layer vision encoder while
    leaving 64 text layers frozen would train the wrong half of the model and still report
    a falling loss. The decoder stack is the one holding the most parameters by a wide
    margin, so pick on that rather than on traversal order, which is an implementation
    detail of whichever transformers version the job happens to have.
    """
    candidates = [
        (name, mod)
        for name, mod in model.named_modules()
        if isinstance(mod, torch.nn.ModuleList) and name.split(".")[-1] == "layers" and len(mod) > 1
    ]
    if not candidates:
        raise ValueError(
            "finetune: could not find the decoder layer list on this model; "
            "trainable_layers only works on a standard stacked decoder"
        )
    name, layers = max(candidates, key=lambda kv: sum(p.numel() for p in kv[1].parameters()))
    if len(candidates) > 1:
        _log(
            f"found {len(candidates)} layer stacks "
            f"({', '.join(n for n, _ in candidates)}); training in {name}, the largest"
        )
    return name, layers


def _freeze_trunk(model, trainable_layers: int) -> int:
    """Freeze everything but the top ``trainable_layers`` decoder layers. Returns the count.

    This is what makes a 27B student trainable on a 141 GB card. The per-rank cost of a
    full finetune here is about 8 bytes per parameter — teacher (bf16) + student (bf16) +
    gradients (bf16) + AdamW8bit moments — and DDP replicates all of it on every rank, so
    adding GPUs does not reduce it. Freezing the trunk removes the gradient and optimizer
    terms for everything below the cut, which is the only half of that budget that is
    actually optional: the two model replicas have to be resident.

    The final norm rides along with the top layers. It sits after the stack, is a single
    ``hidden_size`` vector, and leaving it frozen while retuning the layers feeding it is
    the kind of mismatch that costs quality for no memory saving.

    Embeddings and ``lm_head`` stay frozen deliberately. On these vocabularies they are the
    largest single tensors in the model, and the KL objective already supervises the logits
    they produce through the layers below.
    """
    name, layers = _decoder_layers(model)
    total = len(layers)
    if not 1 <= trainable_layers <= total:
        raise ValueError(
            f"trainable_layers must be between 1 and {total} for this model, "
            f"got {trainable_layers}"
        )

    for p in model.parameters():
        p.requires_grad_(False)
    for layer in layers[total - trainable_layers:]:
        for p in layer.parameters():
            p.requires_grad_(True)

    # The stack's sibling `norm` — `model.norm` next to `model.layers`.
    prefix = name.rsplit(".", 1)[0] if "." in name else ""
    for param_name, p in model.named_parameters():
        head = param_name.rsplit(".", 1)[0]
        if head in (f"{prefix}.norm" if prefix else "norm",):
            p.requires_grad_(True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    _log(
        f"frozen trunk: training top {trainable_layers}/{total} layers of {name} "
        f"({trainable / 1e9:.2f}B trainable, {frozen / 1e9:.2f}B frozen)"
    )
    return trainable


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
    params: list[torch.nn.Parameter],
    max_grad_norm: float = 1.0,
) -> tuple[float, bool]:
    """Clip, step (if the gradients are finite), zero grads.

    Returns ``(lr_applied, stepped)``. ``stepped`` is False when the step was SKIPPED
    because its gradients were not finite.

    Both guards exist because of Qwen2.5-7B on 2026-08-07: without them one gradient spike
    took the weights to NaN between step 1 and step 50, and the run then trained on NaN for
    another ~3,230 steps, saved the checkpoint, uploaded it to YT, evaluated it, and posted a
    table of zeros to the ticket. Every layer reported success. Nothing in eight GPU-hours
    noticed, because nothing was looking.

    Clipping bounds the spike; the finite check makes a spike that still overflows cost one
    skipped step instead of the whole run. `clip_grad_norm_` returns the PRE-clip norm, which
    is the value to test — a norm of inf/NaN means the gradients are already poison and
    stepping on them is what destroys the weights.
    """
    # `params` is required, and empty is an error rather than a silent pass-through. An
    # `if params:` that quietly does nothing would disable BOTH guards — no clipping and no
    # finite check — for a caller that forgot to pass them, which is precisely the
    # silent-success shape these guards exist to remove.
    if not params:
        raise ValueError(
            "_flush_optimizer_step: `params` is empty; clipping and the non-finite check "
            "would both be skipped. Pass the trainable parameters."
        )

    current_lr = _linear_warmup_lr(step_before, warmup_steps, base_lr)
    for group in optimizer.param_groups:
        group["lr"] = current_lr

    grad_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
    stepped = bool(torch.isfinite(grad_norm))

    if stepped:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return current_lr, stepped


def _assert_finite_weights(model, where: str) -> None:
    """Raise if any parameter is non-finite. The last gate before a checkpoint is written.

    A NaN checkpoint is worse than a failed run: it is indistinguishable from a good one
    until someone reads the eval numbers, and by then the weights are on YT under a name that
    claims to be a result. Fail here instead, loudly, while the cause is still on screen.
    """
    bad = [name for name, p in model.named_parameters() if not torch.isfinite(p).all()]
    if bad:
        raise RuntimeError(
            f"{where}: {len(bad)} parameter tensor(s) are not finite "
            f"(first: {bad[0]}). Training diverged; refusing to save a NaN checkpoint."
        )


def _init_ddp(num_gpus: int) -> tuple[int, int, int]:
    """Bring up NCCL and return ``(rank, world_size, local_rank)`` (issue #123).

    When ``num_gpus == 1`` we stay single-device and return ``(0, 1, 0)`` even
    if the process happens to be running under ``torchrun`` — the caller then
    takes the original single-device path unchanged.

    When ``num_gpus > 1`` we require the standard ``torchrun`` env vars
    (``WORLD_SIZE``, ``RANK``, ``LOCAL_RANK``) and fail loudly with a hint if
    they're missing, so a user who forgets to wrap the invocation gets a clear
    error rather than a silent single-GPU run.
    """
    if num_gpus == 1:
        return 0, 1, 0
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"finetune: num_gpus={num_gpus} requires CUDA, but torch.cuda.is_available() is False"
        )
    missing = [k for k in ("WORLD_SIZE", "RANK", "LOCAL_RANK") if k not in os.environ]
    if missing:
        raise RuntimeError(
            f"finetune: num_gpus={num_gpus} requires launching under torchrun "
            f"(missing env: {missing}). Try: "
            f"`torchrun --nproc_per_node={num_gpus} --standalone -m ddi.run ...` "
            f"or `make finetune NUM_GPUS={num_gpus} MODEL=...`."
        )
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != num_gpus:
        raise RuntimeError(
            f"finetune: num_gpus={num_gpus} but WORLD_SIZE={world_size} — "
            "torchrun --nproc_per_node must match num_gpus."
        )
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        # Three sections run on rank 0 alone while every other rank waits in
        # dist.barrier(): the epoch-0 baseline perplexities, the end-of-epoch
        # perplexity, and save_pretrained. Each is far longer than NCCL's
        # 10-minute default watchdog, which aborts the whole job mid-barrier
        # (ALLREDUCE SeqNum=31 ... ran for 600022 milliseconds before timing
        # out). The barrier must outlast the longest single-rank section, so
        # size the timeout for that rather than for a collective.
        timeout_min = int(os.environ.get("DDI_DIST_TIMEOUT_MIN", "240"))
        dist.init_process_group(
            backend="nccl", timeout=timedelta(minutes=timeout_min)
        )
    return rank, world_size, local_rank


def _prefilter_records(records: list[dict], min_docs: int = 2) -> list[dict]:
    """Drop records that cannot produce a gradient.

    Two classes:

    * Empty ``possible_answers`` — no answer span, so no loss. Under DDP, ranks that
      skip while others don't would desync the grad-accum counter and deadlock on
      the next all-reduce. Filtering upfront keeps every rank iterating in lock step.

    * Fewer than ``min_docs`` documents — ``ddi.masks.build_mask`` collapses the whole
      ``ddi_qf``/``ddi_strict``/``ddi_bridge_k`` family to plain ``full`` when there is
      at most one document (masks.py:79-80), because there is nothing to isolate. The
      student and teacher masks are then *identical*, so the KL is exactly zero and the
      record contributes nothing but a forward pass on two 4B models. This is not a
      corner case: the kilt_nq train split is 100% such records (76,945 single-doc plus
      10,427 with no documents at all), so a four-dataset mix was burning roughly a
      quarter of its steps on guaranteed-zero gradients (VIBEQUEUE-2 review, issue R4).

    ``min_docs=1`` restores the old behaviour for a caller who wants it.
    """
    kept = [
        r
        for r in records
        if r.get("possible_answers") and len(r.get("info_contexts") or []) >= min_docs
    ]
    dropped = len(records) - len(kept)
    if dropped:
        _log(
            f"prefilter: dropped {dropped}/{len(records)} records "
            f"(no answer, or fewer than min_docs={min_docs} documents → zero KL)"
        )
    return kept


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
    teacher_first_token_weight: float = 0.0,
    full_anchor_weight: float = 0.0,
    blind_anchor_weight: float = 0.0,
    grad_accum_steps: int = 1,
    warmup_steps: int = 100,
    log_every: int = 50,
    max_steps: int | None = None,
    num_gpus: int = 1,
    min_docs: int = 2,
    trainable_layers: int | None = None,
    holdout_eval_records: int | None = None,
    max_grad_norm: float = 1.0,
    max_skipped_steps: int = 20,
    layout: str = "sequential",
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
    objective_weights = {
        "teacher_first_token_weight": teacher_first_token_weight,
        "full_anchor_weight": full_anchor_weight,
        "blind_anchor_weight": blind_anchor_weight,
    }
    for name, weight in objective_weights.items():
        if weight < 0:
            raise ValueError(f"{name} must be >= 0, got {weight}")
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
    if max_steps is not None and max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")
    if num_gpus < 1:
        raise ValueError(f"num_gpus must be >= 1, got {num_gpus}")
    if trainable_layers is not None and trainable_layers < 1:
        raise ValueError(f"trainable_layers must be >= 1, got {trainable_layers}")
    if max_grad_norm <= 0:
        raise ValueError(f"max_grad_norm must be > 0, got {max_grad_norm}")
    if max_skipped_steps < 0:
        raise ValueError(f"max_skipped_steps must be >= 0, got {max_skipped_steps}")
    if holdout_eval_records is not None and holdout_eval_records < 1:
        raise ValueError(
            f"holdout_eval_records must be >= 1, got {holdout_eval_records}"
        )
    if layout not in ("sequential", "restart", "mixed"):
        raise ValueError(
            f"layout must be 'sequential', 'restart', or 'mixed', got {layout!r}"
        )

    rank, world_size, local_rank = _init_ddp(num_gpus)
    is_dist = world_size > 1
    is_main = rank == 0

    output_dir = Path(output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    _log(f"loading teacher (frozen) + student from {model_name!r} (rank={rank}/{world_size})")
    teacher, tokenizer = inference.load(model_name)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student, _ = inference.load(model_name)
    student.train()
    # Freeze BEFORE the optimizer is built and before the DDP wrap: the optimizer is
    # constructed from `requires_grad` params, and DDP registers its reducer hooks only on
    # params that require grad. Freezing afterwards would leave the optimizer holding
    # tensors it must not step and DDP all-reducing gradients that no longer exist.
    if trainable_layers is not None:
        _freeze_trunk(student, trainable_layers)
    _maybe_grad_checkpoint(student)

    if is_dist:
        device = torch.device(f"cuda:{local_rank}")
        teacher = teacher.to(device)
        student = student.to(device)
    elif torch.cuda.is_available():
        teacher = teacher.to("cuda")
        student = student.to("cuda")
        device = student.device
    else:
        device = student.device

    dtype = student.dtype

    _log(f"mixing records from {len(datasets)} datasets (seed={seed}, cap={max_records})")
    all_records = _mix_records(datasets, seed, max_records)
    train_records, test_records = _holdout_split(all_records, held_out_frac)
    # The holdout perplexities run on RANK 0 ONLY, twice (student + teacher), while the
    # other seven ranks wait at a barrier. At `held_out_frac=0.05` over the full mix that
    # is ~5.5k records x 2 forward passes on one GPU — 15-40 min of an 8-GPU allocation
    # doing single-GPU work, and it is re-paid in full every time the job is restarted.
    # Capping it costs precision on a diagnostic (the headline numbers come from the eval
    # block, not from here) and buys back most of a restart's cost.
    pplx_records = test_records
    if holdout_eval_records is not None and len(test_records) > holdout_eval_records:
        pplx_records = test_records[:holdout_eval_records]
        _log(
            f"holdout perplexity capped to {len(pplx_records)}/{len(test_records)} records "
            f"(holdout_eval_records={holdout_eval_records})"
        )
    # Filter records that would be silently skipped, so every DDP rank iterates
    # the loop the same number of times and no one deadlocks on grad-sync
    # (issue #123). Cheap enough to always run; matches the old behavior of
    # skipping such records inside the loop.
    train_records = _prefilter_records(train_records, min_docs=min_docs)
    _log(
        f"records: {len(train_records)} train / {len(test_records)} test "
        f"(held_out_frac={held_out_frac})"
    )
    if not train_records:
        raise ValueError("finetune: no training records after holdout split")

    if is_dist:
        # Deterministic per-rank shard — strided so consecutive records go to
        # different ranks (preserves the interleaved dataset mix on each rank).
        # Truncate to a length divisible by world_size first, so every rank
        # gets the same iteration count. If we didn't, ranks with the leftover
        # tail records would call backward() more times than the others and
        # deadlock at the next all-reduce (issue #123).
        drop = len(train_records) % world_size
        if drop:
            train_records = train_records[:-drop]
        train_records = train_records[rank::world_size]
        if not train_records:
            raise ValueError(
                f"finetune: rank {rank} has no records after sharding "
                f"(world_size={world_size}); reduce num_gpus or increase max_records"
            )
        _log(f"rank {rank}: {len(train_records)} records after sharding (dropped {drop} tail)")

    trainable_params = [p for p in student.parameters() if p.requires_grad]
    # 8-bit optimizer states on CUDA: full AdamW states in bf16 already push a 7B
    # teacher+student pair close to an 80GB GPU's limit (issue #120 follow-up).
    # bitsandbytes needs CUDA tensors, so CPU-only runs (e.g. unit tests) keep
    # plain torch.optim.AdamW.
    if device.type == "cuda":
        # Imported here, not at module scope: bitsandbytes is a CUDA-oriented package and is
        # now an optional dependency (the `train` extra). A top-level import made the whole
        # trainer module unimportable on any CPU-only host, which is what made
        # tests/test_finetune.py fail at collection time.
        try:
            import bitsandbytes as bnb

            optimizer = bnb.optim.AdamW8bit(trainable_params, lr=lr, weight_decay=weight_decay)
        except (ImportError, ModuleNotFoundError):
            _log("bitsandbytes not installed; falling back to torch.optim.AdamW (more VRAM)")
            optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    # DDP wrap comes AFTER the optimizer sees the raw parameters. DDP doesn't
    # replace parameter tensors (it registers hooks on them), so the optimizer's
    # tensor references remain valid post-wrap.
    if is_dist:
        student = torch.nn.parallel.DistributedDataParallel(
            student, device_ids=[local_rank]
        )
    # `student_module` is the unwrapped model — needed for save_pretrained and for
    # compute_perplexity (which calls model(...) directly, not through DDP scatter).
    student_module = student.module if is_dist else student

    run_prefix = f"finetune:{model_name}:ddi_qf:seed={seed}:{provenance.git_sha[:8]}"

    # ---- Baseline perplexities (epoch 0, before any update) ----
    # Rank 0 only under DDP — the holdout eval is deterministic and lives at the
    # store, not on any tensor, so replicating it across ranks would only
    # duplicate rows. Other ranks wait at the barrier below so training resumes
    # in lock step.
    if test_records and is_main:
        from ddi.perplexity import compute_perplexity

        _log("computing baseline perplexities on holdout")
        pplx_student_pre = compute_perplexity(
            student_module, tokenizer, pplx_records,
            mask_variant="ddi_qf", layout="sequential",
        )
        pplx_teacher = compute_perplexity(
            teacher, tokenizer, pplx_records,
            mask_variant="full", layout="sequential",
        )
        student_module.train()  # compute_perplexity flipped to eval; restore.
        epoch_run_id = f"{run_prefix}:epoch=0"
        append(epoch_run_id, "perplexity_student_ddi_qf_pre", pplx_student_pre,
               provenance, store=store_path)
        append(epoch_run_id, "perplexity_teacher_full", pplx_teacher,
               provenance, store=store_path)
        _log(
            f"epoch 0 baseline: student_ddi_qf={pplx_student_pre:.3f} "
            f"teacher_full={pplx_teacher:.3f}"
        )
    if is_dist:
        dist.barrier()

    # ---- Training loop ----
    global_step = 0
    # Steps dropped because their gradients were not finite. A handful is survivable;
    # a stream of them means the run is diverging and must not produce a checkpoint.
    skipped_steps = 0
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
            # Under DDP every rank must call backward() the same number of times
            # per iteration or the next all-reduce deadlocks. The pre-filter above
            # already dropped empty possible_answers; here we raise on the rarer
            # encode/answer_len failures instead of silently skipping (issue #123).
            if not record["possible_answers"]:
                if is_dist:
                    raise RuntimeError(
                        f"finetune (DDP): record {record.get('id', '?')} has empty "
                        "possible_answers post-prefilter — should not happen; report as a bug."
                    )
                if is_last_record and accum_since_step > 0:
                    _flush_optimizer_step(
                        optimizer, warmup_steps, lr, global_step,
                        params=trainable_params, max_grad_norm=max_grad_norm,
                    )
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
                if is_dist:
                    raise RuntimeError(
                        f"finetune (DDP): encode failed for record {record.get('id', '?')}: {exc!r}"
                    ) from exc
                _log(f"skip record id={record.get('id', '?')}: encode failed: {exc!r}")
                if is_last_record and accum_since_step > 0:
                    _flush_optimizer_step(
                        optimizer, warmup_steps, lr, global_step,
                        params=trainable_params, max_grad_norm=max_grad_norm,
                    )
                    global_step += 1
                    accum_since_step = 0
                    if max_steps is not None and global_step >= max_steps:
                        stop_early = True
                continue
            if answer_len == 0:
                if is_dist:
                    raise RuntimeError(
                        f"finetune (DDP): record {record.get('id', '?')} has zero-length answer"
                    )
                if is_last_record and accum_since_step > 0:
                    _flush_optimizer_step(
                        optimizer, warmup_steps, lr, global_step,
                        params=trainable_params, max_grad_norm=max_grad_norm,
                    )
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
            additive_blind = (
                _build_additive(doc_lengths, query_len, answer_len, "blind", dtype, device)
                if blind_anchor_weight > 0
                else None
            )

            # Mixed-layout training: coin flip per record between sequential/restart.
            # Teacher always uses sequential positions (frozen reference); the student
            # learns to produce the same answer logits under the chosen layout.  This
            # teaches the model to be robust to the position-reset that ddi_cache uses
            # at inference time, without which restart quality collapses (VIBEQUEUE-2).
            if layout == "mixed":
                record_layout = "restart" if torch.rand(1).item() < 0.5 else "sequential"
            else:
                record_layout = layout

            teacher_positions = (
                build_positions(
                    doc_lengths, query_len, answer_len, "sequential"
                )
                .unsqueeze(0)
                .to(device)
            )
            student_positions = (
                build_positions(
                    doc_lengths, query_len, answer_len, record_layout  # type: ignore[arg-type]
                )
                .unsqueeze(0)
                .to(device)
            )

            with torch.no_grad():
                teacher_out = teacher(
                    input_ids=full_ids,
                    attention_mask=additive_teacher,
                    position_ids=teacher_positions,
                    use_cache=False,
                )
                teacher_blind_out = (
                    teacher(
                        input_ids=full_ids,
                        attention_mask=additive_blind,
                        position_ids=teacher_positions,
                        use_cache=False,
                    )
                    if additive_blind is not None
                    else None
                )

            # Next-token shift: logits at position t predict token t+1. The answer
            # region on the input axis is [T-answer_len, T); its predictive logits
            # therefore live at [T-answer_len-1, T-1) on the logits axis.
            T = int(full_ids.shape[-1])
            lo = T - answer_len - 1
            hi = T - 1
            teacher_ans = teacher_out.logits[:, lo:hi, :].float()

            # Under DDP, suppress gradient all-reduce on non-boundary micro-steps —
            # only the backward that immediately precedes optimizer.step() should
            # sync. Otherwise we'd pay world_size × grad_accum_steps all-reduces
            # per optimizer step instead of one (issue #123).
            will_step = (accum_since_step + 1) >= grad_accum_steps or is_last_record
            components: list[tuple[str, float, torch.Tensor, torch.Tensor, torch.Tensor]] = [
                ("ddi_kl_loss", 1.0, additive_student, student_positions, teacher_ans)
            ]
            if full_anchor_weight > 0:
                components.append(
                    (
                        "full_anchor_loss",
                        full_anchor_weight,
                        additive_teacher,
                        teacher_positions,
                        teacher_ans,
                    )
                )
            if blind_anchor_weight > 0:
                assert additive_blind is not None and teacher_blind_out is not None
                components.append(
                    (
                        "blind_anchor_loss",
                        blind_anchor_weight,
                        additive_blind,
                        teacher_positions,
                        teacher_blind_out.logits[:, lo:hi, :].float(),
                    )
                )

            component_values: dict[str, float] = {}
            objective_value = 0.0
            for component_i, (name, weight, mask, positions, target_logits) in enumerate(
                components
            ):
                sync_this_backward = will_step and component_i == len(components) - 1
                sync_ctx = (
                    contextlib.nullcontext()
                    if (sync_this_backward or not is_dist)
                    else student.no_sync()
                )
                # Keep only one student activation graph resident at a time. With three
                # objective paths, summing graph-bearing losses before backward can exceed
                # an 80GB card even though each path fits independently.
                with sync_ctx:
                    student_out = student(
                        input_ids=full_ids,
                        attention_mask=mask,
                        position_ids=positions,
                        use_cache=False,
                    )
                    student_ans = student_out.logits[:, lo:hi, :].float()
                    component_loss = _answer_kl(
                        student_ans, target_logits, kl_temperature
                    )
                    weighted_loss = weight * component_loss
                    if name == "ddi_kl_loss" and teacher_first_token_weight > 0:
                        first_token_loss = _teacher_first_token_ce(student_ans, teacher_ans)
                        component_values["teacher_first_token_loss"] = float(
                            first_token_loss.detach().item()
                        )
                        weighted_loss = weighted_loss + (
                            teacher_first_token_weight * first_token_loss
                        )
                    (weighted_loss / grad_accum_steps).backward()
                value = float(component_loss.detach().item())
                component_values[name] = value
                objective_value += float(weighted_loss.detach().item())
            accum_since_step += 1

            if accum_since_step >= grad_accum_steps or is_last_record:
                current_lr, stepped = _flush_optimizer_step(
                    optimizer, warmup_steps, lr, global_step,
                    params=trainable_params, max_grad_norm=max_grad_norm,
                )
                global_step += 1
                accum_since_step = 0
                if not stepped:
                    skipped_steps += 1
                    _log(
                        f"step {global_step}: non-finite gradients, step SKIPPED "
                        f"({skipped_steps}/{max_skipped_steps} allowed)"
                    )
                    if skipped_steps > max_skipped_steps:
                        raise RuntimeError(
                            f"finetune: {skipped_steps} non-finite gradient steps "
                            f"exceeded max_skipped_steps={max_skipped_steps}; "
                            "training is diverging, aborting instead of saving "
                            "a checkpoint that cannot be trusted"
                        )

                # Always record the FIRST and LAST optimizer step of each epoch, not only
                # every `log_every`-th.
                #
                # `log_every` defaults to 50, so a short run recorded *nothing at all*:
                # VIBEQUEUE-2 row 8 trains on synthetic_qa alone, which yields 57 records
                # and ~7 optimizer steps per rank, so `global_step % 50` was never 0 and
                # the store received zero `kl_loss` rows. `ddi.aggregate._collect_finetune_loss`
                # then reported `backing_rows: 0, data: {}` — a finished 8-GPU job with no
                # loss curve to inspect, and no error anywhere to say why.
                #
                # The last step matters on long runs too: final loss is the number most
                # likely to be read, and it lands on a multiple of `log_every` only by luck.
                # Cost is bounded — at most two extra rows per epoch.
                hit_max_steps = max_steps is not None and global_step >= max_steps
                should_log = (
                    global_step == 1
                    or is_last_record
                    or hit_max_steps
                    or global_step % log_every == 0
                )
                if should_log and is_main:
                    step_rid = f"{run_prefix}:step={global_step}"
                    append(
                        step_rid, "objective_loss", objective_value, provenance,
                        store=store_path,
                    )
                    for metric, value in component_values.items():
                        append(step_rid, metric, value, provenance, store=store_path)
                    # Preserve the historical metric consumed by aggregate.py. It is the
                    # DDI KL component; objective_loss is the weighted total for R1 runs.
                    append(
                        step_rid, "kl_loss", component_values["ddi_kl_loss"], provenance,
                        store=store_path,
                    )
                    append(step_rid, "lr", current_lr, provenance, store=store_path)
                    _log(
                        f"step {global_step} loss={objective_value:.4f} lr={current_lr:.2e} "
                        f"(epoch {epoch + 1}, rec {rec_i + 1}/{len(train_records)})"
                    )

                # Periodic checkpoint every 5000 optimizer steps — survives crashes
                # that would otherwise lose an entire overnight run (VIBEQUEUE-2).
                # Only the latest is kept; the previous .stepN dir is removed to avoid
                # accumulating 15 GB per save (5 checkpoints × 15 GB = 75 GB).
                if is_main and global_step % 5000 == 0 and global_step > 0:
                    ckpt_path = output_dir.parent / f"{output_dir.name}.step{global_step}"
                    _log(f"periodic checkpoint -> {ckpt_path}")
                    student_module.save_pretrained(ckpt_path)
                    tokenizer.save_pretrained(ckpt_path)
                    # Remove previous periodic checkpoint(s) — keep only this one.
                    prev = sorted(
                        output_dir.parent.glob(f"{output_dir.name}.step*")
                    )
                    prev = [p for p in prev if p != ckpt_path]
                    for old in prev:
                        _log(f"removing old checkpoint {old}")
                        shutil.rmtree(old, ignore_errors=True)

                if max_steps is not None and global_step >= max_steps:
                    _log(f"max_steps={max_steps} reached; stopping early")
                    stop_early = True
                    break

        # ---- End-of-epoch perplexities (rank 0 only under DDP) ----
        if test_records and is_main:
            from ddi.perplexity import compute_perplexity

            _log(f"epoch {epoch + 1}: computing post-epoch perplexity")
            pplx_post = compute_perplexity(
                student_module, tokenizer, pplx_records,
                mask_variant="ddi_qf", layout="sequential",
            )
            student_module.train()
            epoch_rid = f"{run_prefix}:epoch={epoch + 1}"
            append(epoch_rid, "perplexity_student_ddi_qf_post", pplx_post,
                   provenance, store=store_path)
            _log(f"epoch {epoch + 1} post: student_ddi_qf={pplx_post:.3f}")
            if layout != "sequential":
                pplx_restart = compute_perplexity(
                    student_module, tokenizer, pplx_records,
                    mask_variant="ddi_qf", layout="restart",
                )
                student_module.train()
                append(epoch_rid, "perplexity_student_ddi_qf_restart", pplx_restart,
                       provenance, store=store_path)
                _log(f"epoch {epoch + 1} post: student_ddi_qf_restart={pplx_restart:.3f}")
        if is_dist:
            dist.barrier()

        if stop_early:
            break

    # ---- Save the trained student as an HF-loadable directory (rank 0 only) ----
    if is_main:
        _log(f"saving checkpoint -> {output_dir}")
        _assert_finite_weights(student_module, "before saving checkpoint")
        student_module.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    if is_dist:
        dist.barrier()
        dist.destroy_process_group()
    _log("finetune: done")
