"""Single-GPU bisect for non-finite gradients in the distillation step.

Written for VIBEQUEUE-2 / Qwen2.5-7B-Instruct, which produces non-finite gradients on
**step 1**, at ``lr=1e-7``, with a finite loss of 0.0006 — i.e. before the optimiser has
touched a weight. That is not divergence; it is the backward pass itself. The same trainer,
revision and settings train Qwen3.5-4B and Qwen3.6-27B without a single skipped step.

The one structural difference between those models is how many layers the DDI mask reaches:

    Qwen2.5-7B-Instruct   Qwen2ForCausalLM    28 of 28 layers are softmax attention
    Qwen3.5-4B-Instruct   Qwen3_5             8 of 32   (the rest are linear_attention)
    Qwen3.6-27B-Instruct  Qwen3_5             16 of 64, and the frozen-trunk run trained 2

So the 7B does ~3.5x more masked-softmax backward per forward than the 4B, and ~14x more
than the 27B run that succeeded. This script varies one thing at a time to find which of
those axes actually carries the NaN.

Runs one real record through the exact `finetune` inner step under a matrix of toggles and
reports, per condition: the loss, the pre-clip gradient norm, and the first parameter whose
gradient is not finite. Nothing is optimised and nothing is written — it is read-only with
respect to every checkpoint.

    python -m ddi.diagnose_grads --model Qwen/Qwen2.5-7B-Instruct --dataset hotpot_qa

Cheap knobs for bisecting depth: ``--truncate-layers N`` keeps only the first N decoder
layers (logits become meaningless, gradient finiteness does not), and ``--records N``
repeats the sweep over N different records so a per-record data effect can be ruled out.
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import os

import torch

from ddi import inference
from ddi.data import load as load_dataset
from ddi.finetune import _encode_record, _prefilter_records
from ddi.masks import build_mask
from ddi.positions import build_positions

_FILLS = {"-inf": float("-inf"), "finfo_min": None}  # None → torch.finfo(dtype).min


def _additive(doc_lengths, query_len, answer_len, variant, dtype, device, fill):
    """`ddi.finetune._build_additive`, with the masked fill value under our control.

    `ddi.masks.to_additive` hardcodes `float("-inf")`. HF's own builders
    (`masking_utils.eager_mask`) use `torch.finfo(dtype).min` instead, and never `-inf`.
    A `-inf` entry is the documented way to get NaN out of a fused attention backward when
    a whole key tile is masked — which is exactly the shape `ddi_qf` produces, since the
    doc_i x doc_j blocks are rectangular and large.
    """
    allow = build_mask(doc_lengths, query_len=query_len, answer_len=answer_len, variant=variant)
    value = _FILLS[fill]
    if value is None:
        value = torch.finfo(dtype).min
    out = torch.full(allow.shape, value, dtype=dtype)
    return out.masked_fill(allow, 0.0).to(device).unsqueeze(0).unsqueeze(0)


def _sdpa_ctx(backend: str):
    if backend == "default":
        return contextlib.nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel

    return sdpa_kernel({"math": SDPBackend.MATH,
                        "efficient": SDPBackend.EFFICIENT_ATTENTION,
                        "flash": SDPBackend.FLASH_ATTENTION}[backend])


def _one_step(student, teacher, tok, record, *, variant, fill, backend, kl_temperature, device):
    """Forward + backward one record. Returns (loss, grad_norm, first_bad_param)."""
    full_ids, doc_lengths, query_len, answer_len = _encode_record(tok, record)
    dtype = student.dtype
    full_ids = full_ids.to(device)
    positions = build_positions(doc_lengths, query_len, answer_len, "sequential")
    positions = positions.unsqueeze(0).to(device)

    m_teacher = _additive(doc_lengths, query_len, answer_len, "full", dtype, device, fill)
    m_student = _additive(doc_lengths, query_len, answer_len, variant, dtype, device, fill)

    student.zero_grad(set_to_none=True)
    with _sdpa_ctx(backend):
        with torch.no_grad():
            t_out = teacher(input_ids=full_ids, attention_mask=m_teacher,
                            position_ids=positions, use_cache=False)
        s_out = student(input_ids=full_ids, attention_mask=m_student,
                        position_ids=positions, use_cache=False)

        T = int(full_ids.shape[-1])
        lo, hi = T - answer_len - 1, T - 1
        t_ans = t_out.logits[:, lo:hi, :].float()
        s_ans = s_out.logits[:, lo:hi, :].float()

        logits_finite = bool(torch.isfinite(s_ans).all()) and bool(torch.isfinite(t_ans).all())

        log_p = torch.nn.functional.log_softmax(s_ans / kl_temperature, dim=-1)
        p_t = torch.nn.functional.softmax(t_ans / kl_temperature, dim=-1)
        loss = torch.nn.functional.kl_div(
            log_p.reshape(-1, log_p.shape[-1]),
            p_t.reshape(-1, p_t.shape[-1]),
            reduction="batchmean",
        ) * (kl_temperature ** 2)
        loss.backward()

    params = [p for p in student.parameters() if p.requires_grad and p.grad is not None]
    norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
    bad = next(
        (n for n, p in student.named_parameters()
         if p.grad is not None and not torch.isfinite(p.grad).all()),
        None,
    )
    n_bad = sum(
        1 for _, p in student.named_parameters()
        if p.grad is not None and not torch.isfinite(p.grad).all()
    )
    student.zero_grad(set_to_none=True)
    return {
        "loss": float(loss),
        "logits_finite": logits_finite,
        "grad_norm": float(norm),
        "grad_finite": bool(torch.isfinite(norm)),
        "n_bad": n_bad,
        "first_bad": bad,
        "seq_len": T,
        "docs": len(doc_lengths),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="hotpot_qa")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--records", type=int, default=3)
    ap.add_argument("--min-docs", type=int, default=2)
    ap.add_argument("--kl-temperature", type=float, default=1.0)
    ap.add_argument("--truncate-layers", type=int, default=0,
                    help="keep only the first N decoder layers (0 = whole model)")
    ap.add_argument("--fills", default="-inf,finfo_min")
    ap.add_argument("--backends", default="default,math")
    ap.add_argument("--variants", default="ddi_qf,full")
    ap.add_argument("--grad-checkpoint", default="0,1")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[diagnose] torch={torch.__version__} device={device} model={args.model}")

    teacher, tok = inference.load(args.model)
    student, _ = inference.load(args.model)
    teacher.to(device).eval()
    student.to(device).train()
    for p in teacher.parameters():
        p.requires_grad_(False)

    if args.truncate_layers:
        for m in (teacher, student):
            m.model.layers = m.model.layers[: args.truncate_layers]
        print(f"[diagnose] truncated to {args.truncate_layers} decoder layers")

    layer_types = getattr(getattr(student, "config", None), "layer_types", None)
    if layer_types is None:
        layer_types = getattr(getattr(student.config, "text_config", None), "layer_types", None)
    if layer_types:
        n_full = sum(1 for t in layer_types if t == "full_attention")
        print(f"[diagnose] layer_types: {n_full} full_attention of {len(layer_types)} layers "
              "— the DDI mask reaches only those")
    else:
        n_layers = len(student.model.layers)
        print(f"[diagnose] no layer_types: all {n_layers} layers are softmax attention")

    records = list(load_dataset(args.dataset, "train", args.seed))
    records = _prefilter_records(records, min_docs=args.min_docs)[: args.records]
    print(f"[diagnose] {len(records)} record(s) from {args.dataset}\n")

    header = (f"{'rec':>3} {'variant':8} {'fill':10} {'sdpa':9} {'ckpt':5} "
              f"{'loss':>10} {'grad_norm':>12} {'ok':>5} {'n_bad':>6}  first_bad")
    print(header)
    print("-" * len(header))

    combos = itertools.product(
        range(len(records)),
        args.variants.split(","),
        args.fills.split(","),
        args.backends.split(","),
        args.grad_checkpoint.split(","),
    )
    rows = []
    for i, variant, fill, backend, ckpt in combos:
        cond = {"record": i, "variant": variant, "fill": fill,
                "sdpa": backend, "grad_checkpoint": ckpt}
        if ckpt == "1":
            student.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        else:
            student.gradient_checkpointing_disable()
        try:
            r = _one_step(student, teacher, tok, records[i], variant=variant, fill=fill,
                          backend=backend, kl_temperature=args.kl_temperature, device=device)
        except Exception as exc:  # a backend that refuses these shapes is a result too
            rows.append({**cond, "error": f"{type(exc).__name__}: {exc}"})
            print(f"{i:>3} {variant:8} {fill:10} {backend:9} {ckpt:5} "
                  f"{'ERROR':>10}  {type(exc).__name__}: {str(exc)[:60]}")
            continue
        rows.append({**cond, **r})
        print(f"{i:>3} {variant:8} {fill:10} {backend:9} {ckpt:5} "
              f"{r['loss']:10.6f} {r['grad_norm']:12.4e} {str(r['grad_finite']):>5} "
              f"{r['n_bad']:>6}  {r['first_bad'] or ''}")

    # `logs/*.json` is the block's json_output contract (nirvana/graphs/scripts.py's
    # _EMIT_JSON_OUTPUT fails the block when the directory is empty), so the sweep has to
    # leave one behind even though its real product is the table above.
    os.makedirs("logs", exist_ok=True)
    summary = {
        "kind": "diagnose_grads",
        "model": args.model,
        "dataset": args.dataset,
        "torch": torch.__version__,
        "truncate_layers": args.truncate_layers,
        "layer_types": list(layer_types) if layer_types else None,
        "rows": rows,
    }
    with open("logs/diagnose_grads.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    n_bad_conditions = sum(1 for r in rows if r.get("grad_finite") is False)
    print(f"\n[diagnose] {n_bad_conditions}/{len(rows)} conditions produced non-finite gradients")


if __name__ == "__main__":
    main()
