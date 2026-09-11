"""Full-model Flex versus SDPA parity diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

_torchvision_library: Any = None


@dataclass(frozen=True)
class ParityMetrics:
    max_abs_error: float
    mean_abs_error: float
    cosine_similarity: float
    token_argmax_agreement: float


COSINE_GATE = 0.999
ARGMAX_GATE = 1.0


def compare_logits(reference: torch.Tensor, candidate: torch.Tensor) -> ParityMetrics:
    """Summarize model-level parity without relaxing the preregistered gate."""
    if reference.shape != candidate.shape:
        raise ValueError(f"logit shapes differ: {reference.shape} != {candidate.shape}")
    if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
        raise ValueError("logits must be finite")
    reference = reference.float()
    candidate = candidate.float()
    delta = (reference - candidate).abs()
    return ParityMetrics(
        max_abs_error=float(delta.max().item()),
        mean_abs_error=float(delta.mean().item()),
        cosine_similarity=float(
            F.cosine_similarity(reference.flatten(), candidate.flatten(), dim=0).item()
        ),
        token_argmax_agreement=float(
            (reference.argmax(dim=-1) == candidate.argmax(dim=-1)).float().mean().item()
        ),
    )


def passes_prefill_gate(metrics: ParityMetrics) -> bool:
    return (
        metrics.cosine_similarity >= COSINE_GATE
        and metrics.token_argmax_agreement == ARGMAX_GATE
    )


def first_divergence(reference: list[dict], candidate: list[dict]) -> dict[str, Any] | None:
    """Return the first layer whose attention output is not bit-identical."""
    if len(reference) != len(candidate):
        raise ValueError("diagnostic layer counts differ")
    for expected, actual in zip(reference, candidate, strict=True):
        if expected["layer"] != actual["layer"]:
            raise ValueError("diagnostic layer ordering differs")
        if not torch.equal(expected["attention_output"], actual["attention_output"]):
            return {
                "layer": expected["layer"],
                "scale": expected["scale"],
                "qkv_equal": {
                    name: bool(torch.equal(expected[name], actual[name]))
                    for name in ("query", "key", "value")
                },
                "max_attention_abs_error": float(
                    (expected["attention_output"].float() - actual["attention_output"].float())
                    .abs()
                    .max()
                ),
                "reference": expected,
                "candidate": actual,
            }
    return None


def _run_prefill(model, input_ids, positions, attention_mask, block_mask, diagnostics=None):
    model._flex_block_mask = block_mask
    model._flex_diagnostics = diagnostics
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=positions,
            use_cache=True,
        )
    model._flex_diagnostics = None
    return output


def _run_cell(model, input_ids, positions, lengths, query_len, variant, *, capture=False):
    from ddi.flex_attention import make_block_mask
    from ddi.masks import build_mask, to_additive

    total = input_ids.shape[-1]
    allow = build_mask(lengths, query_len, 1, variant)
    additive = to_additive(allow[:total, :total], dtype=model.dtype).to(model.device)[None, None]
    block_mask = make_block_mask(lengths, query_len, variant, device=model.device)
    reference_diagnostics = [] if capture else None
    candidate_diagnostics = [] if capture else None
    reference = _run_prefill(
        model, input_ids, positions, additive, None, reference_diagnostics
    )
    candidate = _run_prefill(
        model, input_ids, positions, None, block_mask, candidate_diagnostics
    )
    metrics = compare_logits(reference.logits, candidate.logits)
    reference_cache_len = int(reference.past_key_values.get_seq_length())
    candidate_cache_len = int(candidate.past_key_values.get_seq_length())

    next_token = reference.logits[:, -1].argmax(dim=-1, keepdim=True)
    decode_mask = to_additive(allow[total : total + 1, : total + 1], dtype=model.dtype)
    decode_mask = decode_mask.to(model.device)[None, None]
    decode_position = torch.tensor([[total]], device=model.device)
    cache_position = torch.tensor([total], device=model.device)
    model._flex_block_mask = None
    model._flex_diagnostics = None
    with torch.no_grad():
        reference_decode = model(
            input_ids=next_token,
            attention_mask=decode_mask,
            position_ids=decode_position,
            cache_position=cache_position,
            past_key_values=reference.past_key_values,
            use_cache=True,
        )
        candidate_decode = model(
            input_ids=next_token,
            attention_mask=decode_mask,
            position_ids=decode_position,
            cache_position=cache_position,
            past_key_values=candidate.past_key_values,
            use_cache=True,
        )
    decode_metrics = compare_logits(reference_decode.logits, candidate_decode.logits)
    cache_lengths_equal = reference_cache_len == candidate_cache_len == total
    decode_argmax_equal = bool(
        torch.equal(
            reference_decode.logits[:, -1].argmax(-1),
            candidate_decode.logits[:, -1].argmax(-1),
        )
    )
    result = {
        **asdict(metrics),
        "reference_cache_length": reference_cache_len,
        "candidate_cache_length": candidate_cache_len,
        "cache_lengths_equal": cache_lengths_equal,
        "decode": asdict(decode_metrics),
        "decode_argmax_equal": decode_argmax_equal,
    }
    result["pass"] = passes_prefill_gate(metrics) and cache_lengths_equal and decode_argmax_equal
    return result, reference_diagnostics, candidate_diagnostics


def _install_torchvision_nms_shim() -> None:
    """Permit text-only Transformers imports with the known local torchvision mismatch."""
    global _torchvision_library
    _torchvision_library = torch.library.Library("torchvision", "DEF")
    _torchvision_library.define("nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--n-docs", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--doc-length", type=int, default=128)
    parser.add_argument("--query-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--torchvision-nms-shim", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--failure-artifact", type=Path)
    args = parser.parse_args()

    if args.torchvision_nms_shim:
        _install_torchvision_nms_shim()
    from ddi.inference import load
    from ddi.positions import build_positions

    if not torch.cuda.is_available():
        raise RuntimeError("Flex model parity requires CUDA")

    torch.manual_seed(args.seed)
    model, _ = load(args.model, dtype=args.dtype, backend="flex")
    model = model.to("cuda").eval()

    result = {
        "model": args.model,
        "dtype": args.dtype,
        "seed": args.seed,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "gates": {
            "cosine_similarity": COSINE_GATE,
            "token_argmax_agreement": ARGMAX_GATE,
        },
        "cells": [],
    }
    artifact_written = False
    for n_docs in args.n_docs:
        lengths = [args.doc_length] * n_docs
        total = sum(lengths) + args.query_length
        input_ids = torch.randint(0, model.config.vocab_size, (1, total), device="cuda")
        positions = build_positions(lengths, args.query_length, 0, "sequential")
        positions = positions.to("cuda").unsqueeze(0)
        for variant in ("full", "ddi_qf", "blind"):
            cell, _, _ = _run_cell(
                model, input_ids, positions, lengths, args.query_length, variant
            )
            cell.update(
                variant=variant,
                shape={
                    "n_docs": n_docs,
                    "doc_length": args.doc_length,
                    "query_length": args.query_length,
                },
            )
            result["cells"].append(cell)
            if not cell["pass"] and args.failure_artifact is not None and not artifact_written:
                _, reference_diag, candidate_diag = _run_cell(
                    model,
                    input_ids,
                    positions,
                    lengths,
                    args.query_length,
                    variant,
                    capture=True,
                )
                divergence = first_divergence(reference_diag, candidate_diag)
                payload = {
                    "cell": cell,
                    "input_ids": input_ids.cpu(),
                    "position_ids": positions.cpu(),
                    "first_divergence": divergence,
                }
                args.failure_artifact.parent.mkdir(parents=True, exist_ok=True)
                torch.save(payload, args.failure_artifact)
                cell["failure_artifact"] = str(args.failure_artifact)
                cell["failure_artifact_sha256"] = hashlib.sha256(
                    args.failure_artifact.read_bytes()
                ).hexdigest()
                if divergence is not None:
                    cell["first_divergent_layer"] = divergence["layer"]
                    cell["first_divergent_qkv_equal"] = divergence["qkv_equal"]
                artifact_written = True
    model._flex_block_mask = None
    result["pass"] = all(cell["pass"] for cell in result["cells"])
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            handle.write(rendered)
    print(rendered, end="")
    if not result["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
