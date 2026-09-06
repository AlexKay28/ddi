"""Full-model Flex versus SDPA parity diagnostics."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
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


def _install_torchvision_nms_shim() -> None:
    """Permit text-only Transformers imports with the known local torchvision mismatch."""
    global _torchvision_library
    _torchvision_library = torch.library.Library("torchvision", "DEF")
    _torchvision_library.define("nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--n-docs", type=int, default=2)
    parser.add_argument("--doc-length", type=int, default=128)
    parser.add_argument("--query-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--torchvision-nms-shim", action="store_true")
    args = parser.parse_args()

    if args.torchvision_nms_shim:
        _install_torchvision_nms_shim()
    from ddi.flex_attention import make_block_mask
    from ddi.inference import load
    from ddi.masks import build_mask, to_additive

    if not torch.cuda.is_available():
        raise RuntimeError("Flex model parity requires CUDA")

    torch.manual_seed(args.seed)
    reference, _ = load(args.model, dtype=args.dtype, backend="sdpa")
    candidate, _ = load(args.model, dtype=args.dtype, backend="flex")
    reference = reference.to("cuda").eval()
    candidate = candidate.to("cuda").eval()
    lengths = [args.doc_length] * args.n_docs
    total = sum(lengths) + args.query_length
    input_ids = torch.randint(
        0, reference.config.vocab_size, (1, total), device="cuda"
    )

    result = {
        "model": args.model,
        "dtype": args.dtype,
        "shape": {
            "n_docs": args.n_docs,
            "doc_length": args.doc_length,
            "query_length": args.query_length,
        },
        "seed": args.seed,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "cells": [],
    }
    for variant in ("full", "ddi_qf", "blind"):
        allow = build_mask(lengths, args.query_length, 0, variant)
        additive = to_additive(allow, dtype=reference.dtype).to("cuda")[None, None]
        candidate._flex_block_mask = make_block_mask(
            lengths, args.query_length, variant, device=torch.device("cuda")
        )
        with torch.no_grad():
            expected = reference(
                input_ids=input_ids, attention_mask=additive, use_cache=False
            ).logits
            actual = candidate(input_ids=input_ids, use_cache=False).logits
        metrics = compare_logits(expected, actual)
        result["cells"].append({"variant": variant, **asdict(metrics)})
    candidate._flex_block_mask = None
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
