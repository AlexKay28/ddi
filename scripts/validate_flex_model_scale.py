"""Local-only full-model validation for direct sparse flex BlockMask construction.

The dense DDI reference cannot be materialized at N=25, L=2048. This script
therefore proves patched-model equivalence at a smaller shape, then exercises
the direct BlockMask prefill and one cached decode step at the target shape.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# Permit direct execution without requiring an editable install in the alternate runtime.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The local Python 3.10 environment has a torchvision wheel without its nms extension.
# Transformers probes torchvision during import although this text-only validation never uses it.
try:
    _ = torch.ops.torchvision.nms
except (AttributeError, RuntimeError):
    _TORCHVISION_LIBRARY = torch.library.Library("torchvision", "DEF")
    _TORCHVISION_LIBRARY.define("nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")

from ddi.flex_attention import make_block_mask
from ddi.inference import load
from ddi.masks import build_mask, to_additive


def _elapsed_ms(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end))


def _direct_prefill(model, input_ids: torch.Tensor, block_mask, *, use_cache: bool):
    model._flex_block_mask = block_mask  # type: ignore[attr-defined]
    with torch.no_grad():
        return model(input_ids=input_ids, use_cache=use_cache)


def _dense_prefill(model, input_ids: torch.Tensor, additive: torch.Tensor):
    model._flex_block_mask = None  # type: ignore[attr-defined]
    with torch.no_grad():
        return model(input_ids=input_ids, attention_mask=additive, use_cache=False)


def _proxy_correctness(model, vocab_size: int, device: torch.device) -> dict[str, object]:
    doc_lengths = [128, 128]
    query_len = 64
    total = sum(doc_lengths) + query_len
    input_ids = torch.arange(total, device=device).unsqueeze(0).remainder(vocab_size)
    block_mask = make_block_mask(doc_lengths, query_len, "ddi_qf", device=device)
    allow = build_mask(doc_lengths, query_len, 0, "ddi_qf")
    additive = to_additive(allow, dtype=model.dtype).to(device).unsqueeze(0).unsqueeze(0)

    direct = _direct_prefill(model, input_ids, block_mask, use_cache=True)
    reference = _dense_prefill(model, input_ids, additive)
    logits_delta = (direct.logits - reference.logits).float()
    direct_logits = direct.logits.float().flatten()
    reference_logits = reference.logits.float().flatten()
    cache_len = int(direct.past_key_values.get_seq_length())
    if cache_len != total:
        raise AssertionError(f"flex prefill cache length {cache_len}, expected {total}")

    # qf decode may attend to every prefix token, so one all-zero row is its exact DDI mask.
    next_token = direct.logits[:, -1].argmax(dim=-1, keepdim=True)
    model._flex_block_mask = None  # type: ignore[attr-defined]
    with torch.no_grad():
        direct_decode = model(
            input_ids=next_token,
            attention_mask=torch.zeros((1, 1, 1, total + 1), dtype=model.dtype, device=device),
            position_ids=torch.tensor([[total]], device=device),
            cache_position=torch.tensor([total], device=device),
            past_key_values=direct.past_key_values,
            use_cache=True,
        )
    return {
        "shape": {"N": 2, "L": 128, "Q": 64},
        "prefill_max_abs_error": float(logits_delta.abs().max().item()),
        "prefill_mean_abs_error": float(logits_delta.abs().mean().item()),
        "prefill_cosine_similarity": float(
            torch.nn.functional.cosine_similarity(direct_logits, reference_logits, dim=0).item()
        ),
        "prefill_argmax_agreement": float(
            (direct.logits.argmax(-1) == reference.logits.argmax(-1)).float().mean().item()
        ),
        "decode_finite": bool(torch.isfinite(direct_decode.logits).all()),
        "decode_token": int(direct_decode.logits[:, -1].argmax().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Local HF checkpoint directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not torch._dynamo.is_dynamo_supported():
        raise RuntimeError("torch.compile/Dynamo is unsupported in this environment")

    device = torch.device("cuda")
    torch.manual_seed(0)
    model, _ = load("Qwen/Qwen2.5-7B-Instruct", backend="flex", checkpoint_dir=args.checkpoint)
    model.eval().to(device)
    proxy = _proxy_correctness(model, model.config.vocab_size, device)
    if proxy["prefill_cosine_similarity"] < 0.999:
        raise AssertionError(f"proxy full-model equivalence failed: {proxy}")

    N, L, Q = 25, 2048, 64
    doc_lengths = [L] * N
    total = N * L + Q
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    block_mask = make_block_mask(doc_lengths, Q, "ddi_qf", device=device)
    torch.cuda.synchronize()
    construction = {
        "wall_ms": (time.monotonic() - started) * 1_000,
        "peak_mib": torch.cuda.max_memory_allocated(device) / 2**20,
    }
    input_ids = torch.randint(0, model.config.vocab_size, (1, total), device=device)

    # Compile before recording execution-only latency and peak memory.
    prefill = _direct_prefill(model, input_ids, block_mask, use_cache=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    timings = [
        _elapsed_ms(lambda: _direct_prefill(model, input_ids, block_mask, use_cache=False))
        for _ in range(args.repeats)
    ]
    runtime_peak_mib = torch.cuda.max_memory_allocated(device) / 2**20

    next_token = prefill.logits[:, -1].argmax(dim=-1, keepdim=True)
    model._flex_block_mask = None  # type: ignore[attr-defined]
    with torch.no_grad():
        decoded = model(
            input_ids=next_token,
            attention_mask=torch.zeros((1, 1, 1, total + 1), dtype=model.dtype, device=device),
            position_ids=torch.tensor([[total]], device=device),
            cache_position=torch.tensor([total], device=device),
            past_key_values=prefill.past_key_values,
            use_cache=True,
        )
    result = {
        "environment": {
            "python": __import__("sys").version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "dynamo_supported": torch._dynamo.is_dynamo_supported(),
        },
        "proxy_full_model_correctness": proxy,
        "target_shape": {"N": N, "L": L, "Q": Q, "T": total},
        "block_mask_construction": construction,
        "target_prefill": {
            "latencies_ms": timings,
            "median_ms": sorted(timings)[len(timings) // 2],
            "runtime_peak_mib": runtime_peak_mib,
            "logits_finite": bool(torch.isfinite(prefill.logits).all()),
        },
        "target_decode": {
            "finite": bool(torch.isfinite(decoded.logits).all()),
            "token": int(decoded.logits[:, -1].argmax().item()),
        },
        "boundary": (
            "Target-shape dense logits equivalence is intentionally not attempted: it would "
            "materialize the dense DDI mask/attention path this validation replaces."
        ),
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
