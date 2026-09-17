#!/usr/bin/env python3
"""V101 (VIBEQUEUE follow-up): measure SDPA backend dispatch under 4D additive masks.

The paper currently *qualifies* the realization-gap explanation ("consistent
with the slow math backend; not kernel-profiled"). This script measures it:

1. Build the N=25, L=512, Q=64 prefill batch (the paper's 95%-sparsity shape).
2. Run the same forward under three masks: 2D causal (HF default path),
   4D additive full-causal (byte-equivalent restriction), 4D additive ddi_qf.
3. Profile kernels with torch.profiler and report per-run total time plus the
   attention kernel names actually dispatched (efficient_attention / flash /
   math (unrolled) / cuDNN).

Output: results/v101_sdpa_dispatch_<ts>/summary.json
"""
from __future__ import annotations

import json
import re
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from ddi import inference  # noqa: E402
from ddi.masks import build_mask, to_additive  # noqa: E402
from ddi.prompt import encode_segments  # noqa: E402

N_DOCS = 25
DOC_LEN = 512
QUERY_LEN = 64
WARMUP = 3
TIMED = 10


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fake_docs(tokenizer) -> list[str]:
    # fixed filler text tokenizing to ~DOC_LEN tokens per document
    filler = "The quick brown fox jumps over the lazy dog. " * 90
    docs = []
    for _ in range(N_DOCS):
        ids = tokenizer(filler, add_special_tokens=False).input_ids[:DOC_LEN]
        docs.append(tokenizer.decode(ids))
    return docs


def _additive_from_variant(model, doc_lengths, query_len, variant):
    allow = build_mask(doc_lengths, query_len=query_len, answer_len=0, variant=variant)
    return to_additive(allow, dtype=model.dtype).to(model.device).unsqueeze(0).unsqueeze(0)


def _full_causal_additive(length, dtype, device):
    causal = torch.ones((length, length), dtype=torch.bool, device=device).tril()
    return (
        torch.zeros_like(causal, dtype=dtype).masked_fill_(~causal, -torch.inf)[None, None]
    )


def profile_case(model, ids, positions, attention_mask, label) -> dict:
    with torch.no_grad():
        for _ in range(WARMUP):
            model(input_ids=ids, attention_mask=attention_mask, position_ids=positions)
    torch.cuda.synchronize()

    times = []
    with torch.no_grad(), torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        for _ in range(TIMED):
            model(input_ids=ids, attention_mask=attention_mask, position_ids=positions)
        torch.cuda.synchronize()
    for event in prof.key_averages():
        if event.device_type == torch.profiler.DeviceType.CUDA:
            times.append(event.self_device_time_total)

    kernels = {}
    for event in prof.events():
        name = event.name
        if event.device_type == torch.profiler.DeviceType.CUDA and event.self_device_time_total > 0:
            lowered = name.lower()
            if "mem_efficient" in lowered or "memeff" in lowered or "efficient_attention" in lowered:
                key = "sdpa_mem_efficient"
            elif "flash" in lowered:
                key = "sdpa_flash"
            elif "cudnn" in lowered:
                key = "sdpa_cudnn"
            elif "math" in lowered or "unrolled" in lowered or "softmax" in lowered or "bmm" in lowered or "matmul" in lowered:
                key = "math_path:" + re.sub(r"[^a-z0-9_]+", "_", lowered)[:48]
            else:
                continue
            kernels[key] = kernels.get(key, 0.0) + event.self_device_time_total
    return {"label": label, "kernels_us": dict(sorted(kernels.items(), key=lambda kv: -kv[1]))}


def main() -> int:
    out_dir = ROOT / "results" / f"v101_sdpa_dispatch_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir.mkdir(parents=True, exist_ok=False)

    model, tokenizer = inference.load("Qwen/Qwen2.5-7B-Instruct", dtype="bf16", backend="sdpa")
    model = model.to("cuda").eval()

    docs = _fake_docs(tokenizer)
    query = "Summarize the documents in one word:"
    ids, doc_lengths, query_len, _ = encode_segments(tokenizer, docs, query)
    length = int(ids.shape[-1])
    positions = torch.arange(length, device="cuda").unsqueeze(0)
    ids = ids.to("cuda")

    mask_2d = torch.ones((1, length), dtype=torch.long, device="cuda")
    additive_full = _full_causal_additive(length, model.dtype, "cuda")
    additive_ddi = _additive_from_variant(model, doc_lengths, query_len, "ddi_qf")

    sparsity = 1.0 - sum((l + 1) * l / 2 + query_len * l for l in doc_lengths) / (
        (length + 1) * length / 2
    )

    cases = [
        ("2d_causal_hf_default", mask_2d),
        ("4d_additive_full_causal", additive_full),
        ("4d_additive_ddi_qf", additive_ddi),
    ]
    results = []
    for label, mask in cases:
        started = time.perf_counter()
        case = profile_case(model, ids, positions, mask, label)
        case["wall_s_per_call"] = (time.perf_counter() - started - WARMUP * 0) / TIMED
        results.append(case)
        print(f"{label}: kernels={list(case['kernels_us'])[:4]}")

    summary = {
        "schema": "v101-sdpa-dispatch-v1",
        "created_utc": _utc(),
        "shape": {"n_docs": N_DOCS, "doc_len": DOC_LEN, "query_len": QUERY_LEN, "total": length},
        "pair_sparsity": sparsity,
        "runtime": {
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "dtype": "bfloat16",
            "backend": "sdpa",
            "warmup": WARMUP,
            "timed": TIMED,
        },
        "cases": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"done: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
