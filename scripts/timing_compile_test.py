#!/usr/bin/env python3
"""Quick test: does torch.compile realize speedup from block-diagonal mask sparsity?

Tests a few key (N, L) cells with sdpa vs torch.compile(sdpa) vs torch.compile(sdpa, mode=reduce-overhead).
Compares prefill FTT for full vs ddi_qf masks.

Hypothesis: torch.compile fuses the mask into the attention kernel, potentially
skipping masked QK^T blocks. If realized speedup > 1.5x at high sparsity, it's worth pursuing.
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"

import time
import torch
from ddi.inference import load
from ddi.masks import build_mask, to_additive
from ddi.model import forward_with_mask

MODEL = "results/checkpoints/Qwen__Qwen2.5-7B-Instruct__ddi_qf__8L"
CELLS = [
    # (N, L, sparsity_expected)
    (3, 512, 0.61),
    (9, 512, 0.87),
    (25, 512, 0.95),
    (25, 2048, 0.96),
]
Q = 64
N_REPEATS = 5
WARMUP = 3


def measure_prefill_ms(model, input_ids, additive_mask, n_repeats, warmup):
    times = []
    for i in range(warmup + n_repeats):
        torch.cuda.empty_cache()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            forward_with_mask(model, input_ids, additive_mask)
        end.record()
        end.synchronize()
        if i >= warmup:
            times.append(start.elapsed_time(end))
    times.sort()
    return sum(times) / len(times)


def run_cell(model, N, L, label):
    T = N * L + Q
    input_ids = torch.randint(0, 152064, (1, T), device="cuda")
    doc_lengths = [L] * N

    results = {}

    for mask_name in ["full", "ddi_qf"]:
        allow = build_mask(doc_lengths, query_len=Q, answer_len=0, variant=mask_name)
        additive = to_additive(allow, dtype=torch.bfloat16).to("cuda").unsqueeze(0).unsqueeze(0)
        ms = measure_prefill_ms(model, input_ids, additive, N_REPEATS, WARMUP)
        sparsity = 1.0 - allow.float().mean().item()
        results[mask_name] = {"ms": ms, "sparsity": sparsity}

    full_ms = results["full"]["ms"]
    ddi_ms = results["ddi_qf"]["ms"]
    realized = full_ms / ddi_ms if ddi_ms > 0 else 0
    analytical = 1.0 / (1.0 - results["ddi_qf"]["sparsity"]) if results["ddi_qf"]["sparsity"] > 0 else 1.0

    print(f"  {label:30s}  full={full_ms:8.2f}ms  ddi_qf={ddi_ms:8.2f}ms  "
          f"realized={realized:.2f}x  analytical={analytical:.1f}x  "
          f"sparsity={results['ddi_qf']['sparsity']:.3f}")
    return realized


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"torch: {torch.__version__}")
    print()

    # --- Baseline: sdpa (no compile) ---
    print("=== sdpa (no compile) ===")
    model, _ = load(MODEL, backend="sdpa")
    model.eval()
    model = model.to("cuda")

    baseline = {}
    for N, L, _ in CELLS:
        r = run_cell(model, N, L, f"N={N} L={L}")
        baseline[(N, L)] = r

    del model
    torch.cuda.empty_cache()
    print()

    # --- torch.compile default ---
    print("=== torch.compile (default mode) ===")
    model, _ = load(MODEL, backend="sdpa")
    model.eval()
    model = model.to("cuda")
    model = torch.compile(model)

    # warmup compile (first call triggers compilation)
    print("  (compiling...)")
    dummy_ids = torch.randint(0, 152064, (1, 3 * 512 + Q), device="cuda")
    dummy_allow = build_mask([512] * 3, query_len=Q, answer_len=0, variant="ddi_qf")
    dummy_mask = to_additive(dummy_allow, dtype=torch.bfloat16).to("cuda").unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        forward_with_mask(model, dummy_ids, dummy_mask)
    torch.cuda.synchronize()

    compiled_default = {}
    for N, L, _ in CELLS:
        r = run_cell(model, N, L, f"N={N} L={L}")
        compiled_default[(N, L)] = r

    del model
    torch.cuda.empty_cache()
    print()

    # --- torch.compile reduce-overhead ---
    print("=== torch.compile (mode=reduce-overhead) ===")
    model, _ = load(MODEL, backend="sdpa")
    model.eval()
    model = model.to("cuda")
    model = torch.compile(model, mode="reduce-overhead")

    print("  (compiling...)")
    with torch.no_grad():
        forward_with_mask(model, dummy_ids, dummy_mask)
    torch.cuda.synchronize()

    compiled_ro = {}
    for N, L, _ in CELLS:
        r = run_cell(model, N, L, f"N={N} L={L}")
        compiled_ro[(N, L)] = r

    del model
    torch.cuda.empty_cache()
    print()

    # --- Summary ---
    print("=== SUMMARY ===")
    print(f"{'N':>4} {'L':>5}  {'sdpa':>8} {'compile':>8} {'ro':>8}  {'an_sp':>6}")
    print("-" * 50)
    for N, L, exp_sp in CELLS:
        b = baseline.get((N, L), 0)
        c = compiled_default.get((N, L), 0)
        r = compiled_ro.get((N, L), 0)
        print(f"{N:>4} {L:>5}  {b:>8.2f} {c:>8.2f} {r:>8.2f}  {1.0/(1.0-exp_sp):>6.1f}")


if __name__ == "__main__":
    main()
