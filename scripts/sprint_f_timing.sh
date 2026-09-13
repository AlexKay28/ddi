#!/usr/bin/env bash
# Sprint F (VIBEQUEUE-32): flex_attention realization benchmark.
# Historical harness only: its local correctness check predates and is weaker
# than the exact parity gates in EVIDENCE_TO_CLAIM.md. Do not use it for V49 or
# admissible Flex timing unless ddi.flex_parity first passes every unchanged
# gate for all applicable masks and paths. Current V47 parity fails.
#
# Runs the timing grid: Ns x Ls x {sdpa,flex} x {full,ddi_qf}.
# Correctness check: flex logits vs sdpa logits (bf16 tolerance).
# Also runs blind for completeness.
#
# Usage: bash scripts/sprint_f_timing.sh
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

STORE="results/sprint_f_timing.jsonl"
rm -f "$STORE"
touch "$STORE"

python3 -c "
import sys, json, time, collections
import torch
from ddi.inference import load
from ddi.masks import build_mask, to_additive
from ddi.flex_attention import make_block_mask
from ddi.run import _sparsity, _analytical_speedup, _mask_kwargs

model_name = 'Qwen/Qwen2.5-7B-Instruct'

# ---- Correctness check first (load one at a time to avoid OOM) ----
print('=== Correctness check: flex vs sdpa ===', file=sys.stderr)
model_sdpa, tok = load(model_name, backend='sdpa')
model_sdpa = model_sdpa.to('cuda').eval()

N, L, Q = 5, 512, 64
doc_lengths = [L] * N
T = N * L + Q
ids = torch.randint(0, model_sdpa.config.vocab_size, (1, T), device='cuda')

allow = build_mask(doc_lengths, query_len=Q, answer_len=0, variant='ddi_qf')
add = to_additive(allow, dtype=torch.bfloat16).to('cuda').unsqueeze(0).unsqueeze(0)

with torch.no_grad():
    out_sdpa = model_sdpa(input_ids=ids, attention_mask=add, use_cache=False)
sdpa_logits = out_sdpa.logits.clone()
del model_sdpa, out_sdpa
torch.cuda.empty_cache()

model_flex, _ = load(model_name, backend='flex')
model_flex = model_flex.to('cuda').eval()
bm = make_block_mask(doc_lengths, Q, 'ddi_qf', device='cuda')
with torch.no_grad():
    model_flex._flex_block_mask = bm
    out_flex = model_flex(input_ids=ids, attention_mask=add, use_cache=False)
    model_flex._flex_block_mask = None

delta = (sdpa_logits - out_flex.logits).abs()
max_diff = delta.max().item()
mean_diff = delta.mean().item()
argmax_agreement = (
    sdpa_logits[:, -1].argmax(dim=-1) == out_flex.logits[:, -1].argmax(dim=-1)
).float().mean().item()
print(f'  max logits diff: {max_diff:.6f}; mean: {mean_diff:.6f}; final-token argmax agreement: {argmax_agreement:.3f}')
if mean_diff > 0.1 or argmax_agreement != 1.0:
    raise RuntimeError('flex correctness check failed')
print('  PASS: flex matches sdpa at bf16 tolerance')

del model_flex, out_flex, sdpa_logits
torch.cuda.empty_cache()

# ---- Timing grid ----
print('=== Timing grid ===', file=sys.stderr)
results = []
Ns = [1, 3, 5, 9, 13, 25]
Ls = [128, 512, 2048]
Q = 64
n_repeats = 5
warmup = 3

# Load each backend once, reuse across cells
for backend in ['sdpa', 'flex']:
    model, _ = load(model_name, backend=backend)
    model = model.to('cuda').eval()
    for L in Ls:
        for N in Ns:
            doc_lengths = [L] * N
            T = N * L + Q
            ids = torch.randint(0, model.config.vocab_size, (1, T), device='cuda')
            sparsity = _sparsity(build_mask(doc_lengths, Q, 0, 'ddi_qf'))
            analytical = _analytical_speedup('ddi_qf', N, L, Q)

            for mask in ['full', 'ddi_qf']:
                allow = build_mask(doc_lengths, query_len=Q, answer_len=0, variant=mask)
                add = to_additive(allow, dtype=model.dtype).to('cuda').unsqueeze(0).unsqueeze(0)

                try:
                    if backend == 'flex':
                        bm = make_block_mask(doc_lengths, Q, mask, device='cuda')
                        model._flex_block_mask = bm
                    else:
                        model._flex_block_mask = None

                    # Warmup
                    for _ in range(warmup):
                        with torch.no_grad():
                            _ = model(input_ids=ids, attention_mask=add, use_cache=False)
                    torch.cuda.synchronize()

                    # Measure
                    times = []
                    for _ in range(n_repeats):
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        with torch.no_grad():
                            _ = model(input_ids=ids, attention_mask=add, use_cache=False)
                        torch.cuda.synchronize()
                        times.append((time.perf_counter() - t0) * 1000)
                    r = {
                        'backend': backend, 'mask': mask, 'N': N, 'L': L, 'Q': Q,
                        'ftt_ms': round(sorted(times)[len(times)//2], 1),
                        'sparsity': round(sparsity, 4),
                        'analytical_speedup': round(analytical, 2),
                        'status': 'ok',
                    }
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    r = {
                        'backend': backend, 'mask': mask, 'N': N, 'L': L, 'Q': Q,
                        'sparsity': round(sparsity, 4),
                        'analytical_speedup': round(analytical, 2), 'status': 'oom',
                    }
                results.append(r)
                with open('$STORE', 'a') as f:
                    f.write(json.dumps(r) + '\n')
                ftt_display = r.get('ftt_ms', 'OOM')
                print(f'  {backend:4s} {mask:7s} N={N:2d} L={L:4d}  FTT={ftt_display}ms  sparsity={sparsity:.2f}')

    if backend == 'flex':
        model._flex_block_mask = None
    del model
    torch.cuda.empty_cache()

print(f'Done. {len(results)} cells saved to $STORE')
"
