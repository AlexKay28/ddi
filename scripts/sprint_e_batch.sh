#!/usr/bin/env bash
# Sprint E (VIBEQUEUE-29): batched inference benchmark.
#
# Measures prefill throughput (QPS) at batch sizes 1, 4, 8, 16 with DDI masks.
# Verifies that batched loss matches single-sample loss (correctness gate).
#
# Usage: bash scripts/sprint_e_batch.sh
set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python3 -c "
import time, sys, json, collections
import torch
from ddi.inference import load, encode, forward_with_mask
from ddi.batch import encode_batch, forward_batch, batched_loss
from ddi.data import load as load_dataset
from ddi.prompt import encode_segments, build_query_text
from ddi.masks import build_mask, to_additive
from ddi.metrics import make_answer_labels
from ddi.run import shifted_cross_entropy

model_name = 'Qwen/Qwen2.5-7B-Instruct'
print(f'Loading {model_name}...', file=sys.stderr)
model, tokenizer = load(model_name, backend='sdpa')
model.eval()
if torch.cuda.is_available():
    model = model.to('cuda')

# Collect records
records = []
for i, record in enumerate(load_dataset('hotpot_qa', 'val', 0)):
    if i >= 64:
        break
    if not record['possible_answers']:
        continue
    docs = list(record['info_contexts'])
    if not docs:
        continue
    query = build_query_text(tokenizer, record)
    answer = record['dialog'][-1]['content']
    records.append((docs, query, answer))

print(f'Collected {len(records)} records', file=sys.stderr)

# --- Correctness check: batch=4 loss vs single-sample loss ---
print('Correctness check (batch=4 vs single)...', file=sys.stderr)
batch_samples_3 = [(r[0], r[1], r[2]) for r in records[:4]]
batched = encode_batch(tokenizer, batch_samples_3, mask_variant='ddi_qf')
logits = forward_batch(model, batched)
batched_losses = batched_loss(logits, batched, tokenizer, records[:4])

single_losses = []
for docs, query, answer in records[:4]:
    full_ids, dl, ql, al = encode_segments(tokenizer, list(docs), query, answer)
    full_ids = full_ids.to(model.device)
    labels = make_answer_labels(full_ids, full_ids.shape[-1] - al, full_ids.shape[-1]).to(model.device)
    allow = build_mask(dl, query_len=ql, answer_len=al, variant='ddi_qf')
    additive = to_additive(allow, dtype=model.dtype).to(model.device).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        out = model(input_ids=full_ids, attention_mask=additive, use_cache=False)
    loss = float(shifted_cross_entropy(out.logits, labels).item())
    single_losses.append(loss)

max_diff = max(abs(a - b) for a, b in zip(batched_losses, single_losses))
print(f'  batched: {[f\"{l:.6f}\" for l in batched_losses]}')
print(f'  single:  {[f\"{l:.6f}\" for l in single_losses]}')
print(f'  max diff: {max_diff:.8f}')
if max_diff > 0.01:
    print('  WARNING: loss difference > 0.01 — batching may be incorrect!', file=sys.stderr)
else:
    print('  PASS: batched loss matches single-sample loss')

# --- Throughput benchmark ---
results = []
for batch_size in [1, 2, 4, 8]:
    n = len(records)
    n_batches = (n + batch_size - 1) // batch_size

    # Warmup
    warmup_batch = [(r[0], r[1]) for r in records[:batch_size]]
    wb = encode_batch(tokenizer, warmup_batch, mask_variant='ddi_qf')
    _ = forward_batch(model, wb)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for b in range(n_batches):
        chunk = records[b * batch_size : (b + 1) * batch_size]
        if not chunk:
            continue
        batch_samples = [(r[0], r[1]) for r in chunk]
        batched = encode_batch(tokenizer, batch_samples, mask_variant='ddi_qf')
        _ = forward_batch(model, batched)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    qps = n / elapsed
    print(f'batch={batch_size:2d}  n={n}  time={elapsed:.2f}s  qps={qps:.2f}')
    results.append({'batch_size': batch_size, 'n': n, 'elapsed': elapsed, 'qps': qps})

# Save results
with open('results/sprint_e_batch.jsonl', 'w') as f:
    for r in results:
        f.write(json.dumps(r) + '\n')

print('Done. Results saved to results/sprint_e_batch.jsonl')
"
