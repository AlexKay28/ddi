# V53 Exact-Work Cache Validation

## Reproduction and root cause

On 2026-09-07 the local V51 Qwen2.5-7B tokenizer realizes seed-0 nominal
500-token documents as 513--555 tokens (32 documents). IDs are decoded with
special-token removal, then retokenized. `ddi/run.py` reports requested rather
than realized geometry. Full causal runs A decode forwards; shipped DDI stops
on EOS and returns text without retained output counts. The small V51 behavior
diagnostic is not the timed workload. Cache identities omit checkpoint content,
tokenizer and position configuration. These are measurement-contract failures;
they do not by themselves prove that document KV reuse is incorrect.

## Prespecified protocol

No training, Flex, batching or serving claims. Use existing local V51 base Qwen
weights, bf16 SDPA, one A100 80GB. Seeds 0,1,2; exact pretokenized N=5,L=500,Q=32;
32 documents, Zipf exponent 1.5, 10 warmup + 50 measured requests per mode.
Use the same request stream and token IDs across all modes. Fixed work means
16 emitted greedy tokens, one prefill and 15 decode forwards, ignoring EOS.
This differs intentionally from the old extra unused final forward. Shipped
generation defaults are unchanged. Equal work does not mean equal attention
masks or output tokens between full causal and DDI.

Before timing each seed, independently validate cold (0 hits), warm (5),
permuted (5), partial reuse (4), and every request in the actual stream.
Compare cache and fresh restart DDI both as greedy trajectories and with the
cache teacher-forced along fresh tokens. Preserve full decision logits, exact
input and generated IDs, finite checks, per-step absolute errors, argmax
agreement and top-two margins. Separate natural-EOS generation (same 16-token
cap) is untimed. This synthetic behavior says nothing about natural QA quality.
Numerical gate is fixed at atol=rtol=1e-4; behavioral gate is exact ID equality.
Do not relax either after results. Failures withdraw correctness-preserving
speedup claims; equal-work timings can still be reported as local diagnostics.

Rotate mode order across seeds: cache/recompute/full, recompute/full/cache,
full/cache/recompute. Include masks, disk reads/writes, cold document prefill,
cache assembly/transfer, query prefill and decoding in synchronized wall time.
Exclude input sampling, diagnostics, trace serialization, and model loading.
Retain warmups and their hit states; validation caches never warm timing caches.
Use fresh exclusive output directories; each namespace and document key includes
checkpoint/tokenizer file hashes, effective config, source hashes, software and
hardware manifest identity, and exact IDs. No old cache is read or overwritten.
Report paired seed QPS ratios and df=2 t intervals; three seeds are weak evidence.

## Validation and rollback

First run CPU tiny-Qwen tests for exact geometry, deterministic paired streams,
EOS vs fixed work, cache hit state, permutation/partial reuse and strict numeric
comparison. Then run the full diagnostic campaign and inspect all traces.
Rollback consists only of removing the new script/tests and reversing this
task's paper edits; no production generation implementation changes are needed.

## Source admission boundary

Base SHA: `16dd67d7e53a8168cf6580e6d0197c219e6ba198`. No permission to commit or
push. The script preserves executing Python sources and hashes, the tracked
working diff and its hash, immutable checkpoint/tokenizer content hashes and
effective configuration. These are explicitly local dirty-source diagnostics,
not clean-revision evidence. The clean-revision acceptance item remains
incomplete. Historical V51 raw files must remain unchanged. Paper throughput
claims must be withdrawn rather than replaced with inadmissible evidence.
