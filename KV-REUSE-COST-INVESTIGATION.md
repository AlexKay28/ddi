# Investigate modest document-KV reuse speedup

Filed after backend recovery: https://st.yandex-team.ru/VIBEQUEUE-59
Intended queue: VIBEQUEUE. Parent: VIBEQUEUE-2. Assignee: alexkay28.
Related: VIBEQUEUE-58, VIBEQUEUE-57, VIBEQUEUE-53, VIBEQUEUE-56.

## Question

Why does V58's all-hit reuse yield only 1.093x fixed-work speedup, and where
does avoided document computation exceed persisted-KV loading/assembly costs?
Separate TTFT benefit from end-to-end request benefit. No jobs are authorized
or launched by this draft.

## Starting evidence

V58 evaluated 100 matched HotpotQA requests with a restart-trained Qwen checkpoint
and bf16 SDPA. All 1,000 warm document accesses hit. Fixed-work speedup was
1.093x, 95% CI [1.082, 1.105], or about 8.5% less time. QA non-inferiority failed;
exact generated IDs agreed on 92/100 requests. Assembly integrity passed, while
independently projected document states differed numerically. This remains
diagnostic, not quality-preserving acceleration.

## Checklist

- [ ] Audit V58 raw timings and the exact speedup estimand before new runs.
- [ ] Freeze runtime, checkpoint/tokenizer/source identity, matrix, cache states,
      seed/repeat counts, acceptance criteria and GPU/storage budget.
- [ ] Instrument lookup, read/deserialization, transfer, assembly/allocation,
      document computation, query integration and decoding where separable.
- [ ] Measure synchronized request totals, component spans, residual time and
      instrumentation overhead; do not add overlapping CPU/GPU spans.
- [ ] Measure TTFT, TPOT and request latency separately on matched requests.
- [ ] Run a predetermined short/medium/long-context and short/long-decode matrix
      with cold, partial-hit and all-hit conditions, within the frozen budget.
- [ ] Quantify KV bytes, host/GPU memory, cache-construction amortization and
      the crossover where avoided computation exceeds loading overhead.
- [ ] Retain numerical/output disagreement and available QA outcomes; audit
      paired uncertainty and publish all planned cells, including failures/OOMs.
- [ ] Publish traces, charts and a bounded conclusion to VIBEQUEUE-56.

## Measurement rules

Primary arms use identical tokens, checkpoint, positions and SDPA DDI semantics:
fresh computation versus persisted cache reuse. Flex is out of scope.

If fresh DDI performs a joint document/query forward, preserve that baseline.
Report the joint measurement or explicitly labeled profiler attribution rather
than silently substituting a slower staged baseline to obtain a decomposition.

TTFT starts at request entry and includes stated cache I/O, assembly and prefill.
Natural stopping supplies QA outcomes; a separate equal-work pass controls
emitted-token and decode-forward counts. Counterbalance arm order and retain at
least five measured repeats after warmup. Respect request/seed/cache-history
dependence in uncertainty estimates; repeats are not independent workloads.

Begin with V58's frozen natural-QA inputs. Synthetic exact-length cases may
isolate costs but cannot establish QA quality. Distinguish warm OS page-cache
reads, cold storage where controllable, and GPU-resident reuse as separate
designs. Record prewarming and construction costs separately. Do not claim
concurrent serving RPS from sequential latency.

If testing higher precision, remeasure complete timing and QA, including added
conversion, storage and compute costs. Do not carry the old speedup into a new
precision regime or relax V58's failed gates after seeing results.

## Deliverables

Preserve immutable manifests, exact inputs, raw timing/profiling traces, cache
pre-state/hits, outputs, QA metrics, memory/payload sizes, source snapshots and
independent audit. Commit/push only with explicit authorization; dirty-source
runs remain diagnostic under the existing policy.

Charts: cost breakdown; TTFT versus request speedup by context/decode budget;
load/assembly overhead versus avoided compute; cold-to-warm crossover and
amortization; paired quality/output differences alongside speed.

Possible findings include TTFT gains diluted by decode, I/O/assembly domination,
a long-context crossover, or no material benefit. Derive the explanation from
traces. A successful investigation need not produce a positive speedup claim;
a paper WIN still requires the declared quality and material-speed gates.
