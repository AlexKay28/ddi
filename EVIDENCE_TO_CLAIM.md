# Evidence-To-Claim Framework

**Status:** preregistered decision rules for future evidence. This document
does not promote any existing result to a conclusion. It applies to the R6a
CPU handoff and to GPU work that follows it.

## Current Boundary

- The compiled Flex path currently fails its full-model proxy at `N=2`,
  `L=128`, `Q=64`. The reproducible `ddi.flex_parity` diagnostic on
  Qwen2.5-7B-Instruct, bf16, torch 2.13.0+cu126, and an A100 80GB reports
  cosine/argmax agreement of `0.993342`/`87.19%` for `full`,
  `0.998553`/`95.00%` for `ddi_qf`, and `0.998701`/`94.38%` for `blind`.
  Divergence begins in the first attention layer on identical projected Q/K/V;
  explicit scale, prescaling, and FP32 do not repair it. Raw-kernel checks and
  archived timing observations are therefore not Flex model-parity or
  throughput evidence.
- VIBEQUEUE-49 is deferred outside the bounded paper rather than completed as
  a timing result. Reopening it requires the unchanged gates below; timing must
  not begin until all applicable parity gates pass. The bounded paper admits no
  Flex model-throughput, batching, or serving claim and retains no numerical
  historical Flex result.
- Existing Qwen and Llama quality values, including the legacy checkpoints,
  are descriptive diagnostics unless they satisfy the matched matrix below.
- The historical C4 cache correctness and 1.72x throughput claims are withdrawn: those
  runners populated but did not consume real document K/V. The corrected Qwen2.5-7B bf16
  path exactly matches 16-token greedy text for original, reordered, and 2-hit/1-miss
  requests, but cache-vs-recompute numerical logit identity fails (`max|delta|=0.3125` at
  `N=2,L=32,Q=8`). At `top=32,N=5,L=500,Q=32,A=16`, three pinned seeds give a
  cache/recompute mean speedup of 1.219x (95% t-CI [1.152, 1.286]). This supports only the
  measured shape and implementation, not a general serving claim.
- R4/R5a establish reproducibility plumbing, not a quality or transfer result:
  frozen held-out manifests, per-example outputs, checkpoint revision
  provenance, and seed-independent aggregation identities.
- Cache gap-closing work and intermediate gates are tracked in `VIBEQUEUE-51`.
- The preregistered VIBEQUEUE-52 seed-0 remediation pilot stopped after DDI F1
  regressed by 0.121 on hotpot_qa and 0.041 on musique relative to the matched
  V46 seed-0 checkpoint. It is not a deciding quality cell: the source tree was
  dirty at run time, the stopped checkpoint was removed after hashing, and no
  three-seed expansion or paired bootstrap ran. VIBEQUEUE-48 retains the useful
  powered-expansion and scale plans as deferred future work; those unrun cells
  are not prerequisites for the bounded paper and cannot extend its claims.
- Current quality evidence is bounded to the measured Qwen2.5-7B-Instruct and
  Llama-3.1-8B-Instruct checkpoints, named datasets, frozen manifests, and
  retained per-example outputs. The sole valid natural-QA architecture contrast
  is musique F1. It establishes neither parity nor a practical architecture
  difference, equivalence, architecture invariance, or a model-scale trend.

## Evidence Required For Any Gate

A deciding cell must retain its pinned code revision, model revision and exact
checkpoint revision, config, hardware/software identity, seed, frozen manifest
hash, aggregate row, and per-example outputs. The paired arms must use the
same frozen record IDs for a seed. Missing provenance, an identity that mixes
checkpoint revisions, a regenerated manifest, or a missing required arm makes
the cell unavailable rather than negative evidence.

Metrics are EM, F1, presence, and loss, with loss oriented so that larger is
better when it is compared with the other metrics. EM and F1 are primary;
presence and loss are supporting diagnostics and cannot overturn a primary
metric failure. Report every metric, including controls, rather than selecting
the favorable one.

## Flex Model And Throughput Gates

Flex is admitted only as a model backend after all of these gates pass for the
claimed mask, model revision, precision, cache mode, and batch/layout path:

1. Compare the patched Flex model with the dense SDPA reference on identical
   tokens and positions. Run `full`, `ddi_qf`, and `blind`; include `N=1` and
   the smallest representative multi-document proxy (`N=2`, `L=128`, `Q=64`).
2. For each compared prefill, require finite logits, cache length equal to the
   input length, cosine similarity at least `0.999`, and 100% token-argmax
   agreement. For the cached decode path, require finite logits and identical
   next-token argmax. A pass on an attention-kernel unit test alone is not a
   model pass.
3. Before a batched claim, repeat step 2 on the exact claimed batch size and
   variable-layout path. Before a target-shape claim where dense comparison is
   infeasible, pass the proxy above and demonstrate finite target-shape prefill
   and cached decode; label that target-shape result as proxy-validated, not
   dense-logit-equivalent.

Only after the applicable parity gate passes may timing begin. This is the
reopenable VIBEQUEUE-49 protocol, not a missing prerequisite for the bounded
paper. Time matched
`full` and `ddi_qf` workloads on the same hardware, software build, model,
precision, shape, batch, warmup policy, and decode length. Discard warmups and
collect at least five timed repeats. Report all repeats, median, spread, and a
95% interval for the full/DDI ratio for both prefill and end-to-end latency.

A **material prefill speedup** requires the 95% interval lower bound for
`latency(full) / latency(ddi_qf)` to exceed `1.05`. An end-to-end or serving
throughput claim additionally requires the corresponding end-to-end lower
bound to exceed `1.05` on the exact batch/layout path. A ratio above one that
does not clear this bound may be reported as an inconclusive timing estimate;
it is not a speedup claim. SDPA and raw-kernel measurements remain useful
mechanistic observations but may not be relabeled as compiled-Flex throughput.

## Cross-Architecture Transfer Design

The unit of inference is an architecture-specific, seed-matched contrast, not
a raw score pooled across Qwen and Llama. For each architecture `a` and seed
`s`, evaluate the complete matrix:

| Condition | Required masks |
| --- | --- |
| Base checkpoint | `full`, `ddi_qf`, `blind` |
| Matched `ddi_qf`-tuned checkpoint | `full`, `ddi_qf`, `blind` |

Use the same seed-specific frozen manifest for all six cells. The base and
tuned checkpoints must have distinct seed-independent `evaluation_model`
labels and recorded checkpoint revisions. Complete all three seeds for both
architectures before making a transfer comparison; do not replace a missing
seed with a legacy checkpoint or a different held-out set.

For each dataset and metric, let `S_a` be base `full - blind` headroom and let
`P_a`, `R_a`, and `E_a` be the existing parity, preservation, and end-to-end
gaps from `ddi.parity`, respectively. Report the seed-level values and their
equal-weight three-seed mean. Compare architecture effects through normalized
within-architecture contrasts such as closure (`1 - P_a`), never through raw
EM, F1, presence, loss, latency, or checkpoint scores across architectures.

The transfer comparison has three possible questions, which must not be
collapsed:

| Question | Required rule | Permitted wording |
| --- | --- | --- |
| Does tuned DDI reach parity in an architecture? | That architecture passes the admissibility, parity, preservation, and blind-control gates below. | “Within this architecture and measured cells, tuned DDI is within the preregistered tolerance.” |
| Is the direction replicated across architectures? | Both architectures independently pass the same gate on the same primary metric/dataset cell. | “The within-architecture result replicated in the two measured architectures.” |
| Is there an architecture difference? | The 95% interval for the difference in normalized closure excludes zero and its magnitude is at least `0.10` headroom. | “The measured closure differs between these architectures.” |

The `0.10` headroom threshold is the practical-effect threshold for a
cross-architecture difference. A difference smaller than it is reported as
not practically distinguished even if its interval excludes zero. Failure to
show a difference is not evidence that architectures behave identically.

## Quality, Uncertainty, And Controls

For each architecture, dataset, and primary metric, compute
`Delta = 2 * sample_std(base full seed means)` before inspecting the tuned
comparison. A cell is admissible only if base document headroom is greater than
`max(0.05, 3 * Delta)`. This is the existing `ddi.parity` admissibility rule;
no fallback constant or `--allow-missing-delta` output can support a claim.

Use paired per-example outcomes from the frozen manifest to form a 95% paired
bootstrap interval for each seed contrast, then resample the three seed means
with equal seed weight to form the reported three-seed interval. Report the
three seed values, point estimate, interval, `Delta`, headroom, manifest hash,
and all control values. With only three seeds these intervals are decision
guards, not a claim of precise population estimation.

The quality gates are:

1. **Document-use control:** base `full - blind` clears the admissibility
   threshold. Otherwise report `INADMISSIBLE`; neither parity nor a negative
   DDI result is identified.
2. **Parity:** the paired `full - ddi_qf` gap for the tuned checkpoint is no
   larger than `Delta`, and its 95% interval does not include a degradation
   greater than `Delta`.
3. **Preservation:** base `full - tuned full` is no larger than `Delta`, and
   its interval does not include a degradation greater than `Delta`. An
   improvement to `full` is reported separately and is not a preservation
   failure.
4. **Blind stability:** the tuned-minus-base blind change is no larger than
   `Delta` in absolute value. A moved blind control invalidates closure rather
   than being attributed to a cause without evidence.

For a material quality degradation, require both the interval lower bound and
the point estimate to show a degradation greater than `max(Delta, 0.02)` in a
primary bounded metric. The `0.02` absolute-score floor prevents a numerically
detectable but practically tiny effect from being labeled material. Loss is
reported with its interval but has no cross-dataset absolute practical-effect
threshold.

## Outcome-To-Claim Table

| Outcome | Claim permitted | Claim prohibited |
| --- | --- | --- |
| Flex parity fails or has not run | “Compiled Flex is unavailable as a validated model backend for this path.” | Any Flex model-speedup, serving, or quality claim. |
| Flex parity passes, timing does not clear 1.05 | “The validated path has no material measured speedup at these shapes.” | “Flex is faster” or a general throughput claim. |
| Flex parity and exact-path timing pass | “On the stated hardware and workload, validated Flex DDI achieved the reported prefill/end-to-end speedup.” | General hardware, model, batch, or serving claims beyond measured cells. |
| Quality cell is inadmissible, incomplete, or has moved blind control | “The cell cannot identify DDI quality parity or degradation.” | “No effect,” “parity,” or “DDI fails.” |
| Parity and preservation pass in one architecture | Architecture-scoped within-tolerance result. | Cross-architecture transfer or universal quality preservation. |
| Both architectures independently pass | Two-architecture replication for the named cells. | Architecture invariance or universal transfer. |
| One architecture has a material, interval-supported degradation | Architecture- and cell-bounded negative result. | A universal DDI failure or an explanation not tested by the design. |
| Architecture contrast is below 0.10 headroom or interval crosses zero | “No practically distinguished architecture difference was established.” | “The architectures are equivalent.” |
| Both architectures show material, interval-supported degradation | “A replicated negative result on the measured architectures, datasets, and checkpoints.” | A claim about all decoder-only models, tasks, or deployments. |

All claims must name the model revisions/checkpoint conditions, datasets,
manifests, backend, shapes, and uncertainty method. Negative outcomes remain
reportable evidence when the relevant gate was validly run; unavailable or
inadmissible outcomes remain scope boundaries, not negative findings.
