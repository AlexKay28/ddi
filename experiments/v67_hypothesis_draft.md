# V67 — New frozen hypothesis draft (v0) — **PENDING OWNER APPROVAL; NO RUNS LAUNCHED**

Ticket: VIBEQUEUE-67. This draft exists to satisfy checklist item 1 ("Write new frozen hypothesis and get owner
budget approval before any run"). Nothing below is executed until approval.

## Evidence base this draft builds on

1. **V62 (closed, failed k0):** deployment-matched, answer-supervised k0 REGRESSED (DDI EM .60→.40, F1 .689→.524)
   and was stopped by its rule. The k=1/5 request-time link arms were stopped WITH it — they remain hypotheses,
   not failures.
2. **V65/V66 (closed, audited win):** all-hit reuse is 2.05x faster than monolithic (62.30 vs 127.52 ms), TTFT
   4.28x [4.16, 4.41] at L=512/N=10; the hit-rate/length/decode boundary is measured. Any quality lever must not
   erase this win silently.
3. **V68 (closed, diagnostic n=22):** reuse damage is a two-sided, low-magnitude perturbation — 8 near-tie greedy
   flips / 11 semantic-sized outputs after single-token flips / 3 truncations; gold present in either output on the
   damaged records; reuse strictly GAINS EM on 4 divergent records. This is the same phenomenon as independent
   bf16 re-encoding sensitivity, NOT cache-specific semantic corruption.

Point 3 is why this is a NEW hypothesis and not a silent resume of V62's recipe: V62 trained for an integration
failure mode the anatomy says is not the dominant signature. The dominant signature is near-boundary greedy
instability under independent encoding.

## H1 (primary): request-time link tokens as a stabilizer

H1: Document KVs are context-independent (byte-identity proven in V58/V60/V65) while cross-document integration
happens at request time; k∈{1,5} link tokens give the request-time decoder direct access to sibling-document
context and re-anchor greedy decisions that independent encoding leaves near ties.

- Falsifiable prediction: at equal budget, k1/k5 reduce the divergent-record count and the near-tie flip fraction
  relative to k0-under-the-same-recipe, with recovery concentrated on the divergent population.
- Mechanism constraints (hard): document rows never attend links or other documents; links computed per request,
  never stored in document KVs; mask/position reference tests (2-document hand-checkable cases + causal-leakage
  tests) mandatory before training.

## H2 (fallback / control recipe): perturbation-consistent training

H2: k0 failed because answer-supervised training never taught the model invariance to independent-encoding
perturbations. Revised recipe: train WITH the production perturbation (per-record bf16 re-encode noise /
cache-dropout), multi-hop supervision with natural distractors, random document order during training. This
directly targets the V68 near-tie signature.

## Proposed design (single experiment, both hypotheses at once)

Revised recipe (H2) applied to three EQUAL-budget arms: **k0 (control, recipe changed → re-run required),
k1, k5**. This satisfies the ticket's rule (control re-run because the recipe changed) and tests H1 and H2
jointly: k0-new vs V62-k0-old isolates H2; k1/k5 vs k0-new isolates H1.

- Data: disjoint train/dev/confirmation splits (no V58/V60/V62/V65 eval leakage; the 100 confirmation ids stay
  frozen); fixed teacher; multi-hop supervision with natural distractors; random document order.
- Gates (frozen in the FINAL prereg before any run): paired non-inferiority with the existing −0.02 floor for
  cache-vs-fresh AND chosen full-reference; full-path preservation control tracked; dev selection first, then
  untouched confirmation; training seeds as budget permits (exact count fixed in final prereg).
- Costs counted and reported in the verdict: link-token prefill, storage, request latency vs the V65/V66 win,
  training tokens. If links help quality but erase the latency win → report the tradeoff; no silent re-baselining.
- Stop rules: no dev improvement at fixed budget → stop; quality collapse → stop; no 8-GPU-scale spend.

## Budget request (numbers pending owner approval)

- Hardware: **1× A100 80GB** (same as V62 pilot). No multi-GPU.
- Extrapolation from V62 (100 steps ≈ 7.7 min): 3 arms × ~500 steps ≈ 3 × ~40 min; dev eval ~10 min/arm;
  untouched confirmation eval ~15 min; equivalence/mask tests CPU. Total ≈ **3–4 h**, **hard cap 6 h A100**.
- A 20-step smoke run re-estimates cost BEFORE the full budget commits; if smoke contradicts the estimate, back
  to owner. If the 500-step pilot shows no dev signal, the stop rule fires and remaining budget is not spent.
- Honest caveat: these are extrapolations from one 100-step pilot; the final prereg restates them after the smoke.

## Approval scope

Owner approval = permission to (1) finalize the full prereg (exact steps, seeds, splits, gates, budget),
(2) run the 20-step smoke, (3) run the 3-arm equal-budget training + evals within the capped budget.
Any deviation returns to the owner.
