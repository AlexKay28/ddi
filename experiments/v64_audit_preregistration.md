# V64 Evidence Audit — Step 2: Preregistration (methods only)

- Frozen (UTC): 2026-09-10T09:08:52Z
- Status: **preregistered before outcome recomputation**. No outcome metric is computed or inspected in this step. Step 3 must execute the procedures below exactly as frozen; any deviation requires an append-only amendment in §8 **before** results are unblinded.
- Step chain: step 1 = `results/v64_audit_20260910_Inventory/` (immutable inventory) → **step 2 = this file (+ JSON twin)** → step 3 = recomputation per §2–§3 → step 4 = fill the claims table per §5.
- Additive-only audit: no existing file modified; no GPU; no Tracker writes in this step.

## 1. Frozen inputs (sha256-pinned)

| Path | sha256 | Role |
|---|---|---|
| `experiments/v60_confirmation_manifest.json` | `2dffebf5d04065021dfb2b9e694651b34f0ebfae4f3d4b9aabd672908aa6cb07` | id-freeze sidecar (24 ids, seed 0, `data_role: confirmation_frozen`, `created_utc: 2026-09-10T09:30:00Z`) |
| `results/v60_confirmation24_20260910T093500Z/requests.jsonl` | `ec08d056d6ebe9208cf831e8a63e2d527cc8afec90d377ad9356635767410a5c` | per-record outcomes (24 lines; per arm A/B/C_cold/C_all_hit/D: `quality.{em,f1,presence}`, `ttft_median_seconds`, `ttft_seconds` ×3 repeats, `backend` labels) |
| `results/v60_confirmation24_20260910T093500Z/manifest.json` | `21524c7df6ec564fd855fd957e48efa92bc381d1da5a42b27695c51f5b570f48` | run manifest (verbatim `data_role: "development"` + note) |
| `results/v60_confirmation24_20260910T093500Z/result.json` | `9421b268eefaf7fcf6bc60b80469ba43a8b6868f9c0dfec123647ff259547464` | stored aggregates + `timing_caveat` |
| `scripts/run_canonical_qa_dev.py` | `636793a19f2f40eab6db905a1be6dab9525687d5a7544962379004634428c611` | runner source (hardcoded `data_role` L235/L443; `canonical_backend` default `"eager"` L222) |
| `results/v64_audit_20260910_Inventory/INVENTORY.json` | `ba7850ddafc431d20b2c63eb6220615de6b2c400055e7e027bb2c626a5acb51b` | immutable step-1 inventory (both `data_role` strings verbatim; per-arm label table) |
| `results/v64_audit_20260910_Inventory/AUDIT_BASELINE.md` | `fa594464b9b5642fba282dce713e73be981b53c9297140eca81a7162c60c47f8` | step-1 baseline narrative |

Gate at step-3 start: re-verify every sha above; on any mismatch, stop and re-inventory — do not recompute on drifted inputs.

## 2. Analysis A — QA paired non-inferiority (recomputation)

**Units and ids.** Exactly the 24 record ids frozen in `experiments/v60_confirmation_manifest.json` (pinned verbatim in the JSON twin, `frozen_record_ids`). All 24 enter; **no exclusions**. If any per-record entry is missing or unparsable, the analysis aborts with a count reconciliation against 24 — records are never silently dropped.

**Contrasts** (per-record paired on the same 24 ids; `pilot-canonical minus monolithic A`):

| Contrast | Definition | Status |
|---|---|---|
| `B_minus_A` | arm B (`canonical_per_document_ddi_qf`, cache `disabled`, standard) − arm A (`original_monolithic_ddi_qf`, standard) | **primary** |
| `C_all_hit_minus_A` | arm C (`all_hit` scenario) − arm A | co-reported |
| `C_cold_minus_A` | arm C (`cold` scenario) − arm A | completeness only |

Stored `b_vs_c_equal_all=true` implies B and C per-record outcomes coincide; the co-reported contrasts verify this on the raw records. Any divergence from the primary is itself a finding, not a rule change.

**Statistics.** For each metric (EM, F1) and contrast: `d_i = canonical_i − A_i`; point estimate = mean(`d_i`). Interval = paired bootstrap two-sided 95% percentile CI:

- RNG: `numpy.random.default_rng(47)` (PCG64), fresh instance for this analysis.
- One index matrix `rng.integers(0, 24, size=(10000, 24))` — records resampled **with replacement**, drawn **once** and shared by EM and F1 (paired bootstrap), 10000 resamples.
- Per resample: mean delta per metric over the resampled records.
- CI: `np.quantile(resampled, [0.025, 0.975])` (default linear interpolation).

**Decision rule (frozen).** For the primary contrast `B_minus_A`: non-inferiority holds **iff** the CI lower bound ≥ −0.02 for **BOTH** EM and F1 (mirrors the original −0.02 floor). Classification used in step 3 / step 4:

- **pass** — rule satisfied;
- **inconclusive** — rule not satisfied, but mean delta ≥ 0 for both metrics (direction favorable; interval too wide to clear the floor — underpowered);
- **fail** — rule not satisfied and mean delta < 0 for either metric.

Co-reported contrasts get the same rule, reported alongside; the headline verdict is the primary only. No gate-shopping: a verdict is never taken from the co-reported or completeness rows if the primary does not carry it.

**No post-hoc metric changes.** EM and F1 exactly as stored in `requests.jsonl` (`arms.*.quality.{em,f1}`). `presence` is reported but is outside the decision rule. The stored aggregates contain **no** non-inferiority interval — step 3 produces the first one, from raw per-record outcomes only.

## 3. Analysis B — TTFT timing recomputation

**Estimand.** Ratio of median TTFTs. Median TTFT of an arm = median over the 24 per-record `arms.<X>.ttft_median_seconds` (each itself the median of the 3 stored `ttft_seconds` repeats) — the same definition under which the stored aggregates recompute (step-1 baseline §3.1).

**Ratios** (C := C_all_hit; point estimates to 4 decimals):

| Key | Definition | Status |
|---|---|---|
| `B_over_C` | medianTTFT(B) / medianTTFT(C_all_hit) | the quantity behind the falsified Tracker claim "B→C 1.77x" |
| `C_over_A` | medianTTFT(C_all_hit) / medianTTFT(A) | |
| `A_over_C` | medianTTFT(A) / medianTTFT(C_all_hit) (= 1 / `C_over_A`) | preregistered because "C is X× A" vs "A is X× C" reciprocal confusion is part of the 1.77x defect |
| `C_cold` | medianTTFT(C_cold) reported for completeness | completeness only; ratios vs C_cold (`B_over_C_cold`, `C_cold_over_A`) reported completeness-only, no claim attached |

**Interval.** Paired bootstrap over records: fresh `numpy.random.default_rng(47)`; one index matrix `rng.integers(0, 24, size=(10000, 24))` (records with replacement, paired across arms) drawn once and shared by **all** ratios in this analysis; per resample, each arm's median TTFT is recomputed from the resampled records' `ttft_median_seconds`, then the ratio; CI = `np.quantile(..., [0.025, 0.975])`. 10000 resamples, seed 47 — identical protocol to Analysis A.

**Gate: none.** These are descriptive reconstructions; no pass/fail threshold is attached. Their sole purpose is to attach exact, reproducibly-derived numbers to the step-4 claims table (esp. adjudicating "B→C 1.77x"). Per the VIBEQUEUE-64 description: do not conflate ratio of medians with mean paired request ratios; the latter is **not** computed here (not preregistered, not needed for the claims).

## 4. Metadata adjudication rules (frozen before step 3/4)

### 4.1 `data_role` conflict

- The immutable step-1 inventory records **both verbatim strings**: run `manifest.json`/`result.json` → `"data_role": "development"` (plus `data_role_note` "…must never serve as untouched confirmation") vs `experiments/v60_confirmation_manifest.json` → `"data_role": "confirmation_frozen"`.
- **Adjudication:** `data_role = confirmation_frozen` is **authoritative**, because the id-freeze preceded execution: the freeze manifest (sha `2dffebf5…`, `created_utc 2026-09-10T09:30:00Z`) predates the run artifact (`results/v60_confirmation24_20260910T093500Z`, stamped 09:35:00Z). The freeze is the input that determines *what* was run; the run cannot retroactively reclassify its own inputs.
- The `"development"` string is a **hardcoded runner default** (`scripts/run_canonical_qa_dev.py` L235/L443, unconditional in the dict literals) → recorded as a **provenance defect** of the runner metadata, **not** as evidence of record contamination.
- Scope limit: whether the 24 ids are genuinely untouched (intersections vs V58/V60/V61 observed manifests, `observed_exclusion_count: 100`) is checklist item 2's separate question, decided on its own evidence; this rule adjudicates only which label is authoritative for the frozen-id question.

### 4.2 Backend labels

- **Authoritative source for the executed backend:** per-request records (`requests.jsonl` → `arms[*].backend`, identical across all 24 records): **`sdpa` for all five arms** (A, B, C_cold, C_all_hit, D).
- Consequently A-vs-B/C is a **same-backend** comparison (monolithic vs canonical *execution*), and any "eager" narrative for the canonical arms is a **metadata defect**. The runner default `canonical_backend = "eager"` (L222, overridden by `--canonical-on-sdpa`) explains how the narrative could arise; retained outputs (B/C = `"sdpa"`) show `--canonical-on-sdpa` was in effect.
- Narrative "eager" claims, wherever they appear (prose, comments, review), are entered as metadata defects in the step-4 claims table; per-request records always outrank narrative.
- Corollary: `result.json:timing_caveat`'s "cross-backend … comparisons are inadmissible" phrasing rests on a mislabeled premise (the comparison is same-backend); the caveat is preserved verbatim in the inventory and its wording corrected in step 4.

## 5. Admissible-claims table skeleton (to be filled in step 4)

Columns fixed now; `corrected text` and `evidence sha` are filled **only** in step 4 from the step-3 recomputation. No row may be added or dropped post hoc.

| id | Claim | Prior text (verbatim + where stored) | Corrected text | Evidence sha |
|---|---|---|---|---|
| C1 | Paired EM delta B−A | "+.0417" (prose only: `V59-V63-CONSOLIDATED-REVIEW.md:67`; VIBEQUEUE-64 description; no stored aggregate key) | TBD (step 4) | TBD (step 4) |
| C2 | Paired F1 delta B−A | "+.0220" (same two prose sources) | TBD (step 4) | TBD (step 4) |
| C3 | Paired non-inferiority (EM & F1, floor −0.02) | never stored in any artifact; only point estimates exist | TBD (step 4) | TBD (step 4) |
| C4 | "B→C 1.77x" timing claim | VIBEQUEUE-60 comment claim; not stored in any artifact; consolidated review stores corrected arithmetic ("B/C about 1.888x … C/A about 1.771x", `V59-V63-CONSOLIDATED-REVIEW.md:60-61`) | TBD (step 4) | TBD (step 4) |
| C5 | `data_role` label | `"development"` (run manifest/result.json, hardcoded) vs `"confirmation_frozen"` (freeze manifest) — both verbatim in INVENTORY.json | TBD (step 4) | TBD (step 4) |
| C6 | Backend narrative | "eager" narratives for canonical arms vs retained per-request `backend: "sdpa"` (all five arms) | TBD (step 4) | TBD (step 4) |
| C7 | `b_vs_c_equal_all` | `result.json:aggregate.b_vs_c_equal_all` + per-record flags | TBD (step 4) | TBD (step 4) |

## 6. Reporting and reproducibility

- All step-3 numbers derive only from the §1 sha-pinned files.
- Precision: EM/F1 deltas and CIs to 6 decimals; TTFT in ms to 4 decimals; ratios to 4 decimals; CIs reported as `[lower, upper]`.
- Determinism is fully pinned: RNG type, seed, draw order, shared index matrices, quantile method (§2, §3). A rerun with the same pinned inputs must reproduce every number bit-for-bit.
- Step 3 reports the frozen decision-rule verdicts verbatim, including unfavorable ones; "if underpowered/inconclusive, record that" (checklist item 1) — the test is never altered and no observations are added opportunistically.

## 7. Rollback

- The audit is **additive-only**: step 2 creates exactly two new files and touches nothing else (no existing file edited, no GPU, no Tracker).
- Rollback of step 2 = delete the two new files:
  `rm experiments/v64_audit_preregistration.md experiments/v64_audit_preregistration.json`
- sha256 of both files is printed in the step-2 completion output (session log); the files do not self-embed their own digests.

## 8. Amendments

None. Any later change to §2–§5 must be appended here with a UTC timestamp **before** the corresponding step-3 results are produced, never silently edited.
