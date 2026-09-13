# V68 preregistration — anatomy of KV-cache reuse output differences (VIBEQUEUE-68)

Frozen before classification. Written 2026-09-10 (UTC). CPU-only diagnostic task.

## Purpose

V65 (combined validation, n=100 untouched confirmation ids) FAILED non-inferiority
narrowly: paired deltas (reuse C − fresh A) degraded EM on 5 records and F1 on 8
(results/v65_combined_20260910T102918Z/summary.md). Before any quality intervention,
this task characterizes HOW reuse degrades outputs: near-tie greedy flips (consistent
with the documented bf16 independent-encoding numerical sensitivity) vs systematic
semantic damage. Diagnostic only.

## Inputs (sha-pinned)

| artifact | sha256 |
|---|---|
| results/v65_combined_20260910T102918Z/qa_requests.jsonl | 68b1806e6104705e0da1a45b6c8579c3a06c5e08ecea9a7a2dfd9bb80d649f3a |
| results/v65_combined_20260910T102918Z/manifest.json | 4dc558c97d0f86c93de098afaa8d930ec2f641a203dc47fc1c3c4dc6e78f2cdd |
| experiments/v62_split_manifest.json | 5ad3761106ecaed967b28d21c3919fa1869ebf70bf0e178c1e773a14bf8ee098 |
| results/v60_confirmation24_20260910T093500Z/requests.jsonl | ec08d056d6ebe9208cf831e8a63e2d527cc8afec90d377ad9356635767410a5c (matches V65 manifest pin) |
| results/v60_qa_dev20_sdpa_20260909T103000Z/requests.jsonl | 7430d3306cd60133878d42e418434685c06d09b28161696018d1d3561cc25390 |
| results/v60_qa_dev20_20260909T100000Z/requests.jsonl | 7f4aba0556b2f45cb707f66e458816fc151e45626b1af9ee2041a2e387f90764 |
| scripts/v68_extract_differing.py | bf693a50bae8b7213e816659f2b167b482b037c211588ec9c352593c8f965254 |

Retained fields inspected in qa_requests.jsonl: per-record arms A (monolithic SDPA,
"fresh") and C (all-hit canonical reuse, "cached") each with `prediction`,
`natural_generated_ids`, `fixed_generated_ids`, `fixed_ids_sha256`, `quality`
(em/f1/presence), `stop_reason`; plus `gold_answers`, `prediction_equal`,
`first_token_equal`, `doc_lengths`, `query_length`, `request_index`.
**Logits/logprobs were NOT retained** in the V65 artifact; this prereg therefore
restricts all inference to output text and retained token ids, as the ticket allows.

## (a) Record-selection rule (frozen)

Every confirmation record (the 100 ids of the V62 confirmation arm, as served in
V65 validation 2) where cached prediction != fresh prediction under the frozen dev
eval, i.e. `C.prediction != A.prediction` in qa_requests.jsonl. The retained
`prediction_equal: false` flag is the same rule; the extractor additionally asserts
consistency between string inequality and the flag.

Mechanical identification is deterministic from the pinned artifact and **may happen
before this freeze; classification may not.** Mechanically identified pre-freeze:
100 confirmation records parsed, **22 differing** (sha256 of the extraction output
results/v68_anatomy_20260910T131036Z/differing_records.jsonl =
c7184d0ce3ef323efb8d61c472d016dfbe4640a9dfae96c1fde1eeaca5983928; per-record ids are
in that artifact). No per-record label is frozen here or known to this document.

## (b) Classification rubric (frozen, text terms)

Exactly four labels; every differing record gets exactly one; **"ambiguous" is an
explicit allowed label and must be used whenever the record does not clearly fit a
bucket — never force a bucket.**

- **near-tie greedy flip** — minimal-edit difference (entity swap, synonym,
  word-order, punctuation/function-word change) where both outputs answer in the
  same neighborhood.
- **clear semantic change** — different factual content/answer.
- **truncation/stopping artifact** — one output is a prefix/subset, or the outputs
  differ mainly in length/stopping.
- **ambiguous** — evidence genuinely does not discriminate the above.

Evidence consulted per record (computed mechanically by scripts/v68_extract_differing.py,
sha above): char edit distance, whitespace-token difflib opcodes, token-level prefix
relations (both whitespace and model-token), first-divergence position, gold
containment per side, retained em/f1 per side. Decision guidance: pick the label
whose description dominates the observed difference; when near-tie and truncation
features co-occur (e.g. a one-token prefix relation that reads as a punctuation/
function-word change), judge which reading dominates; if genuinely unclear, label
ambiguous. "Near-tie" here is a *structural/textual* judgment — logits are
unavailable, so no probability margin is verifiable.

## (c) Divergence-position taxonomy (frozen)

- **first-token divergence** — the first sampled generation token differs between
  arms. Primary measure: index 0 of the retained `natural_generated_ids` (model
  token ids); this is the same quantity as the retained `first_token_equal` flag,
  and the extractor asserts agreement.
- **mid-generation divergence** — first differing position > 0 (or pure length
  difference with a common prefix).

Tokenization assumption: the retained ids are Qwen2.5 tokenizer generation ids
counted from 0 = first sampled token after the prompt; retained natural generations
end at the `<|im_end|>` id (151645). The fallback measure stated by the ticket is a
text token diff: whitespace tokenization of the two prediction strings
(`str.split()`), index of first differing whitespace token. Both are computed and
reported; the model-token measure is primary. Divergence position is *reported*,
not itself a label.

## (d) Interpretation rules (frozen)

1. Diagnostic-only language: this analysis describes *what kind* of differences
   occur; it does not establish causes and does not evaluate fixes.
2. Tiny n: 22 differing records out of 100, single seed, single checkpoint
   (Qwen2.5-7B-Instruct ddi_qf 8L restart). No rate claims — counts and shares are
   reported only as descriptive counts of this sample, never as population rates
   or with inferential statistics.
3. No causal claims: nothing here attributes a difference to a mechanism (bf16
   sensitivity, cache state, etc.); "consistent with" language only.
4. Logits unavailable: the numeric margin between the arms' top choices cannot be
   verified; "near-tie" claims are structural analogies, stated as such.
5. Every number traces to a sha-pinned artifact; missing data is reported as
   missing, not imputed.

## (e) V60 cross-reference plan (frozen before looking)

Compare the n=100 divergence-position mix and the presence/absence of clear
semantic changes against the V60 dev-20 residual analysis (3/20 requests differed;
ticket records it as ~50% first-token / ~50% mid-generation, "no semantic
signature") and the V60 confirmation-24 artifact. Verdict options pre-declared:
*agrees*, *agrees on position mix but disagrees on semantic signature*, *disagrees*,
with exact counts stated and no smoothing of mismatches.

## Deliverables (planned)

results/v68_anatomy_<UTC>/: differing_records.jsonl (pre-freeze mechanical),
classifications.jsonl (post-freeze), anatomy_table.md, README.md (provenance,
input+output shas, limitations), interpretation.md; scripts/v68_extract_differing.py,
tests/test_v68_extract.py (pytest subset run pre-freeze: 10 passed).
