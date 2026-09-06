# VIBEQUEUE-52 Natural-QA Remediation Protocol

This protocol was frozen after reproducing VIBEQUEUE-46 and before changing the
training objective or launching another GPU job. It tests a new mechanism; it does
not reinterpret the VIBEQUEUE-46 result.

## Diagnostic Basis

The existing objective distils full-mask teacher logits into the DDI student only
on gold-answer prefixes. Across the matched VIBEQUEUE-46 predictions, 79--96% of
the decisive `full`-correct/`ddi_qf`-wrong cases diverge on the first generated
token, and none of those DDI predictions contains a gold answer. Fine-tuning also
changes 132--252 of 300 blind predictions per natural-QA dataset/architecture.

A no-training counterfactual restored tuned `full` and `blind` outputs exactly to
their base outputs while retaining the observed tuned DDI outputs. No natural-QA
EM/F1 cell passed. Preservation-only training is therefore rejected before GPU use:
the eligible recipe must improve rollout agreement and constrain both control paths.

## Frozen Candidates

Candidates are ordered; a later candidate cannot be selected after seeing its result.

| Candidate | Objective | Decision |
| --- | --- | --- |
| `P0-preservation-only` | Existing answer-position DDI KL plus full/blind anchors | Rejected by the no-training counterfactual. Never launch. |
| `R1-first-token-anchors` | Existing answer-position DDI KL + CE on the full teacher's greedy first answer token + base-teacher KL anchors under `full` and `blind` | Sole GPU-eligible candidate. |

`R1` uses fixed weights `teacher_first_token_weight=1.0`,
`full_anchor_weight=1.0`, and `blind_anchor_weight=1.0`, with KL temperature 1.0.
The first-token target is the frozen full teacher's argmax at the first answer
position, not the gold label. This is the cheapest rollout-aligned intervention
supported by the paired-error signature; it does not add an unregistered gold CE
or a hyperparameter sweep. The anchor teacher is the same frozen base model, under
the corresponding mask. Zero for all three weights remains the historical objective.

## Pilot And Budget

The only pilot is Qwen seed 0:

| Field | Frozen value |
| --- | --- |
| Model | `Qwen/Qwen2.5-7B-Instruct` |
| Datasets | `hotpot_qa`, `musique`, `synthetic_qa` |
| Training records | 2,000 mixed records, 5% deterministic tail holdout |
| Epochs / optimizer steps | 1 / at most 475 (`grad_accum_steps=4`) |
| Trainable parameters | Top 8 decoder layers, as in VIBEQUEUE-46 |
| LR / warmup | `1e-5` / 100 steps |
| Hardware | One GPU; no overlapping arm |
| Output identity | `Qwen__Qwen2.5-7B-Instruct__ddi_qf__v52_r1_pilot_seed0` |
| Evaluation | Existing seed-0 frozen manifests; 100 records per dataset; `full`, `ddi_qf`, `blind`; 64 generated tokens |

Maximum pilot budget is one training and one matched evaluation, estimated at no
more than 2.5 times the per-record VIBEQUEUE-46 training cost because `R1` adds two
student and one teacher forward passes. Abort rather than reduce the objective if it
does not fit one A100 80GB. Do not overwrite or resume any VIBEQUEUE-46 checkpoint.

The pilot proceeds to the full recipe only if all conditions hold:

1. No non-finite or skipped optimizer step; checkpoint tensors are finite.
2. `ddi_qf` F1 improves by at least 0.03 over the matched VIBEQUEUE-46 Qwen seed-0
   tuned checkpoint on both HotpotQA and MuSiQue.
3. Full-path F1 regression from seed-0 base is no greater than that dataset's frozen
   VIBEQUEUE-46 Delta.
4. Absolute blind F1 change from seed-0 base is no greater than that Delta.

Failure of any condition stops `R1` with a bounded negative result. There is no retry
with different weights, learning rate, layer count, record count, or seed.

If the pilot passes, run the same `R1` recipe on 20,000 mixed records for Qwen seeds
0, 1, and 2, each with a unique checkpoint and store. The full-stage maximum is three
sequential one-GPU trainings and three matched evaluations. Llama is out of scope for
VIBEQUEUE-52; architecture breadth remains VIBEQUEUE-48 and cannot be added after
seeing Qwen results.

## Full-Stage Evidence Gate

Use the existing committed seed-specific manifests and the VIBEQUEUE-46 matrix rules:
same base architecture and seed, `full`/`ddi_qf`/`blind`, `max_records=100`,
`max_new_tokens=64`, and metrics `em,f1,presence,loss`. Join by `record_id` and run
the existing paired percentile/equal-weight-seed bootstrap with 10,000 repeats.

Report Delta/admissibility, DDI parity, full-path preservation, blind stability, and
normalized closure for every natural-QA EM/F1 cell. Parity achieved by degrading
`full`, moving `blind`, changing a manifest, or selecting a post-hoc recipe is failure.
Commit the protocol, configs, prediction rows, aggregate JSON, and bounded conclusion,
including negative or unavailable outcomes.
