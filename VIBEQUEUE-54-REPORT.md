# VIBEQUEUE-54 Report

## Scope and reproduction

This CPU-only change preserves the VIBEQUEUE-46 preregistered decision outputs
and thresholds. It does not start V55, train a model, launch a GPU job, or
replace an original statistic.

`ddi.transfer_analysis.analyze` was rerun on the twelve retained paired
prediction files. Both historical `cells` arrays and bootstrap settings match
exactly: Qwen matches `experiments/qwen_transfer_r5d.result.json`; Llama
matches `experiments/llama_transfer_r5d.result.json`. The historical natural-QA
decisions remain: Qwen hotpot EM inadmissible, Qwen hotpot/musique F1 parity
not reached, Llama hotpot EM/F1 invalid because blind moved, and Llama musique
EM/F1 parity not reached. The only original valid architecture decision remains
musique F1: Llama minus Qwen `C = +0.028`, equal-seed interval
`[-0.085, +0.179]`, below the 0.10 practical threshold.

## Root Cause and resolution

The old wording called `C = 1 - (T_F - T_D)/(B_F - B_B)` recovered base
full-to-DDI headroom. Algebraically, its numerator is the *remaining tuned
full--DDI gap*, and its denominator is base full--blind document headroom.
It can therefore be high before DDI improves and can rise when the tuned full
path falls. V54 retains `C` and every historical gate as preregistered outputs,
but calls it the **normalized remaining tuned gap**. It separately reports the
paired DDI training change `T_D-B_D`, full-path change `T_F-B_F`, and gap change
`(T_F-T_D)-(B_F-B_D)`. Positive DDI/full changes are improvements; a negative
gap change narrows the full--DDI gap. Loss is sign-oriented before contrasts;
the source keeps its raw lower-is-better values.

## Raw paired primary outcomes

Each vector is seed 0/1/2 in the order `[B_F, B_D, B_B, T_F, T_D, T_B]`.
These are direct means from per-example predictions, not rounded table input.

| Architecture | Dataset/metric | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- | --- |
| Llama | hotpot EM | [.50,.34,.02,.47,.32,.13] | [.42,.32,.03,.35,.33,.13] | [.49,.32,.03,.43,.33,.13] |
| Llama | hotpot F1 | [.672,.446,.025,.667,.439,.181] | [.577,.442,.034,.543,.446,.187] | [.662,.439,.030,.624,.442,.155] |
| Llama | musique EM | [.30,.14,.00,.33,.17,.00] | [.26,.10,.00,.27,.13,.00] | [.29,.10,.01,.31,.14,.01] |
| Llama | musique F1 | [.419,.219,.011,.459,.265,.008] | [.359,.182,.005,.383,.206,.012] | [.377,.158,.020,.413,.203,.023] |
| Qwen | hotpot EM | [.57,.31,.13,.50,.46,.17] | [.45,.31,.15,.46,.42,.17] | [.46,.24,.13,.44,.37,.15] |
| Qwen | hotpot F1 | [.700,.426,.252,.662,.599,.305] | [.626,.427,.244,.623,.543,.294] | [.631,.376,.227,.616,.491,.249] |
| Qwen | musique EM | [.29,.05,.03,.25,.14,.05] | [.25,.04,.05,.17,.11,.05] | [.20,.06,.03,.22,.10,.07] |
| Qwen | musique F1 | [.377,.148,.109,.376,.252,.107] | [.368,.130,.115,.303,.198,.114] | [.309,.145,.110,.368,.216,.142] |

The complete machine-readable result additionally includes all three seed
values and all six arm means for `presence` and raw `loss`, plus every control
and post-hoc interval: `experiments/transfer_v54_posthoc_sensitivity.json`.
Thus presence and loss were not omitted; they are supporting controls rather
than new primary decision cells.

## Main separated results

| Architecture | Dataset/metric | C | DDI change | Full change | Gap change | Blind change |
| --- | --- | ---: | ---: | ---: | ---: |
| Llama | hotpot EM | .806 | .000 | -.053 | -.053 | +.103 |
| Llama | hotpot F1 | .727 | +.000 | -.025 | -.025 | +.145 |
| Llama | musique EM | .440 | +.033 | +.020 | -.013 | .000 |
| Llama | musique F1 | .480 | +.038 | +.033 | -.006 | +.002 |
| Qwen | hotpot EM | .855 | +.130 | -.027 | -.157 | +.027 |
| Qwen | hotpot F1 | .781 | +.135 | -.019 | -.154 | +.041 |
| Qwen | musique EM | .524 | +.067 | -.033 | -.100 | +.020 |
| Qwen | musique F1 | .452 | +.081 | -.002 | -.083 | +.010 |

The separate numbers clarify that Qwen has positive paired DDI changes and
negative gap changes on all primary cells, while full-path changes are not
uniformly positive. They do not overturn the historical gates: headroom,
blind-control movement, and preservation remain required for a parity claim.

## Post-hoc uncertainty sensitivity

`ddi.transfer_sensitivity` jointly resamples paired example outcomes inside
each seed. At each draw, it selects the same record-ID sample across the two
architectures, preserving the six within-record arms and seed pairing. Every
natural-QA seed has exactly 100 IDs in each architecture and 100 common IDs;
there is no partial-overlap fallback in this result. The procedure uses 10,000
draws and seed 54. It does not resample or claim to represent a population of
training seeds, checkpoints, tasks, or manifests.

For illustration, the conditional 95% intervals for Qwen DDI change are
hotpot EM `[+.090,+.173]`, hotpot F1 `[+.095,+.176]`, musique EM
`[+.030,+.107]`, and musique F1 `[+.042,+.120]`. The corresponding Qwen gap
change intervals are `[-.210,-.103]`, `[-.205,-.104]`, `[-.153,-.050]`, and
`[-.131,-.035]`. Llama hotpot has a moved blind control and remains invalid;
the sensitivity does not repair that gate. All other sensitivity intervals are
recorded in the JSON artifact and are explicitly post-hoc.

## Data lineage and limitations

All twelve deciding prediction JSONLs are locally accessible. Each records
per-example raw metrics, record ID, run ID, seed, code SHA, dirty status,
config hash, model/dataset identity, GPU/runtime identity, timestamp, and
checkpoint revision. `experiments/transfer_v54_inputs.json` pins their paths;
the sensitivity output hashes every input and is hash-pinned by
`paper/evidence.json`. The effective fine-tune YAMLs and checkpoint directory
identities are tracked under `experiments/finetune_{qwen,llama}_r5b_seed*.yaml`.

The nine frozen manifests are tracked in `HEAD` but absent from this checkout;
they remain recoverable, e.g. `git show HEAD:results/transfer_heldout/hotpot_qa__seed0__val.json`.
No checkpoint weight directories are present locally. Consequently this is a
recomputation from retained predictions and tracked protocol/config references,
not a model rerun. The fixed record IDs fully overlap across architectures, but
the trained checkpoints are only three selected instances, so neither the
preregistered equal-seed interval nor the V54 sensitivity is a population CI.

## Commands and verification

Exact commands are in `VIBEQUEUE-54-PROTOCOL.md`. Verification ran the original
reproduction, focused transfer tests, `python3 paper/render.py`, and the full
test/lint/check sequence listed in the final execution report.
