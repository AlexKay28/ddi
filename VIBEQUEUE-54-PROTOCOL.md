# VIBEQUEUE-54 Analysis Protocol

This CPU-only reanalysis preserves the VIBEQUEUE-46 preregistered decision
outputs. It does not train, evaluate a model, or change a frozen manifest.

## Commands

Reproduce the historical decision JSONs from retained paired predictions:

```bash
python3 -m ddi.transfer_analysis \
  --base 0=results/r5b1_qwen25_7b_base_seed0_20260902T1059Z_predictions.jsonl \
  --base 1=results/transfer_qwen_base_seed1_20260902T114019Z_predictions.jsonl \
  --base 2=results/transfer_qwen_base_seed2_20260902T142821Z_predictions.jsonl \
  --tuned 0=results/transfer_qwen_tuned_seed0_predictions.jsonl \
  --tuned 1=results/transfer_qwen_tuned_seed1_predictions.jsonl \
  --tuned 2=results/transfer_qwen_tuned_seed2_predictions.jsonl \
  --output /var/tmp/qwen-transfer-reproduction.json
```

Repeat with the six Llama paths listed in
`experiments/transfer_v54_inputs.json`, then compare the `cells` and
`bootstrap` objects with the checked-in historical result JSONs.

Run the additional, explicitly post-hoc sensitivity:

```bash
python3 -m ddi.transfer_sensitivity \
  --inputs experiments/transfer_v54_inputs.json \
  --output experiments/transfer_v54_posthoc_sensitivity.json
python3 paper/render.py --check
```

## Estimands

For a higher-is-better score (the analysis negates loss before contrasts), let
`B_F`, `B_D`, `B_B`, `T_F`, `T_D`, and `T_B` denote base/tuned full, DDI-QF,
and blind score means. The retained historical statistic is
`C = 1 - (T_F - T_D) / (B_F - B_B)`. It is a normalized remaining tuned gap,
not training recovery.

The post-hoc report additionally gives `T_D - B_D` (DDI training change),
`T_F - B_F` (full-path change), `(T_F - T_D) - (B_F - B_D)` (full--DDI gap
change), and `T_B - B_B` (blind control change). It samples record IDs with
replacement within each fixed seed and samples the same retained IDs across
architectures. Thus it preserves base/tuned, mask, seed, and architecture
pairing while remaining conditional on these three checkpoints and manifests.
