# V70 Confirmation Verdict — One Command

Run this from the repository root the moment `results/v70_confirmation300_k5.json` lands. It is CPU-only, takes well under a minute, and requires no GPU.

```bash
/var/tmp/opencode/v57-conda312/bin/python scripts/v70_analyze_confirmation.py --input results/v70_confirmation300_k5.json
```

The output directory is created automatically as `results/v70_confirmation_analysis_<UTC>` (pass `--output-dir <dir>` to pin it; an existing directory is refused, never overwritten).

## What it does

1. Validates the artifact fail-closed against the frozen preregistration (`experiments/v70_preregistration.md`, amendments 1-3): schema `v70-evaluation-v2`, `split == confirmation300`, `data_role == confirmation_primary`, `record_count == 300`, `k == 5`, selected k5 checkpoint aggregate SHA, hit-only/reopened cache evidence, and `provenance.record_ids` equal to `experiments/v70_confirmation300.json` exactly in order (manifest internal hashes and the frozen `ba7263ae…` / `529f0109…` ID SHAs are re-derived, not trusted).
2. Recomputes the preregistered C-B and C-D gates with `scripts.v70_harness.paired_quality_gate` (10,000 paired resamples, seed 47, margin -0.02) and hard-fails if they do not exactly match the gates embedded in the artifact.
3. Recomputes the same gates on the n=100 original-confirmation subset (manifest positions 1-100) as the V67-comparability column — reported, not a gate.
4. Writes the V68 divergent-22 post-hoc report (`results/v68_anatomy_20260910T131036Z/differing_records.jsonl`): per-ID C-vs-D deltas, predictions, and the recovered count (C EM > 0 or C F1 >= 0.1) — clearly labeled post hoc, never selection.
5. Emits `verdict.json` (all numbers, gate pass/fail, subset comparison, divergent report, full SHA-256 provenance), `verdict.md` (tables plus a preregistration-faithful conclusions paragraph naming exactly which preregistered gates passed/failed), and a `SHA256SUMS` sidecar.

Any validation failure aborts with exit code 2, writes nothing, and prints the reason to stderr — a missing, tampered, or mismatched artifact can never produce a verdict.

## Outputs

- `results/v70_confirmation_analysis_<UTC>/verdict.json` — machine-readable verdict (its SHA-256 is printed to stdout and recorded in `SHA256SUMS`)
- `results/v70_confirmation_analysis_<UTC>/verdict.md` — human-readable verdict
- `results/v70_confirmation_analysis_<UTC>/SHA256SUMS`

The stdout tail is the summary: `gate C-B pass`, `gate C-D pass`, and the overall preregistered gate pass.
