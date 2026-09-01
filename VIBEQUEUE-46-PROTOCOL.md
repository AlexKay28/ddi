# VIBEQUEUE-46 Transfer Protocol

This protocol tests transfer of DDI fine-tuning across two architectures without
turning legacy single-seed checkpoints into evidence.

## Design

| Role | Values | Rule |
| --- | --- | --- |
| Architecture | `Qwen/Qwen2.5-7B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct` | Analyze each architecture separately; raw cross-architecture scores are not treatment effects. |
| Checkpoint condition | `base`, `ddi_qf_tuned` | The paired base is the same architecture and seed as its tuned checkpoint. |
| Seed | `0`, `1`, `2` | Each seed has its own split and held-out manifest. |
| Evaluation mask | `full`, `ddi_qf`, `blind` | `full` is the baseline; `blind` is the document-use null control. |
| Dataset | `hotpot_qa`, `musique`, `synthetic_qa` | Keep all three unless a documented loader failure makes a cell unavailable. |

Fixed settings are the pinned code revision, SDPA quality path, validation split,
sorted ID manifests, `max_records: 100`, `max_new_tokens: 64`, metrics
`em,f1,presence,loss`, and one sequential GPU process. Compiled Flex has failed
its representative parity gate, so this protocol does not make Flex or sparse
throughput claims.

## Checkpoint Identity

Every checkpoint arm must use a unique, seed-independent `evaluation_model` label.
`ddi.run` writes that label into the run ID and provenance, and records the exact
`checkpoint` string as `provenance.checkpoint_revision` on every metric row. This
keeps all three seed rows in one aggregation identity while preventing a checkpoint
arm from sharing the base identity. A missing, base-equal, or seed-qualified label
fails before model load.

The local `results/checkpoints/` Qwen and Llama directories are legacy 8/16-layer
single-seed artifacts. They may be used only for an explicitly labelled diagnostic,
not as the matched `{base,tuned} x {0,1,2}` matrix below.

## Manifest Procedure

1. Restore or verify the seed-specific record caches for all three datasets.
2. Run each architecture's base arm for seed 0 with `transfer_protocol.yaml` plus
   `--set heldout_manifest_mode=write`; inspect
   `results/transfer_heldout/*__seed0__val.json` and commit the files before any tuned arm.
3. Repeat the base-arm bootstrap for seeds 1 and 2. Never delete, regenerate, or change a
   manifest limit after a comparison starts.
4. Run the same manifests for the paired tuned checkpoint in verify-only mode. `ddi.run`
   rejects a checkpoint arm unless `heldout_manifest_mode=verify`, so it cannot create a
   first manifest. Preserve the append-only metric
   and per-example prediction stores, then join candidate outputs by `record_id`.

The manifests are not checked in before bootstrap because invented IDs would make a
deterministic but invalid evaluation set.

## Budget And Decisions

Interpret all resulting evidence under
[`EVIDENCE_TO_CLAIM.md`](EVIDENCE_TO_CLAIM.md); it fixes the Flex,
cross-architecture, uncertainty, and outcome-specific claim boundaries without
asserting a result from this protocol.

Budget the base feasibility stage as six sequential 1-GPU evaluations: two architectures
times three seeds. Do not schedule a tuned run, training, multiple GPUs, or parallel jobs
in the feasibility stage. The command below runs exactly one arm.

Primary outcomes are per-seed paired `full - ddi_qf` changes for EM, F1, and loss;
the per-example store is the audit artifact. Compute the base model's `Delta =
2 * sample_std(full per-seed means)` with `ddi.delta_cli`. A cell is admissible only
when document headroom `full - blind > max(0.05, 3 * Delta)`.

Go to matched tuning only when both architectures have all three base seeds and frozen
manifests, at least one admissible primary cell, and a base DDI gap outside `Delta`.
Otherwise stop with a bounded null or inadmissible conclusion. For tuned checkpoints,
`ddi.parity` must pass both DDI parity and full-path preservation; parity achieved by a
degraded `full` path is a failure.

## One-Arm Command

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python3 -m ddi.run experiments/transfer_protocol.yaml \
  --model Qwen/Qwen2.5-7B-Instruct --seed 0 \
  --set evaluation_model=Qwen2.5-7B-Instruct__base \
  --set heldout_manifest_mode=write
```

For a tuned arm, add `--set checkpoint=results/checkpoints/<checkpoint>` and set the
seed-independent `evaluation_model` `Qwen2.5-7B-Instruct__ddi_qf_tuned`; leave the
template's `heldout_manifest_mode=verify` unchanged. Do not use this command until the
base seed's manifest has been reviewed and committed.

## R5a Evidence And R5b Handoff

R5a changed only the transfer safety path. Quality rows now retain
`provenance.checkpoint_revision`; `ddi.delta_cli` and `ddi.parity` reject a reused
evaluation identity with conflicting checkpoint revisions. The focused CPU test writes
three seeded rows under one identity and verifies both aggregators consume them. A
verify-only manifest test confirms no manifest directory is created when it is absent.

The local CPU test environment passed `80` focused aggregation/manifest tests. The
runner-specific test module could not collect: its Python 3.10 `transformers` import
loads an incompatible `torchvision` and raises `RuntimeError: operator
torchvision::nms does not exist`; Python 3.11 has no `pytest`. No environment was
changed. Tracker is unavailable in this workspace, so this document is the handoff
evidence for later orchestration.

R5b must run each base arm first, one architecture and seed at a time, with the only
creation-capable command:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python3 -m ddi.run experiments/transfer_protocol.yaml \
  --model <architecture> --seed <0|1|2> \
  --set evaluation_model=<architecture-label>__base \
  --set heldout_manifest_mode=write
```

After review and commit of that seed's manifests, R5b must evaluate its matching tuned
checkpoint only with this verify-only command:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python3 -m ddi.run experiments/transfer_protocol.yaml \
  --model <architecture> --seed <0|1|2> \
  --set checkpoint=results/checkpoints/<checkpoint> \
  --set evaluation_model=<architecture-label>__ddi_qf_tuned \
  --set heldout_manifest_mode=verify
```
