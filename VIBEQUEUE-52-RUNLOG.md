# VIBEQUEUE-52 Run Log

## 2026-09-06 R1 Pilot Attempt 1

The frozen Qwen seed-0 pilot completed all 475 optimizer steps in 2,821 seconds
with no reported non-finite or skipped step. Holdout DDI perplexity moved from
7.249 before training to 2.793 after training.

Checkpoint serialization then failed with:

```text
SafetensorError: Error while serializing: I/O error:
Software caused connection abort (os error 103)
```

The 512GB home volume had reached 100% and the Arcadia FUSE mount disconnected.
The partial checkpoint contained shards 1--3 of 4 but no index or tokenizer, so
it was not loadable or evaluable. This is an infrastructure failure after training,
not a result under the preregistered scientific stopping rule.

One exact infrastructure retry is allowed. Preserve the first append-only store
with a `failed_save` suffix, delete only the unusable 9.7GB partial checkpoint,
back the same logical output path with non-FUSE `/var/tmp`, and rerun the identical
frozen config. No weight, seed, dataset, record count, step count, or threshold may
change. A second infrastructure failure stops the pilot as unavailable.

The Tracker comment attempt failed because the tracker MCP backend disconnected
with the mount. Mirror this entry to VIBEQUEUE-52 when the backend recovers.

## 2026-09-06 R1 Pilot Attempt 2 And Stop

After preserving attempt 1's store, removing only its unusable partial checkpoint,
and backing the same logical checkpoint path with non-FUSE storage, the one allowed
exact retry completed all 475 steps and saved successfully in 2,813 seconds. Holdout
DDI perplexity moved from 7.249 to 2.801. The matched nine-cell evaluation then
completed in 371 seconds against the existing seed-0 manifests.

The preregistered expansion gate failed on both natural-QA datasets:

| Dataset | V46 tuned DDI F1 | R1 DDI F1 | R1 - V46 | Full change from base | Blind change from base | Frozen F1 Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HotpotQA | 0.598803 | 0.477444 | -0.121358 | +0.002557 | +0.030367 | 0.082170 |
| MuSiQue | 0.251655 | 0.210761 | -0.040894 | +0.006801 | +0.002990 | 0.074575 |

Both control-path conditions pass, but the required DDI F1 improvement of at least
0.03 fails in the opposite direction. Per protocol, stop without a 20,000-record
run, additional seeds, altered weights, or another recipe. Three-seed paired
bootstrap, aggregate parity, and architecture transfer are unavailable by design;
do not extrapolate them from this pilot.

After hashing all four checkpoint shards and preserving the scored predictions,
the stopped pilot's 15GB checkpoint and its symlink were removed. The home volume
had only 5.6GB free; retaining weights that the frozen gate forbids using would risk
another infrastructure failure. The hashes remain in the result summary, but the
checkpoint is intentionally unavailable.
