#!/usr/bin/env bash
set -euo pipefail

# Mirrors the primed dataset record cache (results/splits/*_records.jsonl,
# see ddi/data/records_cache.py) to YT, so an offline container can restore it
# with `make yt-fetch-records` instead of ever touching the HF Hub.
#
# Unlike models (upload_models_to_yt.sh), mirroring the raw HF hub snapshot
# for datasets is NOT enough to make `datasets.load_dataset()` work offline -
# its offline path needs the library's own prebuilt arrow cache, not just the
# raw source files. The record cache sidesteps that entirely: ddi's loaders
# check `results/splits/` first and never call `datasets.load_dataset()` on a
# cache hit.

PROXY=watt
DEST_ROOT="//home/gena/alexkay28/tickets/ALIGNMENT-2445/data/records_cache"
LOCAL_SPLITS="results/splits"

yt --proxy="$PROXY" create map_node "$DEST_ROOT" -r -i

for f in "$LOCAL_SPLITS"/*_records.jsonl; do
  [ -f "$f" ] || continue
  fname=$(basename "$f")
  dest_path="$DEST_ROOT/$fname"
  size=$(du -sh -L "$f" | cut -f1)
  echo "uploading $fname ($size) ..."
  yt --proxy="$PROXY" write-file "$dest_path" < "$f"
  echo "done: $fname"
done

echo "ALL RECORD CACHES UPLOADED"
