#!/usr/bin/env bash
set -euo pipefail
shopt -s globstar

PROXY=watt
DEST_ROOT="//home/gena/alexkay28/tickets/ALIGNMENT-2445/data"
HF_CACHE="$HOME/.cache/huggingface/hub"

# cache_name (datasets--<cache_name>) -> repo_name (destination dir on YT)
# Keys track the loaders' _HF_DATASET_ID with / -> --, so they must stay namespaced:
# the HF cache dir is datasets--<namespace>--<name>, and a bare key finds no snapshot
# (silently, via the `ls | head -1` below) now that the loaders resolve namespaced ids.
# Destinations are unchanged, so the existing YT mirrors stay where they are.
declare -A DATASETS=(
  ["facebook--kilt_tasks"]="kilt_tasks"
  ["dgslibisey--MuSiQue"]="MuSiQue"
  ["hotpotqa--hotpot_qa"]="hotpot_qa"
)

for cache_name in "${!DATASETS[@]}"; do
  repo_name="${DATASETS[$cache_name]}"
  snapshot=$(ls -d "$HF_CACHE/datasets--$cache_name/snapshots"/*/ | head -1)
  dest_dir="$DEST_ROOT/$repo_name"

  echo "=== $repo_name -> $dest_dir ==="
  yt --proxy="$PROXY" create map_node "$dest_dir" -r -i

  # Unlike model snapshots (flat), dataset snapshots nest files under a config
  # subdir (e.g. kilt_tasks/nq/*.parquet, hotpot_qa/distractor/*.parquet), so
  # walk recursively (globstar, not `find` - sandboxed shells may silently
  # return nothing for `find` on paths outside the project dir) and preserve
  # relative structure.
  for f in "$snapshot"**; do
    [ -f "$f" ] || continue
    rel="${f#"$snapshot"}"
    dest_path="$dest_dir/$rel"
    dest_parent=$(dirname "$dest_path")
    size=$(du -sh -L "$f" | cut -f1)
    echo "  uploading $rel ($size) ..."
    yt --proxy="$PROXY" create map_node "$dest_parent" -r -i
    yt --proxy="$PROXY" write-file "$dest_path" < "$f"
    echo "  done: $rel"
  done
  echo "=== $repo_name complete ==="
done

echo "ALL DATASETS UPLOADED"
