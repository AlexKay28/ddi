#!/usr/bin/env bash
set -euo pipefail

PROXY=watt
DEST_ROOT="//home/gena/alexkay28/tickets/ALIGNMENT-2445/models"
HF_CACHE="$HOME/.cache/huggingface/hub"

declare -A MODELS=(
  ["meta-llama--Llama-3.1-8B-Instruct"]="Llama-3.1-8B-Instruct"
  ["Qwen--Qwen2.5-7B"]="Qwen2.5-7B"
  ["Qwen--Qwen2.5-7B-Instruct"]="Qwen2.5-7B-Instruct"
)

for cache_name in "${!MODELS[@]}"; do
  repo_name="${MODELS[$cache_name]}"
  snapshot=$(ls -d "$HF_CACHE/models--$cache_name/snapshots"/*/ | head -1)
  dest_dir="$DEST_ROOT/$repo_name"

  echo "=== $repo_name -> $dest_dir ==="
  yt --proxy="$PROXY" create map_node "$dest_dir" -r -i

  for f in "$snapshot"*; do
    fname=$(basename "$f")
    dest_path="$dest_dir/$fname"
    size=$(du -sh -L "$f" | cut -f1)
    echo "  uploading $fname ($size) ..."
    yt --proxy="$PROXY" write-file "$dest_path" < "$f"
    echo "  done: $fname"
  done
  echo "=== $repo_name complete ==="
done

echo "ALL MODELS UPLOADED"
