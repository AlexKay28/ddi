set -uo pipefail
# don't use -e: we want one failed model to not kill the whole batch
pip install --upgrade transformers

# Single-host data parallelism (issue #123). Default 1 keeps the offline
# container's single-device path unchanged; set NUM_GPUS=8 in the caller
# environment on hosts with 8× GPUs to run each model's finetune under DDP.
export NUM_GPUS="${NUM_GPUS:-1}"

export YT_PROXY=watt
export HF_HOME="${HF_HOME:-$HOME/hf_home}"
# /mnt/models isn't writable here; use a real writable cache dir
mkdir -p "$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$SOURCE_CODE_PATH"
ls -la

# Restore the dataset records cache once, up front — hotpot_qa/musique/kilt_nq
# loaders need results/splits/*_records.jsonl to work offline (see
# ddi/data/records_cache.py). synthetic_qa is generated locally and doesn't
# need this cache.
if ! make yt-fetch-records; then
  echo "!!! yt-fetch-records failed, hotpot_qa/musique/kilt_nq cells will fail
offline"
fi

# List of models to finetune
models=(
  "Qwen/Qwen2.5-7B-Instruct"
  "Qwen/Qwen3.5-4B-Instruct"
  "Qwen/Qwen3.6-27B-Instruct"
)

# experiments/finetune.yaml hardcodes `output_dir:` (and thus the checkpoint
# path) for a single model — `make finetune MODEL=<hf-id>` only overrides
# `model:`, not `output_dir:`. So every model in this loop trains into the
# SAME checkpoint dir and writes the SAME logs/finetune_summary.json. Move
# both out of the way after each run so later models don't clobber earlier
# ones. Read the fixed path from the YAML instead of hardcoding it so this
# stays in sync if the YAML changes.
FIXED_OUTPUT_DIR="$(grep -oP '^output_dir:\s*\K\S+' experiments/finetune.yaml)"
FIXED_SUMMARY="logs/finetune_summary.json"

for model in "${models[@]}"; do
  echo "=== Finetuning $model ==="
  if ! make yt-fetch MODEL="$model"; then
    echo "!!! yt-fetch failed for $model, skipping finetune"
    continue
  fi

  safe_name="$(echo "$model" | sed 's#/#__#g')"
  dest_ckpt="results/checkpoints/${safe_name}__ddi_qf__seed0"
  dest_summary="logs/finetune_${safe_name}_summary.json"

  # Clear any stale dir/file from a previous unrelated run so we can tell
  # whether *this* run actually produced output.
  rm -rf "$FIXED_OUTPUT_DIR"
  rm -f "$FIXED_SUMMARY"

  if ! make finetune MODEL="$model" NUM_GPUS="$NUM_GPUS"; then
    echo "!!! finetune failed for $model"
    continue
  fi

  if [ -d "$FIXED_OUTPUT_DIR" ]; then
    mv "$FIXED_OUTPUT_DIR" "$dest_ckpt"
    echo "checkpoint -> $dest_ckpt"
  else
    echo "!!! no checkpoint produced at $FIXED_OUTPUT_DIR for $model"
  fi

  if [ -f "$FIXED_SUMMARY" ]; then
    mv "$FIXED_SUMMARY" "$dest_summary"
    cat "$dest_summary" 2>/dev/null || true
  else
    echo "!!! no summary produced at $FIXED_SUMMARY for $model"
  fi
done

echo "=== LOGS READING ==="
ls logs
if [ -d "logs" ] && ls logs/*.json 1>/dev/null 2>&1; then
    printf '[' > "$JSON_OUTPUT_FILE"
    first=true
    for file in logs/*.json; do
        if [ "$first" = true ]; then
            first=false
        else
            printf ',' >> "$JSON_OUTPUT_FILE"
        fi
        cat "$file" >> "$JSON_OUTPUT_FILE"
    done
    printf ']\n' >> "$JSON_OUTPUT_FILE"
else
    echo '[]' > "$JSON_OUTPUT_FILE"
fi
