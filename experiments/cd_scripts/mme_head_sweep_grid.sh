#!/usr/bin/env bash
# Full 32x32 single-head ablation grid. ONLY remains at layer 15.
set -euo pipefail
cd "$(dirname "$0")/../.."

OUT="${GRID_ROOT:-/workspace/outputs/mme_head_sweep_grid}"
SOURCE="${BASELINE_ROOT:-/workspace/outputs/mme_head_sweep_layer15}"
LAYER_START="${LAYER_START:-0}"
LAYER_END="${LAYER_END:-31}"
HEAD_START="${HEAD_START:-0}"
HEAD_END="${HEAD_END:-31}"
ONLY_LAYER="${ONLY_LAYER:-15}"
SEED="${SEED:-5}"

if [[ ! -s "$SOURCE/answers/baseline.jsonl" ||
      ! -s "$SOURCE/metrics/baseline/logs.txt" ]]; then
  echo "Missing reference baseline at $SOURCE" >&2
  echo "Generate ONLY-only baseline first using mme_head_sweep_layer15.sh" >&2
  exit 1
fi
if [[ "$(wc -l < "$SOURCE/answers/baseline.jsonl")" -ne 240 ]]; then
  echo "Expected 240 baseline answers; verify BASELINE_ROOT before running." >&2
  exit 1
fi

for ((layer=LAYER_START; layer<=LAYER_END; layer++)); do
  if ((layer < 0 || layer > 31)); then
    echo "Invalid layer: $layer" >&2; exit 1
  fi
  dest="$OUT/layer_$layer"
  mkdir -p "$dest/answers" "$dest/metrics/baseline"
  # Reuse the matched seed-5 ONLY-only baseline; no repeated inference.
  if [[ ! -s "$dest/answers/baseline.jsonl" ]]; then
    cp "$SOURCE/answers/baseline.jsonl" "$dest/answers/baseline.jsonl"
  fi
  if [[ ! -s "$dest/metrics/baseline/logs.txt" ]]; then
    cp "$SOURCE/metrics/baseline/logs.txt" "$dest/metrics/baseline/logs.txt"
  fi
  # Reuse the 32 already-scored heads from the earlier Layer-15 run.
  if ((layer == 15)) && [[ "${REUSE_LAYER15:-1}" == "1" ]]; then
    for ((head=HEAD_START; head<=HEAD_END; head++)); do
      for relative in "answers/head_$head.jsonl" "metrics/head_$head/logs.txt"; do
        if [[ -s "$SOURCE/$relative" && ! -s "$dest/$relative" ]]; then
          mkdir -p "$(dirname "$dest/$relative")"
          cp "$SOURCE/$relative" "$dest/$relative"
        fi
      done
    done
  fi

  echo "=== Ablation layer $layer / ONLY layer $ONLY_LAYER ==="
  OUTPUT_ROOT="$dest" HEAD_LAYER="$layer" ONLY_LAYER="$ONLY_LAYER" \
    HEAD_START="$HEAD_START" HEAD_END="$HEAD_END" SEED="$SEED" \
    bash experiments/cd_scripts/mme_head_sweep_layer15.sh
done

python experiments/eval/plot_full_head_heatmap.py --root "$OUT"
