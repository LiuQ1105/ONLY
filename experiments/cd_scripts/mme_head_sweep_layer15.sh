#!/usr/bin/env bash
# ONLY Layer 15 + single-head ablation sweep (seed 5).
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/transformers/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

ROOT="${OUTPUT_ROOT:-/workspace/outputs/mme_head_sweep_layer15}"
DATA="${MME_DATA_ROOT:-/workspace/datasets/MME_Benchmark_release_version}"
MODEL="${MODEL_PATH:-/workspace/models/llava-v1.5-7b}"
QUESTION="${QUESTION_FILE:-$DATA/mme_hallucination.jsonl}"
SEED="${SEED:-5}"
LAYER="${ONLY_LAYER:-15}"
HEAD_LAYER="${HEAD_LAYER:-15}"
HEAD_START="${HEAD_START:-0}"
HEAD_END="${HEAD_END:-31}"
mkdir -p "$ROOT/answers" "$ROOT/logs" "$ROOT/metrics"
test -f "$QUESTION" || { echo "Missing $QUESTION" >&2; exit 1; }
test -d "$MODEL" || { echo "Missing model $MODEL" >&2; exit 1; }

run_one() {
  local name="$1" heads="$2"
  local answer="$ROOT/answers/${name}.jsonl"
  local logfile="$ROOT/logs/${name}.log"
  local metric="$ROOT/metrics/${name}"
  if [[ -s "$metric/logs.txt" && -s "$answer" && "${FORCE:-0}" != "1" ]]; then
    echo "SKIP $name (existing answers and scores)"
    return
  fi
  echo "RUN $name heads=$heads"
  # Fail fast instead of silently reusing incomplete results.
  rm -f "$answer"
  mkdir -p "$metric"
  python experiments/eval/mme_llava.py \
    --model-path "$MODEL" \
    --question-file "$QUESTION" \
    --image-folder "$DATA" \
    --answers-file "$answer" \
    --seed "$SEED" \
    --temperature 1.0 \
    --top_p 1 \
    --use_ritual False \
    --use_vcd False \
    --use_m3id False \
    --use_only True \
    --enhance_layer_index "$LAYER" \
    --use_adhh False \
    --ablate_heads "$heads" \
    --ritual_alpha_pos 3.0 \
    --ritual_alpha_neg 1.0 \
    --ritual_beta 0.4 \
    --js_gamma 0.25 > "$logfile" 2>&1
  python experiments/eval/convert_answer_to_mme.py \
    --output_path "$answer" --seed "$SEED" --model llava \
    --log_path "$metric" >> "$logfile" 2>&1
  python experiments/eval/eval_mme.py \
    --results_dir "$metric/mme_answers" --log_path "$metric" \
    --seed "$SEED" >> "$logfile" 2>&1
}

# Use exactly the same generation setup for the baseline and each head.
run_one baseline ""
for ((head=HEAD_START; head<=HEAD_END; head++)); do
  run_one "head_${head}" "${HEAD_LAYER}:${head}"
done
python experiments/eval/summarize_head_sweep.py \
  --root "$ROOT" --head-layer "$HEAD_LAYER" --head-start "$HEAD_START" --head-end "$HEAD_END"
echo "Done: $ROOT"
