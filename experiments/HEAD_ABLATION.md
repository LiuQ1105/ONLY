# ONLY Layer 15: Single Attention Head Ablation

This experiment fixes ONLY at layer 15, disables AD-HH, and independently zeroes each specified attention head on the **ordinary attention path**, leaving ONLY's contrastive/CD branch intact. Head indexes are zero-based. Unlike AD-HH, this is full-head ablation for every query position, with no threshold.

## Setup

Clone this branch and run from a GPU-enabled Docker container with the existing LLaVA-1.5-7B model and MME data mounted at `/workspace/models` and `/workspace/datasets`.

```bash
git clone --branch experiment/layer15-head-ablation https://github.com/LiuQ1105/ONLY.git ONLY_head_sweep
cd ONLY_head_sweep
export PYTHONPATH="$PWD/transformers/src"
# Verify the active transformers actually comes from this checkout:
python -c 'import transformers; print(transformers.__file__)'
```

Default paths:
- Model: `/workspace/models/llava-v1.5-7b`
- Dataset: `/workspace/datasets/MME_Benchmark_release_version`
- Questions: `mme_hallucination.jsonl`
- Output: `/workspace/outputs/mme_head_sweep_layer15`

## Test one head first

```bash
HEAD_START=0 HEAD_END=0 bash experiments/cd_scripts/mme_head_sweep_layer15.sh
```

Then run the remaining heads; the baseline and completed head 0 will be reused:

```bash
HEAD_START=1 HEAD_END=31 bash experiments/cd_scripts/mme_head_sweep_layer15.sh
```

To rerun a range, set `FORCE=1`. To test different ONLY and ablation layers, set `ONLY_LAYER` and `HEAD_LAYER`. For example, `HEAD_LAYER=26` targets head 26:x while retaining ONLY layer 15.

The script generates baseline and head answer JSONL, conversion/scoring logs, a CSV of scores and deltas, and (if matplotlib/numpy are installed) a 1x32 heatmap PNG/PDF.

**Experimental caveats:** The baseline must be generated with the same settings and seed to compare deltas fairly. With sampling enabled, stochastic variability remains; repeat promising candidates across seeds. The MME evaluator in this repo reports only existence, count, position and color (not the full MME benchmark). Compare result scores against this script's own baseline, not a past run with unverified generation settings. The scripts have not been GPU-tested in this GitHub change.
