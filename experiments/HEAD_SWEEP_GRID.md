# 32 × 32 Head Ablation Heatmap (ONLY Layer 15 fixed)

This experiment performs **one independent head ablation at a time**, on
LLaVA attention layers 0–31 and heads 0–31. It **does not use AD-HH**.
ONLY's intervention layer remains fixed at 15 for every run.

- x-axis: ablated head (0–31)
- y-axis: ablated attention layer (0–31)
- cell: MME total score minus the SAME seed-5 ONLY-only baseline (633.34 in the initial head sweep).

**Ablation layer 0–31 is NOT a new ONLY-layer sweep**.
There are 1024 conditions, 240 MME questions per condition (approx. 245,760 model
responses); this can take substantial GPU time.

## Run

Use the same local modified Transformers installation. Run from this checkout:

```bash
cd /workspace/ONLY_head_sweep
export PYTHONPATH="$PWD/transformers/src"
# Verify current source and run just two layers first
LAYER_START=14 LAYER_END=15 bash experiments/cd_scripts/mme_head_sweep_grid.sh
# Once validated, resume across the entire grid (completed runs skip)
bash experiments/cd_scripts/mme_head_sweep_grid.sh
```

Default input paths come from the existing single-layer sweep script.
The launcher expects the existing baseline here:
`/workspace/outputs/mme_head_sweep_layer15`, with a 240-line
`answers/baseline.jsonl` and `metrics/baseline/logs.txt`.
It copies baseline output (not recomputing it) into each layer directory,
and reuses available Layer-15 Head 0–31 runs from the prior sweep.

## Files

```text
/workspace/outputs/mme_head_sweep_grid/
  layer_0/head_sweep_seed5.csv
  ...
  layer_31/head_sweep_seed5.csv
  all_layer_head_scores_seed5.csv
  head_heatmap_32x32_seed5.png
  head_heatmap_32x32_seed5.pdf
```

To preview while some runs remain incomplete:

```bash
python experiments/eval/plot_full_head_heatmap.py \
  --root /workspace/outputs/mme_head_sweep_grid --allow-partial
```

Missing cells appear gray. For a complete heatmap the plot script requires
all 1024 condition scores. Runs can be resumed without setting FORCE.

If you change `ONLY_LAYER`, `SEED`, model, or decoding settings,
generate a **new matched baseline** and use a fresh `GRID_ROOT` to avoid
mixing incomparable results.
