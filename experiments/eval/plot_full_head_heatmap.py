#!/usr/bin/env python3
"""Plot a 32x32 ablation-layer x head-index heatmap relative to ONLY-only."""
import argparse
import csv
import math
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true",
                        help="Render a partial grid with missing cells left gray")
    args = parser.parse_args()
    root = args.root
    data = [[math.nan] * 32 for _ in range(32)]
    details = []
    for layer in range(32):
        csv_file = root / f"layer_{layer}" / "head_sweep_seed5.csv"
        if not csv_file.exists():
            continue
        with csv_file.open(newline="") as f:
            for row in csv.DictReader(f):
                h = int(row["head"])
                l = int(row["layer"])
                if l != layer or not (0 <= h < 32):
                    raise ValueError(f"Unexpected layer/head in {csv_file}: {l}:{h}")
                d = float(row["delta_total"])
                data[layer][h] = d
                details.append((l, h, d, float(row["total"])))

    missing = [(i, j) for i in range(32) for j in range(32)
               if not math.isfinite(data[i][j])]
    if missing and not args.allow_partial:
        raise SystemExit(
            f"Only {1024-len(missing)}/1024 layer/head cells scored. "
            f"Missing first cells: {missing[:10]}. "
            "Use --allow-partial to preview incomplete sweep."
        )
    if not details:
        raise SystemExit("No scores found.")
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    matrix = np.ma.masked_invalid(np.array(data))
    limit = max(float(np.max(np.abs(matrix))), 1.0)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#dddddd")
    fig, ax = plt.subplots(figsize=(15, 12))
    image = ax.imshow(matrix, cmap=cmap,
                      norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
                      interpolation="nearest", aspect="equal")
    ax.set_xticks(range(32))
    ax.set_yticks(range(32))
    ax.set_xlabel("Ablated Head Index")
    ax.set_ylabel("Ablated Layer Index")
    ax.set_title("LLaVA-1.5-7B | ONLY Layer 15 + Single-Head Ablation (seed 5)")
    ax.tick_params(labelsize=8)
    fig.colorbar(image, ax=ax, shrink=0.83,
                 label="MME delta vs ONLY-only baseline (positive = improvement)")
    fig.tight_layout()

    for extension in ("png", "pdf"):
        out = root / f"head_heatmap_32x32_seed5.{extension}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print("Saved:", out)
    plt.close(fig)
    ranked = sorted(details, key=lambda r: r[2], reverse=True)
    out_csv = root / "all_layer_head_scores_seed5.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "head", "delta_total", "total"])
        writer.writerows(sorted(details, key=lambda r:(r[0],r[1])))
    print("Saved:", out_csv)
    print("Scored:", len(details), "/ 1024")
    print("Top 10 ablations:")
    for layer, head, delta, score in ranked[:10]:
        print(f"  L{layer:02d}:H{head:02d}: delta={delta:+.2f}, score={score:.2f}")

if __name__ == "__main__":
    main()
