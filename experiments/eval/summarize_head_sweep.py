#!/usr/bin/env python3
"""Aggregate ONLY+ablation MME scores and create a 1xN heatmap."""
import argparse
import csv
from pathlib import Path

def read_scores(path):
    # eval_mme.py writes one comma-separated line per group to logs.txt:
    # Perception: existence,count, and Cognition: position,color.
    raw = path.read_text().strip().strip(",")
    values = [float(x) for x in raw.split(",") if x.strip()]
    if len(values) != 4:
        raise ValueError(f"Expected four category scores in {path}; got {values}")
    e, count, pos, color = values
    return [e, count, pos, color, e + count, pos + color, sum(values)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--head-layer", type=int, default=15)
    ap.add_argument("--head-start", type=int, default=0)
    ap.add_argument("--head-end", type=int, default=31)
    args = ap.parse_args()
    root = args.root
    names = ["existence", "count", "position", "color",
             "perception_total", "cognition_total", "total"]
    baseline = read_scores(root / "metrics" / "baseline" / "logs.txt")
    rows = []
    for head in range(args.head_start, args.head_end + 1):
        score_path = root / "metrics" / f"head_{head}" / "logs.txt"
        if not score_path.exists():
            print(f"Missing {score_path}; skipping")
            continue
        scores = read_scores(score_path)
        row = {"layer": args.head_layer, "head": head}
        row.update(dict(zip(names, scores)))
        row["delta_total"] = scores[-1] - baseline[-1]
        rows.append(row)
    csv_path = root / "head_sweep_seed5.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "head"] + names + ["delta_total"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Baseline score: {baseline[-1]:.2f}; saved {csv_path}")
    if not rows:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
        import numpy as np
    except ImportError:
        print("Plot skipped: install numpy and matplotlib")
        return
    deltas = [row["delta_total"] for row in rows]
    limit = max(max(abs(x) for x in deltas), 1.0)
    fig, ax = plt.subplots(figsize=(max(13, len(rows)*0.43), 2.7))
    image = ax.imshow(np.asarray([deltas]), cmap="RdBu_r",
                      norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
                      aspect="auto")
    ax.set_yticks([0], [f"Layer {args.head_layer}"])
    ax.set_xticks(range(len(rows)), [str(row["head"]) for row in rows])
    ax.set_xlabel("Ablated attention head")
    ax.set_title(f"ONLY Layer 15 + Head Ablation (seed 5) | baseline {baseline[-1]:.2f}")
    for j, delta in enumerate(deltas):
        ax.text(j, 0, f"{delta:+.1f}", ha="center", va="center",
                rotation=90, fontsize=7, color="black")
    fig.colorbar(image, ax=ax, label="MME score difference from ONLY-only baseline")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        output = root / f"head_heatmap_seed5.{ext}"
        fig.savefig(output, dpi=300, bbox_inches="tight")
        print(f"Saved {output}")
    plt.close(fig)

if __name__ == "__main__":
    main()
