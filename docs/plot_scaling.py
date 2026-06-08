"""
Render the SBayesRC GPU-vs-CPU scaling plot for the README from a CSV of
sweep runs.

Input CSV columns: impl, chain_length, num_chains, burn_in, wall_seconds.
Where impl ∈ {"cpu", "gpu"} and wall_seconds is either a float or "FAILED".

Plot: x = total MCMC samples (chain_length × num_chains), y = wall time.
Solid lines connect the chain-length sweep (num_chains=1) for each impl.
Open markers show the multi-chain points (num_chains > 1, fixed chain_length),
which sit BELOW the corresponding longer-single-chain points on GPU because
of cross-chain d_annoMat sharing.

Usage:
    python docs/plot_scaling.py --csv scaling_results.csv --out docs/scaling.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_csv(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["wall_seconds"] == "FAILED":
                continue
            rows.append({
                "impl":         r["impl"],
                "chain_length": int(r["chain_length"]),
                "num_chains":   int(r["num_chains"]),
                "burn_in":      int(r["burn_in"]),
                "wall_seconds": float(r["wall_seconds"]),
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", default="SBayesRC scaling: 1× H100 vs 64-core Xeon (HM3, 1.15M SNPs)")
    args = ap.parse_args()

    rows = parse_csv(args.csv)
    if not rows:
        raise SystemExit(f"No usable rows in {args.csv}")

    for r in rows:
        r["total_samples"] = r["chain_length"] * r["num_chains"]

    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)

    style = {
        "cpu": {"color": "#d65f5f", "marker": "o", "label": "CPU (64-core Xeon)"},
        "gpu": {"color": "#5f88d6", "marker": "s", "label": "GPU (1× H100)"},
    }

    # --- Main lines: chain-length sweep (num_chains == 1) ---------------------
    for impl in ("cpu", "gpu"):
        line_rows = sorted(
            [r for r in rows if r["num_chains"] == 1],
            key=lambda r: r["total_samples"],
        )
        line_rows = [r for r in line_rows if r["impl"] == impl]
        if not line_rows:
            continue
        xs = [r["total_samples"] for r in line_rows]
        ys = [r["wall_seconds"]  for r in line_rows]
        ax.plot(xs, ys,
                color=style[impl]["color"],
                marker=style[impl]["marker"],
                label=style[impl]["label"],
                linewidth=2.4, markersize=8.5, alpha=0.95, zorder=3)

    # --- Multi-chain points (num_chains > 1) ---------------------------------
    # Plot as open markers + dashed connector. Same colors so they read as
    # the same impl, but visually distinct so the reader can see "wide vs long".
    for impl in ("cpu", "gpu"):
        multi_rows = sorted(
            [r for r in rows if r["num_chains"] > 1 and r["impl"] == impl],
            key=lambda r: r["total_samples"],
        )
        if not multi_rows:
            continue
        xs = [r["total_samples"] for r in multi_rows]
        ys = [r["wall_seconds"]  for r in multi_rows]
        ax.plot(xs, ys,
                color=style[impl]["color"],
                marker=style[impl]["marker"],
                linestyle="--",
                markerfacecolor="white",
                markeredgecolor=style[impl]["color"],
                markeredgewidth=1.8,
                linewidth=1.4, markersize=9, alpha=0.85, zorder=2)

    # --- Annotations: label every point with (chain_length × num_chains) -----
    for r in rows:
        tag = f"{r['chain_length']}×{r['num_chains']}"
        c = style[r["impl"]]["color"]
        dy = 14 if r["impl"] == "gpu" else -16
        ax.annotate(tag,
                    xy=(r["total_samples"], r["wall_seconds"]),
                    xytext=(0, dy),
                    textcoords="offset points",
                    fontsize=7.5, ha="center",
                    color=c, alpha=0.75)

    ax.set_xlabel("Total MCMC samples  (chain_length × num_chains)")
    ax.set_ylabel("Wall time (seconds)")
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Dense, human-readable y-axis ticks instead of just powers of 10.
    yticks = [100, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 5000]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{t}" for t in yticks])
    ax.minorticks_off()
    # And x-axis ticks at every actual config x.
    xticks = sorted({r["total_samples"] for r in rows})
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{t}" for t in xticks])

    ax.grid(True, which="both", linewidth=0.4, alpha=0.5)

    # Explicit 4-entry legend so the marker semantics are unambiguous.
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=style["cpu"]["color"], marker="o", markersize=8,
               linewidth=2.4, label="CPU — single chain (vary length)"),
        Line2D([0], [0], color=style["gpu"]["color"], marker="s", markersize=8,
               linewidth=2.4, label="GPU — single chain (vary length)"),
        Line2D([0], [0], color=style["cpu"]["color"], marker="o", markersize=9,
               markerfacecolor="white", markeredgecolor=style["cpu"]["color"],
               markeredgewidth=1.8, linestyle="--", linewidth=1.4,
               label="CPU — multi-chain (length=500, vary chains)"),
        Line2D([0], [0], color=style["gpu"]["color"], marker="s", markersize=9,
               markerfacecolor="white", markeredgecolor=style["gpu"]["color"],
               markeredgewidth=1.8, linestyle="--", linewidth=1.4,
               label="GPU — multi-chain (length=500, vary chains)"),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="lower right", fontsize=9)
    ax.set_title(args.title)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Wrote {args.out}")

    # Speedup summary table.
    print("\nSpeedup summary:")
    print(f"{'config':<35} {'CPU (s)':>10} {'GPU (s)':>10} {'speedup':>10}")
    by_key = {}
    for r in rows:
        key = (r["chain_length"], r["num_chains"])
        by_key.setdefault(key, {})[r["impl"]] = r["wall_seconds"]
    for (cl, nc), d in sorted(by_key.items()):
        if "cpu" in d and "gpu" in d and d["gpu"] > 0:
            sp = d["cpu"] / d["gpu"]
            print(f"chain_length={cl} num_chains={nc:<5} {d['cpu']:>10.2f} {d['gpu']:>10.2f} {sp:>9.2f}×")


if __name__ == "__main__":
    main()
