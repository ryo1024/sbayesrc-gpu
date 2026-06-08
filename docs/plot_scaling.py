"""
Render the SBayesRC GPU-vs-CPU scaling plot for the README from a CSV of
sweep runs.

Input CSV columns: impl, chain_length, num_chains, burn_in, wall_seconds.
Where impl ∈ {"cpu", "gpu"} and wall_seconds is either a float or "FAILED".

Plot: x = chain_length (or chain_length × num_chains in --mode=full).
Two solid lines (CPU red, GPU blue) over the single-chain length sweep. In
--mode=full, also overlay open markers for the multi-chain points.

Usage:
    # Simple plot for the README: single-chain length sweep only.
    python docs/plot_scaling.py --csv scaling_results.csv \\
        --out docs/scaling.png --mode simple

    # Full plot for docs/components.md: also shows the multi-chain points.
    python docs/plot_scaling.py --csv scaling_results.csv \\
        --out docs/scaling_full.png --mode full
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


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
    ap.add_argument("--mode", choices=("simple", "full"), default="simple",
                    help="simple = single-chain length sweep only (README); "
                         "full = also overlay multi-chain points (components.md)")
    ap.add_argument("--title",
                    default="SBayesRC scaling: 1× H100 vs 64-core Xeon (HM3, 1.15M SNPs)")
    args = ap.parse_args()

    rows = parse_csv(args.csv)
    if not rows:
        raise SystemExit(f"No usable rows in {args.csv}")

    for r in rows:
        r["total_samples"] = r["chain_length"] * r["num_chains"]

    figsize = (6.4, 4.2) if args.mode == "simple" else (8.5, 5.5)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    style = {
        "cpu": {"color": "#d65f5f", "marker": "o", "label": "CPU (64-core Xeon)"},
        "gpu": {"color": "#5f88d6", "marker": "s", "label": "GPU (1× H100)"},
    }

    # --- Main lines: single-chain length sweep --------------------------------
    for impl in ("cpu", "gpu"):
        line_rows = sorted(
            [r for r in rows if r["num_chains"] == 1 and r["impl"] == impl],
            key=lambda r: r["chain_length"],
        )
        if not line_rows:
            continue
        xs = [r["chain_length"]  for r in line_rows]
        ys = [r["wall_seconds"]  for r in line_rows]
        ax.plot(xs, ys,
                color=style[impl]["color"],
                marker=style[impl]["marker"],
                label=style[impl]["label"],
                linewidth=2.4, markersize=8.5, alpha=0.95, zorder=3)

    # --- Multi-chain overlay (only in --mode=full) ---------------------------
    if args.mode == "full":
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
        for r in [r for r in rows if r["num_chains"] > 1]:
            tag = f"{r['chain_length']}×{r['num_chains']}"
            c = style[r["impl"]]["color"]
            dy = 14 if r["impl"] == "gpu" else -16
            ax.annotate(tag,
                        xy=(r["total_samples"], r["wall_seconds"]),
                        xytext=(0, dy), textcoords="offset points",
                        fontsize=7.5, ha="center", color=c, alpha=0.75)
        ax.set_xlabel("Total MCMC samples  (chain_length × num_chains)")
    else:
        ax.set_xlabel("MCMC chain length (iterations)")

    ax.set_ylabel("Wall time (seconds)")
    ax.set_xscale("log")
    ax.set_yscale("log")

    yticks = [200, 300, 500, 700, 1000, 1500, 2000, 3000, 5000]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{t}" for t in yticks])
    ax.minorticks_off()

    if args.mode == "full":
        xticks = sorted({r["total_samples"] for r in rows})
    else:
        xticks = sorted({r["chain_length"] for r in rows if r["num_chains"] == 1})
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{t}" for t in xticks])

    ax.grid(True, which="both", linewidth=0.4, alpha=0.5)

    if args.mode == "full":
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
    else:
        ax.legend(frameon=False, loc="lower right", fontsize=10)

    ax.set_title(args.title, fontsize=11)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Wrote {args.out}")

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
