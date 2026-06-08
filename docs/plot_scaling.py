"""
Render the two-panel scaling plot for the README from a CSV of sweep runs.

Input CSV columns: impl, chain_length, num_chains, burn_in, wall_seconds.
Where impl ∈ {"cpu", "gpu"} and wall_seconds is either a float or "FAILED".

Plot A: x = chain_length, y = wall_seconds, lines = {cpu, gpu}.
        Filtered to num_chains == 1.
Plot B: x = num_chains, y = wall_seconds, lines = {cpu, gpu}.
        Filtered to chain_length == 2000.

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
    ap.add_argument("--title", default="SBayesRC chr22 scaling (1 H100 vs 64-core Xeon)")
    args = ap.parse_args()

    rows = parse_csv(args.csv)
    if not rows:
        raise SystemExit(f"No usable rows in {args.csv}")

    plot_a = sorted([r for r in rows if r["num_chains"] == 1],
                    key=lambda r: (r["impl"], r["chain_length"]))
    # Plot B uses whichever chain_length has the most num_chains points (the
    # one the sweep actually exercised).
    cl_counts: dict[int, int] = {}
    for r in rows:
        if r["num_chains"] >= 1:
            cl_counts[r["chain_length"]] = cl_counts.get(r["chain_length"], 0) + 1
    plot_b_cl = max(cl_counts, key=lambda cl: cl_counts[cl])
    plot_b = sorted([r for r in rows if r["chain_length"] == plot_b_cl],
                    key=lambda r: (r["impl"], r["num_chains"]))

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

    colors = {"cpu": "#d65f5f", "gpu": "#5f88d6"}
    markers = {"cpu": "o", "gpu": "s"}
    labels = {"cpu": "CPU (64-core Xeon)", "gpu": "GPU (1× H100)"}

    for impl in ("cpu", "gpu"):
        xs = [r["chain_length"] for r in plot_a if r["impl"] == impl]
        ys = [r["wall_seconds"] for r in plot_a if r["impl"] == impl]
        if xs:
            ax_a.plot(xs, ys, color=colors[impl], marker=markers[impl],
                      label=labels[impl], linewidth=2, markersize=7)

    ax_a.set_xlabel("MCMC chain length (iterations)")
    ax_a.set_ylabel("Wall time (s)")
    ax_a.set_title("A. Single-chain scaling")
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.grid(True, which="both", linewidth=0.4, alpha=0.5)
    ax_a.legend(frameon=False, loc="lower right")

    for impl in ("cpu", "gpu"):
        xs = [r["num_chains"] for r in plot_b if r["impl"] == impl]
        ys = [r["wall_seconds"] for r in plot_b if r["impl"] == impl]
        if xs:
            ax_b.plot(xs, ys, color=colors[impl], marker=markers[impl],
                      label=labels[impl], linewidth=2, markersize=7)

    ax_b.set_xlabel("Number of MCMC chains")
    ax_b.set_ylabel("Wall time (s)")
    ax_b.set_title(f"B. Multi-chain scaling (length={plot_b_cl})")
    ax_b.grid(True, which="both", linewidth=0.4, alpha=0.5)
    ax_b.legend(frameon=False, loc="upper left")

    fig.suptitle(args.title, fontsize=12)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Wrote {args.out}")

    # Also dump a small speedup summary.
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
