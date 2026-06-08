#!/usr/bin/env python3
"""
Generate deterministic synthetic inputs (.ma summary stats + annotation .txt)
for the SBayesRC end-to-end smoke test from a PLINK .bim file.

The numbers are not biologically meaningful — the goal is to produce inputs
that exercise the GPU pipeline (eigen-LD assembly, MCMC, post-processing)
without requiring a real GWAS or the real annotation matrix.

Usage:
    python tests/gen_smoke_inputs.py \\
        --bim tests/data/1000G_eur_chr22.bim \\
        --ma-out  tests/data/smoke.ma \\
        --annot-out tests/data/smoke.annot.txt \\
        --n 503

Reproducibility:
    Driven by a fixed seed (12345). Same inputs → same outputs.
"""

from __future__ import annotations

import argparse
import csv
import random
import math
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bim",        type=Path, required=True)
    ap.add_argument("--ma-out",     type=Path, required=True)
    ap.add_argument("--annot-out",  type=Path, required=True)
    ap.add_argument("--n",          type=int, default=503,
                    help="Sample size to write in N column of the .ma file.")
    ap.add_argument("--causal-frac", type=float, default=0.1,
                    help="Fraction of SNPs marked Anno=1 (rest are Anno=0).")
    ap.add_argument("--seed",       type=int, default=12345)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # Read BIM: 6 cols, tab-separated → CHR, SNP, CM, BP, A1, A2.
    snps = []
    with args.bim.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            chrom, snp, _cm, bp, a1, a2 = parts[:6]
            snps.append((snp, a1, a2))
    print(f"Read {len(snps)} SNPs from {args.bim}")

    # Write summary stats. Columns: SNP A1 A2 FREQ BETA SE P N.
    with args.ma_out.open("w") as f:
        f.write("SNP\tA1\tA2\tfreq\tb\tse\tp\tN\n")
        for snp, a1, a2 in snps:
            # Allele freq: Beta(2, 2)-like, clamp to [0.05, 0.95] to avoid
            # numerical issues at boundaries.
            freq = max(0.05, min(0.95, rng.betavariate(2, 2)))
            # Effect size: normal(0, 0.01). Most are near 0, occasional larger.
            beta = rng.gauss(0.0, 0.01)
            # SE: scale ~1/sqrt(N), with some heterogeneity.
            se = max(1e-5, 1.0 / math.sqrt(2 * args.n * freq * (1 - freq)) * rng.uniform(0.9, 1.1))
            # P-value approximated from z = beta/se via normal tail. Crude but
            # the value is rarely read by GCTB for SBayesRC.
            z = abs(beta / se)
            p = max(1e-300, math.erfc(z / math.sqrt(2)))
            f.write(f"{snp}\t{a1}\t{a2}\t{freq:.6f}\t{beta:.6e}\t{se:.6e}\t{p:.3e}\t{args.n}\n")
    print(f"Wrote summary stats → {args.ma_out}")

    # Write annotation file. Columns: SNP Intercept Anno.
    with args.annot_out.open("w") as f:
        f.write("SNP\tIntercept\tAnno\n")
        for snp, _a1, _a2 in snps:
            anno = 1 if rng.random() < args.causal_frac else 0
            f.write(f"{snp}\t1\t{anno}\n")
    print(f"Wrote annotation → {args.annot_out}")


if __name__ == "__main__":
    main()
