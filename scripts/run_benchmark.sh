#!/bin/bash
# Reproduce the published 5.04× SBayesRC benchmark on UKB-EUR-Imputed.
#
# Requires:
#  - Patched gctb binary built with USE_GPU=1 (see README.md "How to build")
#  - Eigen-decomposed LD reference at $LDREF (e.g. ukbEUR_Imputed)
#  - rsq0.5.pwld inside the LD ref directory
#  - A trait .ma file at $MA_FILE
#  - An annotation .txt file at $ANNOT_FILE
#  - gene-map .txt file at $GMAP
#
# Reproduces:
#  - SBayesRC with 4 chains × 2000 iter on (height × A_sbrc baseline-LF).
#  - Expected wall on one H100 80 GB with the full stack enabled: ~2h 40m.
#  - CPU baseline on a 64-core Xeon (no GPU) for the same config: ~13h 40m.

set -euo pipefail

GCTB=${GCTB:-/usr/local/bin/gctb}
LDREF=${LDREF:-./ukbEUR_Imputed}
MA_FILE=${MA_FILE:-./height.imputed.ma}
ANNOT_FILE=${ANNOT_FILE:-./A_sbrc.annot.txt}
GMAP=${GMAP:-./gene_map_hg38_hg19.txt}
OUT=${OUT:-./results/sbrc_gpu}
NUM_CHAINS=${NUM_CHAINS:-4}
CHAIN_LENGTH=${CHAIN_LENGTH:-2000}
BURN_IN=${BURN_IN:-500}
THREAD=${THREAD:-16}

mkdir -p "$(dirname "$OUT")"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo
echo "[$(date)] Starting GPU SBayesRC: $NUM_CHAINS chains × $CHAIN_LENGTH iter"
echo

time SBRC_GPU_R=${SBRC_GPU_R:-1} \
     SBRC_GPU_GIBBS=${SBRC_GPU_GIBBS:-1} \
     SBRC_SKIP_FINDBEST=${SBRC_SKIP_FINDBEST:-1} \
     "$GCTB" \
        --gwfm RC \
        --use-gpu \
        --ldm-eigen "$LDREF" \
        --pwld-file rsq0.5.pwld \
        --gwas-summary "$MA_FILE" \
        --annot "$ANNOT_FILE" \
        --gene-map "$GMAP" \
        --num-chains "$NUM_CHAINS" \
        --chain-length "$CHAIN_LENGTH" \
        --burn-in "$BURN_IN" \
        --thread "$THREAD" \
        --out "$OUT"

echo
echo "[$(date)] DONE. Output: ${OUT}.*"
echo "Compare against CPU baseline with: $GCTB --gwfm RC (no --use-gpu)"
