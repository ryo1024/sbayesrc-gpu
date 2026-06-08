#!/bin/bash
# Two-panel SBayesRC scaling sweep. Reproduces docs/scaling.png against your
# local CPU + GPU binaries and LD reference.
#
# Plot A: vary chain-length at num-chains=1.
# Plot B: vary num-chains at chain-length=500.
#
# Idempotent: existing rows in $OUT_DIR/scaling_results.csv are reused, so this
# script is safe to re-launch after SLURM preemption.
#
# Required env (override on the command line):
#   GCTB_GPU   — path to the patched GCTB binary built with USE_GPU=1
#   GCTB_CPU   — path to a vanilla GCTB binary for the CPU baseline
#   LDREF      — eigen-LD reference dir (e.g. ukbEUR_HM3)
#   MA_FILE    — GWAS summary stats (.ma)
#   ANNOT      — annotation file (.txt)
#   GMAP       — gene-map file
#   OUT_DIR    — where to write per-run logs and scaling_results.csv

set -euo pipefail

GCTB_GPU="${GCTB_GPU:?set GCTB_GPU to the patched binary path}"
GCTB_CPU="${GCTB_CPU:?set GCTB_CPU to a vanilla GCTB binary for the baseline}"
LDREF="${LDREF:?set LDREF to your eigen-LD reference dir}"
MA_FILE="${MA_FILE:?set MA_FILE}"
ANNOT="${ANNOT:?set ANNOT}"
GMAP="${GMAP:?set GMAP}"
OUT_DIR="${OUT_DIR:-./scaling_out}"
CSV="${OUT_DIR}/scaling_results.csv"

mkdir -p "${OUT_DIR}"
if [ ! -f "${CSV}" ]; then
    echo "impl,chain_length,num_chains,burn_in,wall_seconds" > "${CSV}"
fi

# No `cd` — every path above is absolute and gctb writes to absolute --out paths.

already_done() {
    local impl="$1" cl="$2" nc="$3"
    grep -q "^${impl},${cl},${nc}," "${CSV}" 2>/dev/null
}

run_one() {
    local impl="$1" chain_length="$2" num_chains="$3" burn_in="$4"
    if already_done "${impl}" "${chain_length}" "${num_chains}"; then
        echo "[$(date +%H:%M:%S)] SKIP ${impl} cl=${chain_length} nc=${num_chains} (already in CSV)"
        return
    fi
    local out="${OUT_DIR}/hm3_${impl}_cl${chain_length}_nc${num_chains}"
    local logf="${out}.log"
    echo "[$(date +%H:%M:%S)] RUN  ${impl} cl=${chain_length} nc=${num_chains} burn_in=${burn_in}"
    local bin extra_env extra_flag
    if [[ "${impl}" == "gpu" ]]; then
        bin="${GCTB_GPU}"
        extra_env=(env SBRC_GPU_R=1 SBRC_GPU_GIBBS=1 SBRC_SKIP_FINDBEST=1)
        extra_flag="--use-gpu"
    else
        bin="${GCTB_CPU}"
        extra_env=(env SBRC_SKIP_FINDBEST=1)
        extra_flag=""
    fi
    local t0 t1
    t0=$(date +%s.%N)
    set +e
    # NOTE: NO `--chr 22` filter — we want the full 1.15M HM3 panel so per-iter
    # cost dominates and the GPU/CPU lines actually diverge.
    OMP_NUM_THREADS=16 "${extra_env[@]}" "${bin}" \
        --gwfm RC ${extra_flag} \
        --ldm-eigen "${LDREF}" \
        --pwld-file rsq0.5.pwld \
        --gwas-summary "${MA_FILE}" \
        --annot "${ANNOT}" \
        --gene-map "${GMAP}" \
        --num-chains "${num_chains}" \
        --chain-length "${chain_length}" \
        --burn-in "${burn_in}" \
        --thread 16 \
        --out "${out}" >"${logf}" 2>&1
    local rc=$?
    set -e
    t1=$(date +%s.%N)
    local wall
    wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b - a}')
    if [[ $rc -ne 0 ]]; then
        echo "  FAILED (rc=$rc); see ${logf}"
        echo "${impl},${chain_length},${num_chains},${burn_in},FAILED" >> "${CSV}"
    else
        echo "  ${wall}s"
        echo "${impl},${chain_length},${num_chains},${burn_in},${wall}" >> "${CSV}"
    fi
}

# Plot A: vary chain-length at num-chains=1.
# Burn-in stays modest so chain-only time is what scales.
for cl in 100 250 500 1000 2000; do
    bi=$((cl/4 > 100 ? 100 : cl/4))
    run_one cpu "$cl" 1 "$bi"
    run_one gpu "$cl" 1 "$bi"
done

# Plot B: vary num-chains at chain-length=500.
for nc in 1 2 4 8; do
    run_one cpu 500 "$nc" 100
    run_one gpu 500 "$nc" 100
done

echo
echo "DONE. Results: ${CSV}"
cat "${CSV}"
