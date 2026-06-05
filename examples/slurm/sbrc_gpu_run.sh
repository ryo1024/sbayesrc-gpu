#!/bin/bash
# GPU variant of the SBayesRC GWFM runner.
#
# Usage:
#   sbatch slurm/phase_sbrc_gpu_run.sh A_sbrc height
#   sbatch slurm/phase_sbrc_gpu_run.sh K_sbrc ldl
#
# Env knobs:
#   NUM_CHAINS=4            # GCTB internal chains; default 4 (matches CPU runner)
#   CHAIN_LENGTH=2000       # default production length
#   BURN_IN=500
#   LDREF_NAME=ukbEUR_Imputed  # or ukbEUR_HM3
#   SBRC_GPU_R=1            # also GPU-dispatch the pass-1 ApproxBayesR sampler.
#                            # Default OFF (RC pass-2 only). Set to 1 to opt-in once
#                            # the GCTB tuneEigenCutoff + ~Model() lifecycle patches
#                            # have been built into the binary at phase_sbrc_gpu/build/gctb_gpu.
#   GCTB_GPU=...            # override binary path
#
# Per the cluster policy: GPU jobs use --qos=scavenge by default for autonomous research.
# We do NOT set --cpus-per-task or --mem so the cluster's default GPU resource bundling applies.
#SBATCH --partition=h100
#SBATCH --gpus=1
#SBATCH --qos=scavenge
#SBATCH --job-name=sbrc_gpu
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/sbrc_gpu_%j.out
#SBATCH --error=slurm/logs/sbrc_gpu_%j.err

set -euo pipefail
# Set REPO_ROOT to wherever your cluster-side data + binaries live.
# Example: REPO_ROOT=/path/to/your/sbrc-work sbatch examples/slurm/sbrc_gpu_run.sh A_sbrc height
cd "${REPO_ROOT:?REPO_ROOT must be set to the cluster-side working directory}"

CONDITION="${1:?condition arg required (e.g. A_sbrc)}"
TRAIT="${2:?trait arg required (e.g. height)}"
NUM_CHAINS="${NUM_CHAINS:-4}"
LDREF_NAME="${LDREF_NAME:-ukbEUR_Imputed}"
CHAIN_LENGTH="${CHAIN_LENGTH:-2000}"
BURN_IN="${BURN_IN:-500}"

GCTB_GPU="${GCTB_GPU:-phase_sbrc_gpu/build/gctb_gpu}"
GCTB_CPU="tools/gctb_2.5.5_Linux/gctb"   # used only for impute-summary cache build
LDREF="data/sbrc_reference/${LDREF_NAME}"
GMAP="data/sbrc_reference/gene_map_hg38_hg19.txt"
MA_FILE="data/sbrc/sumstats_cojo/${TRAIT}.ma"
ANNOT_FILE="data/sbrc/annot/${CONDITION}.annot.txt"
PWLD_FILE="rsq0.5.pwld"

IMPUTED_DIR="data/sbrc/sumstats_imputed/${TRAIT}"
RESULT_DIR="data/sbrc/results_gpu/${CONDITION}/${TRAIT}"   # separate from CPU runs
mkdir -p "$IMPUTED_DIR" "$RESULT_DIR"

# ── Step 1: --impute-summary (per-trait, shared across conditions, CPU-only) ──
IMPUTED_MA="${IMPUTED_DIR}/imputed.ma.imputed.ma"
if [ ! -f "$IMPUTED_MA" ]; then
    echo "[$(date)] Step 1: --impute-summary for $TRAIT (CPU)"
    LOCK="${IMPUTED_DIR}/.lock"
    if mkdir "$LOCK" 2>/dev/null; then
        trap "rmdir '$LOCK' 2>/dev/null" EXIT
        "$GCTB_CPU" \
            --ldm-eigen "$LDREF" \
            --gwas-summary "$MA_FILE" \
            --impute-summary \
            --thread ${SLURM_CPUS_ON_NODE:-16} \
            --out "${IMPUTED_DIR}/imputed.ma"
        trap - EXIT
        rmdir "$LOCK" 2>/dev/null || true
    else
        echo "[$(date)] Waiting for sibling to finish --impute-summary …"
        while [ -d "$LOCK" ] && [ ! -f "$IMPUTED_MA" ]; do sleep 30; done
    fi
else
    echo "[$(date)] Imputed sumstats cached: $IMPUTED_MA"
fi

if [ ! -f "${LDREF}/${PWLD_FILE}" ]; then
    echo "ERROR: pwld file missing at ${LDREF}/${PWLD_FILE}" >&2
    exit 1
fi

# ── Step 2: --gwfm RC --use-gpu ───────────────────────────────────────────
echo "[$(date)] Step 2: --gwfm RC --use-gpu for $CONDITION/$TRAIT, ${NUM_CHAINS} chains, length=$CHAIN_LENGTH"
echo "[$(date)] Binary: $GCTB_GPU  SBRC_GPU_R=${SBRC_GPU_R:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Note: SBRC_GPU_R=1 enables the (still-experimental) ApproxBayesR pass-1 GPU hook.
# Validated to match CPU within MC noise on chr22 and full-Imputed 10-iter.
SBRC_GPU_R="${SBRC_GPU_R:-0}" "$GCTB_GPU" \
    --gwfm RC \
    --use-gpu \
    --ldm-eigen "$LDREF" \
    --pwld-file "$PWLD_FILE" \
    --gwas-summary "$IMPUTED_MA" \
    --annot "$ANNOT_FILE" \
    --gene-map "$GMAP" \
    --num-chains "$NUM_CHAINS" \
    --chain-length "$CHAIN_LENGTH" \
    --burn-in "$BURN_IN" \
    --thread ${SLURM_CPUS_ON_NODE:-16} \
    --out "${RESULT_DIR}/sbrc"

echo "[$(date)] DONE  $CONDITION/$TRAIT"
ls -la "$RESULT_DIR/"
