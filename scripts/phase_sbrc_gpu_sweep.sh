#!/bin/bash
# Array-job launcher for the SBayesRC GWFM sweep on GPU. Submits one SLURM task per
# (condition, trait) pair so multiple GPUs work in parallel.
#
# Usage:
#   bash slurm/phase_sbrc_gpu_sweep.sh                  # default: 3 conds × 2 traits = 6-job MVP gate
#   CONDITIONS="A_sbrc B_sbrc K_sbrc" TRAITS="height ldl" bash slurm/phase_sbrc_gpu_sweep.sh
#   CHAIN_LENGTH=2000 SBRC_GPU_R=1 SBRC_SKIP_FINDBEST=1 bash slurm/phase_sbrc_gpu_sweep.sh
#
# Each submitted task uses --gpus=1 --qos=scavenge per the cluster policy for
# autonomous research GPU work.

set -euo pipefail
cd /mnt/data/artifacts/ryo/glm_stat_gen/glm_finemapping

CONDITIONS="${CONDITIONS:-A_sbrc B_sbrc K_sbrc}"
TRAITS="${TRAITS:-height ldl}"
CHAIN_LENGTH="${CHAIN_LENGTH:-2000}"
BURN_IN="${BURN_IN:-500}"
NUM_CHAINS="${NUM_CHAINS:-4}"
LDREF_NAME="${LDREF_NAME:-ukbEUR_Imputed}"
SBRC_GPU_R_VAL="${SBRC_GPU_R:-0}"
SBRC_SKIP_FINDBEST_VAL="${SBRC_SKIP_FINDBEST:-0}"

echo "Sweep config:"
echo "  CONDITIONS=$CONDITIONS"
echo "  TRAITS=$TRAITS"
echo "  CHAIN_LENGTH=$CHAIN_LENGTH  BURN_IN=$BURN_IN  NUM_CHAINS=$NUM_CHAINS"
echo "  LDREF_NAME=$LDREF_NAME"
echo "  SBRC_GPU_R=$SBRC_GPU_R_VAL  SBRC_SKIP_FINDBEST=$SBRC_SKIP_FINDBEST_VAL"
echo

JOBIDS=()
for COND in $CONDITIONS; do
    for TRAIT in $TRAITS; do
        if [ ! -f "data/sbrc/sumstats_cojo/${TRAIT}.ma" ]; then
            echo "SKIP: $COND × $TRAIT (no .ma file)"
            continue
        fi
        if [ ! -f "data/sbrc/annot/${COND}.annot.txt" ]; then
            echo "SKIP: $COND × $TRAIT (no annot file)"
            continue
        fi
        echo -n "Submitting $COND × $TRAIT ... "
        JID=$(SBRC_GPU_R=$SBRC_GPU_R_VAL \
             SBRC_SKIP_FINDBEST=$SBRC_SKIP_FINDBEST_VAL \
             CHAIN_LENGTH=$CHAIN_LENGTH \
             BURN_IN=$BURN_IN \
             NUM_CHAINS=$NUM_CHAINS \
             LDREF_NAME=$LDREF_NAME \
             sbatch --parsable slurm/phase_sbrc_gpu_run.sh "$COND" "$TRAIT")
        echo "jobid $JID"
        JOBIDS+=("$JID")
    done
done

echo
echo "Submitted ${#JOBIDS[@]} jobs: ${JOBIDS[*]}"
echo
echo "Watch progress:"
echo "  squeue -u \$USER -o '%i %T %M %j' --states=running,pending"
echo "  watch -n30 'squeue -j ${JOBIDS[*]// /,}'"
