#!/usr/bin/env bash
# End-to-end GPU smoke test. Builds an LD eigen-decomp from a vendored 1000G
# chr22 BED file (~2 MB), generates deterministic synthetic GWAS summary
# stats + annotations, then runs the GPU MCMC pipeline for a few iterations
# and checks the output snpRes file exists with the expected number of rows.
#
# Run locally on a GPU machine:
#   tests/test_e2e_gpu.sh
#
# Or inside the Docker image:
#   docker run --gpus all --rm -v $(pwd):/work -w /work sbayesrc-gpu \
#       tests/test_e2e_gpu.sh
#
# Requires:
#   - gctb on PATH (the patched build with USE_GPU=1)
#   - python3 (only standard library, no numpy/pandas)
#   - nvidia GPU + driver
#
# Skips silently if no nvidia-smi is detected (so CI without GPUs doesn't fail).

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
test_data="${repo_root}/tests/data"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "SKIP: nvidia-smi not on PATH (no GPU detected)."
    exit 0
fi
if ! command -v gctb >/dev/null 2>&1; then
    echo "FAIL: gctb not on PATH. Build the patched binary and PATH-add it." >&2
    exit 1
fi

work="$(mktemp -d)"
cleanup() { rm -rf "$work"; }
trap cleanup EXIT

cp "${test_data}"/1000G_eur_chr22.{bed,bim,fam} "${work}/"
cp "${test_data}/ref_b37_1588blocks.pos" "${work}/blocks.pos"

bfile="${work}/1000G_eur_chr22"
nsnp=$(wc -l < "${bfile}.bim" | tr -d ' ')
echo "[e2e] $nsnp SNPs in vendored chr22 BED"

echo "[e2e] Step 1: build block LDM from BED ..."
gctb --bfile "${bfile}" \
     --make-block-ldm \
     --block-info "${work}/blocks.pos" \
     --out "${work}/ldm" >"${work}/step1.log" 2>&1
if [ ! -f "${work}/ldm/ldm.info" ]; then
    echo "FAIL: ldm/ldm.info missing after --make-block-ldm. See ${work}/step1.log" >&2
    tail -20 "${work}/step1.log" >&2
    exit 1
fi
nblocks=$(($(wc -l < "${work}/ldm/ldm.info") - 1))
echo "[e2e]   built $nblocks LD blocks"

echo "[e2e] Step 2: generate synthetic .ma + annotation (fixed seed) ..."
python3 "${repo_root}/tests/gen_smoke_inputs.py" \
    --bim "${bfile}.bim" \
    --ma-out  "${work}/smoke.ma" \
    --annot-out "${work}/smoke.annot.txt" >"${work}/step2.log" 2>&1

echo "[e2e] Step 3: eigen-decompose the LDM (CPU step in upstream gctb) ..."
gctb --ldm "${work}/ldm" \
     --gwas-summary "${work}/smoke.ma" \
     --make-ldm-eigen \
     --out "${work}/ldm" >"${work}/step3.log" 2>&1
mapfile -t eigen_files < <(find "${work}/ldm" -maxdepth 1 -name '*.eigen.bin' -print)
if [ "${#eigen_files[@]}" -eq 0 ]; then
    echo "FAIL: no *.eigen.bin produced. See ${work}/step3.log" >&2
    tail -30 "${work}/step3.log" >&2
    exit 1
fi
neigen=${#eigen_files[@]}
echo "[e2e]   built $neigen eigen-decomp block files"

echo "[e2e] Step 4: run GPU SBayesRC (--gwfm RC --use-gpu, short chain) ..."
SBRC_SKIP_FINDBEST=1 gctb \
    --gwfm RC \
    --use-gpu \
    --ldm-eigen "${work}/ldm" \
    --gwas-summary "${work}/smoke.ma" \
    --annot "${work}/smoke.annot.txt" \
    --num-chains 1 \
    --chain-length 50 \
    --burn-in 10 \
    --thread 4 \
    --out "${work}/result" >"${work}/step4.log" 2>&1

if [ ! -f "${work}/result.snpRes" ]; then
    echo "FAIL: result.snpRes missing. See ${work}/step4.log" >&2
    tail -40 "${work}/step4.log" >&2
    exit 1
fi
nrow=$(($(wc -l < "${work}/result.snpRes") - 1))
echo "[e2e]   snpRes has $nrow rows (expected $nsnp)"
if [ "$nrow" -lt $((nsnp / 2)) ]; then
    echo "FAIL: snpRes row count $nrow is implausibly small (expected ~$nsnp)" >&2
    tail -30 "${work}/step4.log" >&2
    exit 1
fi

echo "[e2e] PASS: end-to-end GPU pipeline produced a valid snpRes."
