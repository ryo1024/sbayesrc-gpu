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

# Numerical regression guard: the chain is short and the inputs are deterministic
# (fixed-seed synthetic), so summary stats should land in stable, narrow bands.
# FP non-associativity in cuBLAS reductions + cuRAND seed handling means we
# can't lock a hash, but we can catch "kernel produced garbage" or "chain
# diverged to all-NaN / all-zero" without false positives on legitimate noise.
python3 - "${work}/result.snpRes" <<'PYEOF' || { echo "FAIL: numerical regression check failed" >&2; tail -30 "${work}/step4.log" >&2; exit 1; }
import math
import sys

# snpRes is whitespace-delimited with a header line. Parse manually — Python's
# csv module doesn't accept delimiter=None for variable-width whitespace.
path = sys.argv[1]
pip_vals, beta_vals = [], []
with open(path) as f:
    header = f.readline().split()
    if "PIP" not in header or "A1Effect" not in header:
        print(f"  snpRes header missing PIP / A1Effect: {header}", file=sys.stderr)
        sys.exit(1)
    pip_i = header.index("PIP")
    eff_i = header.index("A1Effect")
    for line in f:
        parts = line.split()
        try:
            pip = float(parts[pip_i])
            eff = float(parts[eff_i])
        except (ValueError, IndexError):
            continue
        # NaN/Inf would short-circuit the chain — fail loudly.
        if not math.isfinite(pip) or not math.isfinite(eff):
            print(f"  non-finite PIP or effect on line: {line.rstrip()}", file=sys.stderr)
            sys.exit(1)
        pip_vals.append(pip)
        beta_vals.append(eff)

n = len(pip_vals)
if n < 1000:
    print(f"  too few SNPs parsed: {n}", file=sys.stderr)
    sys.exit(1)

# 1. All PIPs in [0, 1].
if not all(0.0 <= p <= 1.0 for p in pip_vals):
    bad = [p for p in pip_vals if not (0.0 <= p <= 1.0)][:5]
    print(f"  PIPs out of [0,1]: examples {bad}", file=sys.stderr)
    sys.exit(1)

# Use loose bands. The chain runs only 50 iter for a quick smoke and the
# synthetic signals are weak, so legitimate posteriors can be very small.
# The goal here is to catch obvious garbage (NaN propagation, "all 1.0",
# "all exactly 0.0", absurd effect sizes), not to validate convergence.

# 2. Mean PIP not pinned at the extremes.
mean_pip = sum(pip_vals) / n
if not (0.0 < mean_pip <= 0.7):
    print(f"  mean PIP {mean_pip:.6f} outside sanity band (0, 0.7]", file=sys.stderr)
    sys.exit(1)

# 3. At least one non-trivially positive PIP (chain not stuck at exactly 0).
max_pip = max(pip_vals)
if max_pip <= 0.0:
    print(f"  max PIP {max_pip:.2e} non-positive — chain stuck at 0", file=sys.stderr)
    sys.exit(1)

# 4. Effect-size RMS finite and not blown up to >1 (PIP-weighted betas
#    should be small for our weak synthetic signals).
beta_rms = math.sqrt(sum(b * b for b in beta_vals) / n)
if not (0.0 <= beta_rms <= 1.0):
    print(f"  effect-size RMS {beta_rms:.2e} outside sanity band [0, 1]", file=sys.stderr)
    sys.exit(1)

print(f"  numerical-regression check: mean PIP={mean_pip:.4f}, max PIP={max_pip:.4f}, "
      f"effect RMS={beta_rms:.2e} — all within bands")
PYEOF

echo "[e2e] PASS: end-to-end GPU pipeline produced a valid snpRes with sane numerics."
