#!/usr/bin/env bash
# One-shot reproducer: clone the pinned GCTB upstream, apply the GPU patch,
# build with USE_GPU=1, and run the end-to-end smoke test. Use this when you
# want to answer "does the GPU pipeline actually work on my box?" with a
# single command instead of the ~6 step manual recipe.
#
# Usage:
#   EIGEN3_INCLUDE_DIR=/path/to/eigen-3.4.0 \
#   BOOST_LIB=/path/to/boost_1_85_0 \
#   GPU_ARCH=sm_90 \
#       scripts/build_and_smoke.sh
#
# Required env:
#   EIGEN3_INCLUDE_DIR  — directory containing Eigen/ headers
#   BOOST_LIB           — directory containing boost/ headers
#
# Optional env:
#   GPU_ARCH            — sm_80 (A100), sm_86 (RTX 30xx), sm_90 (H100, default)
#   GCTB_DIR            — where to clone GCTB (default: ./build/GCTB)
#   GCTB_SHA            — override the pinned upstream SHA
#   SKIP_SMOKE=1        — build but don't run the e2e smoke test
#
# Produces: $GCTB_DIR/scr/gctb (the patched binary). The script also adds
# that directory to PATH for the smoke run.
#
# Idempotent: re-running picks up a partially-built tree. To force a clean
# build, rm -rf $GCTB_DIR.

set -euo pipefail

: "${EIGEN3_INCLUDE_DIR:?Set EIGEN3_INCLUDE_DIR to a directory containing Eigen/ headers}"
: "${BOOST_LIB:?Set BOOST_LIB to a directory containing boost/ headers}"

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
GPU_ARCH="${GPU_ARCH:-sm_90}"
GCTB_DIR="${GCTB_DIR:-${repo_root}/build/GCTB}"
GCTB_SHA="${GCTB_SHA:-cc7fa7d765c83a89c6375946cf77fe50ba1a317e}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"

# 1. Clone GCTB at the pinned SHA if not already present.
if [ ! -d "${GCTB_DIR}/.git" ]; then
    echo "[build_and_smoke] cloning GCTB into ${GCTB_DIR} ..."
    git clone --quiet https://github.com/jianzeng/GCTB.git "${GCTB_DIR}"
fi
echo "[build_and_smoke] checking out GCTB ${GCTB_SHA} ..."
git -C "${GCTB_DIR}" fetch --quiet origin
git -C "${GCTB_DIR}" checkout --quiet "${GCTB_SHA}"

# 2. Apply patch. If it's already applied, --check will fail loudly; we use
#    that to decide whether to skip the apply step.
if git -C "${GCTB_DIR}" apply --check "${repo_root}/src/gctb-gpu-patch.patch" 2>/dev/null; then
    echo "[build_and_smoke] applying patch ..."
    git -C "${GCTB_DIR}" apply "${repo_root}/src/gctb-gpu-patch.patch"
else
    # Either already applied, or genuinely broken. Distinguish by reverse-check.
    if git -C "${GCTB_DIR}" apply --check --reverse "${repo_root}/src/gctb-gpu-patch.patch" 2>/dev/null; then
        echo "[build_and_smoke] patch already applied, skipping."
    else
        echo "[build_and_smoke] ERROR: patch neither applies forward nor matches reverse." >&2
        echo "Try: rm -rf ${GCTB_DIR} and re-run." >&2
        exit 1
    fi
fi

# 3. Copy GPU dispatch sources into scr/ (these aren't in the patch on purpose).
cp "${repo_root}/src/sbrc_gpu.cu"  "${GCTB_DIR}/scr/"
cp "${repo_root}/src/sbrc_gpu.hpp" "${GCTB_DIR}/scr/"

# 4. Build with USE_GPU=1.
echo "[build_and_smoke] make USE_GPU=1 GPU_ARCH=${GPU_ARCH} ..."
make -C "${GCTB_DIR}/scr" \
    USE_GPU=1 GPU_ARCH="${GPU_ARCH}" \
    EIGEN3_INCLUDE_DIR="${EIGEN3_INCLUDE_DIR}" \
    BOOST_LIB="${BOOST_LIB}" \
    -j "$(nproc)"

if [ ! -x "${GCTB_DIR}/scr/gctb" ]; then
    echo "[build_and_smoke] ERROR: build did not produce ${GCTB_DIR}/scr/gctb" >&2
    exit 1
fi
echo "[build_and_smoke] binary: ${GCTB_DIR}/scr/gctb"

# 5. Run end-to-end smoke test, with the freshly-built gctb on PATH.
if [ "${SKIP_SMOKE}" = "1" ]; then
    echo "[build_and_smoke] SKIP_SMOKE=1 set; not running e2e."
    exit 0
fi

export PATH="${GCTB_DIR}/scr:${PATH}"
echo "[build_and_smoke] running tests/test_e2e_gpu.sh ..."
bash "${repo_root}/tests/test_e2e_gpu.sh"
