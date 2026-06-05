#!/usr/bin/env bash
# Verify the GCTB GPU patch still applies cleanly against the pinned upstream SHA.
# No GPU or CUDA toolkit required — only git.
#
# Run:
#   tests/test_patch_applies.sh
#
# Exits 0 if patch applies, 1 otherwise.

set -euo pipefail

GCTB_SHA="${GCTB_SHA:-cc7fa7d765c83a89c6375946cf77fe50ba1a317e}"
GCTB_URL="${GCTB_URL:-https://github.com/jianzeng/GCTB.git}"

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
patch_file="${repo_root}/src/gctb-gpu-patch.patch"

if [ ! -f "$patch_file" ]; then
    echo "FAIL: patch file not found at $patch_file" >&2
    exit 1
fi

work_dir="$(mktemp -d)"
# Expand $work_dir now (it's already set), but use single quotes so shellcheck
# stays happy about late expansion in the trap body.
cleanup() { rm -rf "$work_dir"; }
trap cleanup EXIT

echo "Cloning GCTB@${GCTB_SHA} into $work_dir ..."
git clone --quiet --no-tags "$GCTB_URL" "$work_dir/GCTB"
git -C "$work_dir/GCTB" checkout --quiet "$GCTB_SHA"

echo "Checking patch applies cleanly ..."
if git -C "$work_dir/GCTB" apply --check "$patch_file"; then
    echo "PASS: patch applies cleanly against GCTB@${GCTB_SHA}."
else
    echo "FAIL: patch does not apply to GCTB@${GCTB_SHA}." >&2
    echo "If GCTB upstream moved, bump GCTB_SHA in docs/Dockerfile and CI, and" >&2
    echo "regenerate the patch following CONTRIBUTING.md." >&2
    exit 1
fi

# Also do a real apply so we can confirm sbrc_gpu.{cu,hpp} can be dropped in.
git -C "$work_dir/GCTB" apply "$patch_file"
cp "${repo_root}/src/sbrc_gpu.cu"  "$work_dir/GCTB/scr/"
cp "${repo_root}/src/sbrc_gpu.hpp" "$work_dir/GCTB/scr/"
echo "PASS: post-apply tree builds the expected scr/ layout."

# Quick sanity: the patched HDR list in scr/Makefile should mention sbrc_gpu.hpp.
if ! grep -q 'sbrc_gpu\.hpp' "$work_dir/GCTB/scr/Makefile"; then
    echo "FAIL: scr/Makefile does not reference sbrc_gpu.hpp after patch." >&2
    exit 1
fi
echo "PASS: scr/Makefile references sbrc_gpu.hpp."
