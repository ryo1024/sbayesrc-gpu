#!/usr/bin/env bash
# Verify the Dockerfile builds end-to-end (clone GCTB → apply patch → compile
# with USE_GPU=1). No GPU needed at build time — nvcc only emits PTX/SASS.
#
# Run:
#   tests/test_docker_build.sh
#
# Requires:
#   docker (or podman if you set DOCKER=podman)
#
# This is slower than the other tests (~5-10 min for a fresh build); CI may
# want to skip it on PRs that don't touch src/ or docs/Dockerfile.

set -euo pipefail

DOCKER="${DOCKER:-docker}"
GPU_ARCH="${GPU_ARCH:-sm_90}"
IMAGE_TAG="${IMAGE_TAG:-sbayesrc-gpu:test}"

repo_root="$(cd "$(dirname "$0")/.." && pwd)"

echo "Building $IMAGE_TAG with GPU_ARCH=$GPU_ARCH ..."
"$DOCKER" build \
    --build-arg "GPU_ARCH=${GPU_ARCH}" \
    -t "$IMAGE_TAG" \
    -f "${repo_root}/docs/Dockerfile" \
    "$repo_root"

echo "Verifying gctb binary exists in image ..."
"$DOCKER" run --rm "$IMAGE_TAG" sh -c 'test -x /usr/local/bin/gctb && /usr/local/bin/gctb 2>&1 | head -5'

echo "PASS: docker image $IMAGE_TAG built and gctb binary present."
