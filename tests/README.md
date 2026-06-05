# Tests

Three test suites, ordered by speed:

| Test | Runtime | GPU? | What it catches |
| --- | --- | --- | --- |
| `test_compare_pip_nan` | <1 s | no | Comparator regressions where `-ffast-math` strips NaN handling and post-MCMC `std::sort` would crash. |
| `test_patch_applies.sh` | ~5 s | no | Upstream GCTB moved and the patch no longer applies cleanly to the pinned SHA. |
| `test_docker_build.sh` | ~5-10 min | no (at build time) | Patch+build pipeline regressions; missing toolchain deps; nvcc compile errors. |

## Running locally

```bash
# 1. NaN-safe comparator (single C++ binary)
g++ -O3 -ffast-math -std=c++17 -o tests/test_compare_pip_nan tests/test_compare_pip_nan.cpp
./tests/test_compare_pip_nan

# 2. Patch-applies check (needs git + internet)
tests/test_patch_applies.sh

# 3. Full Docker build (slow; needs docker)
tests/test_docker_build.sh
```

## CI

Run locally via the three scripts above. A GitHub Actions workflow can be added
later (requires a PAT with `workflow` scope or commit it from the GitHub web UI).
A drop-in workflow YAML for `.github/workflows/ci.yml` is recorded in
[`CONTRIBUTING.md`](../CONTRIBUTING.md).

## GPU-required tests

Anything that actually exercises the CUDA kernels (chr22 smoke, full Imputed
bench) lives outside this directory because it requires:

- An NVIDIA GPU with ≥40 GB HBM (chr22 smoke) or ≥80 GB HBM (full Imputed).
- The eigen-decomposed UKB-EUR LD reference (~70 GB on disk).
- An annotation `.txt` and a `.ma` GWAS summary stats file.

See [`scripts/run_benchmark.sh`](../scripts/run_benchmark.sh) for a reproducible
GPU smoke run when you have hardware.

## Updating the GCTB pin

If upstream GCTB moves and you want to follow it:

1. Pick a new SHA, e.g. `git rev-parse HEAD` after fetching the GCTB main branch.
2. Update `GCTB_SHA` in [`docs/Dockerfile`](../docs/Dockerfile) and the default
   in [`tests/test_patch_applies.sh`](test_patch_applies.sh).
3. Regenerate `src/gctb-gpu-patch.patch` against the new SHA (see
   [`../CONTRIBUTING.md`](../CONTRIBUTING.md)).
4. Run `tests/test_patch_applies.sh` to confirm.
