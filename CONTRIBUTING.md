# Contributing to sbayesrc-gpu

## Regenerating the GCTB patch

The patch in `src/gctb-gpu-patch.patch` is a diff against upstream
[GCTB](https://github.com/jianzeng/GCTB) at a pinned SHA. To bump the SHA or to
add a new modification:

```bash
# 1. Clone GCTB at the SHA you want to base the new patch on.
git clone https://github.com/jianzeng/GCTB.git /tmp/GCTB
cd /tmp/GCTB
git checkout <new-sha>

# 2. Apply the existing patch as a starting point.
git apply /path/to/sbayesrc-gpu/src/gctb-gpu-patch.patch

# 3. Edit scr/ files as needed (Makefile, gctb.hpp, model.cpp, etc.).
#    Do NOT touch scr/sbrc_gpu.{cu,hpp} here — those live in this repo's src/,
#    not in the patch.

# 4. Regenerate the patch.
cd /tmp/GCTB
git diff HEAD -- \
    scr/Makefile scr/gctb.cpp scr/gctb.hpp scr/main.cpp \
    scr/model.cpp scr/model.hpp scr/options.cpp scr/options.hpp \
  > /path/to/sbayesrc-gpu/src/gctb-gpu-patch.patch

# 5. Verify the patch applies cleanly.
cd /path/to/sbayesrc-gpu
tests/test_patch_applies.sh
```

If you bumped the SHA, also update:
- `GCTB_SHA` in `docs/Dockerfile`
- `GCTB_SHA` default in `tests/test_patch_applies.sh`

## Adding to the CUDA dispatch

`src/sbrc_gpu.cu` and `src/sbrc_gpu.hpp` are the GPU dispatch layer. They're not
in the patch — they're copied into `scr/` by the Dockerfile (and by hand for
local builds).

When adding a new kernel:

- Public entry points go in `sbrc_gpu.hpp`; implementations in `sbrc_gpu.cu`.
- Add an env-var gate (`SBRC_GPU_<FEATURE>`) for experimental features so prod
  binaries stay safe by default.
- Bias toward reusing the existing per-impl state (`SbrcImpl`, `SbrcAnnoImpl`)
  rather than introducing new globals; the refcounted caches (`g_q_cache`,
  `g_annomat_cache`) keep multi-chain memory under control.

## Running tests

See [`tests/README.md`](tests/README.md).

## Drop-in GitHub Actions workflow

Pushing a `.github/workflows/*.yml` requires a PAT with `workflow` scope. If you
have one, paste this YAML into `.github/workflows/ci.yml`:

```yaml
name: ci
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  workflow_dispatch:
jobs:
  patch-applies:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4
      - run: tests/test_patch_applies.sh
  compare-pip-nan:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4
      - run: sudo apt-get update && sudo apt-get install -y g++
      - run: g++ -O3 -ffast-math -std=c++17 -o tests/test_compare_pip_nan tests/test_compare_pip_nan.cpp
      - run: ./tests/test_compare_pip_nan
  shellcheck:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4
      - run: sudo apt-get update && sudo apt-get install -y shellcheck
      - run: shellcheck tests/*.sh
  docker-build:
    runs-on: ubuntu-22.04
    if: |
      github.event_name == 'workflow_dispatch' ||
      contains(toJSON(github.event.commits.*.modified), 'docs/Dockerfile') ||
      contains(toJSON(github.event.commits.*.modified), 'src/')
    steps:
      - uses: actions/checkout@v4
      - env: { GPU_ARCH: sm_80 }
        run: tests/test_docker_build.sh
```

## Branch / commit conventions

- One change per commit; subject line ≤72 chars; explain *why* in the body.
- Don't squash CI-fixup commits into the feature commit unless they're trivial.
- Sign-off (`git commit -s`) is welcome but not required.

## Validating a real run

The CI tests cover the build and the comparator fix. They do not validate the
chain produces correct posteriors — that requires a GPU and an LD reference.
Before merging anything that touches a kernel or the dispatch math, run:

```bash
# chr22 smoke (~2 min on H100); compares against a CPU baseline.
scripts/run_benchmark.sh   # see env knobs at top of file
```

Numbers to eyeball:

- **Skeptical SNP count**: should be in the same order of magnitude as the CPU
  baseline (usually 0–50 at chr22, 30–400 at full Imputed). If you see thousands,
  the chain is drifting — almost always an RNG-mismatch bug.
- **PVE, identified causal at PIP≥0.9, causal estimate**: should match CPU
  baseline within MC noise (~3% relative).
- **Exit code**: must be 0. A 139 here usually means `comparePIP` regressed and
  NaN PIPs are crashing `std::sort` in `calcCredibleSets`.
