# sbayesrc-gpu

[![ci](https://github.com/ryo1024/sbayesrc-gpu/actions/workflows/ci.yml/badge.svg)](https://github.com/ryo1024/sbayesrc-gpu/actions/workflows/ci.yml)

GPU-accelerated [SBayesRC](https://doi.org/10.1038/s41588-024-01704-y) fine-mapping
via a minimally-invasive patch on top of [GCTB](https://github.com/jianzeng/GCTB).

| | CPU baseline | GPU | Speedup |
| --- | --- | --- | --- |
| `A_sbrc × height` wall (4 chains × 2000 iter, full Imputed) | 13h 38m 53s | **2h 42m 30s** | **5.04×** |

![Scaling plot](docs/scaling.png)

Hardware: one NVIDIA H100 80 GB HBM3 (GPU) vs one 64-core Intel Xeon, 256 GB RAM
(CPU). Posterior summaries match upstream CPU GCTB 2.5.5 within MC noise on
`A_sbrc × height` at production chain length. See
[`docs/components.md`](docs/components.md) for the breakdown of which
algorithm pieces run on GPU.

## What this is

A drop-in replacement for the `--gwfm RC` step of GCTB 2.5.5. Same input formats
(`.ma`, eigen-LD blocks, annotation `.txt`), same output formats (`.snpRes`,
`.gcs`, `.lcs`, `.parRes`, etc.) — downstream pipelines don't need to change.

## How to build

### Prerequisites

- CUDA 12.0+ (tested on CUDA 13.0)
- Compute capability `sm_80` or `sm_90` (Ampere/Hopper). Set `GPU_ARCH=sm_80` if
  using A100.
- Eigen 3.4
- Boost 1.74+ (header-only components: `format`, `math`, `random`, `range`)
- GCTB 2.5.5 source from `https://github.com/jianzeng/GCTB`
- g++ 11+ with C++17 support

### Build

```bash
# 1. Clone GCTB source
git clone https://github.com/jianzeng/GCTB.git
cd GCTB

# 2. Apply the GPU patch
git apply /path/to/sbayesrc-gpu/src/gctb-gpu-patch.patch

# 3. Copy GPU dispatch sources into scr/
cp /path/to/sbayesrc-gpu/src/sbrc_gpu.{cu,hpp} scr/

# 4. Build with GPU support
cd scr
export EIGEN3_INCLUDE_DIR=/path/to/eigen-3.4.0
export BOOST_LIB=/path/to/boost_1_85_0
make USE_GPU=1 GPU_ARCH=sm_90   # or sm_80 for A100

# Binary: scr/gctb
```

Or use the Dockerfile (see [`docs/Dockerfile`](docs/Dockerfile)) for a
reproducible build.

## How to run

The patched `gctb` binary takes the same flags as upstream; add `--use-gpu` to
enable the GPU dispatch. The complete list of env vars and which pipeline
pieces run on GPU is in [`docs/components.md`](docs/components.md). Headline
recipe:

```bash
SBRC_GPU_R=1 SBRC_GPU_GIBBS=1 SBRC_SKIP_FINDBEST=1 ./gctb \
    --gwfm RC \
    --use-gpu \
    --ldm-eigen ukbEUR_Imputed \
    --pwld-file rsq0.5.pwld \
    --gwas-summary height.imputed.ma \
    --annot baseline-LF.annot.txt \
    --gene-map gene_map_hg38_hg19.txt \
    --num-chains 4 \
    --chain-length 2000 \
    --burn-in 500 \
    --thread 16 \
    --out results/sbrc
```

For a one-shot reproduction of the headline benchmark see
[`scripts/run_benchmark.sh`](scripts/run_benchmark.sh).

Example SLURM scripts (the exact ones we used internally) live in
[`examples/slurm/`](examples/slurm/) — they need light adaptation to your
filesystem layout and cluster QoS.

## Scaling

The plot at the top is the single-chain length sweep on the HM3 1.15M-SNP
reference. GPU per-iter cost is ~0.7 s vs CPU ~1.7 s — a steady ~2.5–3×
per-iter advantage that grows as chain length increases and setup amortizes.

The 5.04× headline at full ukbEUR-Imputed (7M SNPs) is the limit of this
trend: larger SNP panel → kernels are further from being overhead-bound →
the per-iter speedup grows.

A multi-chain version of the plot (showing the additional GPU win from
cross-chain `d_annoMat` sharing) lives in
[`docs/components.md`](docs/components.md#full-scaling-plot).

Reproduce: [`scripts/scaling_sweep.sh`](scripts/scaling_sweep.sh) +
[`docs/plot_scaling.py`](docs/plot_scaling.py). Raw measurements:
[`docs/scaling_results.csv`](docs/scaling_results.csv).

## Memory requirements

- The eigen-LD reference Q matrix for `ukbEUR_Imputed` (7M SNPs, 591 LD blocks) is
  ~70 GB at FP32. Fits one H100 (80 GB) with per-chain state (~500 MB) and
  the annotation matrix (~5 GB at 188 annotations × 7M SNPs).
- Each additional chain in `--num-chains>1` adds ~500 MB of per-chain state.
  Four chains + Q + annot ≈ 77 GB, fits an H100 with ~3 GB headroom.
- For smaller GPUs (A100 40 GB), use the `ukbEUR_HM3` reference (~3 GB Q) or
  reduce `--eig-cutoff`.

## Validation

Validated against upstream CPU GCTB 2.5.5 on the height + `A_sbrc` (baseline-LF
188 annotations) configuration at production chain length (4 chains × 2000
iter). All posterior summaries (causal estimate, PVE, identified causals at
PIP ≥ 0.9) match within Monte Carlo noise.

`scripts/run_benchmark.sh` reproduces the benchmark in one SLURM submission.

## License

This work is derivative of GCTB which is licensed under the [MIT
License](https://opensource.org/licenses/MIT). The original GCTB copyright is
attributed to Jian Zeng's lab at the University of Queensland (see GCTB README).

The GPU dispatch additions in this repo (`sbrc_gpu.cu`, `sbrc_gpu.hpp`, and the
patch file) are licensed under the same MIT terms.

## Citation

If you use this in published work, please cite both:

1. **SBayesRC** — Zheng et al. (2024), *Nature Genetics*. doi: 10.1038/s41588-024-01704-y
2. **GCTB** — Lloyd-Jones et al. (2019), *Nature Communications*. doi: 10.1038/s41467-019-12653-0
3. **This repo** (sbayesrc-gpu) — please link this GitHub repo URL.

## Status / known limitations

- **`ApproxBayesR` GPU hook** (the pass-1 sampler used by `--num-chains>1`) is
  validated within MC noise at chr22 + full-Imputed, but gated behind
  `SBRC_GPU_R=1` because earlier revisions triggered a subtle memory bug.
- **Annotation Gibbs sweep on GPU** (cuBLAS `sdot`/`saxpy`, gated behind
  `SBRC_GPU_GIBBS=1`) is what unlocks the 5.04× headline; without it speedup
  is ~2.1×.
- **`SBRC_SKIP_FINDBEST=1`** uses your provided `--gamma`/`--pis` directly
  instead of auto-selecting via `findBestFitModel`. Saves time but changes
  chain initialization — calibrate `ndist`/`gamma` before using.
- **Only `--gwfm RC` is GPU-accelerated.** Other modes (`--sbayes`, etc.) fall
  back to CPU.

## Acknowledgments

Built on top of GCTB 2.5.5 by Jian Zeng's lab. SBayesRC algorithm by Zheng,
Yengo, Zhao, Wang et al. This repo packages GPU acceleration as a ~550-line
patch plus ~1400 lines of CUDA kernels; the core science remains the published
SBayesRC method.
