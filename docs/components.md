# GPU-accelerated components

The patched `gctb` binary executes the `--gwfm RC` pipeline through a CUDA
dispatch layer. The table below lists each piece of the upstream algorithm and
how it runs on GPU. Default behavior is conservative: only the inner Gibbs
sweep is on by default; the rest are env-gated until a caller opts in.

| Component | Status | Mechanism |
| --- | --- | --- |
| `ApproxBayesRC::SnpEffects::sampleFromFC_eigen` (the inner Gibbs sweep over 591 LD blocks × 7M SNPs) | on by default | Custom CUDA kernel, one threadblock per LD block, parallel q-dim reductions. |
| `ApproxBayesR::SnpEffects::sampleFromFC_eigen` (pass-1 sampler in `--num-chains>1`) | opt-in via `SBRC_GPU_R=1` | Reuses the inner-sweep kernel via a uniform-`snpPi` adapter. |
| `annoEffects.sampleFromFC_Gibbs::snpP = Φ(annoMat × α)` | on by default | cuBLAS `sgemv` + custom CDF kernel. |
| `annoEffects.sampleFromFC_Gibbs` latent variable generation (truncated normal per SNP) | on by default | In-kernel cuRAND, inverse-CDF method. |
| `annoEffects.sampleFromFC_Gibbs` 188-annotation Gibbs sweep | opt-in via `SBRC_GPU_GIBBS=1` | cuBLAS `sdot` / `saxpy` per-k with host-side RNG matching CPU `snorm()` so chain dynamics match CPU bit-for-bit modulo FP noise. |
| Multi-chain GPU concurrency | on by default | Per-impl CUDA streams allow chains to overlap. |
| Cross-chain `d_annoMat` sharing | on by default | Refcounted device buffer keyed on `(data ptr, elems)` — one upload of the ~5.5 GB annotation matrix serves all chains, avoids 4× OOM. |
| `findBestFitModel` skip option | opt-in via `SBRC_SKIP_FINDBEST=1` | Skips the 4-model auto-selection of `ndist`; saves ~6–10 min at full Imputed. **Changes chain initialization — calibrate `ndist`/`gamma` before using.** |
| NaN-safe credible-set sort | on by default | `comparePIP` uses bit-pattern NaN detection (survives `-ffast-math`). |

## What's still on CPU

- `findBestFitModel` 4-model comparison MCMC (skippable via `SBRC_SKIP_FINDBEST=1`).
- Setup phase: loading 70 GB Q from disk, building Q = √Λ × Uᵀ per block.
- Post-processing: writing 1.5 GB `.snpRes`, credible-set computation.

At a 10-iter benchmark scale the GPU kernel is only ~5% of total wall time
because setup + post-processing dominates. At production chain lengths
(2000 iter) the kernel becomes ~25–30% of wall time, which is why the
end-to-end speedup grows with chain length and saturates around the
per-iter speedup ratio.

## Env vars (full list)

| Variable | Default | Effect |
| --- | --- | --- |
| `--use-gpu` (CLI flag) | off | Master switch: enables the GPU dispatch. |
| `SBRC_GPU_R=1` | off | Also route the pass-1 `ApproxBayesR` sampler (used by `--num-chains>1`) through GPU. |
| `SBRC_GPU_GIBBS=1` | off | Route the 188-annotation Gibbs sweep through cuBLAS `sdot`/`saxpy`. |
| `SBRC_SKIP_FINDBEST=1` | off | Skip `findBestFitModel` auto-selection of `ndist`. |
| `SBRC_GPU_DEVICE=N` | 0 | Select GPU N for multi-GPU machines or MPS. |

Pass `SBRC_GPU_R=1 SBRC_GPU_GIBBS=1 SBRC_SKIP_FINDBEST=1` to reproduce the
5.04× headline speedup.
