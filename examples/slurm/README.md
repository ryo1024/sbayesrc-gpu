# Example SLURM scripts

These are the exact SLURM scripts we used on our internal CoreWeave Reno cluster
to produce the published 5.04× benchmark. They are checked in **as examples**
rather than as portable, ready-to-run scripts:

- Cluster knobs (`--qos=scavenge`, `--partition=h100`) are specific to our SLURM
  setup. Strip or rewrite them for yours.
- Filesystem layout (e.g. `data/sbrc/sumstats_cojo/${TRAIT}.ma`,
  `tools/gctb_2.5.5_Linux/gctb`) is our internal convention. Adapt to where your
  inputs live.
- Default working directory is set via `cd <repo root>` near the top of each
  script — change it or replace with `cd "$(dirname "$0")/../.."`.

For a portable, parameter-only runner without the SLURM and cluster assumptions,
see [`scripts/run_benchmark.sh`](../../scripts/run_benchmark.sh) at the repo root.

## `sbrc_gpu_run.sh`

Single-trait, single-condition SLURM job. Runs the impute-summary cache build
(CPU) once per trait, then the GPU GWFM step.

```bash
sbatch examples/slurm/sbrc_gpu_run.sh <CONDITION> <TRAIT>
# e.g.
sbatch examples/slurm/sbrc_gpu_run.sh A_sbrc height
```

Env knobs: `NUM_CHAINS`, `CHAIN_LENGTH`, `BURN_IN`, `LDREF_NAME`, `SBRC_GPU_R`,
`SBRC_GPU_GIBBS`, `SBRC_SKIP_FINDBEST`. See script header for defaults.

## `sbrc_gpu_sweep.sh`

Array launcher: submits one SLURM job per (condition, trait) pair. Use when
running the same setup across many traits or many annotation conditions.

```bash
CONDITIONS="A_sbrc B_sbrc K_sbrc" TRAITS="height ldl bmi" \
  bash examples/slurm/sbrc_gpu_sweep.sh
```
