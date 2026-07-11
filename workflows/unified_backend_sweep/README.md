# Unified mmml.md backend sweep

Smoke-tests every driver/sampler + ensemble combination reachable through
`mmml.md.assemble.assemble_and_run` on a single, small, real system: a 4-water
TIP3 box built via the packmol composition builder (the same path
`mmml.cli.run.md_system_unified` uses), scored with the `ml_intra` +
`mm_nonbonded` terms and the bundled example checkpoint
(`examples/sppoky-epoch-0010_params.json`).

| Backend | Driver / sampler | Ensemble |
|---|---|---|
| `jaxmd_min` | `JaxmdDriver` | FIRE minimization |
| `jaxmd_nve` | `JaxmdDriver` | NVE (microcanonical) |
| `jaxmd_nvt` | `JaxmdDriver` | NVT (Nosé–Hoover chain) |
| `jaxmd_npt` | `JaxmdDriver` | NPT (Nosé–Hoover barostat) |
| `rigid_mc` | `RigidBodySampler` | Metropolis Monte Carlo |

This is the "does every backend still actually run, end-to-end, with a real
force field" check for the unified architecture described in
[`docs/md-cg-unification-design.md`](../../docs/md-cg-unification-design.md).
It is deliberately small and fast (single seed range, ~4-water box) — a smoke
test, not a production sampling run.

## What "backend" means here

`RunConfig` has two axes that together select the propagator:
`RunConfig.backend` (currently only `"jaxmd"` is implemented) and
`RunConfig.sampler` (`"md"` → `JaxmdDriver`, `"rigid"` → `RigidBodySampler`).
Within `"md"`, `EnsembleSpec.ensemble` picks `min` / `nve` / `nvt` / `npt`. The
five rows above are every reachable combination as of this workflow's creation;
add a row to `config.yaml`'s `backends:` section (and this table) as new
backends/ensembles land.

## Note on NPT

`jaxmd_npt` bridges jax-md's fractional-coordinate NPT state to the
real-space, box-aware energy terms every step. That bridge is markedly slower
to JIT-compile with a real ML model than the fixed-box ensembles (~4 minutes
vs. ~30 seconds observed for this tiny system) — `config.yaml` gives it a
longer `runtime_min` budget for exactly this reason.

## Dry-run

From this directory:

```bash
uv run --with snakemake --with snakemake-executor-plugin-slurm \
  snakemake -n
```

## Run locally

```bash
export CHARMM_LIB_DIR=/path/to/mmml/setup/charmm
uv run --with snakemake snakemake --profile profiles/local --keep-going
```

Run one setting directly (useful when iterating):

```bash
python scripts/run_setting.py \
  --workflow-config config.yaml --backend jaxmd_nvt --seed 1 \
  --output-dir results/jaxmd_nvt/seed_1 --repo-root ../..
```

## Run on SLURM

Install the Snakemake SLURM executor plugin and edit the `slurm` section of
`config.yaml` if the cluster uses a different partition or GPU request, then:

```bash
uv run --with snakemake --with snakemake-executor-plugin-slurm \
  snakemake --profile profiles/slurm --keep-going
```

## Outputs

Each setting writes `run_config.json`, `stdout.log`, and `status.json` under
`results/<backend>/seed_<seed>/`. `status.json` reports whether the run
completed with finite energies throughout, the initial/final energy, energy
drift, (for `rigid_mc`) the Metropolis acceptance ratio, and elapsed time, or
the captured exception if it failed. Successful completion of all settings
produces `results/summary.csv` and `results/summary.md`; `collect_results.py`
exits non-zero if any setting failed, so `snakemake --keep-going` still
reports the sweep as failed overall when a backend regresses.
