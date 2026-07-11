# cg_jaxmd ALA + 50-water sweep

This workflow runs a small capped alanine system with 50 TIP3 waters for every
valid `cg_jaxmd.py` energy mode and each requested timestep.

| Mode | Intramolecular | Peptide-water |
|---|---|---|
| `mm_mm` | Classical MM | Classical MM |
| `ml_mm` | ML | Classical MM |
| `ml_ml` | ML | ML interaction energy |

The invalid combination `use_ml_intramolecular: false` with
`peptide_water_ml: true` is omitted because `cg_jaxmd.py` rejects it.

The default matrix contains nine jobs: three energy modes crossed with
`dt_fs: [0.5, 0.25, 0.1]`. Simulation lengths, checkpoint, system settings, and
SLURM resources are configured in `config.yaml`.

## Dry-run

From this directory:

```bash
snakemake -n
```

## Run on SLURM

Install the Snakemake SLURM executor plugin and edit the `slurm` section of
`config.yaml` if the cluster uses a different partition or GPU request. Then run:

```bash
snakemake --profile profiles/slurm --keep-going
```

Run one setting with, for example:

```bash
snakemake results/ml_mm/dt_0p25/status.json --profile profiles/slurm
```

## Outputs

Each setting writes its exact `run_config.json`, trajectories, `stdout.log`, and
`status.json` under `results/<mode>/dt_<timestep>/`. Successful completion of all
nine settings produces `results/summary.csv` and `results/summary.md`.
