# Liquid methane PBC MM/ML (Ewald JSON sweep)

Snakemake workflow for **periodic liquid methane** hybrid MM/ML dynamics on
**PyCHARMM** and **JAX-MD**, sweeping **portable JSON checkpoints** at
`T = 100 / 200 / 300 K` with the **native Ewald** long-range Coulomb solver.

Sibling workflows: [pbc_solvent_burst](../pbc_solvent_burst/) (TIP3/MEOH bursts),
[pbc_liquid_density_dyn](../pbc_liquid_density_dyn/) (DCM/ACO liquid density).

## Goal

Pack methane (`METH`, CGenFF `CG331`/`HGA3`) at liquid density, then for each
`(T, checkpoint JSON, backend)` cell run a short equilibration + production
campaign with:

| Knob | Value |
|------|--------|
| `lr_solver` | `ewald` |
| `mm_nonbond_mode` | `periodic_external` |
| `ensemble` | `pbc_nvt` (fixed liquid density across the T ladder) |
| Backends | `pycharmm`, `jaxmd` |

NVT is intentional: methane is supercritical above ~190.6 K at 1 atm, so NPT
at ambient pressure would boil off the box. All temperatures share the same
liquid-density packing.

## Matrix (default `config.yaml`)

| Axis | Values |
|------|--------|
| Solvent | `METH` |
| Temperature | 100, 200, 300 K |
| Box | 20 Å |
| Density | 1.0 × bulk liquid (~0.423 g/cm³) |
| Checkpoints | `des`, `spooky4`, `sppoky10` (see `checkpoints:`) |
| Backends | `pycharmm`, `jaxmd` |

Run tag example: `meth_127_t100_l20_des_pycharmm`.

Smoke matrix (`config.smoke.yaml`): L=16 Å at 0.5× bulk, two JSONs, 1 ps legs.

## Prerequisites

```bash
export JAX_ENABLE_X64=1
# optional single-ckpt override (named matrix still applies unless edited)
# export MMML_CKPT=/path/to/params.json
```

GPU node with PyCHARMM + OpenCL, `packmol`, `snakemake`,
`snakemake-executor-plugin-slurm`.

## Run

```bash
cd workflows/pbc_methane_ewald
bash scripts/preflight.sh
snakemake -n

# Smoke on Slurm
MMML_WORKFLOW_CONFIG=config.smoke.yaml bash scripts/snakemake_slurm.sh

# Full matrix
bash scripts/snakemake_slurm.sh
```

Single cell:

```bash
srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 \
  bash scripts/job_shell.sh meth_127_t100_l20_des_jaxmd
```

## Outputs

```
artifacts/pbc_methane_ewald/{tag}/
  campaign.yaml
  pycharmm_init/ …
  pycharmm_prod_02/handoff/state.npz   # or jaxmd_prod/
  done.txt
```

## Tests

```bash
uv run pytest tests/unit/test_pbc_methane_ewald_campaign.py -q
```

## METH residue

Methane is not in stock CGenFF; MMML adds `RESI METH` (neutral CH₄ using
alkane `CG331`/`HGA3` types) plus `mmml/data/molecules/meth_monomer.pdb`.
Aliases: `CH4`, `methane` → `METH`.
