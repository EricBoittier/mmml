# NH₃–CH₃Cl reaction-path campaign (studix GPU)

Snakemake matrix over the `examples/m` reaction-path toolkit: endpoints, make-box,
NEB, DMC basins, gas/solvated `umbrella-sample`, ADUMB, and MBAR.

All ML jobs use **`examples/m/model_ext.json`** (`checkpoint:` in config).

## Matrix axes

| Axis | Config key | Applied to |
|------|------------|------------|
| Checkpoint | `checkpoint` | every ML job (`--checkpoint` / `MMML_CKPT`) |
| Seeds | `seeds: [...]` | umbrella, ADUMB, DMC |
| Temperatures (K) | `temperatures: [...]` | umbrella, ADUMB (classical NVT) |
| Solvents | `solvents` | make-box, umbrella_sol, adumb_sol |
| Umbrella variants | `umbrella.active` | gas + solvated umbrella (+ MBAR) |

NEB and make-box run once (shared). DMC expands **seeds only** (no classical T).

Default full campaign (`config.yaml`):

- `seeds: [0, 1, 2]`
- `temperatures: [250, 300, 350]`
- `solvents: [tip3, acn, dmso]`
- `umbrella.active: [smoke, medium]`

Artifact layout (under `artifacts/nh3_ch3cl_reaction_path/`):

```text
endpoints/  boxes/  neb/
dmc/{basin}/seed{S}/
umbrella_gas/{variant}/T{T}/seed{S}/[+ mbar/]
umbrella_sol/{solvent}/{variant}/T{T}/seed{S}/[+ mbar/]
adumb_gas/T{T}/seed{S}/
adumb_sol/{solvent}/T{T}/seed{S}/
```

## Prerequisites

```bash
# Checkpoint (required)
ls examples/m/model_ext.json

cd ~/mmml
uv sync --extra gpu    # PyCHARMM + CHARMM for make-box / ADUMB
uv sync --extra mbar   # pymbar for umbrella-mbar

# CHARMM lib (GPU node). job_shell.sh also tries ensure_charmm_mlpot_limits.sh.
export CHARMM_LIB_DIR=${CHARMM_LIB_DIR:-$HOME/.cache/mmml-charmm-build/tier_56000000_nodomdec/lib}
```

If `make_boxes` fails, check `artifacts/.../boxes/stdout.log` for Packmol / PyCHARMM /
`CHARMM_LIB_DIR`. If `mbar_*` fails with `No module named pymbar`, run `uv sync --extra mbar`.

## Dry-run

```bash
cd workflows/nh3_ch3cl_reaction_path
MMML_WORKFLOW_CONFIG=config.smoke.yaml bash scripts/snakemake_local.sh 2 -n
```

## Studix GPU queue

Submit from the **login node** (do not run JAX/PyCHARMM on login):

```bash
cd workflows/nh3_ch3cl_reaction_path

# Smoke (TIP3, T=300, seed=0, ADUMB off)
MMML_WORKFLOW_CONFIG=config.smoke.yaml \
  nohup bash scripts/snakemake_slurm.sh 4 > snakemake_gpu.log 2>&1 &

# Full matrix
nohup bash scripts/snakemake_slurm.sh 8 > snakemake_gpu.log 2>&1 &
tail -f snakemake_gpu.log
```

Slurm profile: `profiles/slurm` → `partition=gpu`, `gpu=1`, `charmm_slot=1` per job.
Concurrent slots come from `slurm.max_jobs` (overridable as the first launcher arg).

Optional in config:

```yaml
slurm:
  mail_user: you@example.com
  nodelist: gpu08,gpu09
```

## Local interactive GPU

```bash
MMML_WORKFLOW_CONFIG=config.smoke.yaml bash scripts/snakemake_local.sh 2
```

## Vary seeds / temperatures

Edit `config.yaml` (or a copy):

```yaml
checkpoint: examples/m/model_ext.json
seeds: [0, 1, 2, 3]
temperatures: [280, 300, 320]
```

Then relaunch with `MMML_WORKFLOW_CONFIG=...`.

## Summary

After jobs finish (or with `--keep-going` partial completion):

```bash
# written by rule collect
cat results/summary.md
```
