# DCM:5 cross-backend MD benchmark

Reproducible **2 ps** smoke benchmark on **DCM:5** comparing ASE, JAX-MD, and PyCHARMM
(`mmml md-system`) across vacuum and PBC integrator modes.

## Do not commit run outputs

Benchmark artifacts live under `results/` and `.snakemake/` — both are **gitignored**.
Commit only workflow source (`Snakefile`, `config.yaml`, `scripts/`, `README.md`, tests).

```bash
git add workflows/dcm5_md_benchmark/Snakefile \
        workflows/dcm5_md_benchmark/config.yaml \
        workflows/dcm5_md_benchmark/scripts/ \
        workflows/dcm5_md_benchmark/README.md \
        workflows/dcm5_md_benchmark/profiles/ \
        tests/unit/test_dcm5_benchmark_config.py \
        tests/unit/test_collect_benchmark.py
# NOT: git add -A inside the workflow directory
```

## Prerequisites

- MMML env with GPU JAX (`uv sync --extra gpu` on cluster nodes)
- DCM PhysNet checkpoint:

```bash
export MMML_CKPT=/path/to/dcm1-.../ckpts/dcm1-...
```

- **PyCHARMM jobs**: OpenMPI-linked libcharmm (use bundled wrapper)
- **Snakemake** (`pip install snakemake` or cluster module)
- `packmol` on PATH (first vacuum cluster build)

If Snakemake uses a Python without `mmml` installed, set:

```bash
export MMML_PYTHON="$PWD/../../.venv/bin/python"   # from workflow dir
```

PyCHARMM runs use [scripts/mmml-charmm-mpirun.sh](../../scripts/mmml-charmm-mpirun.sh) (1 MPI rank).

## Job matrix (15 jobs)

| Category | Jobs |
|----------|------|
| NVE vacuum | `ase_vac_nve`, `jaxmd_vac_nve`, `pycharmm_vac_nve` |
| NVE PBC | `ase_pbc_nve`, `jaxmd_pbc_nve`, `pycharmm_pbc_nve` |
| NVT vacuum | `ase_vac_nvt_nhc`, `ase_vac_nvt_langevin`, `jaxmd_vac_nvt`, `pycharmm_vac_heat_scale`, `pycharmm_vac_heat_hoover` |
| NVT PBC | `ase_pbc_nvt_nhc`, `ase_pbc_nvt_langevin`, `jaxmd_pbc_nvt` |
| NPT PBC | `jaxmd_pbc_npt` (optional — may fail on 2 ps smoke; does not block collect) |

Shared parameters (see [config.yaml](config.yaml)):

- `DCM:5`, `--seed 123`, `--dt-fs 0.25`, **2.0 ps** (8000 steps)
- Vacuum Packmol sphere `R = 6.9 Å`; PBC `--box-size 25`
- PyCHARMM heat: 0 → 240 K (`scale` vs `hoover`)

## Run

```bash
export MMML_CKPT=/path/to/your/dcm1-ckpt-directory   # real path, not this placeholder
bash scripts/preflight.sh                             # fail fast if unset/invalid
cd workflows/dcm5_md_benchmark

# Dry-run
snakemake -n

# Local (1 GPU; PyCHARMM serialized via mpi=1 resource)
snakemake -j4 --resources gpu=2 mpi=1 --keep-going

# With local profile
snakemake --profile profiles/local --keep-going

# Cluster (edit profiles/slurm/config.yaml first)
snakemake --profile profiles/slurm -j20
```

Run a single job:

```bash
bash scripts/job_shell.sh pycharmm_vac_heat_hoover
# or
python3 scripts/run_job.py ase_vac_nve
```

Collect only (works even if some jobs failed — no longer requires every `done.txt`):

```bash
snakemake results/benchmark_summary.csv -f
# or
python3 scripts/collect_benchmark.py
```

## Outputs

Per job: `results/<job_id>/stdout.log` plus backend artifacts under the same directory.

Aggregate:

- `results/benchmark_summary.csv`
- `results/benchmark_report.md`

## Troubleshooting

### `pycharmm_vac_nve` has trajectories but no `done.txt`

Usually NVE stopped early (echeck) or overlap chunk restart handoff failed. The directory may contain many `nve_dcm_5.chunk.*.dcd` files from the default 500-step overlap interval — use `dynamics_overlap_check_interval: 8000` in `config.yaml` (single chunk).

```bash
grep -E 'incomplete|restart step|NVE complete|READYN|echeck' results/pycharmm_vac_nve/stdout.log | tail -20
rm -f results/pycharmm_vac_nve/done.txt
snakemake results/pycharmm_vac_nve/done.txt -c1
```

Sync repo + `pip install -e .` for NVE memory handoff and post-run DCD validation.

### `pycharmm_vac_heat_hoover`: heat stops ~step 500–700 (echeck, T≫240 K)

Typical log: `DYNA>` temperature hundreds of K at 0.125 ps, `restart step 689 < 7600`, one readable DCD frame. Causes: **short mini** + **stale NB lists** after `inbfrq=0` mini, then **Hoover `tmass` too small** for a 25-atom ML cluster (PSF formula ≈80).

Current benchmark defaults: `mini_nstep: 2000`, `bonded_mm_mini`, `ps_heat: 5`, Hoover heat uses **`tmass ≥ 500`** and **`pgamma 0`** with **CHARMM UPDATE before heat**.

```bash
grep -E 'CHARMM UPDATE after mini|tmass=|DYNA>|integrated |echeck' \
  results/pycharmm_vac_heat_hoover/stdout.log | tail -25
rm -f results/pycharmm_vac_heat_hoover/done.txt
bash scripts/job_shell.sh pycharmm_vac_heat_hoover
```

If it still aborts: try `pycharmm_vac_heat_scale` (velocity scaling) or `--no-echeck` once to confirm echeck vs physics (see `mlpot/COMP_AND_HEATING.md`).

### `pycharmm_vac_heat_hoover`: `CRYStal must be used for constant pressure simulations`

Loose-PBC Hoover heat uses CHARMM **CPT** (`pmass=0`). The crystal must stay active after CGENFF pre-minimize. Older code called `crystal free` during MM pre-min; current `mmml` skips that when `--box-size` is set and re-installs the box before heat.

```bash
grep -E 'CHARMM loose PBC|CHARMM crystal ready|CRYStal must be used|HEAT Hoover' \
  results/pycharmm_vac_heat_hoover/stdout.log | tail -15
```

After updating the package, rerun the job. You should see `CHARMM loose PBC` early and either no crystal warning or `CHARMM crystal ready for CPT` before `HEAT Hoover (CPT)`.

### `jaxmd_pbc_npt` fails

NPT is the most fragile mode (barostat + neighbor list + small cluster). Check:

```bash
cat results/jaxmd_pbc_npt/stdout.log | tail -80
grep -E 'ERROR|NaN|error|Partial output' results/jaxmd_pbc_npt/stdout.log
```

Common log messages:

- `First step produced NaN positions` — try larger box (`box_size: 40` in config) or longer `--ps`
- `Energy blow-up` — initial overlap; Packmol placement is now enabled for PBC jobs

This job is **optional**: the summary CSV is produced when the other 14 jobs finish. Use `--keep-going` so one NPT failure does not stop the workflow.

Retry NPT alone:

```bash
bash scripts/job_shell.sh jaxmd_pbc_npt
```

## Pass/fail criteria

| Backend | Pass |
|---------|------|
| ASE | `suite_summary.json` run entry with `log_samples > 0` |
| JAX-MD | `suite_summary_jaxmd.json` → `status == complete`, `nsteps_completed ≥ 7600` |
| PyCHARMM | Log contains `HEAT complete` / `NVE complete`; `restart_step ≥ 7600`; no early echeck abort |

NVT: report `temp_mean_K` within ±15% of 300 K (ASE collector warns otherwise).

NVE: report `etot_drift_eV` (ASE) in CSV notes.

## Log checks (PyCHARMM heat)

```bash
grep -E 'HEAT Hoover|HEAT complete|IHTFRQ|VELOCITIES HAVE BEEN SCALED|integrated |cpt|pmass' \
  results/pycharmm_vac_heat_hoover/stdout.log | tail -20
```

- **Hoover** (`pycharmm_vac_heat_hoover`): uses **loose PBC** (`pbc: true`, `box_size: 55`, `setup: free_nvt`) — CHARMM crystal + **CPT Hoover** (`pmass=0`), ML **without MIC**. Log should show `CHARMM loose PBC`, `HEAT Hoover (CPT)`, and dynamics with `cpt` / `hoover reft` (not `HEAT Hoover (vacuum fallback)`).
- **Scale** (`pycharmm_vac_heat_scale`): explicit `VELOCITIES HAVE BEEN SCALED` every `heat_ihtfrq` steps

## Unit tests (no CHARMM)

```bash
pytest tests/unit/test_dcm5_benchmark_config.py \
       tests/unit/test_collect_benchmark.py -v
```

## Optional longer tier

Edit [config.yaml](config.yaml):

- `--ps-heat 10–20`, `--ps-nve 5`
- `--mini-nstep 150`
- `--dcd-nsavc 250` for denser trajectories

## Notes

- ASE/JAX-MD use BFGS pre-MD minimization; PyCHARMM uses MLpot SD — pretreat paths differ by design.
- Trajectory RMSD cross-backend comparison is out of scope for this smoke suite.
- NPT is JAX-MD only (`pbc_npt`); ASE has no NPT integrator in `md_pbc_suite`.
