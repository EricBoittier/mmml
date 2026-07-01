# DCM and ACO dimer scans (cutoffs + LR solvers)

Rigid **COM-distance scans** for **DCM:2** and **ACO:2** dimers, sweeping **MM nonbond modes**
and **long-range Coulomb backends**. Same philosophy as the DCM:3 NVE cutoff sweep
([`workflows/dcm3_nve_cutoff_sweep`](https://github.com/EricBoittier/mmml/tree/main/workflows/dcm3_nve_cutoff_sweep)) and the
MEOH distance scan ([`scripts/scan_meoh_dimer_distance.py`](https://github.com/EricBoittier/mmml/blob/main/scripts/scan_meoh_dimer_distance.py)),
but focused on **hybrid energy vs COM** with **solver comparison**.

## What you get

Each run writes `scan_1d.npz` with decomposed energies on a COM grid (`d₀₁`):

| Field | Meaning |
|-------|---------|
| `scan_2d_hybrid_energy_kcal` | Total hybrid USER energy |
| `scan_2d_ml_2b_E_kcal` | PhysNet dimer (ML two-body) |
| `scan_2d_mm_E_kcal` | Switched JAX MM |
| `lr_solver_active` | Resolved backend (`mic`, `jax_pme`, …) |
| `mm_nonbond_mode` | `jax_mic` or `periodic_external` |

Vertical lines at **6.5 / 8 / 13 Å** mark default cutoff regions (see [hybrid-potential-regions.md](../../hybrid-potential-regions.md)).

## Prerequisites

```bash
export MMML_CKPT=/path/to/checkpoint   # DCM or ACO PhysNet
uv sync --extra gpu                    # or CPU JAX for smoke tests
# OpenMPI + libcharmm — use scripts/mmml-charmm-mpirun.sh
```

## Full solver sweep (DCM + ACO)

```bash
chmod +x scripts/run_dcm_aco_dimer_lr_scans.sh
./scripts/run_dcm_aco_dimer_lr_scans.sh
```

| `scan_tag` | `mm_nonbond_mode` | `lr_solver` | Notes |
|------------|-------------------|-------------|-------|
| `vacuum_mic` | `jax_mic` | `mic` | Free space; pair-loop MIC only |
| `pbc_mic` | `jax_mic` | `mic` | Periodic box; truncated Coulomb |
| `pbc_jax_pme_ewald` | `jax_mic` | `jax_pme` | + Coulomb & r⁻⁶ tail |
| `pbc_jax_pme_pme` | `jax_mic` | `jax_pme` | PME mesh |
| `pbc_jax_pme_p3m` | `jax_mic` | `jax_pme` | P3M |
| `pbc_jax_pme_ewald_no_disp` | `jax_mic` | `jax_pme` | Coulomb-only LR |
| `pbc_periodic_external_jax_pme_pme` | `periodic_external` | `jax_pme` | Full-box Coulomb + CHARMM LJ |
| `pbc_periodic_external_nvalchemiops` | `periodic_external` | `nvalchemiops_pme` | If `mmml[nvalchemiops-pme]` installed |
| `pbc_periodic_external_scafacos_ewald` | `periodic_external` | `scafacos` | If `SCAFACOS_LIB` set |

Environment knobs:

```bash
BOX_SIZE=40 SCAN_MIN=4.0 SCAN_MAX=15.0 SCAN_STEPS=16 \
  COMPOSITIONS="DCM:2" SKIP_PERIODIC=1 \
  ./scripts/run_dcm_aco_dimer_lr_scans.sh
```

`SKIP_PERIODIC=1` skips `periodic_external` legs (no ScaFaCoS / nvalchemiops required).

## Single manual scan

```bash
export MMML_CKPT=/path/to/dcm_ckpt
./scripts/mmml-charmm-mpirun.sh python scripts/scan_mlpot_dimer_2d_pycharmm.py \
  DCM:2 \
  --scan-1d \
  --scan-tag pbc_jax_pme_ewald \
  --box-size 36 --mlpot-pbc \
  --lr-solver jax_pme --jax-pme-method ewald \
  --scan-2d-min 3.5 --scan-2d-max 14.0 --scan-2d-steps 12 \
  --mm-switch-on 8 --mm-switch-width 5 --ml-switch-width 1.5 \
  --packmol-sphere --packmol-radius 10 \
  --output-dir artifacts/dimer_lr_scans
```

ACO example (same flags, different residue):

```bash
./scripts/mmml-charmm-mpirun.sh python scripts/scan_mlpot_dimer_2d_pycharmm.py \
  ACO:2 --scan-1d --scan-tag vacuum_mic --free-space --lr-solver mic \
  --output-dir artifacts/dimer_lr_scans
```

Trimer / 2D grid (legacy path, e.g. cutoff geometry panels):

```bash
./scripts/run_mlpot_dimer_2d_scans.sh DCM:3
```

## Plot comparison

```bash
uv run python scripts/plot_dimer_lr_scan_compare.py --root artifacts/dimer_lr_scans
# → artifacts/dimer_lr_scans/plots/dcm_2_lr_compare.png
# → artifacts/dimer_lr_scans/plots/aco_2_lr_compare.png
```

## Unit tests (no PyCHARMM)

```bash
uv run pytest tests/unit/test_dimer_lr_scan_lib.py tests/unit/test_trimer_scan_geometry.py -q
```

## Related

- [Long-range solver tutorial](../../long-range-solver-tutorial.md)
- [MLpot switching](../../mlpot-settings.md)
- [`tests/functionality/long_range/`](https://github.com/EricBoittier/mmml/tree/main/tests/functionality/long_range) — backend validation without MD
