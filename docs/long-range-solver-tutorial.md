# Tutorial: long-range Coulomb solvers in MMML

This guide shows how to run **MIC**, **jax-pme** (Ewald / PME / P3M), and **ScaFaCoS**
Coulomb backends in hybrid ML/MM workflows, using the DCM liquid box as a concrete example.

Related:

- [`scripts/run_dcm_liquid_workflow.sh`](../scripts/run_dcm_liquid_workflow.sh) — baseline DCM liquid pipeline
- [`scripts/run_dcm_long_range_workflow.sh`](../scripts/run_dcm_long_range_workflow.sh) — solver comparison sweep
- [`tests/functionality/long_range/`](../tests/functionality/long_range/) — standalone validation
- [`mmml/interfaces/pycharmmInterface/mlpot/LONG_RANGE_ELECTROSTATICS.md`](../mmml/interfaces/pycharmmInterface/mlpot/LONG_RANGE_ELECTROSTATICS.md) — architecture

---

## 1. Two MM nonbond modes

| Mode | LJ | Coulomb | When to use |
|------|----|---------|-------------|
| **`jax_mic`** (default) | Switched JAX pairs (~13 Å) | MIC in pair loop, or **jax-pme** with `--lr-solver jax_pme` | Standard hybrid MLpot; jax-pme Coulomb + r⁻⁶ LJ when enabled |
| **`periodic_external`** | CHARMM IMAGE VDW | **jax-pme** or **ScaFaCoS** full-box Coulomb | Large boxes where JAX MM is off; requires `pbc_*` setup |

In **`jax_mic`** + **`jax_pme`** mode: **r⁻¹²** repulsion stays on the switched pair list (exact Lorentz–Berthelot σ_ij, ε_ij, hybrid λ); **Coulomb** and the **r⁻⁶** LJ tail use jax-pme (Ewald/PME/P3M) with per-atom √C6 and geometric k-space combining (GROMACS-style LJ-PME reciprocal rule).

---

## 2. CLI flags (`md-system`, `mmml md-system`)

| Flag | Values | Default | Meaning |
|------|--------|---------|---------|
| `--mm-nonbond-mode` | `jax_mic`, `periodic_external` | `jax_mic` | MM stack selection |
| `--lr-solver` | `auto`, `mic`, `jax_pme`, `scafacos` | env / `auto` | Coulomb backend |
| `--jax-pme-method` | `ewald`, `pme`, `p3m` | `ewald` | jax-pme variant |
| `--jax-pme-sr-cutoff` | float (Å) | `6.0` | jax-pme real-space cutoff |
| `--scafacos-method` | `ewald`, `p3m`, … | `ewald` | ScaFaCoS `fcs_init` string |
| `--include-mm` / `--no-include-mm` | bool | on | JAX MM pair path (LJ ± Coulomb) |

Environment mirrors:

```bash
export MMML_LR_SOLVER=jax_pme      # mic | jax_pme | scafacos | auto
export JAX_PME_METHOD=pme          # ewald | pme | p3m
export JAX_PME_SR_CUTOFF=6.0
export SCAFACOS_LIB=$HOME/.local/scafacos/lib/libfcs.so
export SCAFACOS_METHOD=ewald
```

YAML keys use underscores: `lr_solver`, `jax_pme_method`, `jax_pme_sr_cutoff`, `mm_nonbond_mode`.

Example config: [`mmml/cli/run/dcm_long_range_solvers.example.yaml`](../mmml/cli/run/dcm_long_range_solvers.example.yaml).

---

## 3. Standalone validation (no PyCHARMM)

Quick check that jax-pme and backends are installed:

```bash
cd ~/mmml
python tests/functionality/long_range/00_check_lr_env.py
pytest tests/functionality/long_range/test_coulomb_backends.py -v
pytest tests/functionality/long_range/test_hybrid_jax_pme_mm.py -v
bash tests/functionality/long_range/run_all.sh
```

ScaFaCoS (optional):

```bash
export SCAFACOS_LIB=$HOME/.local/scafacos/lib/libfcs.so
export LD_LIBRARY_PATH=$HOME/.local/scafacos/lib:$LD_LIBRARY_PATH
python tests/functionality/long_range/04_scafacos_methods.py
```

---

## 4. Example: jax-pme Ewald on a certified DCM box

After [`run_dcm_liquid_workflow.sh`](../scripts/run_dcm_liquid_workflow.sh) produces `~/tests/boxes/dcm60_l32/`:

```bash
export MMML_CKPT=~/mmml/mmml/models/physnetjax/defaults/hf_json/<checkpoint>_portable.json

MMML_MPI_NP=1 ~/mmml/scripts/mmml-charmm-mpirun.sh md-system \
  --config ~/mmml/mmml/cli/run/dcm_long_range_solvers.example.yaml \
  --from-psf ~/tests/boxes/dcm60_l32/model.psf \
  --from-crd ~/tests/boxes/dcm60_l32/model.crd \
  --lr-solver jax_pme \
  --jax-pme-method ewald \
  --output-dir ~/tests/runs/dcm60_jax_pme_ewald \
  --checkpoint "$MMML_CKPT"
```

Compare with truncated MIC (default):

```bash
  --lr-solver mic \
  --output-dir ~/tests/runs/dcm60_mic
```

Swap `--jax-pme-method` to `pme` or `p3m` to test mesh methods.

---

## 5. Example: periodic_external + jax-pme

Turns off JAX real-space MM; Coulomb from jax-pme, LJ from CHARMM IMAGE:

```bash
MMML_MPI_NP=1 ~/mmml/scripts/mmml-charmm-mpirun.sh md-system \
  --setup pbc_npt \
  --composition DCM:60 \
  --from-psf ~/tests/boxes/dcm60_l32/model.psf \
  --from-crd ~/tests/boxes/dcm60_l32/model.crd \
  --mm-nonbond-mode periodic_external \
  --lr-solver jax_pme \
  --jax-pme-method pme \
  --box-size 32 \
  --checkpoint "$MMML_CKPT" \
  -o ~/tests/runs/dcm60_periodic_jax_pme
```

ScaFaCoS variant (requires `libfcs`):

```bash
  --lr-solver scafacos \
  --scafacos-method ewald
```

---

## 6. Automated solver sweep

[`run_dcm_long_range_workflow.sh`](../scripts/run_dcm_long_range_workflow.sh) runs validation, optional liquid-box certification, then one short `md-system` mini per solver configuration.

```bash
# Default: MIC + jax-pme (ewald, pme, p3m) in jax_mic mode
export MMML_CKPT=~/mmml/.../<ckpt>_portable.json
~/mmml/scripts/run_dcm_long_range_workflow.sh

# Custom sweep
LR_SOLVERS=mic,jax_pme \
JAX_PME_METHODS=ewald,p3m \
N_DCM=60 BOX_SIZE=32 \
~/mmml/scripts/run_dcm_long_range_workflow.sh

# Validation only (no MD)
SKIP_MD=1 ~/mmml/scripts/run_dcm_long_range_workflow.sh

# Include ScaFaCoS when installed
LR_SOLVERS=mic,jax_pme,scafacos \
SCAFACOS_METHODS=ewald,p3m \
~/mmml/scripts/run_dcm_long_range_workflow.sh
```

Results land in `~/tests/runs/dcm<N>_l<L>_lr_solvers/solver_comparison.tsv` (includes `hybrid_grms_kcalmol_A` per solver).

### Force validation (same certified box)

After the sweep, compare hybrid GRMS across solvers on the **same** certified PSF/CRD:

```bash
# Post-run: validate TSV from the workflow
python tests/functionality/long_range/07_hybrid_grms_lr_solver_compare.py \
  --summary-tsv ~/tests/runs/dcm60_l32_lr_solvers/solver_comparison.tsv

# Live probe at fixed coordinates (PyCHARMM + checkpoint required)
MMML_MPI_NP=1 ~/mmml/scripts/mmml-charmm-mpirun.sh \
  python tests/functionality/long_range/07_hybrid_grms_lr_solver_compare.py \
  --psf ~/tests/boxes/dcm60_l32/model.psf \
  --crd ~/tests/boxes/dcm60_l32/model.crd \
  --checkpoint "$MMML_CKPT" \
  --box-size 32
```

Expectations:

- All solvers return **finite** hybrid GRMS (no JAX tracer errors under `jax_pme`).
- **jax-pme** methods (`ewald`, `pme`, `p3m`) agree within ~10% rtol on the same geometry.
- **MIC** vs jax-pme may differ (truncated vs full Coulomb); both should be finite.

---

## 7. Expected behaviour

| Comparison | Expectation |
|------------|-------------|
| **MIC vs jax-pme** on large periodic box | \|E_jax_pme\| ≥ \|E_mic\| (truncated MIC underestimates Coulomb) |
| **ewald vs pme vs p3m** (jax-pme) | Energies agree to ~0.1–1% on neutral crystals; P3M may need tuning |
| **LJ r^-12 (pair loop)** | Identical LB mixing with/without jax_pme; **total vdw** differs (r^-6 via k-space) |
| **ScaFaCoS vs jax-pme** | Similar totals on CsCl / dimer tests (~0.1–0.7%) |
| **Hybrid GRMS** (same certified box) | jax-pme methods within ~10% rtol; MIC may differ (see §6 force validation) |

Log line at MLpot startup (verbose):

```
Decomposed MLpot: lr_solver=jax_pme, scafacos=no, jax_pme=yes (jax-pme method=ewald, sr_cutoff=6.0 Å)
```

---

## 8. Troubleshooting

| Issue | Fix |
|-------|-----|
| `jax_pme` falls back to `mic` | Install jax-pme: `uv sync` (pinned in pyproject.toml) |
| `scafacos` unavailable | Build to `~/.local/scafacos`, set `SCAFACOS_LIB` and `LD_LIBRARY_PATH` |
| MIC box too small | Use L ≥ 28–32 Å for DCM; see `run_dcm_liquid_workflow.sh` header |
| `periodic_external` fails | Need `--setup pbc_*`, positive `--box-size`, jax_pme or libfcs |
| Segfault under MPI + ScaFaCoS | Ensure mpi4py uses `COMM_WORLD.handle` (fixed in recent MMML) |
| `TracerArrayConversionError` under `jax_pme` | Upgrade MMML (jax-pme Coulomb uses `jax.pure_callback` inside JIT) |

---

## 9. Next steps

- Production NPT equilibration: extend `MD_STAGES=mini,equi` in the workflow script
- Campaign YAML: `lr_solver` / `jax_pme_method` under `defaults` — see [`docs/md-system-configs.md`](md-system-configs.md)
- Force validation: `07_hybrid_grms_lr_solver_compare.py` (also run automatically at end of `run_dcm_long_range_workflow.sh`)
