# PET-MAD PBC MD: 32 Å liquid ethanol at 300 K

Neat **ethanol** (CGenFF `ETOH`) in a **32 Å** cube at experimental bulk density,
**300 K**, timestep **0.5 fs**, with a metatomic **PET-MAD** model as the all-ML
potential.

Example files: [`examples/pet_mad_etoh_pbc/`](https://github.com/EricBoittier/mmml/blob/main/examples/pet_mad_etoh_pbc/).
Metatomic USER contract: [metatomic.md](../metatomic.md).
Box certification: [liquid-box workflow](../liquid-box-workflow.md).

---

## Why this box

| Quantity | Value |
|----------|-------|
| Cell | cubic **32 Å**, PBC |
| Target ρ | **0.789 g/cm³** (`SOLVENT_BULK_PROPS["ETOH"]`, ~298 K) |
| Count | **ETOH:338** (3042 atoms) |
| T, Δt | **300 K**, **0.5 fs** |
| Eval | `--metatomic-eval-mode whole_system` |

`--box-auto count --composition ETOH:1 --box-size 32 --target-density-g-cm3 0.789`
scales the stoichiometry to 338 molecules. Pin `ETOH:338` in YAML so the count
does not drift.

**Do not** use `fragments` here. That mode is the MMML dimer hybrid
(`Σ E_i + s(r)·(E_AB − E_A − E_B)`). A neat 338-molecule liquid would issue
hundreds of isolated-monomer evals plus dimer pairs on every CHARMM USER call.
`whole_system` is one PET-MAD evaluation on the ML selection.

CHARMM ELEC/VDW/bonded should be off (`--no-include-mm` and
`--charmm-zero-energy-terms vdw,elec,bonded`) so the USER term *is* the PES.

---

## 1. Export PET-MAD

```bash
uv sync --extra metatomic
uv pip install 'metatrain[pet]' huggingface_hub
mtt export lab-cosmo/upet models/pet-mad-xs-v1.5.0.ckpt \
  -o /path/to/pet-mad-xs-v1.5.0.pt
export PET_MAD_CKPT=/path/to/pet-mad-xs-v1.5.0.pt
```

xs 1.5.0 is the smoke default. PET-MAD s is the same CLI with a larger `.pt`.

---

## 2. CHARMM-free ASE smoke (no Packmol)

First-class command: **`mmml metatomic-pbc-md`**. Grid-places 338 copies of
`examples/pet_mad_etoh_pbc/etoh.xyz` in the 32 Å cell and runs Langevin NVT
(or VelocityVerlet NVE) through `load_metatomic_calculator`. This does **not**
go through `setup_calculator` (JAX MM) or CHARMM. `examples/pet_mad_etoh_pbc/ase_pbc_md.py`
is a thin wrapper around the same CLI.

```bash
export PET_MAD_CKPT=/path/to/pet-mad-xs-v1.5.0.pt
JAX_PLATFORMS=cpu MMML_METATOMIC_DEVICE=cpu \
  mmml metatomic-pbc-md --ensemble nvt --n-steps 5
# or: ./examples/pet_mad_etoh_pbc/run_smoke.sh
# or: N_STEPS=2 ENSEMBLE=nve ./examples/pet_mad_etoh_pbc/run_smoke.sh
```

| Check | Pass |
|-------|------|
| `n_molecules` | 338 |
| `n_atoms` | 3042 |
| `density_g_cm3` | ≈ 0.789 (rel 1 %) |
| `dt_fs` / `temperature_K` | 0.5 / 300 |
| `E0_eV`, `E1_eV`, forces | finite |
| `report.json` `"ok"` | `true` |

GPU: `MMML_METATOMIC_DEVICE=cuda`. On CPU, a 3042-atom PET-MAD step is seconds
to tens of seconds; keep `--n-steps` small for a smoke.

### NVE conservation (FIRE + VelocityVerlet)

Lattice packing leaves |F| ~ 10 eV/Å. Minimize first, then NVE at 0.5 fs:

```bash
export PET_MAD_CKPT=/path/to/pet-mad-xs-v1.5.0.pt
mmml metatomic-pbc-md --ensemble nve --minimize-steps 60 --n-steps 400
# or: N_STEPS=400 MINI_STEPS=60 ./examples/pet_mad_etoh_pbc/run_nve.sh
```

| Check | Pass |
|-------|------|
| FIRE | finite E, |F|_max drops vs the lattice start |
| `energy.csv` | PE, KE, Etot, T every step |
| Etot | finite; no T explosion (> 5000 K fails) |
| `drift_eV_per_ps` | reported in `report.json` (judge conservation from the trace) |

400 steps at 0.5 fs is **0.2 ps**. GPU recommended for longer traces.

Unit tests (no torch MD):

```bash
uv run pytest \
  tests/unit/test_box_sizing.py \
  tests/unit/test_pet_mad_etoh_pbc_example.py \
  tests/unit/test_interaction_pes.py \
  -q
```

---

## 2b. Interaction slices and surfaces (no MD)

Rigid COM scans of PET-MAD xs 1.5.0. Interaction energy is
\(E_\mathrm{int}=E(AB)-E(A)-E(B)\) in kcal/mol. The trimer leftover is
\(E_3=E_\mathrm{int}(ABC)-\sum E_\mathrm{int}(IJ)\). If PET were 2-body at the
molecular-fragment level, \(E_3=0\).

```bash
export PET_MAD_CKPT=/path/to/pet-mad-xs-v1.5.0.pt
JAX_PLATFORMS=cpu MMML_METATOMIC_DEVICE=cpu \
  mmml pet-interaction-pes --checkpoint "$PET_MAD_CKPT"
# or: ./examples/pet_mad_etoh_pbc/run_interaction_pes.sh
```

| Check | Pass |
|-------|------|
| JSON | `schema` is `mmml.interaction_pes/v1`; energies finite |
| 1D slices | H-bond and stacked \(E_\mathrm{int}(r)\) from 2.5–12 Å; far-field ≈ 0 past the ~9 Å RF |
| 2D surface | ethanol COM × in-plane rotation; heatmap + isolevels |
| Trimer | \(|E_3|\) is kcal/mol at close spacing (water ~2.85 Å, ethanol ~4.2 Å) and ≈ 0 at 12 Å |

Replot without PET: `--from-json examples/pet_mad_etoh_pbc/data/interaction_pes.json`.
Figures land next to that JSON (`pet_mad_dimer_slices.png`,
`pet_mad_dimer_surface.png`, `pet_mad_trimer_mbe.png`) and under
`docs/images/plots/` when regenerated.

---

## 3. Production: certify MM box, then PET-MAD USER

Phase A is CHARMM MM only (Packmol → density → SD/ABNR). Phase B registers
PET-MAD as the CHARMM USER term.

```bash
export PET_MAD_CKPT=/path/to/pet-mad-xs-v1.5.0.pt
export MMML_NO_CHARMM_MPI=1 MMML_NO_MPI_RERUN=1
export MMML_METATOMIC_DEVICE=cpu   # or cuda

mmml liquid-box --composition ETOH:338 --box-size 32 \
  --target-density-g-cm3 0.789 -o boxes/etoh338_32A
```

Pass for liquid-box: `box.json` `box_side_A ≈ 32`, `model.psf` / `model.crd`
exist, inter-monomer contacts above the prep floor (see `REPORT.md`).

```bash
# 5 NVE steps at 0.5 fs (smoke)
mmml md-system --config examples/pet_mad_etoh_pbc/yaml/pbc_nvt.yaml \
  --job-id nve_smoke \
  --from-psf boxes/etoh338_32A/model.psf \
  --from-crd boxes/etoh338_32A/model.crd \
  --checkpoint "$PET_MAD_CKPT"

# 0.2 ps NVE after mini (conservation)
mmml md-system --config examples/pet_mad_etoh_pbc/yaml/pbc_nvt.yaml \
  --job-id nve \
  --from-psf boxes/etoh338_32A/model.psf \
  --from-crd boxes/etoh338_32A/model.crd \
  --checkpoint "$PET_MAD_CKPT"

# 2 ps heat + 20 ps NVT at 300 K
mmml md-system --config examples/pet_mad_etoh_pbc/yaml/pbc_nvt.yaml \
  --job-id nvt \
  --from-psf boxes/etoh338_32A/model.psf \
  --from-crd boxes/etoh338_32A/model.crd \
  --checkpoint "$PET_MAD_CKPT"
```

Equivalent without YAML (`--box-auto count` derives 338):

```bash
mmml md-system --backend pycharmm --setup pbc_nvt \
  --ml-potential-mode metatomic --metatomic-eval-mode whole_system \
  --no-include-mm --mlpot-pbc \
  --charmm-zero-energy-terms vdw,elec,bonded \
  --box-auto count --composition ETOH:1 --box-size 32 \
  --target-density-g-cm3 0.789 \
  --temperature 300 --dt-fs 0.5 \
  --checkpoint "$PET_MAD_CKPT"
```

| Check | Pass |
|-------|------|
| CHARMM `ENER` | finite `USER` |
| Cell | 32 Å cube (from `box.json` / `--box-size`) |
| Count | ETOH:338 |
| Smoke NVE | `--ps-nve 0.0025` at `--dt-fs 0.5` is 5 steps; completes |
| NVT | heat + equi at 300 K without explosion; DCD written |

`--ps 0.0025` at `--dt-fs 0.5` is 5 dynamics steps.

---

## What this is not

- Not hybrid ML/MM PhysNet MLpot. PET-MAD scores the whole liquid.
- Not JAX-MD (`setup_calculator` with `ml_potential_mode='metatomic'` is MM-only).
- Not a density-equilibration campaign. Fixed **L = 32 Å**; NPT is a later job
  if you need ρ(T,P).

Related: [md-system YAML](../md-system-configs.md),
[TIP3 Ewald smoke](https://github.com/EricBoittier/mmml/blob/main/examples/tip3_50_ewald_smoke/).
