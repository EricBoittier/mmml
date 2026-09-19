# PET-MAD periodic MD: 32 Å liquid ethanol

Worked example: neat ethanol in a **32 Å** cube at experimental liquid density
(**0.789 g/cm³** → **ETOH:338**, 3042 atoms), **300 K**, **0.5 fs**.

Full write-up: [`docs/examples/pet-mad-etoh-pbc.md`](../../docs/examples/pet-mad-etoh-pbc.md).
Metatomic USER notes: [`docs/metatomic.md`](../../docs/metatomic.md).

Use **`--metatomic-eval-mode whole_system`**. Fragment mode would evaluate 338
isolated monomers plus dimers on every CHARMM USER call.

## CHARMM-free ASE smoke

```bash
export PET_MAD_CKPT=/path/to/pet-mad-xs-v1.5.0.pt
mmml metatomic-pbc-md --ensemble nvt --n-steps 5
# or: ./examples/pet_mad_etoh_pbc/run_smoke.sh
```

Pass: `report.json` has `"ok": true`, finite eV energies/forces, `n_molecules=338`,
`density_g_cm3 ≈ 0.789`, 5 × 0.5 fs steps at 300 K.

## CHARMM-free NVE conservation

FIRE-minimize the lattice box, assign 300 K velocities, then VelocityVerlet
at 0.5 fs. Forces are autograd of the energy (`non_conservative=False`).

```bash
export PET_MAD_CKPT=/path/to/pet-mad-xs-v1.5.0.pt
mmml metatomic-pbc-md --ensemble nve --minimize-steps 60 --n-steps 400
# or: N_STEPS=400 MINI_STEPS=60 ./examples/pet_mad_etoh_pbc/run_nve.sh
```

Writes `energy.csv`, `nve_energy.png`, and `report.json` with `drift_eV_per_ps`
and `drift_meV_per_atom_ps`. Default 400 steps is 0.2 ps.

## PyCHARMM (`md-system`)

Certify the box under MM, then all-ML PET-MAD USER:

```bash
export PET_MAD_CKPT=/path/to/pet-mad-xs-v1.5.0.pt

mmml liquid-box --composition ETOH:338 --box-size 32 \
  --target-density-g-cm3 0.789 -o boxes/etoh338_32A

mmml md-system --config examples/pet_mad_etoh_pbc/yaml/pbc_nvt.yaml \
  --job-id nve_smoke \
  --from-psf boxes/etoh338_32A/model.psf \
  --from-crd boxes/etoh338_32A/model.crd \
  --checkpoint "$PET_MAD_CKPT"
```

Production NVT (`--job-id nvt`): 2 ps heat + 20 ps equilibration at 300 K, 0.5 fs.

The same count without pinning `ETOH:338`:

```bash
mmml md-system --backend pycharmm \
  --ml-potential-mode metatomic --metatomic-eval-mode whole_system \
  --no-include-mm --mlpot-pbc \
  --box-auto count --composition ETOH:1 --box-size 32 \
  --target-density-g-cm3 0.789 \
  --temperature 300 --dt-fs 0.5 \
  --checkpoint "$PET_MAD_CKPT"
```

## Interaction slices and surfaces (CHARMM-free)

PET is many-body (transformer over neighbor tokens; PET-MAD xs RF ≈ 9 Å). A
2-body molecular reconstruction cannot clone it. `mmml pet-interaction-pes`
scans rigid OH···O dimers/trimers (O–O, Cs acceptor flap) with the same checkpoint (single points, no MD):

```bash
export PET_MAD_CKPT=/path/to/pet-mad-xs-v1.5.0.pt
JAX_PLATFORMS=cpu MMML_METATOMIC_DEVICE=cpu \
  mmml pet-interaction-pes --checkpoint "$PET_MAD_CKPT"
# or: ./examples/pet_mad_etoh_pbc/run_interaction_pes.sh
# replot: FROM_JSON=examples/pet_mad_etoh_pbc/data/interaction_pes.json \
#   ./examples/pet_mad_etoh_pbc/run_interaction_pes.sh
```

| Figure | What it shows |
|--------|----------------|
| `pet_mad_dimer_slices.png` | Water / ethanol / acetone \(E_\mathrm{int}(r)\) vs O–O (kcal/mol vs Å); linear OH···O vs acceptor–acceptor (acetone: C=O vs methyl); wall inset; PET RF at 9 Å |
| `pet_mad_dimer_angular.png` | \(E_\mathrm{int}(\theta)\) at \(r_e\); \(\theta\) = donor–H–acceptor (in-plane vs out-of-plane) |
| `pet_mad_dimer_surface.png` | Ethanol \(E_\mathrm{int}(r,\theta)\) heatmap + isolevels + marked minimum |
| `pet_mad_trimer_mbe.png` | Linear vs cyclic: \(E_\mathrm{int}(ABC)\), \(\sum E_\mathrm{int}(IJ)\), residual \(E_3\), and \(E_3/E_\mathrm{int}\) |

Pass: JSON schema `mmml.interaction_pes/v2`; far-field \(E_\mathrm{int}\to 0\) past ~9–12 Å; trimer \(|E_3|\) kcal/mol at close O–O and ~0 at 12 Å. Unit tests (dummy calculator, no torch): `uv run pytest tests/unit/test_interaction_pes.py -q`.
