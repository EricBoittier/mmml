# Metatomic in MMML (PyCHARMM MLpot + ASE)

Metatomic supplies a TorchScript `AtomisticModel` (typically a `.pt` file) as an
ASE calculator. MMML does **not** put torch inside a `jax.jit` spherical
function. The CHARMM USER term is an ASE adapter; optional JAX MM stays in a
separate MM-only `setup_calculator` path.

Install (optional extra; CI does not require torch):

```bash
uv sync --extra metatomic
```

Device: `MMML_METATOMIC_DEVICE` (default `cpu`). Do not set CUDA at import time.

Units: ASE/metatomic energy is **eV**, forces **eV/Å**. CHARMM USER energy and
`dx/dy/dz` are **kcal/mol** and **kcal/mol/Å** (`EV_TO_KCAL_MOL`).

## Two evaluation modes

| `--metatomic-eval-mode` | USER term | Use when |
|---|---|---|
| `fragments` (default) | Isolated-monomer metatomic + switched dimer interaction `s(r_com)·(E(AB)−E(A)−E(B))` | The MMML ML/MM scheme (same split as PhysNet MLpot) |
| `whole_system` | One metatomic evaluation on the ML selection | All-ML USER; CHARMM ELEC/VDW should already be off under the jax_mic energy policy |

Fragment PBC: monomer B is wrapped by an **exact MIC lattice shift of its COM**
relative to A. That shift is **not** differentiated (same MD-safe rule as JAX
`wrap_dimer_monomer_b`). Differentiating the wrap blows up `|F|` near ±L/2.

The ML handoff `s(r_com)` is the NumPy twin of `ml_switch_scale` (1 inside the
ML region, 0 past `mm_switch_on`). Forces include the product-rule term
`-E_int · ds/dR`.

## PyCHARMM contract

`pycharmm.MLpot` only needs `get_pycharmm_calculator(...)` and Fortran-shaped
`calculate_charmm` (energy kcal/mol; `dx[ai] -= F`). `MetatomicMlpotModel`
fills that slot.

`setup_calculator(ml_potential_mode='metatomic')` builds the **JAX MM
spherical_fn only**. Pass `doML=False` and `doML_dimer=False`. Metatomic ML
lives in `MetatomicMlpotCalculator`, not in the jitted spherical_fn.

Optional JAX MM: `build_metatomic_mlpot_model(..., do_mm=True)` tries an MM-only
factory. If that fails (missing PSF / CGenFF tables), USER is **ML-only** and
the run prints a warning. Keep CHARMM ELEC/VDW or pass `--no-include-mm`.

Auto-detect: a `.pt` / `.pth` file, or a directory containing `model.pt` /
`exported.pt` / `metatomic.pt` / `atomistic-model.pt`, selects this path even
when `--ml-potential-mode` is unset.

## CLI

```bash
# PyCHARMM USER = fragment metatomic ML/MM (no CHARMM in this agent session)
mmml md-system --backend pycharmm \
  --ml-potential-mode metatomic \
  --checkpoint /path/to/export.pt \
  --metatomic-eval-mode fragments \
  --composition DCM:2 --setup pycharmm_minimize

# All-ML USER
mmml md-system --backend pycharmm \
  --ml-potential-mode metatomic \
  --checkpoint /path/to/export.pt \
  --metatomic-eval-mode whole_system \
  --no-include-mm

# ASE scans (no CHARMM)
mmml dimer-scan DCM DCM --calculator metatomic --checkpoint /path/to/export.pt
mmml ic-scan --calculator metatomic --checkpoint /path/to/export.pt --structure mol.xyz ...
mmml pet-interaction-pes --checkpoint /path/to/export.pt
```

Python:

```python
from mmml.interfaces.pycharmmInterface.mlpot import build_metatomic_mlpot_model

model = build_metatomic_mlpot_model(
    "export.pt",
    atomic_numbers,
    atoms_per_monomer,
    n_monomers,
    do_mm=False,  # skip JAX MM in tests
)
calc = model.get_pycharmm_calculator(ml_atom_indices=range(n_atoms))
# energy_kcal = calc.calculate_charmm(...)
```

Public imports: `mmml.interfaces.calculators.load_metatomic_calculator`,
`AseFragmentHybridCalculator`, `have_metatomic`, `is_metatomic_checkpoint`.

Energy/forces provider: `ProviderSpec(name="metatomic", options={"checkpoint": ...})`.

## Pass / fail (unit tests; no CHARMM)

```bash
uv run pytest \
  tests/unit/test_metatomic.py \
  tests/unit/test_ase_fragment_hybrid.py \
  tests/unit/test_metatomic_mlpot.py \
  tests/unit/test_energy_forces_providers.py \
  tests/unit/test_dimer_scan_cli.py \
  tests/unit/test_interaction_pes.py \
  tests/unit/test_md_system_pycharmm_cmd.py \
  -q
```

- Checkpoint suffix / export-dir detection does not import torch.
- NumPy `ml_switch_scale` matches JAX `ml_switch_scale`.
- MIC wrap of monomer B matches JAX `wrap_dimer_monomer_b` (exact, detached).
- Dummy ASE calculator: fragment `E = Σ E_i + s(r)(E_AB − E_A − E_B)`; CHARMM
  dummy `calculate_charmm` returns kcal/mol and subtracts forces into `dx/dy/dz`.
- `--ml-potential-mode metatomic` and `--metatomic-eval-mode` parse and forward
  through `build_pycharmm_command`.

## Hub models (PET-MAD / UPET)

Export a TorchScript `.pt` from the [lab-cosmo/upet](https://huggingface.co/lab-cosmo/upet)
or [lab-cosmo/pet-mad](https://huggingface.co/lab-cosmo/pet-mad) checkpoints with
metatrain (`mtt export`), then pass that file as `--checkpoint`.

```bash
uv sync --extra metatomic
uv pip install 'metatrain[pet]' huggingface_hub
mtt export lab-cosmo/upet models/pet-mad-s-v1.0.2.ckpt -o pet-mad-s-v1.0.2.pt
mtt export lab-cosmo/upet models/pet-mad-xs-v1.5.0.ckpt -o pet-mad-xs-v1.5.0.pt
mtt export lab-cosmo/upet models/pet-mols-s-v1.0.0.ckpt -o pet-mols-s-v1.0.0.pt
```

Smoke (ASE water dimer, CHARMM-free) is `tests/functionality/metatomic/eval_hub_models.py`.
Pass: finite eV energies/forces, fragment hybrid matches `s(r)·(E_AB−E_A−E_B)`,
dummy `calculate_charmm` kcal/mol equals `E_ev * EV_TO_KCAL_MOL`.

PET-MAD reports formation-like totals (~−14 eV / water). PET-MOLS reports a much
deeper electronic reference (~−2000 eV / water); that is the checkpoint, not a
unit bug. CHARMM USER will inherit that offset.

## Cost vs bundled JAX PhysNet

PET-MAD / PET-MOLS are much larger than the in-repo PhysNet DES-dimers
checkpoint (`examples/ckpts_json/DESdimers_params.json`: features=32,
`max_degree=1`, 2 message-passing iterations, 16 RBF, 6 Å cutoff, ZBL, **~19 k**
parameters, 680 KB JSON). The Hub PET exports are **~3–4.5 M** parameters
(14–20 MB `.pt`).

On CPU (no GPU, `JAX_PLATFORMS=cpu`), after ASE-cache reset and a 1e-4 Å
position jitter, median energy+forces:

| System | PhysNet whole / fragments | PET-MAD xs 1.5.0 | PET-MAD s 1.0.2 / PET-MOLS s |
|---|---|---|---|
| Water dimer (6 atoms) | 2.7 ms / 8.4 ms | 9.9 ms / 27 ms (**3.6×**) | 10 ms / 26 ms (**3.7×**) |
| Acetone dimer (20 atoms) | 3.8 ms / 9.9 ms | 17 ms / 42 ms (**4.5×**) | 34 ms / 63 ms (**9×**) |

`fragments` is three sequential evals (`E(A)+E(B)+s·(E(AB)−E(A)−E(B))`), which
is what `--metatomic-eval-mode fragments` pays per CHARMM USER call. Production
PhysNet MLpot batches those fragments in one jitted apply, so the PhysNet USER
path is cheaper than the sequential numbers above. A GPU would move PET much
more than this tiny PhysNet. Re-run:

```bash
JAX_PLATFORMS=cpu MMML_METATOMIC_DEVICE=cpu \
  uv run python tests/functionality/metatomic/compare_jax_cost.py
```

Local CHARMM smoke (serial `libcharmm`; `rebuild_charmm_mlpot.sh --no-mpi`):

```bash
export MMML_NO_CHARMM_MPI=1 MMML_NO_MPI_RERUN=1 MMML_METATOMIC_DEVICE=cpu
uv run python tests/functionality/metatomic/pycharmm_md_smoke.py --run \
  --checkpoint export.pt --residue ACO --n-molecules 2 --spacing 5.0 \
  --mini-nstep 3 --nstep 5 --no-echeck
```

Or `md-system --backend pycharmm --ml-potential-mode metatomic --checkpoint export.pt`
with a tiny dimer (`--composition DCM:2` or `--residue ACO --n-molecules 2`,
`--setup pycharmm_minimize` then a few `--ps` of `free_nve`).

Pass: CHARMM `ENER` includes a finite USER term; SD and short NVE complete.
With `--metatomic-eval-mode fragments` and a monomer `cons_fix`, fixed-monomer
RMSD ≈ 0 after SD pass 2 (same criterion as PhysNet MLpot).

## Periodic liquid ethanol (32 Å, 300 K, 0.5 fs)

Worked example: neat `ETOH:338` in a 32 Å cube at 0.789 g/cm³ (experimental
bulk), **`whole_system`** PET-MAD. CHARMM-free ASE MD is first-class CLI:

```bash
export PET_MAD_CKPT=/path/to/pet-mad-xs-v1.5.0.pt
# CHARMM-free ASE (grid pack, no Packmol / CHARMM)
mmml metatomic-pbc-md --ensemble nvt --n-steps 5
# NVE conservation: FIRE mini, then VelocityVerlet (0.2 ps default)
mmml metatomic-pbc-md --ensemble nve --minimize-steps 60 --n-steps 400
# wrappers: ./examples/pet_mad_etoh_pbc/run_smoke.sh  and  run_nve.sh

# Production: MM-certify the box, then all-ML USER
mmml liquid-box --composition ETOH:338 --box-size 32 \
  --target-density-g-cm3 0.789 -o boxes/etoh338_32A
mmml md-system --config examples/pet_mad_etoh_pbc/yaml/pbc_nvt.yaml \
  --job-id nvt --from-psf boxes/etoh338_32A/model.psf \
  --from-crd boxes/etoh338_32A/model.crd --checkpoint "$PET_MAD_CKPT"
```

`--box-auto count --composition ETOH:1 --box-size 32 --target-density-g-cm3 0.789`
is the same 338-molecule count. Do **not** use `fragments` on this box.
YAML `nve` is 0.2 ps after mini. Details, pass/fail, and YAML:
[PET-MAD ethanol PBC](examples/pet-mad-etoh-pbc.md).

Interaction PES (CHARMM-free single points; no MD):

```bash
JAX_PLATFORMS=cpu MMML_METATOMIC_DEVICE=cpu \
  mmml pet-interaction-pes --checkpoint /path/to/pet-mad-xs-v1.5.0.pt
```

Writes linear OH···O vs acceptor–acceptor \(E_\mathrm{int}(r)\) slices
(O–O; acetone is C=O vs methyl–methyl), an angular cut at \(r_e\), an ethanol
\(r\times\theta\) surface (\(\theta\) = donor–H–acceptor), and a linear/cyclic
trimer leftover \(E_3\). Plots are ICML-styled and reproducible from JSON
(`--from-json`). See the [ethanol PBC example](examples/pet-mad-etoh-pbc.md).

## PET-MAD teacher → PhysNet student (acetone)

Weight copy is impossible (TorchScript PET vs 19 k-parameter JAX PhysNet).
This path **labels** acetone geometries with PET and trains the bundled
PhysNet architecture on those labels.

Pool: bundled ACO PDB + DMC extxyz, Cartesian noise, C=O stretches, random
relative orientations, and COM scans covering the four PES regions (repulsive /
well / shoulder / long-range) plus a far-field replica. Default labels are the
hybrid pieces: monomer `E - E_eq` and unswitched dimer `E(AB)-E(A)-E(B)`
(forces match). The ML/MM switch is **not** baked into `E`; MLpot applies it
at MD time.

```bash
# 1. Label (CHARMM-free). --preset smoke is tiny; md is the MD-oriented mix.
JAX_PLATFORMS=cpu MMML_METATOMIC_DEVICE=cpu \
  uv run mmml pet-physnet-distill \
    --checkpoint /tmp/mmml-metatomic-models/pet-mad-xs-v1.5.0.pt \
    --out-dir ./acetone_pet_distill --preset smoke

# 2. Train the student (warm-start DESdimers architecture, no live .pt teacher)
uv run mmml physnet-train --config ./acetone_pet_distill/physnet-train.yaml

# 3. Held-out teacher vs student (same NPZ units, eV)
uv run mmml physnet-evaluate \
  --checkpoint ./ckpts/acetone_pet_student \
  --data ./acetone_pet_distill/valid.npz

# 4. Downstream MD (you run this; not in agent sessions)
uv run mmml md-system --backend pycharmm \
  --ml-potential-mode physnet \
  --checkpoint ./ckpts/acetone_pet_student \
  --residue ACO --n-molecules 2 --setup pycharmm_minimize
```

Pass for (1): `train.npz` / `valid.npz` have finite `E`/`F`, `_mmml_units` is
eV / eV/Å, `report.json` records teacher path and counts.
Pass for (2–3): valid force MAE well below a raw DESdimers-on-PET baseline.
Pass for (4): finite USER, short NVE without explosion; with `cons_fix` on one
monomer, RMSD ≈ 0 after SD pass 2.

Do not pass the PET `.pt` as `--teacher-checkpoint` on `physnet-train` — that
flag loads a Flax tree. The NPZ *is* the teacher.

### Batched TorchScript teacher and larger PETs

Labelling defaults to `--teacher-backend torchscript`
(`mmml/distill/batched_teacher.py`). Monomers, dimers and both dimer
fragments go into one request. Structures are sorted by size, packed under
`--max-atoms-per-batch` / `--max-systems-per-batch`, and each pack is one
`AtomisticModel.forward` over a `list[System]`. Forces are `-dE/dR` from one
backward pass. `--teacher-backend ase` keeps the per-structure
`MetatomicCalculator` path.

On an RTX 5090 with PET-MAD xs, the `md` acetone pool (548 geometries, 1320
structures with fragments) labels in about 0.9 s once warm. The ASE path runs at
about 16 ms per structure. The two paths agree to float32 noise
(|ΔE| < 1e-5 eV, |ΔF| < 1e-4 eV/Å).

The `metatomic` extra installs `upet` (downloads and exports the UPET family)
and `metatrain` (fine-tuning). To export another teacher:

```bash
python -c "from upet import list_upet; list_upet()"
python -c "from upet import save_upet; \
  save_upet(model='pet-omol', size='m', version='1.0.0', output='pet-omol-m-v1.0.0.pt')"
mmml pet-physnet-distill --checkpoint pet-omol-m-v1.0.0.pt \
  --out-dir ./acetone_omol_m --preset md --max-atoms-per-batch 2048
```

Lower `--max-atoms-per-batch` if an `l`/`xl` model runs out of GPU memory.
