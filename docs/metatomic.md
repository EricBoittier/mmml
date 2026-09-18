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

Local CHARMM smoke (user machine; not run in the agent):

1. `uv sync --extra metatomic`
2. Export a metatomic `.pt` (see [metatomic](https://github.com/metatensor/metatomic)).
3. `md-system --backend pycharmm --ml-potential-mode metatomic --checkpoint export.pt`
   with a tiny dimer (`--composition DCM:2`, `--setup pycharmm_minimize`).
4. Pass: CHARMM `ENER` includes a USER term; with `--metatomic-eval-mode fragments`
   and a monomer `cons_fix`, fixed-monomer RMSD ≈ 0 after SD pass 2 (same criterion
   as PhysNet MLpot).
