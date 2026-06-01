# CHARMM `MLpot` exploration scripts

Runnable scripts to bring up `pycharmm.MLpot` alongside the existing ASE PhysNet path (`get_ase_calc` / `get_pyc`). Run them **in order** from the repository root.

**Default cluster:** acetone dimer **ACO × 2 → 20 atoms**. `ic.build()` alone is nearly 1D (y≈z≈0); builders use bundled `mmml/generate/sample/pdb/aco_monomer.pdb` for 3D monomer geometry, then place monomers on a grid with `--spacing` (default 4 Å COM separation).

## Prerequisites

- `CHARMM_HOME` and `CHARMM_LIB_DIR` (via `mmml/CHARMMSETUP` or environment)
- PyCHARMM importable (`import_pycharmm` path)
- JAX, e3x, ASE
- A PhysNet checkpoint (same as other mmml tests)

```bash
export MMML_CKPT=/path/to/DESdimers   # or examples/ckpts_json/DESdimers_params.json
```

Optional: install the package editable so imports resolve:

```bash
pip install -e .
```

## Scripts (run in order)

| Script | Needs CHARMM | Purpose |
|--------|--------------|---------|
| `00_check_environment.py` | lib only | Verify imports, `mlpot_*` symbols, checkpoint |
| `01_callback_vs_ase_no_charmm.py` | no | Compare `PyCharmm_Calculator.calculate_charmm` vs `get_ase_calc` on same geometry |
| `02_mlpot_register_smoke.py` | yes | Build PSF cluster, register `MLpot`, run `energy.show()` |
| `03_energy_compare.py` | yes | Tabulate ASE vs direct callback vs CHARMM energy after `MLpot` |
| `run_all.sh` | — | Run 00→03; stops on first failure |

```bash
cd /path/to/mmml
./tests/functionality/mlpot/run_all.sh

# Or individually:
python tests/functionality/mlpot/00_check_environment.py
python tests/functionality/mlpot/01_callback_vs_ase_no_charmm.py --residue ACO --n-molecules 2
python tests/functionality/mlpot/02_mlpot_register_smoke.py --residue ACO --n-molecules 2
python tests/functionality/mlpot/03_energy_compare.py --residue ACO --n-molecules 2 --rtol 0.05
```

## What we learn at each step

1. **00** — Environment blockers (missing lib, no checkpoint, no `mlpot_set_func` in your CHARMM build).
2. **01** — Whether `get_pyc` / `calculate_charmm` matches ASE **before** CHARMM coupling (units, pair list, indices).
3. **02** — Whether `MLpot` registers and CHARMM accepts a full `ENER` call without crashing.
4. **03** — Numeric agreement between ASE energy and CHARMM-reported terms (likely `USER` or total `ENER`).

## Known gaps (to narrow in follow-up)

- `get_pycharmm_calculator()` in `helper_mlp.py` ignores arguments from `MLpot` and returns a pre-built calculator (OK only when all atoms are ML and in order).
- `PyCharmm_Calculator` does not yet use ML–MM pair lists (`idxu`/`idxv`) for embedding electrostatics.
- `get_pyc` uses `pycharmm_conversion` to convert model output from eV → kcal/mol (same factor as `ev2kcalmol` in ASE calculators).

## Steps 4–5 and pytest (stubs)

| Script / test | Purpose |
|---------------|---------|
| `04_mlpot_minimize_stub.py` | Full-system MLpot + `cons_fix` on `--fix-resid 1`; `--run --save` writes outputs |
| `05_mlpot_dynamics_stub.py` | Short NVE with MLpot (`--run`) |
| `test_mlpot_energy_matches_ase.py` | Pytest equivalent of script 03 |

```bash
pytest tests/functionality/mlpot/test_mlpot_energy_matches_ase.py -q
```

## Library module

Reusable API: `mmml/interfaces/pycharmmInterface/mlpot/` — see `mlpot/README.md`.
