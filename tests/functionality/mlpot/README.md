# CHARMM `MLpot` exploration scripts

Runnable scripts to bring up `pycharmm.MLpot` alongside the existing ASE PhysNet path (`get_ase_calc` / `get_pyc`). Run them **in order** from the repository root.

**Default cluster:** acetone **ACO × N** (10 atoms per monomer). `ic.build()` alone is nearly 1D; builders use bundled `mmml/generate/sample/pdb/aco_monomer.pdb` and place monomers on a grid with `--spacing` (default 4 Å). Each CGenFF **resid** = one monomer for constraints.

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
| `04_mlpot_minimize_stub.py` | yes | SD minimization + optional `--save` outputs |
| `05_mlpot_dynamics_stub.py` | yes | Short NVE with DCD/restart (`--run`) |
| `06_spatial_mpi_tier2_smoke.py` | optional | Tier 2 spatial MPI env + hybrid callback under `mpirun` (see `docs/pycharmm-mpi.md`) |
| `07_md_system_spatial_mpi_mini.py` | optional | Full `md-system --ml-spatial-mpi` mini dry-run / cluster smoke |
| `08_serial_vs_mpirun_md_system.py` | optional | A/B serial `python md-system` vs `mpirun np=1` (`upinb` claim); writes JSON report |
| `run_all.sh` | — | Run 00→03; `RUN_EXTENDED=1` adds 04–05 |

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

## `mmml md-system` (PyCHARMM backend)

The same minimize / NVE workflows are available via the main CLI (no `--run` stub):

```bash
# Packmol sphere + MMFP (30 monomers in R=22 Å, flat-bottom R=20 Å)
mmml md-system --setup free_nve --backend pycharmm \
  --composition ACO:30 \
  --packmol-radius 22 \
  --flat-bottom-radius 20 \
  --packmol-tolerance 2.0 \
  --seed 42 \
  --ps 0.5 --mini-nstep 20

# Mixed composition in a Packmol sphere (explicit --packmol-sphere)
mmml md-system --setup free_nve --backend pycharmm \
  --composition ACO:15,MEOH:15 \
  --packmol-sphere --packmol-radius 25 \
  --flat-bottom-radius 22 \
  --output-dir artifacts/pycharmm_mlpot/aco15_meoh15

# Two-pass SD + short NVE (acetone tetramer, MMFP sphere R=20 Å)
mmml md-system --setup free_nve --backend pycharmm --residue ACO --n-molecules 4 \
  --flat-bottom-radius 20 --ps 0.5 --mini-nstep 20 --fix-resids 1,3

# SD minimization only
mmml md-system --setup pycharmm_minimize --composition ACO:2 --mini-nstep 30

# Direct module entry (same as backend dispatch)
python -m mmml.cli.run.md_pbc_suite.pycharmm_mlpot --phase full --residue ACO --n-molecules 2 --ps 0.1
```

Outputs default to `artifacts/pycharmm_mlpot/` (`cluster_for_vmd_*.psf`, `nve_*.dcd`). Use `--flat-bottom-radius` (maps to CHARMM MMFP `--fb-rad`) for vacuum droplet restraints.

CHARMM print / heating cadence (`nprint`, `dyn-nprint`, `ihtfrq`): see [`CHARMM_SETTINGS.md`](../../mmml/interfaces/pycharmmInterface/mlpot/CHARMM_SETTINGS.md).

## Steps 4–5 and pytest

```bash
# Minimization (you already validated 3D XYZ export)
# Default: PRNLev=5, nprint=50 (CHARMM console). Use --quiet to reduce output further.
python tests/functionality/mlpot/04_mlpot_minimize_stub.py --run --save --nstep 10
# Tetramer: free SD then constrained SD on monomers 1 and 3:
python tests/functionality/mlpot/04_mlpot_minimize_stub.py --run --save --n-molecules 4 --fix-resids 1,3 --nstep 20
# Trimer, no constraints:
python tests/functionality/mlpot/04_mlpot_minimize_stub.py --run --n-molecules 3 --no-fix
# VMD: topology MUST be cluster_for_vmd_<tag>.psf (bonds intact, same atom count as DCD).
# Do NOT use mini_full_mlpot_*.psf in VMD (MLpot strips bonds; CHARMM-only).
# 4-mer example:
# vmd tests/functionality/mlpot/output/minimize/cluster_for_vmd_aco_4mer.psf \
#     tests/functionality/mlpot/output/minimize/mini_full_mlpot_aco_4mer.dcd
# Denser minimization DCD: every step (default) or every 5 SD steps:
python tests/functionality/mlpot/04_mlpot_minimize_stub.py --run --save --dcd-nsavc 1
python tests/functionality/mlpot/04_mlpot_minimize_stub.py --run --save --dcd-nsavc 5

# Short NVE (~5 fs at 0.25 fs timestep)
python tests/functionality/mlpot/05_mlpot_dynamics_stub.py --run --nstep 20
# 4-mer: free mini, constrained mini on 1,3, then NVE:
python tests/functionality/mlpot/05_mlpot_dynamics_stub.py --run --n-molecules 4 --fix-resids 1,3 --mini-nstep 30 --nstep 50
# Freeze monomers 1–2 during NVE only (after default pre-minimize):
python tests/functionality/mlpot/05_mlpot_dynamics_stub.py --run --no-fix --constrain-resids 1,2 --nstep 50
# DCD every step (default): --dcd-nsavc 1
# DCD every 0.001 ps (4 steps at 0.25 fs): --dcd-interval-ps 0.001
python tests/functionality/mlpot/05_mlpot_dynamics_stub.py --run --nstep 100 --dcd-nsavc 1
python tests/functionality/mlpot/05_mlpot_dynamics_stub.py --run --nstep 100 --dcd-interval-ps 0.00025
# Tighter energy kill (stop before MMFP/PRESS RMS runaway):
python tests/functionality/mlpot/05_mlpot_dynamics_stub.py --run --nstep 200000 --echeck 50
# Looser (NPT-style):
python tests/functionality/mlpot/05_mlpot_dynamics_stub.py --run --echeck 500

# Non-PBC MMFP flat-bottom sphere (production dyna.inp style):
python tests/functionality/mlpot/05_mlpot_dynamics_stub.py --run --n-molecules 4 \
  --fb-rad 20 --fb-forc 1 --fix-resids 1 --mini-nstep 30 --nstep 100

# Core + extended in one shot
RUN_EXTENDED=1 ./tests/functionality/mlpot/run_all.sh

# Pytest (script 03 logic; needs CHARMM + checkpoint)
pytest tests/functionality/mlpot/test_mlpot_energy_matches_ase.py -q
pytest tests/functionality/mlpot/test_pycharmm_conversion.py -q
```

## `md-system --dyna-probe` (evaluate + 1-step NVE)

Compares all PyCHARMM force lanes **before and after** one NVE integration step — useful when
`evaluate-npz` energy matches MP2 but `charmm_total` forces do not.

```bash
REF=artifacts/dcm_mp2_round2/new-dcm-round-2-only_MP2_41950_dimers_psf_order.npz
FRAME=16566
OUT=artifacts/eval_smoke/frame_${FRAME}/dyna_probe

mmml md-system --dyna-probe \
  --evaluate-npz "$REF" --evaluate-frame "$FRAME" \
  --evaluate-reference-npz "$REF" --evaluate-reference-frame "$FRAME" \
  --composition DCM:2 --backend pycharmm --setup free_nve \
  --ml-switch-width 0.1 --mm-switch-on 6.0 --mm-switch-width 3.0 \
  --dyna-probe-nstep 1 --dyna-probe-dt-fs 0.5 --no-echeck \
  --output-dir "$OUT" --quiet

python -c "
import json
d=json.load(open('${OUT}/dyna_probe.json'))
for s in d['snapshots']:
    rc=s.get('reference_compare',{})
    print(s['label'], 'force_rmse', rc.get('force_rmse_eV_A'),
          'spherical', rc.get('force_rmse_spherical_fn_eV_A'),
          'charmm', rc.get('force_rmse_charmm_total_eV_A'))
"
```

Pass criteria: `force_rmse_spherical_fn_eV_A` ≈ ASE evaluate-npz (~0.003 for frame 16566);
`force_rmse_charmm_total_eV_A` exposes the static CHARMM export bug if still ~0.25.
Post-dyna `cross_lane_force_rmse_eV_A` shows whether DYNA changes the mismatch.

## Serial vs mpirun probe (`08`)

Validates whether serial `python md-system` is safe on your MPI-linked CHARMM stack
(see `docs/pycharmm-mpi.md`). **Passed on gpu09 (June 2026)** with matching outputs.

```bash
python tests/functionality/mlpot/08_serial_vs_mpirun_md_system.py --dry-run

export MMML_CKPT=/path/to/checkpoint.json
python tests/functionality/mlpot/08_serial_vs_mpirun_md_system.py --run-both \
  --checkpoint "$MMML_CKPT" \
  --output-dir artifacts/serial_vs_mpirun_$(date +%Y%m%d_%H%M%S)
```

Config: `mmml/cli/run/md_system.serial_mpi_probe.yaml` (ACO:2 hybrid mini). JSON report
includes hostname, UTC timestamp, env snapshot, and per-run `elapsed_s`.

## Library module

Reusable API: `mmml/interfaces/pycharmmInterface/mlpot/` — see `mlpot/README.md`.
