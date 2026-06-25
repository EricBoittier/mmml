# MLpot workflows (`mmml.interfaces.pycharmmInterface.mlpot`)

Helpers for **PhysNet + CHARMM MLpot**, based on the validated scripts in `tests/functionality/mlpot/`.

**Nonbond lists:** how CHARMM and MMML neighbor lists interact, update frequencies, and NVE stability — see [NONBOND_LISTS.md](NONBOND_LISTS.md).

## Quick start (energy only)

```python
import ase
from mmml.interfaces.pycharmmInterface.mlpot import (
    load_physnet_mlpot_bundle,
    register_mlpot,
    select_all_atoms,
    setup_default_nbonds,
)

z, r = ...  # from cluster build or PSF
atoms = ase.Atoms(numbers=z, positions=r)
setup_default_nbonds()

params, model, pyCModel = load_physnet_mlpot_bundle("examples/ckpts_json/DESdimers_params.json", len(z), atoms)
ctx = register_mlpot(pyCModel, z, select_all_atoms())
try:
    import pycharmm.energy as energy
    energy.show()
finally:
    ctx.unset()
```

## Workflow order (CLI / ``run_workflow``)

1. Build cluster: each monomer **CHARMM-minimized** (template + SD/ABNR) → written to Packmol → sphere pack → PSF reorder  
2. **Cluster CHARMM MM SD/ABNR** (CGENFF only, no MLpot) — ``--charmm-pre-minimize`` (default on)  
3. **Register MLpot** + PhysNet  
4. **MLpot SD** (optional second pass with ``cons_fix``) then dynamics  

Skip step 2 with ``--no-charmm-pre-minimize``. Tune with ``--charmm-sd-steps`` / ``--charmm-abnr-steps``.

## Numbered minimize snapshots (staged workflow)

When ``--save`` is on (default), each geometry checkpoint is written under
``artifacts/pycharmm_mlpot/<run>/`` with a **sequence prefix** and a **kind** label.
A manifest ``minimize_snapshots_<tag>.json`` lists every snapshot.

| Seq | Filename stem | Kind | Potential |
|-----|---------------|------|-----------|
| 00 | ``00_packmol_cluster_<tag>`` | packmol | Packmol placement (PSF+PDB) |
| 01 | ``01_charmm_mm_<tag>`` | MM | CGENFF SD/ABNR before MLpot |
| 02 | ``02_mlpot_mmml_<tag>`` | MMML | MLpot PhysNet SD (USER term) |
| 03 | ``03_bonded_mm_after_mini_<tag>`` | bonded_MM | Bonded-only recovery after mini (if run) |
| 04 | ``04_bonded_mm_after_heat_<tag>`` | bonded_MM | Bonded-only recovery after heat (if run) |
| 10+ | ``10_rescue_<context>_<tag>`` | intra_rescue | Overlap / intra close-contact rescue |

Legacy names ``mini_full_mlpot_<tag>.*`` are still copied from snapshot **02** for older scripts.

## Minimization (example pattern)

MLpot on the **whole** system; two SD passes — free, then `cons_fix` on selected monomers:

```python
from pathlib import Path
from mmml.interfaces.pycharmmInterface.mlpot import (
    MinimizeWithMlpotConfig,
    minimize_with_mlpot,
    select_by_resid,
)

# Call after register_mlpot(..., select_all_atoms(), ...)
minimize_with_mlpot(
    MinimizeWithMlpotConfig(
        fixed_ml_selection=select_by_resid(1),
        nstep=500,
        save=True,
        pdb_path=Path("charmm_data/mini.pdb"),
        crd_path=Path("charmm_data/mini.crd"),
        psf_path=Path("charmm_data/mini.psf"),
        energy_json_path=Path("charmm_data/mini_energy.json"),
        xyz_path=Path("charmm_data/mini.xyz"),
        dcd_path=Path("charmm_data/mini.dcd"),
        dcd_nsavc=1,
    )
)
```

Set ``save=True`` to write coordinates, PSF, energy JSON, XYZ, and a **DCD** of the SD trajectory
(``iuncrd`` / ``nsavc`` on ``minimize.run_sd``). Pass 1 is free; pass 2 uses ``cons_fix`` when
``fixed_ml_selection`` is set. Both passes append to the same DCD.

Prefer **CRD** over PDB when reloading minimized structures (avoids ML nonbond exclusion issues with PDB).

## MD stages

```python
from pathlib import Path
from mmml.interfaces.pycharmmInterface.mlpot import (
    CharmmTrajectoryFiles,
    build_heat_dynamics,
    build_nve_dynamics,
    build_cpt_equilibration_dynamics,
    run_dynamics_with_io,
)

data = Path("charmm_data")

# Heating (NVT) — stub: set ``if False`` in production until tested
heat_io = CharmmTrajectoryFiles(
    restart_write=data / "heat.res",
    trajectory=data / "heat.dcd",
)
run_dynamics_with_io(build_heat_dynamics(duration_ps=10.0), heat_io)

# NVE
nve_io = CharmmTrajectoryFiles(
    restart_read=data / "heat.res",
    restart_write=data / "nve.res",
    trajectory=data / "nve.dcd",
)
run_dynamics_with_io(build_nve_dynamics(duration_ps=50.0), nve_io)

# NPT equilibration
equi_io = CharmmTrajectoryFiles(
    restart_read=data / "nve.res",
    restart_write=data / "equi.res",
    trajectory=data / "equi.dcd",
)
run_dynamics_with_io(build_cpt_equilibration_dynamics(duration_ps=50.0), equi_io)
```

Chained production restarts: `production_restart_chain(data_dir, n_segments=10)`.

## Non-PBC flat-bottom sphere (MMFP)

Same as production `dyna.inp` — quartic wall outside ``droff``:

```python
from mmml.interfaces.pycharmmInterface.mlpot import apply_flat_bottom_workflow

apply_flat_bottom_workflow(radius=20.0, force=1.0, center_at_origin=True)
```

Test scripts: ``--fb-rad 20 --fb-forc 1`` (optional ``--no-fb-center`` if the sphere center is not at the origin).

## Energy drift guard (ECHECK)

``build_nve_dynamics`` defaults to ``echeck=100`` kcal/mol (``dyna.inp`` heating/production).
Dynamics stops early if the total energy jump exceeds the tolerance — use before MMFP/PRESS RMS blows up.

Script 05: ``--echeck 100`` (default; auto-loosened for large clusters — DCM:90 → 4500 kcal/mol),
``--no-scale-echeck`` to keep the explicit value, ``--no-echeck`` to disable.

## Partial ML / MM

See `partial_mm.py` — segment registration works; **ML–MM pair electrostatics** (`idxu`/`idxv`) raise `NotImplementedError` until implemented in `PyCharmm_Calculator`.

## CLI (`mmml md-system --backend pycharmm`)

Vacuum MLpot workflows are wired into the main MD CLI (same logic as scripts 04–05):

```bash
mmml md-system --setup free_nve --backend pycharmm --residue ACO --n-molecules 4 \
  --flat-bottom-radius 20 --ps 0.5 --fix-resids 1,3

mmml md-system --setup pycharmm_minimize --composition ACO:2 --mini-nstep 30

python -m mmml.cli.run.md_pbc_suite.pycharmm_mlpot --phase dynamics --ensemble nve --help
```

Implementation: `mmml/cli/run/md_pbc_suite/pycharmm_mlpot.py`, `mlpot/run_workflow.py`, shared flags in `mlpot/cli_common.py`.

## Periodic boundaries (PBC) with MIC

For ``--setup pbc_*``, the PyCHARMM backend:

1. Installs CHARMM crystal + IMAGE (``pbc_env.py``) for the cubic box.
2. Passes the same box side into ``setup_calculator(cell=L)`` for the decomposed monomer/dimer MLpot path (**MIC-only**, matching the ASE/JAX-MD hybrid calculator — no coordinate wrapping during energy evaluation).

**Loose PBC (``--setup free_*`` + ``--box-size``):** CHARMM gets crystal + CPT (e.g. Hoover heat with ``pmass=0``), but ML stays **open-boundary** (no MIC, free-space dimer lists) unless ``--mlpot-pbc`` is set. Use a box large enough that the cluster does not interact with its periodic images.

Log lines to expect:

```text
# Full PBC (pbc_* setups)
PBC cubic box: 20.000 Å
MLpot MIC PBC: cubic L=20.000 Å

# Loose PBC (free_* + --box-size)
CHARMM loose PBC: cubic L=55.000 Å (ML open boundary; no MIC)
MLpot: open boundary (CHARMM box L=55.000 Å; no MIC)
```

**Restart / staged continuation:** pass the same ``--box-size`` as the original run when using ``--skip-cluster-build`` or loading mini artifacts. Auto box from geometry can differ from an earlier explicit box.

**NpT (CPT ``equi`` / ``prod``):** before each CPT stage and on every MLpot callback, the decomposed calculator reads the current CHARMM cubic box (``pbound_get_size``) and passes it into the JAX MIC path so ML dimer switching tracks box resizing during CPT dynamics. Stage handoffs only update the ML ``_cell`` side — they do **not** re-run ``prepare_charmm_pbc`` / ``update_bnbnd`` (unsafe with MLpot registered; CPT restarts restore CHARMM PBC).

```text
MLpot MIC PBC synced to CHARMM L=39.821 Å (was 40.000 Å)
```

Single-monomer ``PyCharmm_Calculator`` (``n_monomers=1``) does not yet use MIC; multi-monomer Packmol clusters use the decomposed path above.

```bash
mmml md-system --setup pbc_nvt --backend pycharmm \
  --composition DCM:20 --box-size 20 --packmol-radius 5 \
  --mini-nstep 100 --output-dir artifacts/pycharmm_mlpot/dcm20_pbc
```

**Cross-backend campaigns** (PyCHARMM equil → JAX-MD prod): see [docs/handoff.md](../../../../docs/handoff.md) for box/velocity carry-over, cutoff consistency, and `handoff_quality_gate`.

## Single-point evaluation (`--evaluate-npz`)

Run one MMML energy/force evaluation at a fixed geometry without dynamics — useful for smoke-testing each backend in its runtime environment:

```bash
mmml md-system \
  --evaluate-npz path/to/geometry.npz \
  --composition DCM:9 \
  --checkpoint examples/ckpts_json/DESdimers_params.json \
  --backend ase \
  --setup free_nve \
  --output-dir artifacts/eval_smoke
```

NPZ fields:

| Key | Required | Notes |
|-----|----------|-------|
| `positions` | yes | `(N, 3)` Å |
| `atomic_numbers` | no | `(N,)`; omit if `--composition` rebuilds topology |
| `pbc`, `cell` | no | periodic box for PBC setups |
| `charges` | no | `(N,)` partial charges applied to CHARMM PSF before eval |
| `at_codes` / `iac` | no | `(N,)` LJ type indices (0- or 1-based) for switched MM |
| `epsilon`, `sigma` | no | recorded in output metadata; MM uses CGenFF via `at_codes` |

Backends: `ase` (ASE calculator), `jaxmd` (ASE + JIT spherical kernel), `pycharmm` (CHARMM MLpot callback). Results: `<output-dir>/evaluate.json`.

## Cutoff optimization

Three entry points share `mmml.interfaces.pycharmmInterface.hybrid_reference` and canonical cutoff names (`ml_switch_width`, `mm_switch_on`, `mm_switch_width`; legacy aliases `ml_cutoff`, `mm_cutoff`):

| Tool | NPZ format | Topology |
|------|------------|----------|
| `mmml md-system --evaluate-npz` | single frame: `positions` | `--composition` or NPZ `atomic_numbers` |
| `mmml md-system --optimize-cutoffs --reference-npz` | trajectory: `R`, optional `E`, `F` | `--composition` required |
| `python -m mmml.cli.misc.opt_mmml` | trajectory: `R`, optional `E`, `F` | PDB + `--n-atoms-monomer` |

Grid search example (ASE hybrid calculator, no dynamics):

```bash
mmml md-system \
  --optimize-cutoffs \
  --reference-npz path/to/qm_traj.npz \
  --composition DCM:2 \
  --checkpoint examples/ckpts_json/DESdimers_params.json \
  --ml-switch-width-grid 1.5,2.0,2.5 \
  --mm-switch-on-grid 6.0,7.0 \
  --mm-switch-width-grid 0.5,1.0 \
  --max-frames 50 \
  --output-dir artifacts/cutoff_fit
```

Results: `<output-dir>/optimize_cutoffs.json` with `best` and per-grid `results` (canonical keys plus `ml_cutoff`/`mm_cutoff` aliases).


## Tests

```bash
./tests/functionality/mlpot/run_all.sh          # 00–03 validated
python tests/functionality/mlpot/04_mlpot_minimize_stub.py --run
python tests/functionality/mlpot/05_mlpot_dynamics_stub.py --run
pytest tests/functionality/mlpot/test_mlpot_energy_matches_ase.py -q
pytest tests/unit/test_mlpot_pbc_cell.py -q
```

## Profiling

### Built-in MLpot Profiling
To enable lightweight timing metrics showing callback counts, JAX/ML evaluation time, and CHARMM overhead:

```bash
mmml md-system ... --mlpot-profile
# or on the PyCHARMM MLpot driver:
python -m mmml.cli.run.md_pbc_suite.pycharmm_mlpot --phase dynamics --mlpot-profile ...
```

Environment variables (same effect when set before Python starts):

```bash
export MMML_MLPOT_PROFILE=1
export MMML_JAX_COMPILE_TIMERS=1
```

### Python cProfile on MD System CLI
You can profile the python-level code (coordinate synchronizations, exclusions setup, CLI arguments parsing) when running standard configs:
```bash
./.venv/bin/python -m cProfile -o md_system.prof -m mmml.cli md-system <arguments>
```
Analyze the results using `snakeviz`:
```bash
./.venv/bin/pip install snakeviz
./.venv/bin/snakeviz md_system.prof
```

### Deep JAX Device Profiling
If you need to profile GPU kernel dispatch overhead, host-device transfers, or JIT compilation bottlenecks inside the JAX network itself:

**JAX Profiler API:** Wrap the region of interest in your script (e.g. inside a test, script, or workflow driver):
```python
import jax

# Start the trace server (writes to the specified directory)
jax.profiler.start_trace("/tmp/tensorboard_logs")

# Run steps to profile (e.g. step dynamics or evaluation)
# ...

# Stop tracing
jax.profiler.stop_trace()
```
You can then visualize the profile using TensorBoard:
```bash
pip install tensorboard tensorboard-plugin-profile
tensorboard --logdir /tmp/tensorboard_logs
```

