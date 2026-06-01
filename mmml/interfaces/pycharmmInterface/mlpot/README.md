# MLpot workflows (`mmml.interfaces.pycharmmInterface.mlpot`)

Helpers for **PhysNet + CHARMM MLpot**, based on the validated scripts in `tests/functionality/mlpot/`.

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

## Minimization (example pattern)

Matches ammonia / segment-fixed ML minimization scripts:

```python
from pathlib import Path
from mmml.interfaces.pycharmmInterface.mlpot import (
    MinimizeWithMlpotConfig,
    minimize_with_mlpot,
    select_by_seg_id,
)

data = Path("charmm_data")
minimize_with_mlpot(
    MinimizeWithMlpotConfig(
        fixed_ml_selection=select_by_seg_id("AMM1"),
        nstep=500,
        pdb_path=data / "mini_ammonia.pdb",
        crd_path=data / "mini_ammonia.crd",
        skip_if_crd_exists=True,
    )
)
```

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

## Partial ML / MM

See `partial_mm.py` — segment registration works; **ML–MM pair electrostatics** (`idxu`/`idxv`) raise `NotImplementedError` until implemented in `PyCharmm_Calculator`.

## Tests

```bash
./tests/functionality/mlpot/run_all.sh          # 00–03 validated
python tests/functionality/mlpot/04_mlpot_minimize_stub.py --run
python tests/functionality/mlpot/05_mlpot_dynamics_stub.py --run
pytest tests/functionality/mlpot/test_mlpot_energy_matches_ase.py -q
```
