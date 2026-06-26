# ScaFaCoS interface

[ScaFaCoS](https://github.com/scafacos/scafacos) (“Scalable Fast Coulomb Solvers”) is a parallel C library for electrostatic and gravitational problems in periodic boundary conditions. MMML can use it as an **optional long-range Coulomb backend** alongside the default truncated minimum-image (MIC) path in JAX.

Project home: [www.scafacos.de](http://www.scafacos.de)

## Installation

ScaFaCoS is **not distributed on PyPI**. Build from source or install via your HPC module system.

### Build from source (typical Linux cluster)

```bash
git clone --recursive https://github.com/scafacos/scafacos.git
cd scafacos
./bootstrap
mkdir build && cd build
../configure --prefix=$HOME/.local/scafacos --enable-shared
make -j
make install
```

Enable the shared library for MMML:

```bash
export SCAFACOS_LIB=$HOME/.local/scafacos/lib/libfcs.so
export LD_LIBRARY_PATH=$HOME/.local/scafacos/lib:$LD_LIBRARY_PATH
```

ScaFaCoS is built with MPI. MMML uses `mpi4py`’s communicator handle (`MPI.COMM_WORLD` by default). Install the MMML CHARMM extra when running under MPI:

```bash
uv sync --extra charmm-interface
```

### Verify availability

```python
from mmml.interfaces.scafacosInterface import have_scafacos

print("ScaFaCoS:", have_scafacos())
```

Or from the shell:

```bash
python -c "from mmml.interfaces.long_range_backend import describe_lr_solver; print(describe_lr_solver())"
```

## Selecting the backend

| Mechanism | Example |
|-----------|---------|
| Environment | `export MMML_LR_SOLVER=scafacos` |
| Auto (default) | `scafacos` if `libfcs` loads, else `jax_pme` if importable, else `mic` |
| Explicit MIC only | `MMML_LR_SOLVER=mic` |

Additional ScaFaCoS options:

| Variable | Default | Meaning |
|----------|---------|---------|
| `SCAFACOS_LIB` | (search path) | Absolute path to `libfcs.so` |
| `SCAFACOS_ROOT` | — | Directory containing `lib/libfcs.so` |
| `SCAFACOS_METHOD` | `p2nfft` | Solver string passed to `fcs_init` (`p3m`, `p2m`, `ewald`, …) |

Method availability depends on how ScaFaCoS was configured (`./configure --help`).

## Python API

### One-shot evaluation

```python
import numpy as np
from mmml.interfaces.scafacosInterface import compute_scafacos_coulomb

n = 4
L = 30.0  # Å cubic box
pos = np.random.uniform(0, L, (n, 3))
chg = np.array([0.5, -0.5, 0.25, -0.25])

result = compute_scafacos_coulomb(pos, chg, box_length_A=L, method="p2nfft")
print(result.energy_kcalmol, result.forces_kcalmol_A.shape)
```

### Session lifecycle (tuning + multiple steps)

```python
from mmml.interfaces.scafacosInterface import ScaFaCoSSession

with ScaFaCoSSession(method="p3m") as fcs:
    fcs.configure_cubic_box(box_length_A=40.0, n_atoms=100)
    fcs.set_parameter("p3m_cutoff", "10.0")
    out = fcs.run_coulomb(positions_A, charges_e)
```

### Backend factory (MMML integration)

```python
from mmml.interfaces.pycharmmInterface.long_range_backend import (
    create_lr_solver,
    pick_lr_solver,
)

print(pick_lr_solver())          # auto-resolved name
solver = create_lr_solver("scafacos")
result = solver.compute(pos, chg, box_length_A=40.0)
```

## C API mapping

MMML’s `scafacos_session.py` wraps the public ScaFaCoS frontend (see upstream `fcs_interface_p.h`):

| ScaFaCoS C call | Python wrapper |
|-----------------|----------------|
| `fcs_init` | `ScaFaCoSSession.__init__` |
| `fcs_set_common` | `configure_cubic_box` |
| `fcs_set_parameter` | `set_parameter` |
| `fcs_run` | `run_coulomb` |
| `fcs_destroy` | `close` / context manager exit |

**Units:** positions in Å, charges in \(e\), forces returned in kcal/mol/Å (CHARMM convention, \(k = 332.063711\) kcal·Å/e²).

**Energy:** computed as \(-\frac{1}{2}\sum_i q_i \phi_i\) from ScaFaCoS potentials, scaled to kcal/mol.

## Relationship to MLpot

Hybrid ML/MM dynamics today:

1. **CHARMM** — IMAGE lists + short-range `cdie` nonbonds; ML atoms have ELEC/VDW zeroed via BLOCK (jax_mic mode).
2. **JAX MM** — switched LJ + **truncated MIC Coulomb** to ~13 Å (`mm_switch_on + mm_switch_width`).
3. **ScaFaCoS (optional)** — full periodic Coulomb for selected atom subsets; intended as a k-space supplement above the JAX cutoff.

### `periodic_external` mode (`--mm-nonbond-mode periodic_external`)

Wired into `mmml md-system` / staged PyCHARMM workflows:

- **JAX real-space LJ and Coulomb are off** (`doMM=False` in the ML JIT path).
- **Coulomb** — ScaFaCoS in the MLpot callback (`periodic_mm_external.py`).
- **Lennard-Jones** — CHARMM periodic VDW (BLOCK keeps VDW on, ELEC off). ScaFaCoS does **not** implement LJ.

Requires `--setup pbc_*`, `libfcs`, and adequate `--box-size` (validated in `periodic_mm.py`).

See [LONG_RANGE_ELECTROSTATICS.md](../pycharmmInterface/mlpot/LONG_RANGE_ELECTROSTATICS.md) for box rules and limitations.

## License note

ScaFaCoS is licensed under GPL/LGPL. MMML’s ctypes wrapper is MIT-licensed; linking against `libfcs` at runtime is your responsibility on shared clusters. Consult your site policies and the ScaFaCoS `COPYING.*` files.

## References

- Repository: https://github.com/scafacos/scafacos  
- Manual: `doc/manual.pdf` in the ScaFaCoS source tree after `make doc`  
- Doxygen: https://www.scafacos.de/doxygen/  
- Citation: see https://www.scafacos.de/publications.html
