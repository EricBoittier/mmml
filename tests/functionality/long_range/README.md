# Long-range Coulomb validation

Functional tests and examples for MMML external Coulomb solvers, comparing:

| Backend | Role |
|---------|------|
| **MIC** (all-pairs or truncated @ 13 Å) | Default JAX MM real-space path |
| **jax-pme** (Ewald / PME / P3M) | Pure-JAX reference (jax-pme test patterns) |
| **ScaFaCoS** (`libfcs`) | Optional HPC backend (ewald, p3m, …) |

## Quick start

```bash
# Environment probe
python tests/functionality/long_range/00_check_lr_env.py

# Full ladder (MIC → jax-pme → ScaFaCoS if installed)
bash tests/functionality/long_range/run_all.sh

# Pytest only
JAX_PLATFORMS=cpu pytest tests/functionality/long_range/test_coulomb_backends.py -v
```

## ScaFaCoS

Build and point MMML at ``libfcs.so``.  Shared builds need plugin libraries on
``LD_LIBRARY_PATH``; HPC module installs usually handle this.  Run validation
under MPI when possible:

```bash
export SCAFACOS_LIB=$HOME/.local/scafacos/lib/libfcs.so
export LD_LIBRARY_PATH=$HOME/.local/scafacos/lib:$LD_LIBRARY_PATH
mpiexec -n 1 python tests/functionality/long_range/04_scafacos_methods.py
```

Minimal configure (no GSL / pnfft):

```bash
../configure --prefix=$HOME/.local/scafacos --enable-shared \
  --disable-fcs-p2nfft --disable-fcs-memd --disable-fcs-wolf
```

ScaFaCoS tests **auto-enable** when ``~/.local/scafacos/lib/libfcs.so`` passes a
smoke run (``scafacos_runtime_ok``); set ``MMML_SCAFACOS_TESTS=0`` to force skip.
Requires ``mpi4py`` and ``LD_LIBRARY_PATH`` including the ScaFaCoS plugin directory.

### Comparison results (local build, Python 3.13)

| System | MIC | MIC trunc | jax-pme Ewald | ScaFaCoS ewald |
|--------|-----|-----------|---------------|----------------|
| Ion dimer 5 Å / L=30 | −66.41 | −66.41 | −67.08 | −67.08 |
| CsCl L=10 | −38.34 | −38.34 | −67.59 | −67.59 |
| Random 8-mer L=12 | −320.34 | −320.34 | −358.87 | −358.88 |

Truncated MIC (13 Å) underestimates full periodic Coulomb; ScaFaCoS ewald/p3m match
jax-pme to ~0.1–0.7%. ``direct`` is validated on dimers only (not bulk crystals).

### Hybrid MM with jax-pme (LJ + electrostatics)

Set ``MMML_LR_SOLVER=jax_pme`` (or ``lr_solver: jax_pme`` in YAML) to keep **switched
Lennard-Jones** on the JAX pair path and evaluate **Coulomb** with jax-pme
(Ewald / PME / P3M via ``JAX_PME_METHOD``):

```bash
export MMML_LR_SOLVER=jax_pme
export JAX_PME_METHOD=ewald   # or pme, p3m
pytest tests/functionality/long_range/test_hybrid_jax_pme_mm.py -v
python tests/functionality/long_range/06_hybrid_jax_pme_mm.py  # PyCHARMM ACO:2 cluster
```

## Test systems

Defined in `_common.py` (mirroring jax-pme `tests/test_ewald.py`):

- **Ion dimer** — analytic 1/r check for MIC in a large box
- **CsCl / NaCl cubic** — Madelung constant vs literature
- **Random neutral cluster** — truncated MIC vs full Ewald

## See also

- `examples/long_range/compare_coulomb_backends.py` — standalone comparison script
- `mmml/interfaces/pycharmmInterface/mlpot/LONG_RANGE_ELECTROSTATICS.md` — MLpot integration
- `mmml/interfaces/scafacosInterface/README.md` — ScaFaCoS install
