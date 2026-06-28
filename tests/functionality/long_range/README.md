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

Build and point MMML at `libfcs.so`:

```bash
git clone --recursive https://github.com/scafacos/scafacos.git
cd scafacos && ./bootstrap && mkdir build && cd build
../configure --prefix=$HOME/.local/scafacos --enable-shared --disable-fcs-p2nfft
make -j && make install

export SCAFACOS_LIB=$HOME/.local/scafacos/lib/libfcs.so
export LD_LIBRARY_PATH=$HOME/.local/scafacos/lib:$LD_LIBRARY_PATH
export SCAFACOS_METHOD=p3m   # or ewald, p3m, …
```

Then re-run `04_scafacos_methods.py` and the ScaFaCoS pytest markers.

## Test systems

Defined in `_common.py` (mirroring jax-pme `tests/test_ewald.py`):

- **Ion dimer** — analytic 1/r check for MIC in a large box
- **CsCl / NaCl cubic** — Madelung constant vs literature
- **Random neutral cluster** — truncated MIC vs full Ewald

## See also

- `examples/long_range/compare_coulomb_backends.py` — standalone comparison script
- `mmml/interfaces/pycharmmInterface/mlpot/LONG_RANGE_ELECTROSTATICS.md` — MLpot integration
- `mmml/interfaces/scafacosInterface/README.md` — ScaFaCoS install
