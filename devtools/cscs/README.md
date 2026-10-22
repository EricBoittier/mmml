# mmml on CSCS clariden (GH200, aarch64)

Container + thin venv + EDF (see `~/cscs/CSCS.md` §3).

- `Dockerfile` — `nvidia/cuda:13.0.2-base-ubuntu24.04` + uv Python 3.13 + `requirements-frozen.txt`
  (CUDA-13 JAX 0.10.2, torch 2.14.1, cupy 13, metatrain, pyscf CPU) + gfortran/cmake/OpenMPI/FFTW.
- `requirements-frozen.txt` — regenerate on the login node:
  `uv pip compile ../../pyproject.toml --extra gpu --extra nl-gpu --extra metatomic --extra charmm-interface --extra jax-pme-solver --extra md --extra quantum --extra ml --extra dev --extra experiments --extra plotting --extra cli --extra nvalchemiops-pme -c constraints.txt --python-version 3.13 --python-platform aarch64-manylinux_2_39 --no-header --no-annotate -o requirements-frozen.txt`
  `constraints.txt` pins `torch>=2.14`: without it uv picks torch 2.10, whose aarch64 PyPI wheel is CPU-only.
  JAX pinned to 0.10.2 (0.11.0 NaN-poisons jitted second-order AD, CSCS.md §3.1).
  Not included: `quantum-gpu` (gpu4pyscf has no aarch64 wheel).
- `build.sbatch` — builds the image, then `setup-venv.sh` inside it: editable mmml in `~/venvs/mmml`,
  `libcharmm.so` + packmol into this checkout. Install the EDF first: `cp mmml.toml ~/.edf/`.
- After a `git pull` touching CHARMM: `srun -A ab010 -p debug --environment=mmml bash devtools/cscs/setup-venv.sh`.

Use: `srun -A ab010 --environment=mmml bash -c 'source ~/venvs/mmml/bin/activate && mmml ...'`.
libcharmm.so links the image's OpenMPI/FFTW, so it only works inside this container.
