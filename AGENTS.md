# AGENTS.md

## Cursor Cloud specific instructions

This repo (`mmml`) is a Python (>=3.13) ML + molecular-mechanics toolkit managed with `uv`,
plus a FastAPI + React molecular-viewer GUI under `mmml/gui/`. Standard commands live in the
`README.md`, `Makefile`, and `docs/development.md`; the notes below only capture non-obvious
caveats for working in the cloud VM.

### Python environment
- The startup update script runs `uv sync --extra dev`, which creates `.venv` with Python 3.13
  (the system `python3` is 3.12 and is the wrong version). Always run project code via
  `uv run ...` (e.g. `uv run python`, `uv run pytest`, `uv run mmml ...`).
- `uv` is installed at `~/.local/bin/uv` and symlinked into `/usr/local/bin/uv`, so it is on PATH
  for both login and non-login shells.
- `uv.lock` is gitignored (see also `.cursor/rules/`); ignore local lock churn after `uv sync`.

### Tests / lint / docs
- Tests: `uv run pytest tests/` passes (currently 533 passed / 39 skipped). Tests needing a live
  PyCHARMM build or a GPU auto-skip when `pycharmm` is not importable — no extra flags needed.
  To deselect them explicitly: `-m "not pycharmm and not gpu and not mlpot"`.
- Per `.cursor/rules/`, do NOT run CHARMM / PyCHARMM dynamics, Packmol box builds, or full
  `mmml md-system` MD in the agent — those libs/GPU are unavailable. Prefer the unit tests above.
- Lint: `make lint` (`uv run ruff check mmml/ scripts/`) currently reports pre-existing
  F401/F841 errors (lint debt in the repo); a clean exit is not expected from an unmodified tree.
- Docs: `uv run mkdocs build --strict` succeeds. The red `mkdocs-material 2.0` banner it prints is
  an upstream informational notice, not a build failure.
- No GPU is present in the cloud VM; JAX runs on CPU. ML checkpoints (`mmml/physnetjax/ckpts`)
  are not bundled, so README energy-calculator examples that load a checkpoint will not run as-is.

### GUI (molecular viewer)
- `mmml/gui/viewer/node_modules` is tracked in git but ships incomplete; a fresh checkout fails
  `vite build` until deps are restored. The update script runs `npm install --prefix mmml/gui/viewer`
  to repair it (idempotent). `dist/` is gitignored and must be built before production serving.
- Production serve (single port): `cd mmml/gui/viewer && npm run build`, then
  `uv run mmml gui --data-dir <dir> --no-browser` → http://127.0.0.1:8000.
- Dev mode (hot reload): `uv run mmml gui --data-dir <dir> --dev --no-browser` (API on :8000) plus
  `cd mmml/gui/viewer && npm run dev` (Vite on :5173, proxies `/api` to :8000).
- Viewing `.pdb` / `.npz` / `.traj` files needs no GPU or checkpoints. Sample files live in
  `mmml/generate/sample/pdb/`.

### CHARMM / pyCHARMM (live build)
- The CHARMM source ships as `setup/charmm.tar.xz`; the extracted tree, the build output, and the
  generated `CHARMMSETUP` file are all gitignored, so they do NOT survive a fresh checkout and are
  intentionally NOT part of the update script (a CHARMM compile is too heavy/brittle for startup).
- To build a CPU-only (serial) `libcharmm.so` for the pyCHARMM tests, system packages are needed:
  `gcc-14 g++-14 gfortran-14 libstdc++-14-dev libfftw3-dev` (the prebuilt `.so` in the tarball was
  linked against OpenMPI + a custom gcc-14 and will not load here). Then:
  ```bash
  cd setup/charmm && bash ../install.sh   # extracts tarball + writes CHARMMSETUP (CHARMM_HOME/LIB_DIR)
  mkdir -p build_agent && cd build_agent
  CC=gcc-14 CXX=g++-14 FC=gfortran-14 ../configure --as-library --without-mpi --without-openmm
  make -j"$(nproc)"                       # ~6 min; produces build_agent/libcharmm.so
  cd .. && ln -sf build_agent/libcharmm.so libcharmm.so   # pyCHARMM loads $CHARMM_LIB_DIR/libcharmm.so
  ```
- `import pycharmm` loads `libcharmm.so` at import time, so it requires `CHARMM_LIB_DIR` (and
  `CHARMM_HOME`) pointing at `setup/charmm`. `mmml.interfaces.pycharmmInterface.import_pycharmm`
  additionally reads the repo-root `CHARMMSETUP` file for those two paths. `import mmml` does NOT
  import pycharmm, so the rest of the package works without a CHARMM build.
- Run the CHARMM-related tests with the env set:
  `CHARMM_HOME=$PWD/setup/charmm CHARMM_LIB_DIR=$PWD/setup/charmm uv run pytest -m "pycharmm and not gpu"`.
  The `pycharmm and gpu` tests load ML checkpoints and force JAX GPU compilation; they fail here with
  `ptxas not found` (no CUDA/GPU in the VM), not because of CHARMM.

### Packaging (uv tool / pip, with pyCHARMM)
- The wheel bundles both the `mmml` and `pycharmm` top-level packages plus the `mmml` console scripts
  (`mmml`, `mmml-pycharmm-two-residue-sample`, `mmml-spectra-md`); no pyproject changes are needed.
- Install as a uv tool: `uv tool install .` (exposes the `mmml` CLI on PATH in an isolated venv).
- Install as a pip package: `pip install .` (or the built wheel). `uv build` writes the sdist/wheel.
- In both installs `import pycharmm` works only when `CHARMM_LIB_DIR`/`CHARMM_HOME` are exported and a
  built `libcharmm.so` exists — the shared library is environment state, not bundled in the wheel.
