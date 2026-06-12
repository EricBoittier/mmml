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
