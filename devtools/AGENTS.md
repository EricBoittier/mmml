# AGENTS.md

## Cursor Cloud specific instructions

This repo (`mmml`) is a Python (>=3.13) ML + molecular-mechanics toolkit managed with `uv`,
plus a FastAPI + React molecular-viewer GUI under `mmml/gui/`. Standard commands live in the
`README.md`, `Makefile`, and `docs/development.md`; the notes below only capture non-obvious
caveats for working in the cloud VM.

### Python environment
- The startup update script runs `uv sync --extra dev`, which creates `.venv` with Python 3.13 or 3.14 free-threaded
  (the system `python3` is 3.12 and is the wrong version). Always run project code via
  `uv run ...` (e.g. `uv run python`, `uv run pytest`, `uv run mmml ...`).
- `uv` is installed at `~/.local/bin/uv` and symlinked into `/usr/local/bin/uv`, so it is on PATH
  for both login and non-login shells.
- `uv.lock` is gitignored (see also `.cursor/rules/`); ignore local lock churn after `uv sync`.

### Tests / lint / docs
- Tests: `uv run pytest tests/` passes (currently 1500+ passed / ~60 skipped). Tests needing a live
  PyCHARMM build or a GPU auto-skip when `pycharmm` is not importable — no extra flags needed.
  To deselect them explicitly: `-m "not pycharmm and not gpu and not mlpot"`.
- **Before finishing a change, run the tests you touched** — CI installs the latest dependency
  pins from `pyproject.toml` (e.g. ASE ≥ 3.26), which may differ from a stale local venv:
  ```bash
  uv run pytest tests/unit/test_calculator_minimize.py -q   # example: touched minimize code
  uv run pytest tests/unit/ -q                                # fast pre-push sweep
  ```
  When behavior is version-gated (third-party API changes), update the matching unit test in the
  **same commit** so CI does not regress (see `ase_optimizer_dual_unit_logfile` / ASE 3.26).
- Per `.cursor/rules/`, do NOT run CHARMM / PyCHARMM dynamics, Packmol box builds, or full
  `mmml md-system` MD in the agent — those libs/GPU are unavailable. Prefer the unit tests above.
- Lint: `make lint` (`uv run ruff check mmml/ scripts/`) currently reports pre-existing
  F401/F841 errors (lint debt in the repo); a clean exit is not expected from an unmodified tree.
- Docs: `uv run mkdocs build --strict` succeeds. The red `mkdocs-material 2.0` banner it prints is
  an upstream informational notice, not a build failure.
- No GPU is present in the cloud VM; JAX runs on CPU. ML checkpoints (`mmml/physnetjax/ckpts`)
  are not bundled, so README energy-calculator examples that load a checkpoint will not run as-is.

### GUI (molecular viewer)
- `mmml/gui/viewer/node_modules` is gitignored (not committed); the update script runs
  `npm install --prefix mmml/gui/viewer` to create it (idempotent). `dist/` is gitignored and must
  be built before production serving.
- Production serve (single port): `cd mmml/gui/viewer && npm run build`, then
  `uv run mmml gui --data-dir <dir> --no-browser` → http://127.0.0.1:8000.
- Dev mode (hot reload): `uv run mmml gui --data-dir <dir> --dev --no-browser` (API on :8000) plus
  `cd mmml/gui/viewer && npm run dev` (Vite on :5173, proxies `/api` to :8000).
- Viewing `.pdb` / `.npz` / `.traj` files needs no GPU or checkpoints. Sample files live in
  `mmml/generate/sample/pdb/`.

### Repository size & Git LFS
- Do NOT commit large/regenerable binaries. `.gitignore` already covers `node_modules/`, `*.dcd`,
  `*.traj`, `*.pov-state`, `*.h5`, `*.xyz`, `*.png`, checkpoints, etc.; add new large artifacts there
  (or host them externally) rather than committing them.
- Git history is heavy (~3.8 GB pack) from binaries committed in the past (training checkpoints,
  notebooks with outputs, `.pov-state`, `.dcd/.traj`, EF param JSONs, a vendored `hdf5-*` tree).
  Deleting files now does not shrink a clone — history must be rewritten. Use
  `scripts/slim_repo_history.sh` (a maintainer-run, backup-first `git filter-repo` flow; it never
  force-pushes for you). A conservative purge tested 3.8 GB → 3.2 GB; an aggressive one → ~0.95 GB.
- Fast checkout without rewriting history: `git clone --filter=blob:none <url>` (partial) or
  `git clone --depth 1 <url>` (shallow).
- Reclaiming Git LFS quota requires deleting the historical LFS objects (GitHub does not GC them
  automatically) — rewrite history, then recreate the repo or use GitHub's LFS admin tooling.
