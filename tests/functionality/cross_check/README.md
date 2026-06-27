# Cross-check functionality smoke tests

Manual verification for `mmml cross-check` on a GPU/QC node. Do not run in agent sessions.

## Prerequisites

```bash
cd ~/mmml && source .venv/bin/activate
# Optional extras:
# uv sync --extra quantum-gpu      # pyscf reference
# uv sync --extra quantum-crosscheck  # tblite xTB
# ORCA / Molpro on PATH
```

## Smoke: reference NPZ + xTB backend

```bash
mmml cross-check \
  -i tests/fixtures/cross_check/water_frames.npz \
  --reference-npz tests/fixtures/cross_check/water_frames.npz \
  --backend xtb \
  --max-frames 1 \
  -o /tmp/cross_check_xtb_smoke
```

Pass: `cross_check_summary.json` exists; xTB energy MAE is finite.

## Smoke: ORCA QM backend (requires ORCA)

```bash
mmml cross-check \
  -i tests/fixtures/cross_check/water_frames.npz \
  --reference-npz tests/fixtures/cross_check/water_frames.npz \
  --backend orca --functional PBE --basis def2-SVP \
  --max-frames 1 \
  -o /tmp/cross_check_orca_smoke
```

## Unit tests (CI-safe)

```bash
pytest tests/unit/test_cross_check.py tests/unit/test_orca_qm.py
```
