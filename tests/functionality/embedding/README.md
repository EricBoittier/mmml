# Solvated peptide MD embedding (`mmml md-embedding`)

CHARMM-node smoke steps for partial MLpot on a peptide in explicit TIP3 solvent.
Unit tests live in `tests/unit/test_md_embedding_cli.py` (no CHARMM).

Design: [`docs/examples/md-embedding-design.md`](../../../docs/examples/md-embedding-design.md).

## Prerequisites

- PyCHARMM + CHARMM (same as `md-system` / `liquid-box`)
- JAX + GPU optional for training; CPU smoke is fine for 30-epoch aaa model
- `uv sync` in the `mmml` repo root

## 1. Train (no CHARMM)

```bash
cd /path/to/mmml
uv run mmml md-embedding train -o artifacts/md_embedding/aaa
```

By default the train phase uses **`mmml fix-and-split`** (`--preserve-units`) for a reproducible
90/10 split and `units_manifest.json` under `splits/`. Use `--simple-split` to skip fix-and-split.
Structure plots (ASE bonds, docs style) land in `figures/peptide_frame0.png` unless `--no-plot`.

Pass criteria:

- `artifacts/md_embedding/aaa/train_manifest.json` exists
- `train.npz` / `valid.npz` split from aaa.ama NPZ
- After full train (omit `--skip-train`): `aaa_smoke_params.json` or checkpoint path in manifest
- `uv run pytest tests/unit/test_md_embedding_cli.py tests/unit/test_aaa_ama_dataset.py -q`

Fast split-only check:

```bash
uv run mmml md-embedding train -o artifacts/md_embedding/aaa --skip-train
```

## 2. Build box (CHARMM)

```bash
uv run mmml md-embedding build -o artifacts/md_embedding/aaa --n-waters 10 --box-side-A 28
```

Pass criteria:

- `model.psf`, `model.crd`, `box.json` under output dir
- `figures/embedding_box.png` and `figures/embedding_peptide.png` (unless `--no-plot`)
- `box.json` records `ml_seg_id: PEPT`, `training_n_atoms: 34`, `n_peptide_atoms: 42`
- Optional: `bonded_report.json` with finite bonded terms

## 3. Run partial MLpot (CHARMM + checkpoint)

```bash
uv run mmml md-embedding run -o artifacts/md_embedding/aaa \
  --checkpoint artifacts/md_embedding/aaa/aaa_smoke_params.json \
  --mini-nstep 20
```

Pass criteria:

- CHARMM prints finite `ENER` after MLpot registration
- `run_manifest.json` with `charmm_total_energy_kcalmol` set
- GRMS decreases vs start when `--mini-nstep` > 0 (qualitative)

## Topology note

Training uses **34** aaa.ama atoms; the bundled CGENFF `TRIA` build has **42** peptide atoms.
Do not expect NPZ energy/force parity until PSF atom order matches the training topology.

## Phase 2 (not in this smoke)

ML–MM electrostatic shell via CHARMM `idxu`/`idxv` is not wired yet; `mlmm_cutoff` flags are
forward-compatible only.
