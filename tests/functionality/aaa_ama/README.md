# aaa.ama — CHARMM bonded reference & partial ML/MM

Companion to [aaa-ama workflow](../../../docs/examples/aaa-ama-workflow.md).

## Prerequisites

- PyCHARMM / `CHARMM_HOME`
- Optional: clone [MMunibas/aaa.ama](https://github.com/MMunibas/aaa.ama) for `dyna.sol.py` and local `aaa.psf` / water PDBs

## 1. Analyze NPZ (no CHARMM)

```bash
uv run python scripts/analyze_aaa_ama_dataset.py --download
uv run pytest tests/unit/test_aaa_ama_dataset.py -q
```

## 2. Protein bonded energies (CHARMM reference)

Reports CGENFF/protein bonded term energies for ACE–ALA×3–CT3 (42 atoms in
`top_all36_prot`).  **Note:** `dataset_aaa.npz` uses **34 atoms** — the training
topology from the upstream ML workflow may differ from this protein PSF; align
PSF atom order with `Z` before comparing energies to NPZ labels.

```bash
export CHARMM_HOME=... CHARMM_LIB_DIR=... LD_LIBRARY_PATH=...
./scripts/mmml-charmm-mpirun.sh python tests/functionality/aaa_ama/report_charmm_bonded.py
```

Pass: script prints bonded components (`BOND`, `ANGL`, `DIHE`, `IMPR`, `CMAP`, `total`).

## 3. Partial ML/MM (peptide ML + water MM)

Production pattern from upstream `dyna.sol.py`:

- `PEPT` segment → PhysNet / MLpot (`ml_selection=select_by_seg_id('PEPT')`)
- `WAT` / TIP3 → pure CHARMM MM

In MMML use `register_mlpot_partial_mm` (see workflow doc). ML–MM pair
electrostatics are **not** implemented yet — segment-only registration only.

## Related

| Topic | Path |
|-------|------|
| Full workflow doc | `docs/examples/aaa-ama-workflow.md` |
| Partial ML API | `mmml/interfaces/pycharmmInterface/mlpot/partial_mm.py` |
| Tri-alanine CGENFF box (42 atoms) | `docs/trialanine-water-box.md` |
