# aaa.ama peptide dataset → CHARMM → PhysNet → partial ML/MM

End-to-end example using the public [MMunibas/aaa.ama](https://github.com/MMunibas/aaa.ama) repository: download the **34-atom capped tri-alanine** ML training set, inspect energies and forces, rebuild the solvated system in CHARMM, train a small PhysNet model, and register **peptide ML + water MM** (the mixed embedding pattern from upstream `dyna.sol.py`).

**Canonical CLI:** [`mmml md-embedding`](md-embedding-design.md) (`train` → `build` → `run`) automates the same pipeline; the sections below remain useful for manual steps and inspection.

Functionality tests: [`tests/functionality/aaa_ama/README.md`](../../tests/functionality/aaa_ama/README.md) and [`tests/functionality/embedding/README.md`](../../tests/functionality/embedding/README.md).

## 1. Fetch and identify the dataset

The NPZ lives at [`aaa_model/dataset_aaa.npz`](https://github.com/MMunibas/aaa.ama/tree/main/aaa_model) (~18 MB). MMML can download and summarize it:

```bash
uv run python scripts/analyze_aaa_ama_dataset.py --download
```

Bundled summary (after running the script): `mmml/data/external/aaa_ama_dataset_summary.json`.

| Field | Value |
|-------|--------|
| Frames | 12 500 |
| Atoms / frame | 34 |
| Net charge | +1 e |
| Stoichiometry | C₉H₁₈N₃O₄ |
| Molecule | ACE–ALA×3–CT3 peptide (training topology) |
| Arrays | `N`, `Z`, `R`, `E`, `F`, `Q`, `D` (eV, eV/Å, e·Å) |

**Species in the NPZ:** one chemical species (the peptide). Histograms below group **by element** (H, C, N, O) as a practical per-species view of forces. The NPZ does **not** include TIP3 waters — those are MM-only in the solvated CHARMM setup.

### Structure (frame 0)

![Peptide frame 0](../../images/examples/aaa-ama/peptide_frame0.png)

### Energy and forces

![Energy histogram](../../images/examples/aaa-ama/energy_histogram.png)

![Force magnitude histogram](../../images/examples/aaa-ama/force_histogram.png)

![Force magnitude by element](../../images/examples/aaa-ama/force_by_element.png)

![Energy trace](../../images/examples/aaa-ama/energy_trace.png)

Regenerate figures:

```bash
uv run python scripts/analyze_aaa_ama_dataset.py --download
```

Unit tests (no CHARMM, no 18 MB download required if summary is committed):

```bash
uv run pytest tests/unit/test_aaa_ama_dataset.py -q
```

## 2. Recreate the system in CHARMM

The upstream driver [`aaa_model/dyna.sol.py`](https://github.com/MMunibas/aaa.ama/blob/main/aaa_model/dyna.sol.py) builds:

1. **Peptide** — `read.psf_card("aaa.psf")`, segment `PEPT`
2. **Solvent** — TIP3 waters from `init.wat.pdb`, segment `WAT`
3. **PBC** — cubic crystal + IMAGE; SHAKE on water; heavy-atom COM restraint on solute
4. **Minimization** — CGENFF MM, then PhysNet after `pycharmm.MLpot` on `PEPT` only

Clone the repo and run with your CHARMM toppar (paths in the script assume a local `toppar.str`):

```bash
git clone https://github.com/MMunibas/aaa.ama.git
cd aaa.ama/aaa_model
# supply aaa.psf, aaa.crd, init.wat.pdb, step5.pdb per your CHARMM build
python dyna.sol.py
```

### MMML bonded reference (protein PSF)

For a **pure MM bonded** reference on ACE–ALA×3–CT3 using `top_all36_prot` (42 atoms — atom count may differ from the 34-atom training topology):

```bash
export CHARMM_HOME=... CHARMM_LIB_DIR=... LD_LIBRARY_PATH=...
./scripts/mmml-charmm-mpirun.sh python tests/functionality/aaa_ama/report_charmm_bonded.py
```

Example output shape:

```text
CHARMM bonded energies (kcal/mol) — ACE–ALA×3–CT3, seg PEPT
  bond       <...>
  angl       <...>
  dihe       <...>
  impr       <...>
  cmap       <...>
  total      <...>
```

**Important:** compare NPZ `E`/`F` labels only after you have a PSF whose atom order matches `Z` in the NPZ. The bundled [tri-alanine water box](../trialanine-water-box.md) uses **42-atom CGENFF `TRIA`** — related chemistry, different atomization.

For JAX vs CHARMM bonded cross-checks on CGENFF, see [cgenff-jax-clone.md](../cgenff-jax-clone.md).

## 3. Train a small PhysNet model

One-shot via `md-embedding` (download, split, smoke train, manifest):

```bash
mmml md-embedding train -o artifacts/md_embedding/aaa
```

Manual equivalent: the NPZ matches MMML's standard schema (`E`, `F`, `R`, `Z`, `N`). Split train/validation, then train with a **small** model (smoke / teaching run):

```bash
# Optional: hold out 10% frames
uv run python - <<'PY'
import numpy as np
from pathlib import Path
d = np.load("mmml/data/external/dataset_aaa.npz", allow_pickle=True)
n = len(d["E"])
idx = np.arange(n)
rng = np.random.default_rng(0)
rng.shuffle(idx)
split = int(0.9 * n)
train = {k: d[k][idx[:split]] for k in d.files}
valid = {k: d[k][idx[split:]] for k in d.files}
out = Path("artifacts/aaa_ama")
out.mkdir(parents=True, exist_ok=True)
np.savez(out / "train.npz", **train)
np.savez(out / "valid.npz", **valid)
print("wrote", out / "train.npz", out / "valid.npz")
PY

mmml physnet-train \
  --data artifacts/aaa_ama/train.npz \
  --valid-data artifacts/aaa_ama/valid.npz \
  --ckpt-dir artifacts/aaa_ama/checkpoints \
  --tag aaa_smoke \
  --num-atoms 34 \
  --max-atomic-number 8 \
  --features 32 \
  --num-epochs 30 \
  --batch-size 16 \
  --energy-weight 1.0 \
  --forces-weight 50.0 \
  --cutoff 5.0 \
  --num-iterations 2 \
  --n-res 1
```

After training, export for MLpot (Orbax → JSON if needed):

```bash
mmml orbax-to-json --checkpoint artifacts/aaa_ama/checkpoints/aaa_smoke --output artifacts/aaa_ama/aaa_params.json
```

See [`physnet-train`](../cli/commands/physnet-train.md) for full flags. Upstream `dyna.sol.py` uses TensorFlow 1.x checkpoints under `./model/aaa` with `run_aaa.inp` — MMML uses **PhysNetJAX**; retrain or convert checkpoints before mixing formats.

## 4. Mixed MM / ML-MM calculator (peptide + water)

Upstream registers PhysNet **only on `PEPT`**; waters keep CHARMM nonbonded:

```python
selection = pycharmm.SelectAtoms(seg_id='PEPT')
pycharmm.MLpot(ml_selection=selection, ...)
```

### MMML equivalent (partial ML / MM)

```python
from mmml.interfaces.pycharmmInterface.mlpot.partial_mm import (
    PartialMlMmConfig,
    register_mlpot_partial_mm,
)
from mmml.interfaces.pycharmmInterface.mlpot.setup import (
    load_physnet_mlpot_bundle,
    select_by_seg_id,
)

# After loading solvated PSF (PEPT + WAT) and building ASE atoms for PEPT:
z_pept = ase_pept.get_atomic_numbers()
params, model, pyCModel = load_physnet_mlpot_bundle(
    "artifacts/aaa_ama/aaa_params.json",
    len(z_pept),
    ase_pept,
    n_monomers=1,
)

ctx = register_mlpot_partial_mm(
    pyCModel,
    z_pept,
    PartialMlMmConfig(ml_seg_id="PEPT", ml_charge=1.0, ml_fq=True),
)
```

Then `energy.show()` / `minimize.run_sd` / dynamics as in [`mlpot/README.md`](../../mmml/interfaces/pycharmmInterface/mlpot/README.md).

### Current limitations

| Feature | Status |
|---------|--------|
| Peptide ML + water MM (segment selection) | API: `register_mlpot_partial_mm` |
| ML–MM pair electrostatics (`idxu`/`idxv`) | Not implemented (`NotImplementedError`) |
| Homogeneous hybrid liquid (`md-system`, all ML monomers) | Production path — see [hybrid-potential-regions.md](../hybrid-potential-regions.md) |
| Some monomers hybrid + some pure MM in one **cluster** liquid | Not supported in `md-system` yet |

For solvated peptide MD, the **partial segment** pattern above matches [aaa.ama `dyna.sol.py`](https://github.com/MMunibas/aaa.ama/blob/main/aaa_model/dyna.sol.py). For bulk liquids (DCM, ACO, …), use full hybrid `md-system` on a single species.

## 5. Suggested validation checklist

1. `analyze_aaa_ama_dataset.py` — plots + JSON summary
2. `pytest tests/unit/test_aaa_ama_dataset.py`
3. CHARMM `report_charmm_bonded.py` — bonded term table
4. `physnet-train` smoke — loss decreases on train/valid NPZ
5. Register `PEPT` MLpot on minimized solvated PSF — `ENER` finite, SD lowers GRMS
6. (Optional) Compare ML energy on one NPZ frame to label `E` after unit alignment

## References

- [aaa.ama repository](https://github.com/MMunibas/aaa.ama)
- [Hybrid potential regions](../hybrid-potential-regions.md)
- [Tri-alanine water box](../trialanine-water-box.md) (CGENFF MM cross-check)
- [Partial ML module](../../mmml/interfaces/pycharmmInterface/mlpot/partial_mm.py)
