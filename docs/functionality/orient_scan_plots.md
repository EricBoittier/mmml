# ACO orientation scans and path-atlas plots

Rigid dimer orientation scans (`S² × SO(3) × r`) diagnose spurious wells on the
hybrid PES. This page covers **running the scan**, **plotting the figure set**,
and **saving outputs under a training-run UUID**.

Related: [Dimer scans (COM distance / LR solvers)](dimer_scans/README.md).

## Scripts

| Script | Role |
|--------|------|
| [`scripts/scan_dimer_orientations.py`](https://github.com/EricBoittier/mmml/blob/main/scripts/scan_dimer_orientations.py) | Batched `hybrid_forward` ray scan → `rays.csv` + `summary.json` |
| [`scripts/plot_orient_hemisphere_annotated.py`](https://github.com/EricBoittier/mmml/blob/main/scripts/plot_orient_hemisphere_annotated.py) | Hemisphere maps, ASE overlays, path atlas, 1D slices |
| [`scripts/run_gfn2_nms_hybrid.sh`](https://github.com/EricBoittier/mmml/blob/main/scripts/run_gfn2_nms_hybrid.sh) | Train → freeze → orientation gate (cluster workflow) |

Plotting needs only CPU + ASE + matplotlib (no CHARMM / GPU). The scan itself
needs a JAX checkpoint (GPU recommended for ~240 rays).

## UUID layout

Store every figure set for a checkpoint under that run’s UUID:

```text
artifacts/orient_plots/<uuid>/
  rays.csv                         # copy of the primary scan (optional)
  scan_summary.json                # from scan summary.json
  shared_scales.json               # locked colour / axis scales
  path_atlas_<label>.png
  equirectangular_maps_<label>.png
  hemisphere_*.png
  perspectives_gallery_<label>.png
  slice_dir*_with_ase_<label>.png
  FLIP_<label>.md                  # auto index of outputs
  README.md                        # how this folder was made
```

Example (GFN2-NMS epoch-222 gate):

```text
UUID=f448e34c-cca7-43f6-8b8e-f4986b9403eb
artifacts/orient_plots/$UUID/
```

Use the same UUID string in `--label` so filenames carry the run id:

```bash
--label "gfn2nms-${UUID}"
```

Large PNGs live under `artifacts/` (gitignored). Keep the recipe in-repo via
this page; optionally mirror the folder to portable storage or the cluster.

## 1. Run the orientation scan

Match the gate in `run_gfn2_nms_hybrid.sh` (ACO example):

```bash
UUID=f448e34c-cca7-43f6-8b8e-f4986b9403eb
CKPT=/path/to/ckpts/gfn2_nms/gfn2nms-${UUID}/epoch-222   # or a frozen copy
DATA=/path/to/out_combined_dedup/..._test.npz
OUT=/path/to/orient_gfn2nms_${UUID:0:8}

uv run python scripts/scan_dimer_orientations.py \
  --checkpoint "$CKPT" \
  --data "$DATA" \
  --resid ACO \
  --n-directions 10 \
  --n-orientations 24 \
  --n-r 36 \
  --mm-switch-on 6.0 \
  --use-ema \
  --out "$OUT"
```

Outputs: `$OUT/rays.csv`, `$OUT/summary.json`.

Companion surfaces for flip comparisons (same geometry grid when possible):

- ML hybrid: e.g. `orient_6A/rays.csv`
- GFN2 / xTB reference: e.g. `orient_xtb/rays.csv`
- Validate oris: `validate_ACO/rays_ACO.csv` (spliced into atlas tags when present)

## 2. Plot the primary (full figure set)

```bash
UUID=f448e34c-cca7-43f6-8b8e-f4986b9403eb
DATA=/Volumes/PortableSSD/DATA/acodcm          # or your data root
PLOT=artifacts/orient_plots/${UUID}
mkdir -p "$PLOT"
cp "$DATA/orient_gfn2nms_f448e34c/summary.json" "$PLOT/scan_summary.json"

uv run python scripts/plot_orient_hemisphere_annotated.py \
  --rays "$DATA/orient_gfn2nms_f448e34c/rays.csv" \
  --label "gfn2nms-${UUID}" \
  --monomer "$DATA/pdb/aco.pdb" \
  --ref-rays "$DATA/orient_xtb/rays.csv" \
  --match-rays "$DATA/orient_xtb/rays.csv" \
  --match-rays "$DATA/orient_6A/rays.csv" \
  --validate "$DATA/validate_ACO/rays_ACO.csv" \
  --out "$PLOT"
```

### Important: shared scales

`--match-rays` locks colour bars and `E(r)` / `r` axes across flip partners.
Pass **companions only** (not the primary again): primary is always included
from `--rays`. Repeating the primary in `--match-rays` skews percentiles.

After regenerating companions with overlapping match sets, freeze scales once:

```bash
uv run python - <<'PY'
from pathlib import Path
import json
from dataclasses import asdict
from scripts.plot_orient_hemisphere_annotated import build_shared_scales
from scripts.plot_orient_hemisphere_surfaces import _load_rays

DATA = Path("/Volumes/PortableSSD/DATA/acodcm")
OUT = Path("artifacts/orient_plots/f448e34c-cca7-43f6-8b8e-f4986b9403eb")
shared = build_shared_scales(
    _load_rays(DATA / "orient_gfn2nms_f448e34c/rays.csv"),
    _load_rays(DATA / "orient_6A/rays.csv"),
    _load_rays(DATA / "orient_xtb/rays.csv"),
    how="min",
)
(OUT / "shared_scales.json").write_text(json.dumps(asdict(shared), indent=2) + "\n")
print(asdict(shared))
PY
```

## 3. Flip companions (path atlas only)

Same `--ref-rays` / `--validate` / shared companions so tags and colour maps line up:

```bash
uv run python scripts/plot_orient_hemisphere_annotated.py \
  --rays "$DATA/orient_6A/rays.csv" --label ML --atlas-only \
  --monomer "$DATA/pdb/aco.pdb" \
  --ref-rays "$DATA/orient_xtb/rays.csv" \
  --match-rays "$DATA/orient_xtb/rays.csv" \
  --match-rays "$DATA/orient_gfn2nms_f448e34c/rays.csv" \
  --validate "$DATA/validate_ACO/rays_ACO.csv" \
  --out "$PLOT"

uv run python scripts/plot_orient_hemisphere_annotated.py \
  --rays "$DATA/orient_xtb/rays.csv" --label GFN2 --atlas-only \
  --monomer "$DATA/pdb/aco.pdb" \
  --ref-rays "$DATA/orient_xtb/rays.csv" \
  --match-rays "$DATA/orient_6A/rays.csv" \
  --match-rays "$DATA/orient_gfn2nms_f448e34c/rays.csv" \
  --validate "$DATA/validate_ACO/rays_ACO.csv" \
  --out "$PLOT"
```

Open `path_atlas_gfn2nms-<uuid>.png`, `path_atlas_ML.png`, and
`path_atlas_GFN2.png` side by side.

## What each figure is

| File | Content |
|------|---------|
| `path_atlas_*.png` | Tags A–D from physical surface minima (prefer `--ref-rays`); complete graph among tags; 3-frame filmstrips `A \| mid \| B`; maps + `E(r)` panels |
| `equirectangular_maps_*.png` | Energy on approach / orientation projections |
| `hemisphere_annotated_dashboard_*.png` | Six hemispheres + legend + tags |
| `hemisphere_ase_ring_*.png` | ASE overlays around hemispheres |
| `perspectives_gallery_*.png` | Tagged dimers from several cameras |
| `slice_dir*_with_ase_*.png` | 1D `E(r)` slices with matching letter tags |
| `projection_explainer_*.png` | How the projections are built |
| `FLIP_*.md` | Auto-generated file list + regenerate snippet |

CLI knobs worth knowing:

- `--ref-rays` — star physical minima (typically GFN2)
- `--ref-top-n` — how many deepest ref wells to star (default 12)
- `--atlas-only` — skip non-atlas panels (companions)
- `--how min|mean` — aggregate used for shared scales (default `min`)
- `--label` — badge + filename suffix

## Example run (epoch-222 GFN2-NMS)

| Field | Value |
|-------|--------|
| UUID | `f448e34c-cca7-43f6-8b8e-f4986b9403eb` |
| Checkpoint | `…/gfn2_nms/gfn2nms-<uuid>/epoch-222` (EMA) |
| Scan | 10 dirs × 24 oris × 36 `r`, `mm_switch_on=6.0` |
| Rays | 240 evaluated; **17.5%** spurious; deepest **−4.54** kcal/mol |
| Local plots | `artifacts/orient_plots/f448e34c-cca7-43f6-8b8e-f4986b9403eb/` |

Pass criteria for a healthy gate are project-specific; the scan summary’s
`frac_rays_spurious` and `deepest_kcal` are the usual first look before reading
the path atlas.

## 4. Evaluate the test set (same UUID)

Pin the frozen epoch (do not pass the live run root — that picks the latest
epoch) and evaluate the in-distribution NPZ. GFN2-NMS energies are already
relative — do **not** pass `--subtract-atom-energies`.

```bash
UUID=f448e34c-cca7-43f6-8b8e-f4986b9403eb
ACODCM=/mmhome/boittier/home/mmml_tutorial/acodcm
FROZEN=$ACODCM/ckpts/_gate_frozen_e222   # contains only epoch-222
OUT=$ACODCM/eval_gfn2nms_${UUID}

mmml physnet-evaluate \
  --checkpoint "$FROZEN" \
  --data "$ACODCM/gfn2_nms_test.npz" \
  -o "$OUT/gfn2_nms_test_e222" \
  --batch-size 16 --plots --use-ema
```

Sync locally:

```bash
rsync -avz studix:$OUT/ artifacts/orient_plots/${UUID}/eval/
```
