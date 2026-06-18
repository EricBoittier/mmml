# MLpot settings reference (plots)

Visual reference for ML/MM handoff cutoffs and staged heating. Figures are generated locally by:

```bash
uv run python scripts/plot_mlpot_settings.py
```

Output directory: `docs/images/mlpot-settings/`.

CLI flags: [`mmml/interfaces/pycharmmInterface/cutoffs.py`](https://github.com/EricBoittier/mmml/blob/main/mmml/interfaces/pycharmmInterface/cutoffs.py) (`--mm-switch-on`, `--mm-switch-width`, `--ml-switch-width`). Dynamics: [`CHARMM_SETTINGS.md`](https://github.com/EricBoittier/mmml/blob/main/mmml/interfaces/pycharmmInterface/mlpot/CHARMM_SETTINGS.md).

## ML/MM cutoff presets

Complementary handoff (default): \(s_{\mathrm{ML}} + s_{\mathrm{MM}} = 1\) over `[mm_switch_on - ml_switch_width, mm_switch_on]`, then \(s_{\mathrm{MM}}\) tapers to 0 at `mm_switch_on + mm_switch_width`.

| Preset | `--mm-switch-on` | `--mm-switch-width` | `--ml-switch-width` | Used by |
|--------|------------------|---------------------|---------------------|---------|
| **Code default (`extended_mm5`)** | **8.0** | **5.0** | **1.5** | `md-system` / PyCHARMM MLpot / `run_dcm9_stability.sh` |
| Legacy narrow ML | 7.0 | 5.0 | 0.1 | Conservative fallback (pre–round-4 sweep) |
| Wide ML taper | 7.0 | 5.0 | 1.0 | Example: softer ML→MM at fixed 7 / 5 |
| Extended handoff (narrow MM) | 8.0 | 3.0 | 1.5 | Example; unstable on some DCM:3 geoms |

**Production default (June 2026):** `extended_mm5` (**8 / 5 / 1.5 Å**) from the DCM:3 NVE cutoff sweep (`workflows/dcm3_nve_cutoff_sweep`, 5 ps validation). Lowest sane mean smoothness across all four trimer COM geometries. Legacy **7 / 5 / 0.1** remains a safe fallback if needed.

### Overlay comparison

![ML and MM scale factors for four presets](images/mlpot-settings/cutoffs_comparison.png)

### Per-preset schematics

**Code default (8 / 5 / 1.5 Å)** — `extended_mm5`

![Code default cutoffs](images/mlpot-settings/cutoffs_code-default.png)

**Legacy narrow ML (7 / 5 / 0.1 Å)** — previous default

![Legacy narrow ML cutoffs](images/mlpot-settings/cutoffs_dcm9-stability.png)

Sparse ML dimer evaluation uses COM distance &lt; `mm_switch_on` (8 Å with current default). MM list outer reach is roughly `mm_switch_on + mm_switch_width` (13 Å).

**Wide ML taper (7 / 5 / 1.0 Å)**

![Wide ML taper cutoffs](images/mlpot-settings/cutoffs_wide-ml-taper.png)

**Extended handoff (8 / 3 / 1.5 Å)**

![Extended handoff cutoffs](images/mlpot-settings/cutoffs_extended-handoff.png)

### Complementary vs legacy MM window

Default runs use complementary handoff. Legacy (`--no-complementary-handoff`) uses a separate MM on/off window (not recommended for new workflows).

![Complementary vs legacy MM scaling at default cutoffs](images/mlpot-settings/cutoffs_complementary_vs_legacy.png)

## Staged heating

`--n-heat-segments N` splits `--ps-heat` into short chained restarts with overlap rescue between segments. DCM:9 script default: **N=4** (20 ps → 5 ps per segment, 0→240 K ramp).

![Staged heat ramp: 0 to 240 K over 20 ps](images/mlpot-settings/heat_staged_ramp.png)

| Segments | ps per segment | Segment end targets (K) |
|----------|----------------|------------------------|
| 1 | 20.0 | 240 |
| 4 | 5.0 | 60, 120, 180, 240 |
| 8 | 2.5 | 30, 60, …, 240 |

Example:

```bash
./scripts/run_dcm9_stability.sh
# or
mmml md-system ... --n-heat-segments 4 --ps-heat 20 \
  --heat-firstt 0 --heat-finalt 240
# cutoffs default to --mm-switch-on 8 --mm-switch-width 5 --ml-switch-width 1.5
```

## ML compute dtype (float32 vs float64)

PhysNet checkpoints are stored in **float32**. The hybrid calculator evaluates ML/MM interior math in a single JAX dtype (default **float32**). CHARMM I/O and returned total energies/forces stay **float64**.

Precedence: `--ml-compute-dtype` → `MMML_ML_DTYPE` → `JAX_ENABLE_X64=1` → float32.

To run ML interior in float64 (experimental; model not re-validated in f64):

```bash
export JAX_ENABLE_X64=1
mmml md-system ... --ml-compute-dtype float64
# or: export MMML_ML_DTYPE=float64  (with JAX_ENABLE_X64=1)
```

`JAX_ENABLE_X64` must be set **before** Python starts (e.g. in the shell or `scripts/mmml-charmm-mpirun.sh`). f32 checkpoints are promoted to f64 on load when f64 is requested.

If you saw smoother heating with “XLA” enabled, that was likely **`JAX_ENABLE_X64=1`**, not `XLA_FLAGS` compiler options alone — explicit `dtype=jnp.float32` in the ML path previously blocked x64 until this centralization.

## Medium PBC dense liquids (500–2000 monomers)

Before long production runs on equilibrated periodic boxes, validate the sparse dimer cap:

```bash
python scripts/validate_mlpot_sparse_dimers.py \
  --crd path/to/mini_full_mlpot_TAG.crd \
  --n-monomers 1000 --atoms-per-monomer 10 --box-size 40
```

See [Medium PBC workflow](mlpot-medium-pbc.md) for caps, `ml_batch_size` defaults, and JAX-MD handoff.
