# Plotting Features Checklist

## ✅ All Features Implemented and Verified

### 1. Validation Scatter Plots (validation_plots_*.png)
- ✅ **Energy scatter** - standard true vs pred
- ✅ **Forces scatter** - standard true vs pred
- ✅ **Dipole scatter (PhysNet)** - standard true vs pred
- ✅ **Dipole scatter (DCMNet)** - standard true vs pred, separate from PhysNet
- ✅ **ESP scatter (PhysNet)** - with **median ± 6 SD bounds** (not full range)
- ✅ **ESP scatter (DCMNet)** - with **median ± 6 SD bounds** (not full range)
- ✅ **Error histograms** - for all metrics (energy, forces, dipoles, ESP)
- ✅ **Perfect prediction line** (y=x) on all scatter plots
- ✅ **MAE/RMSE/R² metrics** in titles

### 2. Centered Scatter Plots (validation_centered_*.png)
- ✅ **Energy (centered)** - (true - mean) vs (pred - mean)
- ✅ **Forces (centered)** - removes mean bias
- ✅ **Dipole PhysNet (centered)** - shows correlation without bias
- ✅ **Dipole DCMNet (centered)** - shows correlation without bias
- ✅ **ESP PhysNet (centered)** - with **median ± 6 SD bounds**
- ✅ **ESP DCMNet (centered)** - with **median ± 6 SD bounds**
- ✅ **Correlation coefficient (R)** shown in titles
- ✅ **Zero reference lines** (x=0, y=0) for centered data

### 3. Comprehensive ESP Analysis (esp_analysis_comprehensive_*.png)
- ✅ **Hexbin density plots** - PhysNet & DCMNet
- ✅ **Residual plots** - error vs true value
- ✅ **2D histograms** - density heatmaps
- ✅ **Bland-Altman plots** - mean vs difference with ±1.96 SD limits
- ✅ **Q-Q plots** - tests error normality (with scipy fallback)
- ✅ **Error percentiles** - PhysNet vs DCMNet comparison
- ✅ **Cumulative distributions** - CDF of absolute errors

### 4. Per-Sample ESP Examples (esp_example_*_*.png)
- ✅ **True ESP** (1D scatter by grid point)
- ✅ **DCMNet ESP prediction** (1D scatter)
- ✅ **PhysNet ESP prediction** (1D scatter)
- ✅ **Error plots** for both models
- ✅ **Fixed colorscale**: [-0.01, 0.01] Ha/e for ALL ESP plots
- ✅ **Symmetric error scales** around 0
- ✅ **RMSE and R² metrics** displayed

### 5. 3D ESP Visualization (esp_example_*_3d*.png)
- ✅ **True ESP in 3D** with atom positions
- ✅ **PhysNet ESP in 3D** with atom labels
- ✅ **DCMNet ESP in 3D** with atom labels AND distributed charges
- ✅ **Fixed colorscale**: [-0.01, 0.01] Ha/e
- ✅ **Atom markers**: Black spheres with colored edges (yellow/lime/cyan)
- ✅ **Atom labels**: Atomic numbers (Z) with background boxes
- ✅ **Distributed charges**: Triangle markers (^) colored by charge magnitude
- ✅ **Proper coordinate frame**: No Bohr conversion (vdw_surface already in Angstroms)
- ✅ **No artificial centering**: Atoms shown at actual positions

### 6. Multi-Scale Error Plots (esp_example_*_error_scales*.png)
- ✅ **3 rows**: 100%, 95th, 75th percentile error ranges
- ✅ **2 columns**: PhysNet vs DCMNet side-by-side
- ✅ **2D projections** (X vs Z)
- ✅ **Symmetric error colorscales** around 0
- ✅ **Atom position markers** on all subplots (black circles with colored edges)
- ✅ **Proper coordinates**: Atoms at actual positions in grid

### 7. Radial ESP Error Plots (esp_radial_*_*.png) 🆕
- ✅ **One subplot per atom** showing error vs distance
- ✅ **Combined plot** with all atoms (log scale for |error|)
- ✅ **Atomic radius markers**:
  - Orange dashed: r_cov (covalent radius)
  - Blue dashed: 2×r_cov (exclusion cutoff)
  - Red shaded zone: 0 to 2×r_cov (excluded region)
- ✅ **Both PhysNet and DCMNet errors** shown
- ✅ **Zero error reference line**
- ✅ **Element-specific cutoffs** (different for C, O, H)

### 8. Distributed Charge Visualization (charges_example_*_*.png)
- ✅ **4 views**: 3D + XY, XZ, YZ projections
- ✅ **Atom markers**: Black circles labeled with atomic numbers
- ✅ **Charge positions**: Colored by magnitude (RdBu_r)
- ✅ **Symmetric colorscale** for charges
- ✅ **Equal aspect ratio** on 2D projections

### 9. Per-Atom Charge Detail (charges_detail_*_*.png)
- ✅ **One subplot per atom**
- ✅ **Relative positions**: Charges shown relative to parent atom
- ✅ **Lines connecting** atom to distributed charges
- ✅ **Charge values labeled** on each point
- ✅ **Sum of charges** shown in title
- ✅ **Center marked** at (0, 0) for reference

## Filtering Applied During Training (Not Plotting)

### Training & Validation Loss Computation:
- ✅ **Atomic radius filtering**: Excludes grid points < 2×covalent_radius (DEFAULT)
- ✅ **Optional fixed distance**: `--esp-min-distance` > 0 adds extra margin
- ✅ **Optional magnitude filter**: `--esp-max-value` excludes high |ESP| values

### Plotting (No Filtering):
- ✅ All plots use **unfiltered data** (`esp_min_distance=0.0`, `esp_max_value=1e10`)
- ✅ Shows full ESP distribution for diagnostic purposes
- ✅ Radial plots visualize the exclusion zones (red shaded regions)

## Key Settings Confirmed

### Colorscales:
- ✅ ESP values: **Fixed at [-0.01, 0.01] Ha/e** (not data-dependent)
- ✅ ESP errors: Symmetric around 0 (data-dependent max)
- ✅ Charges: Symmetric around 0 (RdBu_r colormap)

### Plot Bounds:
- ✅ ESP scatter plots: **Median ± 6 SD** (focuses on main distribution)
- ✅ Centered plots: **Median ± 6 SD** for ESP
- ✅ Other metrics: Full range with perfect prediction line

### Coordinate Frames:
- ✅ All spatial data in **Angstroms**
- ✅ No Bohr conversion (vdw_surface already in Angstroms)
- ✅ Atoms positioned at true locations
- ✅ Grid points properly aligned with atoms

## Plots Generated Every 10 Epochs (Default)

Total plots per validation cycle:
1. `validation_plots_epoch*.png` - Main scatter + histogram grid (4×3)
2. `validation_centered_epoch*.png` - Centered correlation plots (2×3)
3. `esp_analysis_comprehensive_epoch*.png` - Statistical analysis (3×4)
4. For each ESP example (default 2):
   - `esp_example_*_epoch*.png` - 1D ESP comparisons (2×3)
   - `esp_example_*_3d_epoch*.png` - 3D ESP visualization (1×3)
   - `esp_example_*_error_scales_epoch*.png` - Multi-scale errors (3×2)
   - `esp_radial_*_epoch*.png` - Radial error analysis (1×4)
   - `charges_example_*_epoch*.png` - Charge distribution (2×2)
   - `charges_detail_*_epoch*.png` - Per-atom detail (1×n_atoms)

**Total:** ~15 plots per validation cycle (every 10 epochs by default)

## Verification Commands

Check plots exist:
```bash
ls -lh /home/ericb/mmml/mmml/physnetjax/ckpts/co2_joint_physnet_dcmnet/plots/
```

View specific plot:
```bash
# Example for epoch 20
eog /home/ericb/mmml/mmml/physnetjax/ckpts/co2_joint_physnet_dcmnet/plots/esp_radial_0_epoch20.png
```

