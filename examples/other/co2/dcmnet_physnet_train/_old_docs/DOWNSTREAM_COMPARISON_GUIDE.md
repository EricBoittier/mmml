# Downstream Task Comparison Guide

Comprehensive head-to-head comparison of DCMNet vs Non-Equivariant models on all downstream tasks.

---

## Features

This script performs a complete comparison on:

### 1. **Harmonic Vibrational Analysis**
- Geometry optimization
- Normal mode frequencies
- IR intensities from dipole derivatives
- Raman activities from polarizability derivatives

### 2. **Molecular Dynamics**
- NVT ensemble simulations
- Temperature and energy statistics
- Dipole tracking at every timestep

### 3. **Anharmonic IR Spectra**
- IR spectrum from MD dipole autocorrelation
- Captures temperature broadening
- Real anharmonic effects

### 4. **Charge Distribution Analysis**
- Charges as function of CO2 internal coordinates
- Theta (bond angle): 160°-180°
- R1, R2 (bond lengths): 1.0-1.3 Å
- 3D surface plots showing charge variation

---

## Quick Start

### 1. Quick Comparison (5-10 minutes)

```bash
python compare_downstream_tasks.py \
    --checkpoint-dcm comparisons/test1/dcmnet_equivariant/best_params.pkl \
    --checkpoint-noneq comparisons/test1/noneq_model/best_params.pkl \
    --quick \
    --output-dir downstream_quick
```

**What it does:**
- ✅ Harmonic analysis (frequencies, IR, Raman)
- ✅ Geometry optimization
- ❌ MD (skipped for speed)
- ❌ Charge analysis (skipped)

**Output:**
- `spectroscopy_comparison.png` - IR and Raman spectra
- `downstream_results.json` - All metrics

**Time:** ~5 minutes

---

### 2. Full Comparison (30-60 minutes)

```bash
python compare_downstream_tasks.py \
    --checkpoint-dcm comparisons/test1/dcmnet_equivariant/best_params.pkl \
    --checkpoint-noneq comparisons/test1/noneq_model/best_params.pkl \
    --full \
    --md-steps 50000 \
    --output-dir downstream_full
```

**What it does:**
- ✅ Harmonic analysis
- ✅ MD simulations (50,000 steps)
- ✅ Anharmonic IR from MD
- ❌ Charge analysis (add flag if desired)

**Output:**
- `spectroscopy_comparison.png` - All spectra with MD
- `downstream_results.json` - Complete metrics

**Time:** ~30-60 minutes

---

### 3. With Charge Surface Analysis

```bash
python compare_downstream_tasks.py \
    --checkpoint-dcm dcmnet/best_params.pkl \
    --checkpoint-noneq noneq/best_params.pkl \
    --full \
    --analyze-charges \
    --theta-range 160 180 20 \
    --r-range 1.0 1.3 20 \
    --md-steps 50000 \
    --output-dir downstream_complete
```

**What it does:**
- ✅ Everything from full comparison
- ✅ Charge distribution over CO2 configuration space
- ✅ 3D surface plots (theta vs r)

**Output:**
- `spectroscopy_comparison.png`
- `charge_surfaces.png` - 9 subplots of charge analysis
- `downstream_results.json`

**Time:** ~60-90 minutes

---

## Output Files

### 1. `spectroscopy_comparison.png`

4-panel comparison plot:

**Top Left:** Harmonic IR Spectrum
- Stick spectra for both models
- Shows fundamental vibrational frequencies
- DCMNet (blue) vs Non-Eq (purple)

**Top Right:** MD IR Spectrum (Anharmonic)
- Smooth curves from dipole autocorrelation
- Includes temperature broadening
- Real anharmonic effects

**Bottom Left:** Raman Spectrum
- Stick spectra from polarizability derivatives
- Shows Raman-active modes
- Approximate from charge model

**Bottom Right:** Comparison Statistics
- Frequencies for both models
- Optimization statistics
- MD temperature and energy
- Timing information

### 2. `charge_surfaces.png` (if `--analyze-charges`)

9-panel charge analysis:

**Row 1:** Charges vs Bond Angle (at r ≈ 1.16 Å)
- C, O1, O2 charges vs theta
- Shows angular dependence
- Scatter plots for both models

**Row 2:** Charges vs Bond Length (at theta ≈ 180°)
- C, O1, O2 charges vs r
- Shows bond stretching effects
- Scatter plots for both models

**Row 3:** 2D Contour Plots (theta vs r)
- C, O1, O2 charge surfaces
- Contour plots showing full 2D dependence
- DCMNet surfaces (interpolated)

### 3. `downstream_results.json`

Complete JSON with:
```json
{
  "dcmnet": {
    "frequencies": [667.3, 667.3, 1345.6, 2349.2],
    "ir_intensities": [...],
    "raman_intensities": [...],
    "optimized_energy": -123.456,
    "optimization_steps": 15,
    "md_energy_std": 0.0234,
    "md_temperature_mean": 299.8,
    "harmonic_time": 45.2,
    "md_time": 234.5
  },
  "noneq": {
    ...
  }
}
```

---

## Options

### Model Checkpoints
```bash
--checkpoint-dcm PATH      # DCMNet checkpoint (required)
--checkpoint-noneq PATH    # Non-Eq checkpoint (required)
```

### Analysis Modes
```bash
--quick                    # Fast mode (harmonic only, ~5 min)
--full                     # Full mode (harmonic + MD, ~30-60 min)
--analyze-charges          # Add charge surface analysis
```

### MD Parameters
```bash
--md-steps N               # Number of MD steps (default: 10000)
--temperature T            # Temperature in K (default: 300)
--timestep DT              # Timestep in fs (default: 0.5)
```

### Charge Analysis Parameters
```bash
--theta-range MIN MAX N    # Angle range (default: 160 180 20)
--r-range MIN MAX N        # Bond length range (default: 1.0 1.3 20)
```

### Output
```bash
--output-dir DIR           # Output directory (default: downstream_comparison)
--dpi N                    # Plot resolution (default: 200)
```

---

## Interpreting Results

### Harmonic Frequencies

**CO2 Normal Modes:**
1. **Bend (2x):** ~667 cm⁻¹ (degenerate)
2. **Symmetric stretch:** ~1345 cm⁻¹ (IR inactive, Raman active)
3. **Asymmetric stretch:** ~2349 cm⁻¹ (IR active)

**Good model should:**
- Match these frequencies within ~50 cm⁻¹
- Show correct IR/Raman selection rules
- Predict similar values for both models (if data is good)

### MD IR Spectrum

**Advantages over harmonic:**
- Includes anharmonic effects
- Temperature-dependent broadening
- Overtones and combinations visible

**What to look for:**
- Peak positions should match harmonic roughly
- Broadening indicates realistic dynamics
- Stable energy/temperature shows good model

### Charge Surfaces

**What you're seeing:**
- How atomic charges vary with geometry
- Charge transfer during vibrations
- Model's physical consistency

**Equivariant model should:**
- Show smooth charge variation
- Reasonable charge magnitudes (-1 to +1 typically)
- Physically motivated trends

**Non-equivariant might:**
- Show artifacts from orientation changes
- Discontinuities in charge surfaces
- Unphysical charge patterns

---

## Example Workflow

### Step 1: Quick Test

```bash
# Fast sanity check (5 min)
python compare_downstream_tasks.py \
    --checkpoint-dcm dcmnet/best_params.pkl \
    --checkpoint-noneq noneq/best_params.pkl \
    --quick
```

**Check:**
- Do frequencies look reasonable?
- Are IR/Raman patterns correct?
- Any errors or warnings?

### Step 2: Full Comparison

```bash
# Complete spectroscopy comparison (30 min)
python compare_downstream_tasks.py \
    --checkpoint-dcm dcmnet/best_params.pkl \
    --checkpoint-noneq noneq/best_params.pkl \
    --full \
    --md-steps 50000
```

**Check:**
- Does MD IR match harmonic?
- Are dynamics stable?
- Any performance differences?

### Step 3: Charge Analysis

```bash
# Add charge surface analysis (60 min)
python compare_downstream_tasks.py \
    --checkpoint-dcm dcmnet/best_params.pkl \
    --checkpoint-noneq noneq/best_params.pkl \
    --full \
    --analyze-charges \
    --md-steps 50000
```

**Check:**
- Are charge surfaces smooth?
- Any unphysical artifacts in non-eq?
- Does equivariance help charges?

---

## Troubleshooting

### Issue: Optimization doesn't converge

**Solution:** Models might need better training. Check:
```bash
# Verify model loads correctly
python -c "import pickle; print(pickle.load(open('best_params.pkl', 'rb')))"
```

### Issue: MD energy drift

**Symptoms:** Energy increases over time in MD

**Solution:** This is expected for NVE. Use NVT for thermostat.

### Issue: Raman calculation fails

**Note:** Raman is approximate from charges. Warnings are normal.

**Solution:** Script will use zeros and continue. Check warning message.

### Issue: Charge surface plots look strange

**Check:**
- Are there NaN values? (some geometries might fail)
- Is the configuration space reasonable? (don't go too far from equilibrium)
- Try smaller ranges first

---

## Performance Comparison

### What to compare:

**Accuracy:**
- Frequency differences (cm⁻¹)
- Energy differences (eV)
- MD stability (energy conservation)

**Speed:**
- Harmonic analysis time
- MD time per step
- Total wall time

**Physics:**
- Charge distribution smoothness
- IR/Raman selection rules
- Equivariance in MD

---

## Advanced Usage

### Custom Molecule

To analyze a different molecule, modify the script:

```python
# In run_harmonic_analysis() and run_md_analysis()
if molecule == 'H2O':
    atoms = Atoms('H2O', positions=[[...]])
```

### Longer MD for Better IR Resolution

```bash
# 100 ps simulation
python compare_downstream_tasks.py \
    --checkpoint-dcm dcmnet/best_params.pkl \
    --checkpoint-noneq noneq/best_params.pkl \
    --full \
    --md-steps 200000 \
    --timestep 0.5
```

Time: ~100,000 steps = ~2-4 hours depending on hardware

### High-Resolution Charge Surface

```bash
python compare_downstream_tasks.py \
    --checkpoint-dcm dcmnet/best_params.pkl \
    --checkpoint-noneq noneq/best_params.pkl \
    --analyze-charges \
    --theta-range 150 180 50 \
    --r-range 0.9 1.4 50 \
    --quick  # Skip MD to save time
```

This creates 50×50×50 = 125,000 configurations!

---

## For Publications

### Recommended Settings

```bash
python compare_downstream_tasks.py \
    --checkpoint-dcm dcmnet/best_params.pkl \
    --checkpoint-noneq noneq/best_params.pkl \
    --full \
    --analyze-charges \
    --md-steps 100000 \
    --theta-range 160 180 30 \
    --r-range 1.05 1.25 30 \
    --dpi 300 \
    --output-dir paper_figures/downstream
```

**Creates publication-quality figures:**
- High DPI (300)
- Sufficient statistics (100k MD steps)
- Smooth charge surfaces (30×30 grid)

**Time:** 2-3 hours

---

## Expected Results for CO2

### Typical Values

**Harmonic Frequencies:**
- Bend: 650-680 cm⁻¹
- Symmetric stretch: 1300-1360 cm⁻¹
- Asymmetric stretch: 2300-2380 cm⁻¹

**MD at 300K:**
- Average T: 295-305 K
- Energy std: 0.02-0.05 eV
- IR peaks: Broadened by ~20-50 cm⁻¹

**Charges:**
- C: +0.5 to +0.8 e
- O: -0.25 to -0.4 e each
- Sum: ≈ 0 (neutrality)

---

## Summary

This script provides a **complete downstream task comparison** for your trained models:

✅ **Harmonic** - Frequencies, IR, Raman  
✅ **Anharmonic** - MD with realistic broadening  
✅ **Charge Analysis** - Full configuration space  
✅ **Head-to-head** - Direct DCMNet vs Non-Eq comparison  
✅ **Publication-ready** - High-quality plots

**Use it to:**
1. Validate model training
2. Compare equivariant vs non-equivariant
3. Generate paper figures
4. Understand model physics

**Perfect complement to the training comparison script!** 🎉

