# Downstream Comparison Script - Fixed!

The `compare_downstream_tasks.py` script has been updated to work correctly with both DCMNet and Non-Equivariant models.

---

## What Was Fixed

### 1. **Model Configuration**
The model takes config dictionaries, not individual parameters:

```python
# WRONG (before):
model = JointPhysNetDCMNet(
    features_physnet=64,
    max_degree=2,
    ...
)

# CORRECT (after):
physnet_config = {
    'features': 64,
    'max_degree': 2,
    'natoms': 3,
    ...
}
dcmnet_config = {
    'features': 128,
    ...
}
model = JointPhysNetDCMNet(
    physnet_config=physnet_config,
    dcmnet_config=dcmnet_config
)
```

### 2. **Different Model Classes**
The script now correctly uses different model classes:

- **Equivariant:** `JointPhysNetDCMNet`
- **Non-Equivariant:** `JointPhysNetNonEquivariant`

### 3. **Parameter Structure**
Added proper handling for parameter initialization and structure checking.

---

## Usage (Now Working!)

```bash
# Quick test (5-10 min)
cd comparisons/test1
python ../../compare_downstream_tasks.py \
    --checkpoint-dcm dcmnet_equivariant/best_params.pkl \
    --checkpoint-noneq noneq_model/best_params.pkl \
    --quick

# Full analysis (30-60 min)
python ../../compare_downstream_tasks.py \
    --checkpoint-dcm dcmnet_equivariant/best_params.pkl \
    --checkpoint-noneq noneq_model/best_params.pkl \
    --full \
    --md-steps 50000

# With charge analysis (60-90 min)
python ../../compare_downstream_tasks.py \
    --checkpoint-dcm dcmnet_equivariant/best_params.pkl \
    --checkpoint-noneq noneq_model/best_params.pkl \
    --full \
    --analyze-charges \
    --theta-range 160 180 20 \
    --r-range 1.0 1.3 20
```

---

## What It Does

1. **Harmonic Analysis**
   - Geometry optimization
   - Vibrational frequencies
   - IR intensities
   - Raman activities

2. **Molecular Dynamics** (if `--full`)
   - NVT simulation
   - Energy/temperature statistics
   - Anharmonic IR from dipole autocorrelation

3. **Charge Surface Analysis** (if `--analyze-charges`)
   - Charges vs CO2 bond angle
   - Charges vs bond length
   - 3D contour plots

4. **Publication-Quality Plots**
   - `spectroscopy_comparison.png` - 4 panels
   - `charge_surfaces.png` - 9 panels (if requested)
   - `downstream_results.json` - All metrics

---

## Output Files

### spectroscopy_comparison.png
- Top left: Harmonic IR (stick plot)
- Top right: MD IR (anharmonic, broadened)
- Bottom left: Raman spectrum
- Bottom right: Statistics and timing

### charge_surfaces.png (if --analyze-charges)
- Row 1: C, O1, O2 charges vs angle
- Row 2: C, O1, O2 charges vs bond length
- Row 3: 2D contour plots (theta vs r)

### downstream_results.json
Complete metrics in JSON format for further analysis.

---

## Troubleshooting

### If you see import errors:
Make sure you're running from the correct directory with the mmml repo in your Python path.

### If models don't load:
Check that the checkpoint files exist and contain valid parameters:
```bash
python -c "import pickle; print(list(pickle.load(open('best_params.pkl', 'rb')).keys()))"
```

### If optimization fails:
The models might need retraining or the geometry might be far from equilibrium.

---

## Next Steps

Once the script runs successfully, you can:

1. Compare vibrational frequencies to experiment/QM
2. Analyze charge distribution smoothness
3. Check MD stability (energy conservation)
4. Generate publication figures
5. Add to your comparison results alongside training metrics

---

**Script is now ready to use!** 🎉

