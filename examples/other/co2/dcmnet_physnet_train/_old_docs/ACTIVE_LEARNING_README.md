# Active Learning for Model Improvement

This directory includes tools for **active learning** - using unstable/challenging MD structures to improve the model.

## Overview

When MD simulations become unstable (explosions, high forces), these structures represent regions where the model's potential energy surface is inaccurate. By collecting these structures and obtaining high-quality reference data (from QM calculations), you can retrain the model to handle these challenging cases.

## Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  1. Run MD Simulations                                      │
│     → Automatically save unstable structures                │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Collect & Filter Structures                             │
│     → active_learning_manager.py --list                     │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  3. Export for QM Calculations                              │
│     → active_learning_manager.py --export-xyz               │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  4. Run QM Calculations (ORCA, Gaussian, etc.)              │
│     → prepare_qm_inputs.py (helper script)                  │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  5. Merge with Training Data                                │
│     → Convert QM results to NPZ format                      │
│     → Add to training dataset                               │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  6. Retrain Model                                           │
│     → trainer.py with augmented dataset                     │
└─────────────────────────────────────────────────────────────┘
```

## Files Created During MD

When MD becomes unstable, structures are automatically saved to:
```
<output_dir>/active_learning/
├── unstable_20250102_143022_step114.npz    # Exploded structure
├── high_force_20250102_143045_step230.npz  # High force (>5 eV/Å)
└── ...
```

Each NPZ file contains:
- `positions`: Atomic positions (Å)
- `velocities`: Atomic velocities (Å/fs)
- `forces`: Predicted forces (eV/Å)
- `atomic_numbers`: Atom types
- `energy`: Predicted energy (eV)
- `dipole_physnet`, `dipole_dcmnet`: Predicted dipoles
- Metadata: `step`, `time_fs`, `max_force`, `temperature`, `reason`, etc.

## Usage Examples

### 1. List All Saved Structures

```bash
python active_learning_manager.py \
  --source ./md_*/active_learning \
  --list
```

Output:
```
ACTIVE LEARNING STRUCTURE SUMMARY
======================================================================
Total structures: 47

By failure reason:
  explosion: 12
  high_force: 35

By instability type:
  force: 12
  warning: 35

Max force statistics (eV/Å):
  Min: 5.12
  Max: 105.34
  Mean: 8.67
  Median: 6.45

Top 10 most challenging structures...
```

### 2. Export Unique Structures for QM Calculations

```bash
python active_learning_manager.py \
  --source ./md_*/active_learning \
  --export-xyz \
  --deduplicate \
  --min-force 3.0 \
  --max-structures 100 \
  --output ./qm_candidates
```

This creates:
```
qm_candidates/
├── structure_0000.xyz     # XYZ file for QM
├── structure_0000.json    # Metadata
├── structure_0001.xyz
├── structure_0001.json
└── ...
```

### 3. Run QM Calculations

**Option A: ORCA (recommended)**
```bash
# Create ORCA input files
python prepare_qm_inputs.py \
  --xyz-dir ./qm_candidates \
  --qm-software orca \
  --method PBE0 \
  --basis def2-TZVP \
  --output ./orca_inputs

# Run ORCA (batch)
cd orca_inputs
for inp in *.inp; do
  orca $inp > ${inp%.inp}.out
done
```

**Option B: Gaussian**
```bash
python prepare_qm_inputs.py \
  --xyz-dir ./qm_candidates \
  --qm-software gaussian \
  --method PBE0 \
  --basis 6-311G** \
  --output ./gaussian_inputs
```

### 4. Convert QM Results to Training Format

```bash
# Parse ORCA outputs
python convert_qm_to_npz.py \
  --orca-outputs ./orca_inputs/*.out \
  --output ./qm_training_data.npz

# Merge with existing training data
python merge_datasets.py \
  --original ../physnet_train_charges/energies_forces_dipoles_train.npz \
  --new ./qm_training_data.npz \
  --output ../physnet_train_charges/energies_forces_dipoles_train_v2.npz
```

### 5. Retrain Model

```bash
python trainer.py \
  --train-efd ../physnet_train_charges/energies_forces_dipoles_train_v2.npz \
  --train-esp ../dcmnet_train/grids_esp_train.npz \
  --valid-efd ../physnet_train_charges/energies_forces_dipoles_valid.npz \
  --valid-esp ../dcmnet_train/grids_esp_valid.npz \
  --epochs 500 \
  --batch-size 100 \
  --name retrained_active_learning
```

## Configuration Options

### MD Simulation (jaxmd_dynamics.py)

```python
run_jaxmd_simulation(
    ...
    save_unstable_structures=True,    # Enable active learning
    active_learning_dir=Path('./my_al_data'),  # Custom directory
)
```

### Detection Thresholds

Structures are saved when:
- **Explosion**: `max_force > 10.0 eV/Å` OR `max_velocity > 1.0 Å/fs` OR `max_position > 100 Å`
- **High Force Warning**: `max_force > 5.0 eV/Å` (proactive)

You can modify these in `jaxmd_dynamics.py` lines 677 and 729.

## Tips for Best Results

1. **Diverse Conditions**: Run MD at multiple temperatures (100-500 K) to sample different regions
2. **Different Starting Geometries**: Perturb initial structures
3. **Quality over Quantity**: 50-100 high-quality QM structures > 1000 low-quality ones
4. **QM Level**: Use PBE0/def2-TZVP or ωB97X-D3/def2-TZVP for accuracy
5. **Iterative Process**: Retrain → run MD → collect new failures → retrain again

## QM Software Recommendations

| Software | Pros | Cons | Recommended Level |
|----------|------|------|-------------------|
| ORCA | Free, fast, excellent defaults | Needs compilation | PBE0/def2-TZVP |
| Gaussian | Widely used, stable | Commercial | PBE0/6-311G** |
| Q-Chem | Modern, efficient | Commercial | ωB97X-D3/def2-TZVP |
| Psi4 | Free, Python API | Slower | PBE0/def2-TZVP |

## Monitoring Active Learning Progress

After retraining, test the new model:

```bash
# Run MD with new model
python run_production_md.py \
  --checkpoint /path/to/retrained/model \
  --nsteps 1000000 \
  --analyze-ir \
  --output-dir ./test_retrained

# Compare active learning statistics
python active_learning_manager.py \
  --source ./test_retrained/active_learning \
  --list
```

**Success metrics:**
- ✅ Fewer explosions
- ✅ Lower max forces
- ✅ Longer stable trajectories
- ✅ More physical IR spectra

## Advanced: Uncertainty-Based Active Learning

For more sophisticated active learning, you can:

1. **Train an ensemble** of models (different random seeds)
2. **Compute prediction variance** across ensemble
3. **Save high-uncertainty structures** (even if forces are reasonable)
4. **Prioritize QM calculations** on high-uncertainty + high-force structures

This is not yet implemented but could be added to `jaxmd_dynamics.py`.

## Troubleshooting

**Q: Simulations still crash after retraining?**
- Check if QM data covers the problematic regions
- Increase diversity of saved structures (more temperatures)
- Verify QM calculation converged properly

**Q: Too many structures saved?**
- Increase force thresholds (line 677, 729 in jaxmd_dynamics.py)
- Use `--deduplicate` and `--max-structures` when exporting
- Focus on highest-force structures first

**Q: QM calculations too expensive?**
- Start with smaller basis set (def2-SVP instead of TZVP)
- Use semi-empirical methods (GFN2-xTB) for pre-screening
- Parallelize across multiple nodes

## References

- **Active Learning in Chemistry**: https://doi.org/10.1021/acs.chemrev.1c00107
- **SchNet with Active Learning**: https://doi.org/10.1038/s41467-019-12875-2
- **Force Uncertainty for AL**: https://doi.org/10.1063/5.0036522

