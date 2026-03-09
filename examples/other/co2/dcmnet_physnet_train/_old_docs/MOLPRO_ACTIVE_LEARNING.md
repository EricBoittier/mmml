# Molpro Active Learning Integration

Complete workflow for using Molpro QM calculations to improve your ML model based on unstable MD structures.

## Overview

Your existing Molpro infrastructure is **perfect** for active learning! This guide shows how to:
1. Collect unstable structures from MD simulations
2. Generate Molpro inputs compatible with your existing job system
3. Run high-quality MP2/aug-cc-pVTZ calculations
4. Convert results to training format
5. Retrain and validate improvements

## Quick Start

### 1. Run MD and Collect Unstable Structures

```bash
# Run MD (auto-saves unstable structures)
python run_production_md.py \
  --checkpoint /path/to/ckpt \
  --molecule CO2 \
  --nsteps 1000000 \
  --timestep 0.01 \
  --output-dir ./md_300K

# Structures auto-saved to: ./md_300K/active_learning/*.npz
```

### 2. Generate Molpro Job Files

```bash
python create_molpro_jobs_from_al.py \
  --al-structures ./md_*/active_learning/*.npz \
  --output-dir ./molpro_al_jobs \
  --job-name co2_al_mp2 \
  --max-structures 50 \
  --sort-by-force \
  --ntasks 16 \
  --mem 132G
```

This creates:
```
molpro_al_jobs/
├── al_0000_f6.7.inp       # Molpro input (Cartesian geom)
├── al_0001_f8.2.inp
├── ...
├── co2_al_mp2.sbatch      # SLURM submission script
└── logs/                  # Will contain outputs
```

### 3. Submit Molpro Jobs

```bash
cd molpro_al_jobs
sbatch co2_al_mp2.sbatch
```

Monitor with:
```bash
squeue -u $USER
tail -f logs/al_*.err
```

### 4. Convert Results to Training Format

After jobs complete:

```bash
python convert_molpro_to_training.py \
  --molpro-outputs ./molpro_al_jobs/logs/*.molpro.out \
  --cube-dir ./molpro_al_jobs/cubes \
  --merge-with ../physnet_train_charges/energies_forces_dipoles_train.npz \
  --output ../physnet_train_charges/energies_forces_dipoles_train_v2.npz \
  --skip-errors
```

This:
- Parses Molpro outputs (energy, forces, dipole)
- Reads ESP/density cubes
- Converts units (Hartree → eV, Bohr → Å, etc.)
- Merges with existing training data

### 5. Retrain Model

```bash
python trainer.py \
  --train-efd ../physnet_train_charges/energies_forces_dipoles_train_v2.npz \
  --train-esp ../dcmnet_train/grids_esp_train.npz \
  --valid-efd ../physnet_train_charges/energies_forces_dipoles_valid.npz \
  --valid-esp ../dcmnet_train/grids_esp_valid.npz \
  --epochs 500 \
  --batch-size 100 \
  --name retrained_v2
```

### 6. Test Improvements

```bash
# Run MD with new model
python run_production_md.py \
  --checkpoint ./ckpts/retrained_v2 \
  --molecule CO2 \
  --nsteps 2000000 \
  --output-dir ./md_test_v2

# Compare statistics
python active_learning_manager.py \
  --source ./md_test_v2/active_learning \
  --list
```

**Success:** Fewer explosions, lower max forces, longer stable trajectories!

## Comparison: Your Grid Scan vs Active Learning

### Grid Scan (Current Approach)
```bash
# Your existing params.sh:
R1_LIST=(1.0 1.05 1.1 ... 1.5)      # 20 points
R2_LIST=(1.0 1.05 1.1 ... 1.5)      # 20 points  
ANG_LIST=(160 165 170 ... 200)     # 9 points
# → 3600 total structures
```

**Pros:**
- Systematic coverage
- Known geometries
- Easy to parallelize

**Cons:**
- Many irrelevant geometries
- Misses critical regions where model fails
- Expensive (3600 × MP2/aug-cc-pVTZ!)

### Active Learning (New Approach)

```bash
# Collect only problematic structures from MD
→ 50-100 structures where model actually fails
```

**Pros:**
- ✅ Targets model weaknesses
- ✅ 50-100 structures vs 3600 (70× fewer!)
- ✅ Covers real dynamics, not just interpolation
- ✅ Iterative improvement

**Cons:**
- Requires initial model
- Less systematic

### Recommended Hybrid Approach

1. **Initial Training**: Use your grid scan data (great!)
2. **Active Learning Round 1**: Run MD, collect 50 failures, run MP2, retrain
3. **Active Learning Round 2**: Run MD again, collect new failures, repeat
4. **Convergence**: When MD runs stably for long trajectories → done!

## File Format Details

### Active Learning Structure (NPZ)

```python
data = np.load('unstable_20250102_143022_step114.npz')
# Keys:
#   positions: (n_atoms, 3) - Å
#   velocities: (n_atoms, 3) - Å/fs
#   forces: (n_atoms, 3) - eV/Å (model prediction)
#   atomic_numbers: (n_atoms,)
#   energy: float - eV (model prediction)
#   dipole_physnet: (3,) - e·Å
#   dipole_dcmnet: (3,) - e·Å
#   max_force: float - eV/Å
#   reason: str - 'explosion' or 'high_force'
#   step: int
#   time_fs: float
```

### Molpro Input (Generated)

```molpro
***, Active Learning Structure al_0001_f8.2
memory,1800,m
angstrom
symmetry,nosym
orient,noorient

basis=aug-cc-pVTZ

geometry={
  C ,   0.00000000,   0.00000000,   0.00000000
  O ,   0.00000000,   0.00000000,   1.23456789
  O ,   0.00000000,   0.00000000,  -1.23456789
}

{df-hf;  wf,charge=0,spin=0}
{put,molden,al_0001_f8.2.hf.molden}

{df-mp2; expec}
forces

{cube,cubes/density/density_al_0001_f8.2.cube; density; step,0.2,0.2,0.2}
{cube,cubes/esp/esp_al_0001_f8.2.cube; potential; step,0.2,0.2,0.2}
{put,molden,al_0001_f8.2.mp2.molden}

table, energy, dipx, dipy, dipz, grms, gmax
table, save, logs/results_al_0001_f8.2.csv
```

### Training Data (NPZ)

```python
data = np.load('energies_forces_dipoles_train_v2.npz')
# Keys:
#   R: (n_structures, n_atoms, 3) - positions in Å
#   Z: (n_structures, n_atoms) - atomic numbers
#   E: (n_structures,) - energies in eV
#   F: (n_structures, n_atoms, 3) - forces in eV/Å
#   D: (n_structures, 3) - dipoles in e·Å
```

## Unit Conversions (Automatic)

The converter handles all unit conversions:

| Property | Molpro → Training |
|----------|-------------------|
| Energy | Hartree → eV (×27.2114) |
| Forces | Ha/Bohr → eV/Å (×51.422) |
| Positions | Bohr → Å (×0.529177) |
| Dipole | Debye → e·Å (×0.2082) |
| ESP | Ha/e → Ha/e (no change) |

## Troubleshooting

### "Could not find MP2 energy in output"

Check Molpro output for errors:
```bash
grep -i "error\|warning\|fail" logs/*.molpro.out
```

Common issues:
- SCF convergence failure → increase maxiter or try different guess
- Basis set linear dependency → reduce basis or adjust geometry
- Memory issues → increase memory allocation

### "Conversion failed: Atom count mismatch"

Some structures may have different atom counts. Use:
```bash
python convert_molpro_to_training.py ... --skip-errors
```

### "Still getting explosions after retraining"

Possible causes:
1. **Not enough data**: Add more structures (100+ recommended)
2. **QM calculation failed**: Check that Molpro converged properly
3. **Wrong regions**: Run MD at multiple temperatures to sample diverse configs
4. **Model capacity**: May need more PhysNet layers or features

## Advanced: Automated Pipeline

Create a monitoring script:

```bash
#!/bin/bash
# auto_retrain.sh

# 1. Run MD
python run_production_md.py --checkpoint ./current_model --output-dir ./md_iter$ITER

# 2. Check if we have enough new failures
N_FAILURES=$(ls ./md_iter$ITER/active_learning/*.npz | wc -l)

if [ $N_FAILURES -gt 50 ]; then
    echo "Found $N_FAILURES failures, starting retraining cycle..."
    
    # 3. Generate Molpro jobs
    python create_molpro_jobs_from_al.py \
        --al-structures ./md_iter$ITER/active_learning/*.npz \
        --output-dir ./molpro_iter$ITER \
        --max-structures 100
    
    # 4. Submit jobs
    cd ./molpro_iter$ITER
    JOB_ID=$(sbatch co2_al_mp2.sbatch | awk '{print $4}')
    
    # 5. Wait for completion and retrain
    sbatch --dependency=afterok:$JOB_ID retrain_after_qm.sbatch
fi
```

## Cost Analysis

### Typical Active Learning Cycle

| Stage | Structures | Time/Structure | Total Time |
|-------|-----------|----------------|------------|
| MD simulation | - | - | 1-2 hours |
| Failure collection | 50 | instant | - |
| Molpro MP2/aug-cc-pVTZ | 50 | 30 min | **25 hours** |
| Conversion + retrain | - | - | 2-3 hours |

**Total:** ~1 day per cycle

### Cost Savings vs Grid Scan

- Grid: 3600 structures × 30 min = **1800 hours** (75 days!)
- Active Learning: 50 structures × 30 min × 3 cycles = **75 hours** (3 days)

**24× faster**, and better coverage of failure modes!

## Next Steps

1. ✅ Run MD with current model
2. ✅ Collect unstable structures (automatic)
3. ✅ Generate Molpro jobs: `create_molpro_jobs_from_al.py`
4. ✅ Submit to cluster
5. ✅ Convert results: `convert_molpro_to_training.py`
6. ✅ Retrain model
7. ✅ Test improvements
8. ✅ Repeat until stable

## References

- **Molpro Manual**: https://www.molpro.net/info/2015.1/doc/manual/
- **Active Learning Review**: https://doi.org/10.1021/acs.chemrev.1c00107
- **Our Active Learning Setup**: `ACTIVE_LEARNING_README.md`

