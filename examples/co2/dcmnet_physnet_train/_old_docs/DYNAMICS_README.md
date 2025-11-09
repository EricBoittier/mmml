# Molecular Dynamics & Vibrational Analysis

Complete guide for using the joint PhysNet-DCMNet model for dynamics simulations and vibrational analysis.

## Features

### 🎵 **Vibrational Analysis (Harmonic)**
- Geometry optimization
- Normal mode frequencies from numerical Hessian
- IR spectra from dipole derivatives
- Comparison of PhysNet (point charges) vs DCMNet (distributed multipoles)

### 🌡️ **Molecular Dynamics**
- NVE (constant energy) ensemble
- NVT (Langevin thermostat) ensemble
- Dipole moment tracking at every timestep
- **IR spectra from dipole autocorrelation** (anharmonic effects!)

### 📊 **Analysis**
- Energy/temperature evolution
- Statistical distributions
- IR spectra (both harmonic and from MD)
- Dipole autocorrelation functions
- Publication-quality plots

---

## Quick Start

### 1. Full Vibrational Analysis

```bash
python dynamics_calculator.py \
  --checkpoint /path/to/checkpoint \
  --molecule CO2 \
  --optimize --frequencies --ir-spectra \
  --output-dir ./vibrational_analysis
```

**Outputs:**
- `CO2_optimized.xyz` - Optimized geometry
- `ir_spectrum.png` - Harmonic IR spectrum (PhysNet vs DCMNet)
- `vibrations/` - Normal mode data

---

### 2. Molecular Dynamics with IR Spectrum

```bash
# Short NVT simulation (5 ps at 300 K)
python dynamics_calculator.py \
  --checkpoint /path/to/checkpoint \
  --molecule CO2 \
  --md --ensemble nvt --temperature 300 \
  --timestep 0.5 --nsteps 10000 \
  --output-dir ./md_nvt_300K
```

**Outputs:**
- `md_nvt.traj` - Full trajectory (view with `ase gui`)
- `md_nvt_results.png` - Energy/temperature plots
- `ir_spectrum_md.png` - **IR spectrum from dipole autocorrelation**

**Key Features:**
- Saves dipole moments at **every timestep**
- Computes dipole autocorrelation function
- Fourier transforms to get IR spectrum
- Captures **anharmonic effects** and **temperature broadening**

---

### 3. Longer MD for Better IR Resolution

```bash
# 50 ps simulation for high-resolution IR
python dynamics_calculator.py \
  --checkpoint /path/to/checkpoint \
  --molecule CO2 \
  --md --ensemble nvt --temperature 300 \
  --timestep 0.5 --nsteps 100000 \
  --output-dir ./md_long
```

**Tips for IR from MD:**
- Longer simulations → better frequency resolution
- Typical: 10-50 ps for small molecules
- Timestep 0.5 fs is good for vibrations up to ~3000 cm⁻¹
- Use 0.25 fs for high-frequency modes (e.g., C-H stretch)

---

### 4. Everything at Once

```bash
python dynamics_calculator.py \
  --checkpoint /path/to/checkpoint \
  --molecule CO2 \
  --optimize \
  --frequencies --ir-spectra \
  --md --ensemble nvt --temperature 300 \
  --timestep 0.5 --nsteps 20000 \
  --output-dir ./complete_analysis
```

This will:
1. ✅ Optimize geometry
2. ✅ Calculate harmonic frequencies & IR
3. ✅ Run MD with dipole tracking
4. ✅ Compute IR from dipole autocorrelation
5. ✅ Generate all plots

---

## IR Spectrum Methods Compared

### Harmonic Approximation (Normal Modes)
- **Method**: Numerical Hessian + dipole derivatives
- **Pros**: Fast, gives exact frequencies for harmonic potential
- **Cons**: No anharmonicity, no temperature effects
- **Best for**: Initial characterization, identifying modes

### MD Dipole Autocorrelation
- **Method**: Fourier transform of dipole time series
- **Pros**: Includes anharmonicity, temperature broadening, mode coupling
- **Cons**: Requires long simulations, lower frequency resolution
- **Best for**: Realistic spectra, comparing to experiment

---

## Understanding the Outputs

### `ir_spectrum.png` (Harmonic)
- **Stick spectrum**: Individual mode frequencies
- **Broadened spectrum**: Lorentzian line shapes
- **Colors**: Blue = PhysNet, Orange = DCMNet

### `ir_spectrum_md.png` (From MD)
- **Top row**: Dipole autocorrelation functions
  - Decay shows dephasing timescales
  - Oscillations indicate vibrational modes
- **Bottom row**: IR spectra at different ranges
  - 0-5000 cm⁻¹: Full range
  - 500-3500 cm⁻¹: Vibrational region (zoomed)

---

## Advanced Usage

### Custom Molecule from XYZ

```bash
python dynamics_calculator.py \
  --checkpoint /path/to/checkpoint \
  --geometry my_molecule.xyz \
  --optimize --frequencies --ir-spectra \
  --output-dir ./my_molecule_analysis
```

### NVE Ensemble (Energy Conservation Test)

```bash
python dynamics_calculator.py \
  --checkpoint /path/to/checkpoint \
  --molecule CO2 \
  --md --ensemble nve --temperature 300 \
  --timestep 0.5 --nsteps 50000 \
  --output-dir ./md_nve
```

Check the energy plot - should be constant!

### Use DCMNet Dipole for Forces

```bash
python dynamics_calculator.py \
  --checkpoint /path/to/checkpoint \
  --molecule CO2 \
  --use-dcmnet-dipole \
  --md --ensemble nvt --temperature 300 \
  --timestep 0.5 --nsteps 10000 \
  --output-dir ./md_dcmnet_dipole
```

---

## Expected Results for CO2

### Vibrational Frequencies
- **~660 cm⁻¹**: Bending mode (IR active)
- **~1431 cm⁻¹**: Asymmetric stretch (IR inactive for linear CO2)
- **~2603 cm⁻¹**: Symmetric stretch (IR active)

### PhysNet vs DCMNet
- **PhysNet**: Point charges at atomic centers
  - Simpler, faster
  - Good for dominant charge transfer effects
  
- **DCMNet**: Distributed multipoles (3 per atom)
  - More accurate ESP representation
  - Captures charge distribution around atoms
  - Better for anisotropic environments

### IR Intensities
The intensity of each mode depends on `|dμ/dQ|²`:
- Modes with large dipole derivative → strong IR absorption
- Symmetric modes in symmetric molecules → weak/zero intensity
- DCMNet may show different intensities due to better charge distribution

---

## Computational Cost

| Task | Time (CO2, 3 atoms) | Notes |
|------|---------------------|-------|
| Optimization | ~10 sec | BFGS, 3-5 steps |
| Frequencies | ~2 min | 3N×2 energy+force evaluations |
| IR from modes | ~5 min | 3N-6 dipole derivative calculations |
| MD (10k steps) | ~3 min | With dipole saving |
| IR from MD | ~1 sec | Post-processing |

**Scaling**: Time scales with:
- Number of atoms (N)
- Number of MD steps
- Model evaluation cost

---

## Tips & Tricks

### For Better IR Spectra from MD

1. **Equilibration**: Run 1-2 ps without saving, then restart with dipole saving
2. **Longer trajectories**: 50-100 ps gives better resolution
3. **Multiple trajectories**: Average spectra from independent runs
4. **Temperature**: Higher T → more intensity, broader peaks

### For Accurate Frequencies

1. **Optimize first**: Frequencies at non-stationary points have imaginary components
2. **Smaller delta**: Use `--vib-delta 0.005` for more accurate Hessian
3. **Check convergence**: `fmax < 0.001` for optimization

### Viewing Trajectories

```bash
# View with ASE GUI
ase gui md_nvt.traj

# Animate
ase gui md_nvt.traj --animate

# Get specific frame
ase gui md_nvt.traj@-1  # Last frame
```

---

## Troubleshooting

### "No dipole data found"
- Make sure MD runs with `save_dipoles=True` (default)
- Check that calculator has `dipole_physnet` and `dipole_dcmnet` properties

### Energy not conserved in NVE
- Reduce timestep (try 0.25 fs)
- Check for gradient clipping in training
- Verify forces are conservative

### Imaginary frequencies
- Not fully optimized - reduce `--fmax` to 0.001
- Or structure is at saddle point - check geometry

### IR spectrum looks wrong
- Too short trajectory - increase `--nsteps`
- Wrong timestep - use 0.5 fs or smaller
- Check autocorrelation decay - needs to reach ~zero

---

## Citation

If you use this code for research, please cite:
- PhysNet: [doi:10.1021/acs.jctc.8b00908](https://doi.org/10.1021/acs.jctc.8b00908)
- DCMNet: [Your DCMNet reference]
- ASE: [doi:10.1088/1361-648X/aa680e](https://doi.org/10.1088/1361-648X/aa680e)

---

## Examples Gallery

### CO2 at 300 K
![IR from MD](ir_spectrum_md.png)
- Clear bending mode at 660 cm⁻¹
- Temperature broadening visible
- PhysNet and DCMNet agree well

### Vibrational Analysis
![Harmonic IR](ir_spectrum.png)
- Sharp harmonic peaks
- DCMNet shows slightly different intensities
- Useful for mode assignment

---

**Happy simulating!** 🚀🔬✨

