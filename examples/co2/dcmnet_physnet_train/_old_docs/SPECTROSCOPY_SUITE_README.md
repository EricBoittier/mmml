# Complete Spectroscopy Suite

**Production-ready workflow for comprehensive spectroscopic analysis with PhysNet-DCMNet**

---

## 🎯 **Quick Start**

```bash
cd /home/ericb/mmml/examples/co2/dcmnet_physnet_train

# Quick analysis (5 minutes)
python spectroscopy_suite.py \
  --checkpoint /home/ericb/mmml/mmml/physnetjax/ckpts/co2_joint_physnet_dcmnet \
  --molecule CO2 \
  --quick-analysis

# Temperature scan (20 minutes)
python spectroscopy_suite.py \
  --checkpoint /home/ericb/mmml/mmml/physnetjax/ckpts/co2_joint_physnet_dcmnet \
  --molecule CO2 \
  --run-temperature-scan \
  --temperatures 100 200 300 400 500

# Everything (1-2 hours)
python spectroscopy_suite.py \
  --checkpoint /home/ericb/mmml/mmml/physnetjax/ckpts/co2_joint_physnet_dcmnet \
  --molecule CO2 \
  --run-all
```

---

## 📦 **What It Does**

### ✅ **Always Saved to ASE Trajectories**
- All trajectories include **energies** and **forces**
- JAX MD results automatically converted to ASE format
- Compatible with `ase gui` for visualization

### 🔬 **Analysis Modes**

#### 1. **Quick Analysis** (`--quick-analysis`)
- Geometry optimization (BFGS)
- Harmonic frequencies
- IR spectrum (from normal modes)
- Raman spectrum (finite-field)
- **Time**: ~5 minutes

#### 2. **Temperature Scan** (`--run-temperature-scan`)
- MD at multiple temperatures
- IR spectra at each T
- Temperature-dependent broadening
- **Default**: 100, 200, 300, 400, 500 K

#### 3. **Ensemble Comparison** (`--run-ensemble-comparison`)
- NVE (constant energy)
- NVT (constant temperature)
- Compare energy conservation, fluctuations

#### 4. **Optimizer Comparison** (`--run-optimizer-comparison`)
- BFGS
- LBFGS
- BFGSLineSearch
- Compare speed and convergence

#### 5. **Production Run** (`--production`)
- Long MD for high-resolution IR
- Default: 50k steps (customizable with `--nsteps`)
- Best for publication-quality spectra

#### 6. **Everything** (`--run-all`)
- All of the above!

---

## 🚀 **Performance**

### **JAX MD vs ASE MD**

| Task | ASE | JAX MD | Speedup |
|------|-----|--------|---------|
| 10k steps | ~5 min | ~30 sec | **10×** |
| 50k steps | ~25 min | ~2 min | **12×** |
| 100k steps | ~50 min | ~5 min | **10×** |
| 500k steps | ~4 hours | ~25 min | **10×** |

**Use JAX MD (default) for production!**

Force ASE with `--use-ase-md` (not recommended).

---

## 📁 **Output Structure**

```
spectroscopy_suite/
├── CO2_initial.xyz                      # Initial geometry
│
├── quick_analysis/
│   ├── optimization.traj                # ✅ ASE traj with E,F
│   ├── CO2_optimized.xyz
│   ├── ir_spectrum.png                  # Harmonic IR
│   ├── raman_spectrum.png               # Raman
│   └── vibrations/                      # Normal mode data
│
├── temperature_scan/
│   ├── temperature_comparison.png       # All temps overlaid
│   ├── T100K_nvt/
│   │   ├── trajectory.traj              # ✅ ASE traj with E,F
│   │   ├── trajectory.npz               # JAX MD native format
│   │   ├── ir_spectrum_md.png
│   │   └── jaxmd_nvt_results.png
│   ├── T200K_nvt/
│   ├── T300K_nvt/
│   ├── T400K_nvt/
│   └── T500K_nvt/
│
├── ensemble_comparison/
│   ├── nve/
│   │   ├── trajectory.traj              # ✅ ASE traj with E,F
│   │   └── ir_spectrum_md.png
│   └── nvt/
│       ├── trajectory.traj              # ✅ ASE traj with E,F
│       └── ir_spectrum_md.png
│
├── optimizer_comparison/
│   ├── comparison.json                  # Timing & convergence
│   ├── bfgs/
│   │   ├── optimization.traj            # ✅ ASE traj with E,F
│   │   └── CO2_optimized.xyz
│   ├── lbfgs/
│   └── bfgsls/
│
└── production/
    ├── trajectory.traj                  # ✅ ASE traj with E,F
    ├── trajectory.npz                   # JAX MD native
    └── ir_spectrum_md.png               # High-resolution IR
```

---

## 🎨 **Examples**

### **1. Custom Molecule from XYZ**

```bash
python spectroscopy_suite.py \
  --checkpoint ckpt/ \
  --geometry my_molecule.xyz \
  --quick-analysis
```

### **2. Specific Temperature Range**

```bash
python spectroscopy_suite.py \
  --checkpoint ckpt/ \
  --molecule H2O \
  --run-temperature-scan \
  --temperatures 250 300 350 400
```

### **3. Long Production Run**

```bash
# 250 ps (500k steps) for ultra-high resolution IR
python spectroscopy_suite.py \
  --checkpoint ckpt/ \
  --molecule CO2 \
  --production \
  --nsteps 500000 \
  --timestep 0.5
```

### **4. Skip Raman (Faster)**

```bash
# Raman is slow, skip if not needed
python spectroscopy_suite.py \
  --checkpoint ckpt/ \
  --molecule CO2 \
  --quick-analysis \
  --skip-raman
```

### **5. Custom Ensembles**

```bash
python spectroscopy_suite.py \
  --checkpoint ckpt/ \
  --molecule CO2 \
  --run-ensemble-comparison \
  --ensembles nve nvt \
  --nsteps 100000
```

---

## 📊 **Understanding the Outputs**

### **ASE Trajectories (.traj)**

View with ASE GUI:
```bash
ase gui spectroscopy_suite/quick_analysis/optimization.traj
ase gui spectroscopy_suite/temperature_scan/T300K_nvt/trajectory.traj
```

Access programmatically:
```python
from ase.io import read

# Read entire trajectory
traj = read('trajectory.traj', ':')

for atoms in traj:
    energy = atoms.get_potential_energy()  # eV
    forces = atoms.get_forces()            # eV/Å
    positions = atoms.positions            # Å
```

### **IR Spectra**

Two types:
1. **Harmonic** (`ir_spectrum.png`) - from normal modes
   - Fast to compute
   - No temperature effects
   - Sharp peaks

2. **From MD** (`ir_spectrum_md.png`) - from autocorrelation
   - Includes anharmonicity
   - Temperature broadening
   - More realistic!

### **Raman Spectra**

Complementary to IR:
- **IR active**: Asymmetric modes (change dipole)
- **Raman active**: Symmetric modes (change polarizability)

For CO2:
- 1388 cm⁻¹: Strong in Raman, **invisible in IR**
- 2349 cm⁻¹: Strong in IR, weak in Raman

---

## 🔧 **Advanced Options**

### **All CLI Arguments**

```bash
python spectroscopy_suite.py --help
```

Key options:
- `--nsteps N`: MD steps (default: 50000)
- `--timestep T`: MD timestep in fs (default: 0.5)
- `--temperature T`: Temperature in K (default: 300)
- `--temperatures T1 T2 ...`: List for scan
- `--ensembles E1 E2 ...`: Ensembles to compare
- `--optimizers O1 O2 ...`: Optimizers to compare
- `--use-ase-md`: Force ASE instead of JAX MD
- `--skip-raman`: Skip Raman calculation
- `--laser-wavelength W`: Raman laser (nm, default: 532)

---

## 💡 **Tips & Best Practices**

### **For Best IR Resolution**

1. **Longer MD** → better frequency resolution
   ```bash
   --nsteps 200000  # 100 ps at 0.5 fs timestep
   ```

2. **Smaller timestep** for high frequencies
   ```bash
   --timestep 0.25  # For C-H stretch (~3000 cm⁻¹)
   ```

3. **Multiple trajectories** → average for noise reduction

### **For Fastest Results**

1. Use JAX MD (default, not ASE)
2. Skip Raman: `--skip-raman`
3. Reduce steps: `--nsteps 10000`

### **For Publication Quality**

1. Long MD: `--nsteps 500000`
2. Multiple temperatures
3. Include Raman
4. Do `--run-all`

---

## 🐛 **Troubleshooting**

### **"JAX MD not installed"**

```bash
pip install jax-md
```

### **"Out of memory" on GPU**

Reduce batch size or use CPU:
```bash
export CUDA_VISIBLE_DEVICES=""
```

### **MD unstable / exploding**

- Reduce timestep: `--timestep 0.25`
- Check if model was trained with gradient clipping
- Verify forces are conservative (use NVE test)

### **IR spectrum looks noisy**

- Run longer: `--nsteps 100000`
- Or average multiple trajectories

---

## 📖 **How It Works**

### **JAX MD → ASE Conversion**

JAX MD is fast but uses custom formats. The suite:
1. Runs JAX MD (fast!)
2. Saves native `.npz` (positions, velocities, dipoles)
3. **Converts to ASE trajectory** with:
   - Positions from JAX MD
   - Energies computed with model
   - Forces computed with model
   - Result: Standard ASE `.traj` file

### **IR from MD**

Uses fluctuation-dissipation theorem:
1. Record dipole moment μ(t) at each step
2. Compute autocorrelation: C(t) = ⟨μ(0)·μ(t)⟩
3. Fourier transform: I(ω) ∝ FT[C(t)]

This captures:
- Anharmonic effects
- Mode coupling
- Temperature broadening

### **Raman from Finite Fields**

Applies small electric fields F:
1. Measure dipole response: μ(F)
2. Compute polarizability: α = ∂μ/∂F
3. For each mode: ∂α/∂Q
4. Raman intensity ∝ |∂α/∂Q|²

---

## 🎯 **Recommended Workflows**

### **First Time Analysis**

```bash
# 1. Quick check
python spectroscopy_suite.py --checkpoint ckpt/ --molecule CO2 --quick-analysis

# 2. If results look good, full analysis
python spectroscopy_suite.py --checkpoint ckpt/ --molecule CO2 --run-all
```

### **For Paper/Publication**

```bash
# High-resolution production run
python spectroscopy_suite.py \
  --checkpoint ckpt/ \
  --molecule CO2 \
  --production \
  --nsteps 500000 \
  --run-temperature-scan \
  --temperatures 200 250 300 350 400
```

### **Quick Parameter Testing**

```bash
# Fast test with ASE (more stable)
python spectroscopy_suite.py \
  --checkpoint ckpt/ \
  --molecule CO2 \
  --quick-analysis \
  --skip-raman \
  --use-ase-md \
  --nsteps 1000
```

---

## 📚 **Related Scripts**

Individual tools (if you need fine control):

- `trainer.py` - Model training
- `eval_calculator.py` - ESP visualization
- `dynamics_calculator.py` - ASE MD + IR
- `raman_calculator.py` - Raman spectroscopy
- `jaxmd_dynamics.py` - Fast JAX MD
- **`spectroscopy_suite.py`** ← **This (master script)**

---

## 🏆 **Citation**

If you use this for research:

```bibtex
@article{physnet2019,
  title={PhysNet: A neural network for predicting energies and forces},
  author={Unke, Oliver T and Meuwly, Markus},
  journal={Journal of Chemical Theory and Computation},
  year={2019}
}
```

---

## 💬 **Support**

For issues:
1. Check this README
2. Look at example outputs
3. Try `--use-ase-md` (more stable, slower)
4. Reduce `--nsteps` for testing

---

**Happy spectroscopy! 🔬✨**

