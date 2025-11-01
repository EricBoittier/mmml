# Joint PhysNet-DCMNet Training

This directory contains a joint training script that trains PhysNet and DCMNet simultaneously with end-to-end gradient flow.

## Architecture

1. **PhysNet** predicts atomic charges (supervised by molecular dipole D)
2. Those charges become **monopoles** for DCMNet
3. **DCMNet** predicts distributed multipoles for ESP fitting
4. Full gradient flow: ESP loss → DCMNet → charges → PhysNet

## Usage

Basic usage:
```bash
python trainer.py \
  --train-efd ../physnet_train_charges/energies_forces_dipoles_train.npz \
  --train-esp ../dcmnet_train/grids_esp_train.npz \
  --valid-efd ../physnet_train_charges/energies_forces_dipoles_valid.npz \
  --valid-esp ../dcmnet_train/grids_esp_valid.npz \
  --epochs 100 \
  --batch-size 1 \
  --learning-rate 0.00001 \
  --grad-clip-norm 1.0 \
  --plot-results
```

**Note**: Add `--plot-results` to create validation scatter plots and ESP visualizations after training.

Recommended settings for stable training:
```bash
python trainer.py \
  --train-efd ../physnet_train_charges/energies_forces_dipoles_train.npz \
  --train-esp ../dcmnet_train/grids_esp_train.npz \
  --valid-efd ../physnet_train_charges/energies_forces_dipoles_valid.npz \
  --valid-esp ../dcmnet_train/grids_esp_valid.npz \
  --epochs 100 \
  --batch-size 10 \
  --learning-rate 0.00001 \
  --energy-weight 1.0 \
  --forces-weight 50.0 \
  --dipole-weight 25.0 \
  --esp-weight 1000000.0 \
  --mono-weight 100.0 \
  --grad-clip-norm 1.0 \
  --name co2_joint_stable
```

## Performance Notes

### Speed Optimization ✅

**Edge lists are now pre-computed once before training!** This provides a massive speedup:

- **Before**: ~140s/epoch (edge lists computed 8000 times per epoch)
- **After**: ~10-20s/epoch expected (edge lists computed once at startup)

The `precompute_edge_lists()` function:
1. Computes edge lists for all 8000 training samples once (~10 seconds)
2. Stores them as object arrays in the dataset
3. `prepare_batch_data()` just extracts and concatenates pre-computed edges

**Additional speedup options:**
1. **Increase batch size** - Requires handling variable numbers of ESP grid points per molecule
2. **Use compiled edge construction** - Vectorize with JAX/NumPy operations

### Gradient Clipping

Gradient clipping is enabled by default (`--grad-clip-norm 1.0`) to prevent exploding gradients. Disable with `--grad-clip-norm 0` if not needed.

## Loss Components

- **Energy**: PhysNet energy prediction (eV)
- **Forces**: PhysNet force prediction (eV/Å)
- **Dipole**: PhysNet dipole prediction from charges (Debye)
- **ESP**: DCMNet electrostatic potential fitting (Hartree/e)
- **Monopole**: Constraint that sum of DCMNet distributed charges = PhysNet charge per atom

## Hyperparameters

### PhysNet
- `--physnet-features`: Number of features (default: 64)
- `--physnet-iterations`: Message passing iterations (default: 5)
- `--physnet-basis`: Radial basis functions (default: 64)
- `--physnet-cutoff`: Interaction cutoff in Å (default: 6.0)

### DCMNet
- `--dcmnet-features`: Number of features (default: 32)
- `--dcmnet-iterations`: Message passing iterations (default: 2)
- `--dcmnet-basis`: Radial basis functions (default: 32)
- `--dcmnet-cutoff`: Interaction cutoff in Å (default: 10.0)
- `--n-dcm`: Distributed multipoles per atom (default: 3)

### Loss Weights
Start with balanced weights (all 1.0) and adjust based on loss magnitudes:
- `--energy-weight`: Energy loss weight (default: 1.0)
- `--forces-weight`: Forces loss weight (default: 50.0)
- `--dipole-weight`: Dipole loss weight (default: 25.0)
- `--esp-weight`: ESP loss weight (default: 10000.0)
- `--mono-weight`: Monopole constraint weight (default: 1.0)

### Dipole Source
Choose which dipole to use for the dipole loss:
- `--dipole-source physnet` (default): Use PhysNet's dipole computed from charges (D = Σ q_i · r_i)
- `--dipole-source dcmnet`: Use DCMNet's dipole computed from distributed multipoles

**PhysNet dipole**: Direct from atomic charges and positions. Provides direct supervision for charge prediction.

**DCMNet dipole**: From distributed multipoles. Tests if DCMNet's multipole decomposition is physically consistent with the molecular dipole.

Example comparing both:
```bash
# Train with PhysNet dipole (default - supervise charges)
python trainer.py ... --dipole-source physnet

# Train with DCMNet dipole (supervise multipole decomposition)
python trainer.py ... --dipole-source dcmnet
```

## Visualization

Create validation plots after training with `--plot-results`:

### Scatter Plots
Creates a 2x2 grid showing:
- **Energy**: True vs Predicted (eV)
- **Forces**: True vs Predicted (eV/Å)
- **Dipoles**: True vs Predicted (Debye)
- **ESP**: True vs Predicted (Hartree/e)

Each plot includes:
- Perfect prediction line (red dashed)
- Mean Absolute Error (MAE)
- Grid for easy reading

### ESP Examples
Creates detailed visualizations for individual molecules showing:
- **True ESP**: Reference electrostatic potential on VDW surface
- **Predicted ESP**: Model-predicted ESP
- **Error**: Difference between predicted and true ESP

Options:
- `--plot-samples N`: Number of validation samples to include in scatter plots (default: 100)
- `--plot-esp-examples N`: Number of detailed ESP examples to create (default: 2)

Plots are saved to: `{checkpoint_dir}/{experiment_name}/plots/`

**Plotting After Training:**
```bash
python trainer.py \
  --train-efd ../physnet_train_charges/energies_forces_dipoles_train.npz \
  --train-esp ../dcmnet_train/grids_esp_train.npz \
  --valid-efd ../physnet_train_charges/energies_forces_dipoles_valid.npz \
  --valid-esp ../dcmnet_train/grids_esp_valid.npz \
  --epochs 50 \
  --plot-results \
  --plot-samples 200 \
  --plot-esp-examples 5
```

This creates plots **once after training completes**:
- `validation_scatter.png`: 4 scatter plots
- `esp_example_0.png` through `esp_example_4.png`: Detailed ESP visualizations

**Plotting During Training:**
```bash
python trainer.py \
  --train-efd ../physnet_train_charges/energies_forces_dipoles_train.npz \
  --train-esp ../dcmnet_train/grids_esp_train.npz \
  --valid-efd ../physnet_train_charges/energies_forces_dipoles_valid.npz \
  --valid-esp ../dcmnet_train/grids_esp_valid.npz \
  --epochs 100 \
  --plot-freq 10 \
  --plot-samples 100 \
  --plot-esp-examples 2
```

This creates plots **every 10 epochs during training**:
- `validation_scatter_epoch10.png`, `validation_scatter_epoch20.png`, etc.
- `esp_example_0_epoch10.png`, `esp_example_1_epoch10.png`, etc.

This allows you to monitor training progress visually and see how predictions improve over time!

## ASE Calculator and Applications

The `ase_calculator.py` script creates an ASE calculator from the trained joint model for downstream applications.

### Features

1. **Geometry Optimization**: Optimize molecular structures with PhysNet forces
2. **Vibrational Frequencies**: Compute normal modes via numerical Hessian
3. **IR Spectra**: Calculate IR intensities from dipole derivatives (both PhysNet and DCMNet)
4. **Dipole Comparison**: Compare dipoles from atomic charges vs distributed multipoles

### Usage

**Full workflow (optimization + frequencies + IR):**
```bash
python ase_calculator.py \
  --checkpoint mmml/physnetjax/ckpts/co2_joint_physnet_dcmnet/best_params.pkl \
  --molecule co2 \
  --optimize \
  --frequencies \
  --ir-spectra \
  --output-dir ase_results/co2
```

**Just optimization:**
```bash
python ase_calculator.py \
  --checkpoint path/to/best_params.pkl \
  --molecule co2 \
  --optimize \
  --fmax 0.01
```

**Just frequencies:**
```bash
python ase_calculator.py \
  --checkpoint path/to/best_params.pkl \
  --molecule co2 \
  --frequencies
```

### Output

**Geometry Optimization:**
- Initial and final structures
- Optimization trajectory log
- Optimized geometry in XYZ format

**Frequency Calculation:**
- Vibrational frequencies (cm⁻¹)
- Normal modes
- Identification of imaginary/real modes

**IR Spectra:**
- `ir_spectrum_physnet.png`: IR from PhysNet dipole (charges × positions)
- `ir_spectrum_dcmnet.png`: IR from DCMNet dipole (distributed multipoles)
- `ir_spectrum_comparison.png`: Side-by-side comparison
- Stick and broadened spectra for both methods

### IR Spectrum Comparison

The script computes IR intensities using dipole derivatives:

**PhysNet IR:**
- Dipole from atomic charges: μ = Σ q_i · r_i
- Direct charge-based dipole moment

**DCMNet IR:**
- Dipole from distributed multipoles: μ = Σ Σ q_ij · r_ij
- More accurate multipole-based dipole

**Comparison shows:**
- Whether distributed multipoles improve IR predictions
- Consistency between charge-based and multipole-based approaches
- Differences in predicted intensities

### Example for CO2

```bash
# Train model first
python trainer.py --train-efd ... --epochs 100 --name co2_joint

# Then run ASE calculations
python ase_calculator.py \
  --checkpoint mmml/physnetjax/ckpts/co2_joint/best_params.pkl \
  --molecule co2 \
  --optimize \
  --frequencies \
  --ir-spectra \
  --output-dir co2_analysis
```

Expected output:
- Optimized CO2 geometry (linear)
- 4 vibrational modes (3 real: symmetric stretch, asymmetric stretch, 2× bend)
- IR spectrum with characteristic CO2 peaks
- Comparison showing both dipole methods
