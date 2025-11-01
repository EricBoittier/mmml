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
Recommended weights for effective training:
- `--energy-weight`: Energy loss weight (default: 1.0)
- `--forces-weight`: Forces loss weight (default: 50.0)
- `--dipole-weight`: Dipole loss weight (default: 25.0)
- `--esp-weight`: ESP loss weight (default: 10000.0)
- `--mono-weight`: Monopole constraint weight (default: 100.0)

**Note:** The monopole constraint (distributed charges sum to atomic charge) needs high weight (100+) to be properly enforced.

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

### Energy Mixing (Experimental)

**`--mix-coulomb-energy`**: Mix PhysNet energy with DCMNet Coulomb energy using a learnable λ parameter.

**Idea:**
- PhysNet predicts total energy (all interactions)
- DCMNet distributed charges → compute explicit Coulomb energy
- Learn mixing: `E_total = E_physnet + λ × E_coulomb(DCMNet)`
- Forces come from gradient of mixed energy

**Benefits:**
- Separates electrostatic component (DCMNet) from other physics (PhysNet)
- λ is learned during training (initialized to 0.1)
- Can improve energy predictions if Coulomb term is accurate
- **Works with any batch size** (uses `jax.vmap` for batched computation)

**Usage:**
```bash
python trainer.py \
  --train-efd ... \
  --train-esp ... \
  --mix-coulomb-energy \
  --batch-size 10 \
  --mono-weight 100.0 \
  --esp-weight 1000000.0
```

**Output shows:**
```
Coulomb Mixing:
  λ (learned): 0.123
  E_coulomb: -12.45 eV
```

**When to use:**
- When you want explicit electrostatic energy component
- To test if distributed charges improve energy predictions
- Experimental - may help or hurt depending on data

## Training Output

The training script provides detailed metrics with **validation set statistics** for context:

```
Epoch 50/500 (1.8s)
  Train Loss: 3216.732230
    Energy: 715.649172
    Forces: 4.951418
    Dipole: 0.057234
    ESP: 0.002252
    Monopole: 0.000409
    Total Charge: 0.001260
  Valid Loss: 3081.512695
  Coulomb Mixing:  # (only if --mix-coulomb-energy is enabled)
    λ (learned): 0.100000
    E_coulomb: -127.453 eV
    MAE Energy: 41.224107 eV  (950.648528 kcal/mol) [μ=-5104.420, σ=1.777 eV]
    MAE Forces: 1.147455 eV/Å  (26.460883 kcal/mol/Å) [μ=-0.000, σ=8.427 eV/Å]
    MAE Dipole (PhysNet): 0.050436 e·Å  (0.242257 D) [μ=0.123, σ=0.456 e·Å]
    MAE Dipole (DCMNet): 0.059249 e·Å  (0.284588 D) [μ=0.123, σ=0.456 e·Å]
    RMSE ESP (PhysNet): 0.069926 Ha/e  (43.879494 (kcal/mol)/e) [μ=0.001234, σ=0.023456 Ha/e]
    RMSE ESP (DCMNet): 0.067553 Ha/e  (42.389904 (kcal/mol)/e) [μ=0.001234, σ=0.023456 Ha/e]
    Total Charge Violation: 0.001000
```

**Interpretation:**
- **[μ=mean, σ=std]**: Validation set statistics for each metric
- **MAE/σ ratio**: Indicates model quality
  - < 0.5: Excellent (error < 50% of std)
  - 0.5-1.0: Good
  - 1.0-2.0: Needs improvement
  - \> 2.0: Model not learning well
- **Dual metrics**: Both PhysNet and DCMNet dipoles/ESP are reported for comparison
- **Multiple units**: Primary units + converted (e.g., eV and kcal/mol)

## Visualization

Create validation plots after training with `--plot-results`:

### Validation Plots (Scatter + Histograms)
Creates a comprehensive 4×3 grid with:

**Row 1 - Scatter Plots (True vs Predicted):**
- **Energy** (eV)
- **Forces** (eV/Å)
- **Dipole - PhysNet** (e·Å)

**Row 2 - Error Histograms:**
- Energy prediction errors
- Forces prediction errors
- Dipole (PhysNet) prediction errors

**Row 3 - More Scatter Plots:**
- **Dipole - DCMNet** (e·Å)
- **ESP - PhysNet** (Hartree/e)
- **ESP - DCMNet** (Hartree/e)

**Row 4 - More Error Histograms:**
- Dipole (DCMNet) prediction errors
- ESP (PhysNet) prediction errors
- ESP (DCMNet) prediction errors

**Features:**
- Scatter plots: Perfect prediction line (red dashed), MAE/RMSE, R² for ESP
- Histograms: Error distribution, zero-error reference line, standard deviation
- Color-coded by model: PhysNet (blue/default), DCMNet (orange), ESP variants (green/purple)
- Consistent scales for easy comparison

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
