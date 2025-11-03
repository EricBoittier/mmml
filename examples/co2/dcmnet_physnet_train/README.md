# Joint PhysNet-DCMNet Training with Learnable Dipole/ESP Mixing

This directory contains a comprehensive toolkit for joint PhysNet–DCMNet training with flexible loss configurations, learnable mixing networks, EMA-parameterized validation, and extensive analysis workflows.

## 🎯 Key Capabilities

### 1. **Flexible Loss Configuration**
- **Configurable supervision**: Train on PhysNet, DCMNet, or mixed dipole/ESP predictions
- **Multiple loss terms**: Combine multiple loss terms with individual weights and metrics (L2, MAE, RMSE)
- **JSON/YAML configs**: Define complex loss configurations in external files
- **Example**: Supervise both raw and mixed outputs simultaneously for ensemble learning

### 2. **Learnable Charge Orientation Mixing**
- **E(3)-equivariant mixer**: Neural network that learns mixing weights from charge distributions using spherical harmonics
- **Orientation-aware**: Encodes 3D charge orientations via weighted spherical harmonic features
- **Combines modes**: `mixed_dipole = λ·DCMNet + (1-λ)·PhysNet` where λ is learned per-molecule
- **ESP blending**: Applies the same learnable mixing to ESP predictions

### 3. **Exponential Moving Average (EMA)**
- **Smoother validation**: All validation uses EMA-smoothed weights (decay=0.999)
- **Better generalization**: Averaged weights often outperform final snapshots
- **Checkpointing**: Saved checkpoints use EMA parameters automatically

### 4. **Comprehensive Validation**
- **Any batch size**: ESP RMSE computed efficiently across all validation batches
- **3D visualization**: ESP errors visualized in 3D space with symmetric color scales
- **Charge diagnostics**: Per-molecule charge analysis and distribution visualization
- **Mixed metrics**: Track λ values and mixed prediction quality

## Architecture

```
Input → PhysNet → Atomic Charges → DCMNet → Distributed Multipoles
       ↓                                 ↓                       ↓
  E, F, D(Phys)                  D(DCM)                    ESP(Phys/DCM)
                                           ↓                      ↓
                                   [Charge Mixer] → Mixed D/ESP
```

## Data Units

**All spatial coordinates are in Angstroms:**

**Units throughout:**
- Atom positions (`R`): Angstroms
- ESP grid positions (`vdw_surface`): **Angstroms** (already correct in data files)
- ESP values (`esp`): Hartree/e (atomic units)
- Energies (`E`): eV
- Forces (`F`): eV/Å
- Dipoles (`Dxyz`): e·Å

**Expected ESP RMSE for well-trained models:**
- **CO2 (small, rigid)**: < 0.005 Ha/e (~3 kcal/mol/e)
  - Excellent: < 0.002 Ha/e (~1 kcal/mol/e)
  - Good: 0.002-0.005 Ha/e (1-3 kcal/mol/e)
  - Acceptable: < 0.01 Ha/e (~6 kcal/mol/e)
- **Larger/flexible molecules**: 0.005-0.015 Ha/e
- **> 0.02 Ha/e**: Indicates training issues or model capacity problems

**ESP calculation:**
- `calc_esp` expects positions in Angstroms
- Internally converts distances to Bohr: `ESP = q / (r_Angstrom × 1.88973)`
- Returns ESP in Hartree/e (atomic units)

**Coordinate Frame Alignment:**
The trainer automatically aligns ESP grids with molecular coordinates:
- Computes atom COM and grid COM for each molecule
- Shifts grid by `-(grid_COM - atom_COM)` so they share the same center
- Ensures ESP calculations use correct spatial relationships
- Critical for accurate ESP predictions!

Without alignment, atoms and grid can be offset by ~3 Å, causing completely wrong ESP values.

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

### Atomic Energy Subtraction (Default ON)

**By default**, the trainer subtracts reference atomic energies from molecular energies:
- Converts absolute energies → formation/atomization energies
- Makes energies relative to isolated atoms
- Improves training by putting energies on a more consistent scale
- Helps with chemical interpretation

**Reference atomic energies** (PBE/def2-TZVP in eV):
- H: -13.587, C: -1029.499, N: -1484.274, O: -2041.878
- F: -2713.473, P: -8978.229, S: -10831.086, Cl: -12516.444

**To disable** (use absolute energies):
```bash
python trainer.py ... --no-subtract-atom-energies
```

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

### Flexible Loss Configuration

**NEW**: Configure loss terms flexibly via CLI or JSON/YAML files!

**Basic CLI Options:**
- `--dipole-loss-sources {physnet,dcmnet,mixed} ...`: Specify dipole supervision sources
- `--esp-loss-sources {physnet,dcmnet,mixed} ...`: Specify ESP supervision sources
- `--dipole-metric {l2,mae,rmse}`: Metric for dipole losses (default: l2)
- `--esp-metric {l2,mae,rmse}`: Metric for ESP losses (default: l2)
- `--loss-config PATH`: Load loss configuration from JSON or YAML file

**Examples:**

Train on individual modes:
```bash
# Only PhysNet dipole
python trainer.py ... --dipole-loss-sources physnet

# Only DCMNet ESP
python trainer.py ... --esp-loss-sources dcmnet

# Mix both sources with equal weight
python trainer.py ... --dipole-loss-sources physnet dcmnet
```

Train with learnable mixed mode:
```bash
# Enable charge mixer and train on mixed predictions
python trainer.py ... --dipole-loss-sources mixed --esp-loss-sources mixed
```

Train on both raw and mixed simultaneously:
```bash
# Learn mixing while supervising both raw and mixed outputs
python trainer.py ... --dipole-loss-sources physnet dcmnet mixed
```

**Advanced: Loss Configuration File**

Create `loss_config.yaml`:
```yaml
dipole:
  - source: physnet
    weight: 25.0
    metric: l2
    name: dipole_physnet_raw
  - source: dcmnet
    weight: 25.0
    metric: l2
    name: dipole_dcmnet_raw
  - source: mixed
    weight: 50.0
    metric: mae
    name: dipole_mixed

esp:
  - source: dcmnet
    weight: 10000.0
    metric: l2
    name: esp_dcmnet_raw
  - source: mixed
    weight: 5000.0
    metric: rmse
    name: esp_mixed
```

Then use: `python trainer.py ... --loss-config loss_config.yaml`

**Legacy Options** (still supported):
- `--dipole-source physnet` (default): Single dipole source
- `--dipole-source dcmnet`: Use DCMNet dipole
- `--dipole-weight`: Weight for legacy single dipole term
- `--esp-weight`: Weight for legacy single ESP term

### ESP Grid Point Filtering (Atomic Radius-Based, Default ON)

**By default**, the trainer excludes ESP grid points too close to atoms using **element-specific atomic radii**:
- Points very close to nuclei have near-singular Coulomb potentials
- These extreme values can dominate the loss and destabilize training
- Using atomic radii accounts for different element sizes (H vs C vs O)

**Default behavior:** Exclude grid points within **2 × covalent_radius** of any atom

**Example cutoff distances:**
- H (r = 0.31 Å): exclude if d < 0.62 Å
- C (r = 0.76 Å): exclude if d < 1.52 Å  
- O (r = 0.66 Å): exclude if d < 1.32 Å

**Add extra fixed distance** (on top of radius-based):
```bash
python trainer.py ... --esp-min-distance 0.5  # Adds 0.5 Å to radius-based cutoff
```

**How it works:**
1. For each atom, compute cutoff = 2 × covalent_radius
2. For each grid point, check distance to all atoms
3. Exclude if within cutoff of ANY atom
4. Optional: Add fixed `--esp-min-distance` for extra margin
5. Filtering applied to both training and validation RMSE
6. Plotting uses all points (no filtering) for visualization

**Why 2× radius?** 
- 1× radius is inside the electron cloud (extreme ESP)
- 2× radius is near the VDW surface (reasonable ESP values)
- Element-specific: H is smaller than C/O

### Learnable Charge Orientation Mixing ⭐ **NEW**

**Automatically enabled** when using `--dipole-loss-sources mixed` or `--esp-loss-sources mixed`.

**What it does:**
- Neural network (`ChargeOrientationMixer`) learns per-molecule mixing weights (λ) from charge distributions
- Uses E(3)-equivariant features: weighted spherical harmonics + radial statistics
- Combines PhysNet (atom-centered) and DCMNet (distributed multipole) predictions
- Outputs: `mixed = λ·DCMNet + (1-λ)·PhysNet` where λ is learned per-molecule

**Key features:**
- **E(3)-equivariant**: Respects rotational symmetry using spherical harmonics
- **Orientation-aware**: Encodes 3D charge distributions via weighted SH coefficients
- **Per-molecule adaptation**: λ varies based on molecular charge geometry
- **Automatic scaling**: λ ∈ [0,1] via sigmoid activation

**Architecture:**
```
PhysNet charges ──┐
                  ├→ [Orientation Features (SH + radial)] → MLP → [λ_dipole, λ_ESP]
DCMNet charges ──┘
```

**Usage example:**
```bash
# Train with mixed dipole and ESP
python trainer.py \
  --train-efd ../physnet_train_charges/energies_forces_dipoles_train.npz \
  --train-esp ../dcmnet_train/grids_esp_train.npz \
  --valid-efd ../physnet_train_charges/energies_forces_dipoles_valid.npz \
  --valid-esp ../dcmnet_train/grids_esp_valid.npz \
  --dipole-loss-sources mixed \
  --esp-loss-sources mixed \
  --dipole-weight 1000.0 \
  --esp-weight 100000.0 \
  --epochs 1000 \
  --name co2_mixed
```

**Expected output:**
```
Dipole loss terms:
  - mixed: source=mixed, metric=l2, weight=1000.0
ESP loss terms:
  - mixed: source=mixed, metric=l2, weight=100000.0

Training...
    lambda_dipole_mean: 0.643
    lambda_esp_mean: 0.521
```

**When λ ≈ 0**: Model favors PhysNet predictions  
**When λ ≈ 1**: Model favors DCMNet predictions  
**When 0 < λ < 1**: Model learns an optimal blend

### Exponential Moving Average (EMA) ⭐ **NEW**

**Automatic smoothing** of validation parameters for better generalization.

**What it does:**
- Maintains exponentially weighted average of all parameters: `EMA = 0.999 × EMA + 0.001 × params`
- All validation evaluations use EMA parameters (not current params)
- Checkpoints save EMA-averaged parameters
- Reduces validation noise from stochastic gradient updates

**Benefits:**
- More stable validation curves
- Better generalization (standard in modern deep learning)
- Smoother convergence monitoring

**Configuration:**
Currently hardcoded to `ema_decay = 0.999` (adjust in `trainer.py` if needed)

**No extra flags needed** - enabled automatically!

### Energy Mixing (Legacy/Experimental)

**`--mix-coulomb-energy`**: Mix PhysNet energy with DCMNet Coulomb energy using a single global λ.

**Different from charge mixing** - this mixes energies, not dipole/ESP:
- PhysNet predicts total energy
- DCMNet distributed charges → compute explicit Coulomb energy
- Learn mixing: `E_total = E_physnet + λ × E_coulomb(DCMNet)`

**Usage:**
```bash
python trainer.py ... --mix-coulomb-energy
```

**Note**: This is a separate feature from the charge orientation mixing described above.

## Training Output

The training script provides detailed metrics with **validation set statistics** for context.

**Notes:** 
- Energies are **formation energies** (relative to isolated atoms) by default. Use `--no-subtract-atom-energies` to train on absolute energies.
- ESP grid points < 1.0 Å from atoms are **excluded from loss** by default (prevents singularities). Use `--esp-min-distance 0` to disable.

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
    MAE Energy: 2.224 eV  (51.3 kcal/mol) [μ=-5.12, σ=1.78 eV]  # Formation energies!
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

### Charge Diagnostics

Every 10th print interval (and epoch 1), the trainer prints charge statistics:

```
💡 Charge Diagnostics (first validation sample):
  PhysNet charges: [-0.3245, +0.3245] e, sum=0.0000
  DCMNet charges:  [-0.0821, +0.0652] e, sum=-0.0001
  DCMNet charge signs: 8 positive, 4 negative (out of 12)
  ESP (DCMNet): [-0.0234, 0.1567] Ha/e
  ESP (PhysNet): [-0.2106, 0.1531] Ha/e
  ESP (Target):  [-0.0380, 0.7116] Ha/e
```

**⚠️ Warning Signs:**
- **DCMNet ESP only positive:** DCMNet may be predicting all positive charges (bug!)
- **Charge sum ≠ 0:** Monopole or total charge constraint not enforced
- **All same sign:** Check if monopole constraint weight is sufficient

**Diagnostic Tool:** Run `diagnose_charges.py` for detailed per-molecule charge analysis:
```bash
python diagnose_charges.py --checkpoint path/to/best_params.pkl --n-samples 5
```

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

**With Mixed Modes** (when using `--dipole-loss-sources mixed`):
- Additional scatter plots for mixed dipole and mixed ESP
- λ value histograms showing learned mixing weights
- Comparison of raw vs mixed prediction quality

**Features:**
- Scatter plots: Perfect prediction line (red dashed), MAE/RMSE, R² for ESP
- Histograms: Error distribution, zero-error reference line, standard deviation
- Color-coded by model: PhysNet (blue/default), DCMNet (orange), Mixed (red), ESP variants (green/purple)
- Consistent scales for easy comparison

### ESP Examples (2D + 3D)
Creates detailed visualizations for individual molecules:

**2D Plots** (`esp_example_N.png`):
- 2×3 grid comparing True, PhysNet, and DCMNet ESP
- Row 1: True ESP, DCMNet ESP, DCMNet Error
- Row 2: Empty, PhysNet ESP, PhysNet Error
- **With mixed modes**: Additional panels for Mixed ESP and Mixed Error
- Color-coded scatter plots with RMSE and R²
- Shared color scales for easy comparison

**3D Spatial Plots** (`esp_example_N_3d.png`): ⭐ **Enhanced for mixed modes**
- 3D scatter plots showing ESP on VDW surface in real space
- Three panels: True ESP, PhysNet ESP, DCMNet ESP
- **With mixed modes**: Additional panel for Mixed ESP in 3D space
- Points positioned at actual grid coordinates (X, Y, Z in Ångström)
- Color indicates ESP value (same **symmetric** scale centered at 0 across all panels)
- Visualize how well ESP is reproduced in 3D space

**Multi-Scale Error Plots** (`esp_example_N_error_scales.png`):
- 3 rows × 2 columns showing ESP errors at different percentile cutoffs
- Row 1: 100% range (full error distribution)
- Row 2: 95th percentile (removes outliers)
- Row 3: 75th percentile (focuses on typical errors)
- Left column: PhysNet errors, Right column: DCMNet errors
- **With mixed modes**: Optional 3rd column for Mixed ESP errors
- All scales **symmetric around 0** for balanced visualization
- Helps identify error patterns without outliers dominating the colorscale

**Distributed Charge Visualization** (`charges_example_N.png`):
- Shows DCMNet's learned distributed charges around atoms
- **2×2 grid:** 3D view + XY/XZ/YZ projections
- Black spheres = atoms (labeled with atomic number)
- Colored small spheres = distributed charges
- Color scale: **symmetric around 0** (red = positive, blue = negative)
- Visualize how DCMNet distributes charge around each atom

**Per-Atom Charge Detail** (`charges_detail_N.png`):
- One panel per atom showing its distributed charges
- Atom at origin (0,0) with charges positioned relative to it
- Lines connect atom to each distributed charge
- Each charge labeled with its magnitude
- Shows total charge per atom (Σq)
- Helps understand local charge distribution patterns

Options:
- `--plot-samples N`: Number of validation samples to include in scatter plots (default: 100)
- `--plot-esp-examples N`: Number of detailed ESP examples to create (default: 2)
  - Each example generates: 2D ESP plots, 3D ESP plots, error scale plots, charge distribution, and per-atom charge detail

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
<<<<<<< HEAD

## Toolbox Overview

This directory ships a full workflow around the joint PhysNet–DCMNet model. The summaries below collect the most-used entry points and sample invocations. All commands assume you run them from `examples/co2/dcmnet_physnet_train/` unless stated otherwise.

### Training & Evaluation

- `trainer.py` — joint training of PhysNet + DCMNet with flexible loss weights and plotting hooks.
  ```bash
  python trainer.py \
    --train-efd ../physnet_train_charges/energies_forces_dipoles_train.npz \
    --train-esp ../dcmnet_train/grids_esp_train.npz \
    --valid-efd ../physnet_train_charges/energies_forces_dipoles_valid.npz \
    --valid-esp ../dcmnet_train/grids_esp_valid.npz \
    --epochs 100 --batch-size 10 --name co2_joint
  ```
- `train_default.sh` — quick-start shell wrapper that wires in default datasets and hyperparameters:
  ```bash
  ./train_default.sh
  ```
- `run_full_evaluation.sh` — end-to-end metrics + plots (wraps `evaluate_splits.py` and `plot_evaluation_results.R`).
  ```bash
  ./run_full_evaluation.sh \
    --checkpoint ./ckpts/model \
    --train-efd ./data/train_efd.npz --valid-efd ./data/valid_efd.npz \
    --train-esp ./data/train_esp.npz --valid-esp ./data/valid_esp.npz \
    --output-dir ./evaluation
  ```
- `evaluate_splits.py` — programmatic error reports and bond/angle features for train/valid/test.
  ```bash
  python evaluate_splits.py \
    --checkpoint ./ckpts/model \
    --train-efd ./data/train_efd.npz --valid-efd ./data/valid_efd.npz \
    --train-esp ./data/train_esp.npz --valid-esp ./data/valid_esp.npz \
    --output-dir ./evaluation
  ```
- `eval_calculator.py` — single-molecule evaluator with ESP surface plots and dipole breakdowns.
  ```bash
  python eval_calculator.py \
    --checkpoint ./ckpts/model \
    --molecule CO2 \
    --esp-data ../dcmnet_train/grids_esp_valid.npz --esp-index 0 \
    --output-dir ./evaluation_results
  ```

### Dynamics & Spectroscopy Workflows

- `dynamics_calculator.py` — ASE-based optimization, vibrational analysis, IR spectra, and MD drivers.
  ```bash
  python dynamics_calculator.py \
    --checkpoint ./ckpts/model --molecule CO2 \
    --frequencies --ir-spectra --optimize
  ```
- `ase_calculator.py` — lightweight wrapper to expose the joint model as an ASE calculator for geometry, IR, and dipole studies.
  ```bash
  python ase_calculator.py \
    --checkpoint ./ckpts/model --molecule CO2 --optimize --frequencies --ir-spectra \
    --output-dir ./ase_results/co2
  ```
- `run_production_md.py` — conservative production MD runner with optional IR post-processing.
  ```bash
  python run_production_md.py \
    --checkpoint ./ckpts/model --molecule CO2 \
    --nsteps 1000000 --analyze-ir --output-dir ./md_ir_long
  ```
- `compute_response_properties.py` — IR/Raman/VCD/SFG spectra from saved MD trajectories.
  ```bash
  python compute_response_properties.py \
    --trajectory ./md_ir_long/trajectory.npz \
    --checkpoint ./ckpts/model --compute-ir --compute-raman --output-dir ./all_spectra
  ```
- `spectroscopy_suite.py` — orchestration script for full scans across temperatures, ensembles, and spectroscopy modes.
  ```bash
  python spectroscopy_suite.py \
    --checkpoint ./ckpts/model --molecule CO2 --quick-analysis
  ```
- `jaxmd_dynamics.py` — GPU-accelerated JAX MD integrator for long, fine-grained trajectories.
  ```bash
  python jaxmd_dynamics.py \
    --checkpoint ./ckpts/model --molecule CO2 \
    --ensemble nvt --temperature 300 --nsteps 100000 --output-dir ./jaxmd_nvt
  ```
- `convert_npz_to_traj.py` and `extract_ir_from_trajectory.py` — utilities to convert NPZ runs to ASE format and back out IR spectra without retracing the MD.
  ```bash
  python convert_npz_to_traj.py md_ir_long/trajectory.npz --output md_ir_long/trajectory.traj
  python extract_ir_from_trajectory.py --trajectory md_ir_long/trajectory.npz --output-dir md_ir_long
  ```
- `run_ase_example.sh` — handy presets showcasing optimization, frequency, and IR pipelines with a single command.
  ```bash
  ./run_ase_example.sh
  ```

### Visualization Utilities

- `visualize_normal_modes.py` — plots and trajectories for vibrational eigenmodes from ASE caches.
  ```bash
  python visualize_normal_modes.py --raman-dir ./raman_analysis --output-dir ./mode_animations
  ```
- `visualize_charges_during_vibration.py` — snapshots of distributed multipoles along a selected mode.
  ```bash
  python visualize_charges_during_vibration.py \
    --checkpoint ./ckpts/model --raman-dir ./raman_analysis --mode-index 7 --n-frames 20
  ```
- `visualize_all_modes.py` — batch driver that loops over every mode and reuses the charge visualization helper.
  ```bash
  python visualize_all_modes.py --checkpoint ./ckpts/model --raman-dir ./raman_analysis --n-frames 20
  ```
- `remake_raman_plot.py` — rebuild Raman figures (single or multi-wavelength) from cached results.
  ```bash
  python remake_raman_plot.py --raman-dir ./raman_analysis --multi-wavelength
  ```

### Active Learning & Data Preparation

- `active_learning_manager.py` — manage, deduplicate, and export challenging MD structures for QM follow-up.
  ```bash
  python active_learning_manager.py \
    --export-xyz --source "./md_runs/*/active_learning" --output ./qm_candidates --max-structures 100
  ```
- `prepare_qm_inputs.py` — build ORCA/Gaussian/Q-Chem/Psi4 inputs from XYZ ensembles.
  ```bash
  python prepare_qm_inputs.py \
    --xyz-dir ./qm_candidates --qm-software orca --method PBE0 --basis def2-TZVP --output ./orca_inputs
  ```
- `convert_molpro_to_training.py` — turn Molpro cube/log output into NPZ training packs (with optional merge).
  ```bash
  python convert_molpro_to_training.py \
    --molpro-outputs ./logs/*.out --cube-dir ./cubes --output ./qm_training_data.npz
  ```
- `create_molpro_jobs_from_al.py` — generate Molpro input decks and SLURM arrays from active-learning structures.
  ```bash
  python create_molpro_jobs_from_al.py \
    --al-structures "./md_*/active_learning/*.npz" --output-dir ./molpro_al_jobs --ntasks 16 --mem 132G
  ```
- `gas_phase_calculator.py` and `gas_phase_jaxmd.py` — simulate gas-phase ensembles; see `GAS_PHASE_README.md` for workflows.
  ```bash
  python gas_phase_calculator.py \
    --checkpoint ./ckpts/model --n-molecules 10 --temperature 300 --pressure 1.0 --output-dir ./gas_phase
  ```

### Diagnostics & Utilities

- `diagnose_charges.py` — inspect PhysNet/DCMNet charge balance issues on validation samples.
  ```bash
  python diagnose_charges.py \
    --checkpoint ./ckpts/model --valid-esp ../dcmnet_train/grids_esp_valid.npz \
    --valid-efd ../physnet_train_charges/energies_forces_dipoles_valid.npz --n-samples 5
  ```
- `align_esp_frames.py` — detect and correct ESP/geometry frame misalignment.
  ```bash
  python align_esp_frames.py \
    --checkpoint ./ckpts/model --valid-efd ../physnet_train_charges/energies_forces_dipoles_valid.npz \
    --valid-esp ../dcmnet_train/grids_esp_valid.npz
  ```
- `inspect_checkpoint.py` / `save_config_from_checkpoint.py` — recover model configuration metadata from historical checkpoints.
  ```bash
  python inspect_checkpoint.py --checkpoint ./ckpts/model/best_params.pkl
  python save_config_from_checkpoint.py --checkpoint-dir ./ckpts/model --natoms 60 --max-atomic-number 28
  ```
- `remove_rotations_and_compute_ir.py`, `match_harmonic_to_md.py`, `interpolate_qm_charges.py`, `analyze_ir_spectrum.py` — targeted analysis utilities for refining IR/Raman comparisons and cleaning MD trajectories.

### External Dependencies

- **ASE** for geometry, vibrational analysis, and MD backends (`pip install ase`).
- **matplotlib** for all plotting scripts (`pip install matplotlib`).
- **R + tidyverse packages** for `plot_evaluation_results.R` when using the evaluation pipeline.
- **JAX MD** (`pip install jax-md`) for GPU-accelerated dynamics.
=======
>>>>>>> 284d4671 (asdf)
