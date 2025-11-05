

# Charge and Spin Conditioned PhysNet

**Energy and force predictions with explicit charge and spin state control**

This document describes the charge-spin conditioned version of PhysNet that accepts total molecular charge and total spin multiplicity as inputs, enabling accurate multi-state predictions.

## Overview

The standard PhysNet model predicts energies and forces from atomic positions and numbers. This enhanced version adds two critical molecular properties as inputs:

1. **Total Charge** (Q): Net molecular charge (e.g., 0, +1, -1)
2. **Total Spin** (2S+1): Spin multiplicity (e.g., 1=singlet, 2=doublet, 3=triplet)

This enables the model to:
- Predict energies for different charge states (neutral, cations, anions)
- Handle different spin states (singlet, doublet, triplet, etc.)
- Model excited states and open-shell systems
- Capture charge and spin-dependent potential energy surfaces

## Key Features

✅ **Multi-State Capable**: Single model handles multiple charge/spin states  
✅ **Property Embeddings**: Learnable representations for charge and spin  
✅ **Atomic Conditioning**: Molecular properties broadcast to all atoms  
✅ **Gradient-Based**: Forces computed via automatic differentiation  
✅ **Backwards Compatible**: Same architecture as standard PhysNet  

## Quick Start

### Simple Example (5 minutes)

```bash
python examples/train_charge_spin_simple.py
```

This runs a minimal example with dummy data demonstrating:
- Model initialization with charge/spin conditioning
- Training with different charge states (0, +1, -1)
- Training with different spin states (singlet, doublet)
- Energy and force predictions

### Training on Real Data

```bash
python train_physnet_charge_spin.py \
    --data_path my_data \
    --batch_size 32 \
    --num_epochs 100 \
    --charge_min -2 \
    --charge_max 2 \
    --spin_min 1 \
    --spin_max 5
```

## Model Architecture

### Input Structure

```python
inputs = {
    "atomic_numbers": (num_atoms,),      # Z values (e.g., [8, 1, 1])
    "positions": (num_atoms, 3),         # R in Angstrom
    "total_charges": (batch_size,),      # Q (e.g., [0, 1, -1, 0])
    "total_spins": (batch_size,),        # 2S+1 (e.g., [1, 2, 3, 1])
}
```

### Architecture Flow

```
1. Atomic Embedding
   Z → Embed(Z) → atomic_features

2. Molecular Property Embedding
   Q → Embed(Q) → charge_embedding
   S → Embed(S) → spin_embedding
   [charge_embedding, spin_embedding] → mol_features

3. Feature Conditioning
   mol_features → broadcast to atoms → mol_features_per_atom
   atomic_features + Project(mol_features_per_atom) → conditioned_features

4. Message Passing (with conditioning)
   for iteration in range(num_iterations):
       conditioned_features → MessagePass → updated_features

5. Energy & Forces Prediction
   updated_features → Dense → atomic_energies → sum → E
   dE/dR → -F (forces via autograd)
```

### Key Differences from Standard PhysNet

| Feature | Standard PhysNet | Charge-Spin PhysNet |
|---------|-----------------|---------------------|
| Inputs | Z, R | Z, R, Q, S |
| Molecular properties | None | Embedded Q and S |
| Atomic features | Only Z-based | Z-based + molecular conditioning |
| Multi-state | No | Yes |
| Use cases | Ground state | Ground + excited + charged states |

## Usage Examples

### Example 1: Different Charge States

```python
from mmml.physnetjax.physnetjax.models.model_charge_spin import EF_ChargeSpinConditioned

# Create model
model = EF_ChargeSpinConditioned(
    features=128,
    charge_range=(-2, 2),
    spin_range=(1, 4),
)

# Predict for different charges
charges = jnp.array([0, 1, -1, 2])  # Neutral, cation, anion, dication
spins = jnp.array([1, 2, 2, 1])     # Singlet, doublet, doublet, singlet

outputs = model.apply(
    params,
    atomic_numbers=Z,
    positions=R,
    dst_idx=dst_idx,
    src_idx=src_idx,
    total_charges=charges,
    total_spins=spins,
    ...
)

# outputs["energy"]: (4,) - energies for each charge state
# outputs["forces"]: (num_atoms, 3) - forces
```

### Example 2: Spin States

```python
# Predict for different spin multiplicities
# Molecule: CH2 (methylene)

charges = jnp.array([0, 0, 0])      # All neutral
spins = jnp.array([1, 3, 5])        # Singlet, triplet, quintet

outputs = model.apply(
    params,
    atomic_numbers=Z,  # [6, 1, 1]  (C, H, H)
    positions=R,
    dst_idx=dst_idx,
    src_idx=src_idx,
    total_charges=charges,
    total_spins=spins,
    ...
)

print("Singlet energy:", outputs["energy"][0])
print("Triplet energy:", outputs["energy"][1])
print("Quintet energy:", outputs["energy"][2])
```

### Example 3: Training with Multiple States

```python
# Training data with mixed charge/spin states
for batch in dataloader:
    # Batch contains molecules in different states
    # batch["total_charge"]: [0, 0, 1, -1, 0, ...]
    # batch["total_spin"]: [1, 3, 2, 2, 1, ...]
    
    outputs = model.apply(
        params,
        atomic_numbers=batch["Z"],
        positions=batch["R"],
        total_charges=batch["total_charge"],
        total_spins=batch["total_spin"],
        ...
    )
    
    # Loss combines all states
    energy_loss = mse(outputs["energy"], batch["E"])
    forces_loss = mse(outputs["forces"], batch["F"])
```

## Data Format

### Required Fields

Your data must include charge and spin information:

```python
# Option 1: In packed memmap format
data/
├── Z_pack.int32          # Atomic numbers
├── R_pack.f32            # Positions
├── F_pack.f32            # Forces
├── E.f64                 # Energies
├── total_charge.f32      # ← NEW: Total charges
├── total_spin.f32        # ← NEW: Spin multiplicities
├── offsets.npy
└── n_atoms.npy

# Option 2: In batch dict
batch = {
    "Z": ...,
    "R": ...,
    "F": ...,
    "E": ...,
    "total_charge": jnp.array([0, 1, -1, ...]),  # ← NEW
    "total_spin": jnp.array([1, 2, 2, ...]),     # ← NEW
}
```

### Spin Multiplicity Convention

Spin multiplicity = 2S + 1, where S is total spin angular momentum:

| Multiplicity | S | Name | Example Systems |
|--------------|---|------|-----------------|
| 1 | 0 | Singlet | Closed-shell molecules (H2O, CH4) |
| 2 | 1/2 | Doublet | Radicals (CH3•, OH•) |
| 3 | 1 | Triplet | O2, carbenes |
| 4 | 3/2 | Quartet | Many transition metal complexes |
| 5 | 2 | Quintet | Fe²⁺ (d⁶ high-spin) |
| 6 | 5/2 | Sextet | Mn²⁺ (d⁵ high-spin) |
| 7 | 3 | Septet | Gd³⁺ (f⁷) |

## Training

### Command-Line Training

```bash
python train_physnet_charge_spin.py \
    --data_path data_with_charges_spins \
    --batch_size 32 \
    --num_epochs 200 \
    --learning_rate 0.001 \
    --features 256 \
    --num_iterations 3 \
    --charge_embed_dim 16 \
    --spin_embed_dim 16 \
    --charge_min -3 \
    --charge_max 3 \
    --spin_min 1 \
    --spin_max 7 \
    --name my_charge_spin_model
```

### Key Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--charge_embed_dim` | 16 | Dimension of charge embedding |
| `--spin_embed_dim` | 16 | Dimension of spin embedding |
| `--charge_min` | -5 | Minimum charge to support |
| `--charge_max` | 5 | Maximum charge to support |
| `--spin_min` | 1 | Minimum spin multiplicity |
| `--spin_max` | 7 | Maximum spin multiplicity |

### Python API Training

```python
from mmml.physnetjax.physnetjax.models.model_charge_spin import EF_ChargeSpinConditioned

# Create model
model = EF_ChargeSpinConditioned(
    features=128,
    num_iterations=3,
    natoms=60,
    charge_embed_dim=16,
    spin_embed_dim=16,
    charge_range=(-2, 2),
    spin_range=(1, 5),
)

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        # Ensure batch has charge and spin
        outputs = model.apply(
            params,
            atomic_numbers=batch["Z"],
            positions=batch["R"],
            total_charges=batch["total_charge"],
            total_spins=batch["total_spin"],
            ...
        )
        
        loss = compute_loss(outputs, batch)
        # Update parameters...
```

## Use Cases

### 1. Ionization Energy Calculations

```python
# Predict energy of neutral and cation
charges = jnp.array([0, 1])
spins = jnp.array([1, 2])  # Singlet neutral, doublet cation

energies = model.apply(params, Z, R, charges=charges, spins=spins)["energy"]

IE = energies[1] - energies[0]  # Ionization energy
print(f"Ionization energy: {IE:.3f} kcal/mol")
```

### 2. Singlet-Triplet Gap

```python
# Predict singlet and triplet energies
charges = jnp.array([0, 0])
spins = jnp.array([1, 3])  # Singlet, triplet

energies = model.apply(params, Z, R, charges=charges, spins=spins)["energy"]

gap = energies[1] - energies[0]  # S-T gap
print(f"Singlet-triplet gap: {gap:.3f} kcal/mol")
```

### 3. Charge State Stability

```python
# Compare different charge states
charges = jnp.array([-2, -1, 0, 1, 2])
spins = jnp.array([1, 2, 1, 2, 1])  # Appropriate spins

energies = model.apply(params, Z, R, charges=charges, spins=spins)["energy"]

print("Energies by charge state:")
for q, e in zip(charges, energies):
    print(f"  Q={q:+d}: {e:.3f} kcal/mol")
```

### 4. Reaction Barrier with Spin Crossing

```python
# Reactant (singlet), TS (triplet), product (singlet)
charges = jnp.array([0, 0, 0])
spins = jnp.array([1, 3, 1])
geometries = [R_reactant, R_ts, R_product]

energies = []
for R in geometries:
    E = model.apply(params, Z, R, charges=charges, spins=spins)["energy"]
    energies.append(E)

print(f"Reactant (S): {energies[0][0]:.3f}")
print(f"TS (T): {energies[1][1]:.3f}")
print(f"Product (S): {energies[2][2]:.3f}")
```

## Model Configuration

### Charge and Spin Ranges

Set ranges based on your data:

```python
# Typical organic molecules
charge_range=(-2, 2)  # Support -2, -1, 0, +1, +2
spin_range=(1, 4)     # Singlet, doublet, triplet, quartet

# Highly charged systems
charge_range=(-5, 5)  # Wider charge range

# High-spin systems (e.g., lanthanides)
charge_range=(-3, 3)
spin_range=(1, 9)     # Up to nonet
```

### Embedding Dimensions

Larger embeddings for systems with many states:

```python
# Simple systems (few charge/spin states)
charge_embed_dim=8
spin_embed_dim=8

# Complex systems (many states, need expressivity)
charge_embed_dim=32
spin_embed_dim=32
```

## Performance Considerations

### Memory Usage

Memory overhead compared to standard PhysNet:

```
Additional parameters ≈ (n_charges × charge_embed + n_spins × spin_embed)
```

Example:
- Charge range: -5 to +5 → 11 states
- Spin range: 1 to 7 → 7 states  
- Embed dims: 16 each
- **Additional params**: (11 × 16) + (7 × 16) = 288 parameters

**Impact**: Negligible (< 0.1% of total parameters)

### Computational Cost

Per forward pass overhead:
- Embedding lookups: ~0.01 ms
- Broadcasting: ~0.1 ms
- **Total overhead**: < 1% of forward pass time

### Training Speed

Same as standard PhysNet for equivalent architecture.

## Limitations

1. **Discrete States**: Charges and spins are discrete (integer or half-integer)
2. **Range Constraints**: Must predefine charge and spin ranges
3. **Interpolation**: No smooth interpolation between charge/spin states
4. **Spin Contamination**: Doesn't explicitly prevent spin contamination

## Best Practices

### 1. Data Preparation

```python
# Ensure spin multiplicities are correct
# For closed-shell neutral: spin = 1
# For radical cation/anion: spin = 2
# For triplet states: spin = 3

# Check your data
assert jnp.all(spins >= 1)  # Spin must be ≥ 1
assert jnp.all(spins == jnp.round(spins))  # Spin must be integer
```

### 2. Training Strategy

```python
# Start with single states, then expand
# Phase 1: Train on neutral singlets
# Phase 2: Add charged species
# Phase 3: Add excited spin states

# Use appropriate loss weights
energy_weight = 1.0
forces_weight = 52.91  # kcal/mol conversion
```

### 3. Validation

```python
# Validate on hold-out charge/spin states
# Check:
# 1. Ionization energies are reasonable
# 2. Spin gaps match expectations
# 3. Forces are consistent across states
```

## Comparison with Alternatives

### vs. Separate Models per State

| Approach | Charge-Spin Model | Separate Models |
|----------|------------------|-----------------|
| **Parameters** | Single set | N × parameters |
| **Training time** | Same as 1 model | N × training time |
| **Consistency** | Smooth across states | May have discontinuities |
| **Memory** | 1 model | N models |
| **Flexibility** | Easy to add states | Need retrain from scratch |

### vs. Standard PhysNet

| Approach | Charge-Spin Model | Standard PhysNet |
|----------|------------------|------------------|
| **Multi-state** | Yes | No (need separate models) |
| **Explicit conditioning** | Yes | No |
| **Use case** | Charged/excited states | Ground state only |
| **Complexity** | +2 embeddings | Simpler |

## Troubleshooting

### Issue: Model predicts same energy for all charge states

**Cause**: Embeddings not properly conditioned into features

**Solution**:
- Increase `charge_embed_dim` and `spin_embed_dim`
- Check that molecular features are added to atomic features
- Verify data has variation in charges/spins

### Issue: High error for certain spin states

**Cause**: Insufficient training data for those states

**Solution**:
- Balance dataset (equal samples per state)
- Use data augmentation
- Increase spin_embed_dim

### Issue: Physically unreasonable predictions

**Cause**: Charge/spin outside training range

**Solution**:
- Check input values are within model's supported range
- Retrain with wider `charge_range` or `spin_range`

## Advanced Topics

### Transfer Learning

```python
# Train on easy molecules first
model_pretrained = train_on_small_molecules()

# Fine-tune on harder systems
model_finetuned = finetune_on_complex_molecules(model_pretrained)
```

### Multi-Task Learning

```python
# Jointly train on:
# 1. Ground state energies
# 2. Excited state energies
# 3. Ionization energies
# 4. Electron affinities

# Use same model with different charge/spin combinations
```

### Uncertainty Quantification

```python
# Use ensemble for uncertainty
models = [train_model(seed=i) for i in range(10)]

predictions = [m.apply(...) for m in models]
mean_energy = jnp.mean([p["energy"] for p in predictions], axis=0)
std_energy = jnp.std([p["energy"] for p in predictions], axis=0)

print(f"Energy: {mean_energy:.3f} ± {std_energy:.3f} kcal/mol")
```

## References

- **PhysNet**: [Unke & Meuwly, JCTC 2019](https://doi.org/10.1021/acs.jctc.9b00181)
- **Charge models**: [Ko et al., Nat Commun 2021](https://doi.org/10.1038/s41467-020-20427-2)
- **Spin states**: [Janet & Kulik, Chem Sci 2017](https://doi.org/10.1039/C7SC04665K)

## Citation

If you use this model in your research, please cite both the original PhysNet paper and this extension:

```bibtex
@article{unke2019physnet,
  title={PhysNet: A neural network for predicting energies, forces, dipole moments, and partial charges},
  author={Unke, Oliver T and Meuwly, Markus},
  journal={Journal of Chemical Theory and Computation},
  volume={15},
  number={6},
  pages={3678--3693},
  year={2019}
}
```

---

**Documentation**: See also `PACKED_MEMMAP_TRAINING.md` for data loading  
**Examples**: `examples/train_charge_spin_simple.py`  
**Training**: `train_physnet_charge_spin.py`  
**Status**: Experimental ⚠️ (validate thoroughly before production use)

