# Hybrid Calculator Force Debugging Guide

## Observed Issue
The hybrid calculator is producing incorrect forces compared to reference forces:
- Many atoms have **zero forces** in computed but **non-zero** in reference (atoms 2, 3, 5, 7, 11, 12, etc.)
- Significant magnitude differences even where forces are non-zero
- Some atoms have forces in computed but zero in reference

## Critical Failure Modes to Check

### 1. **Switching Function Application to Forces** ⚠️ CRITICAL
**Location**: `mmml/utils/hybrid_optimization.py:610-614`, `mmml/pycharmmInterface/mmml_calculator.py:1911-1954`

**Problem**: Forces are being **scaled** by switching functions rather than computing proper gradients:
```python
ml_forces = ml_forces * ml_scale  # WRONG!
mm_forces = mm_forces * mm_scale  # WRONG!
```

**Why this is wrong**: Switching functions depend on positions, so forces should include gradients of the switching function:
```
F_total = -d/dR [E_ML * s_ML(R) + E_MM * s_MM(R)]
       = -[dE_ML/dR * s_ML(R) + E_ML * ds_ML/dR + dE_MM/dR * s_MM(R) + E_MM * ds_MM/dR]
       = F_ML * s_ML(R) + E_ML * (-ds_ML/dR) + F_MM * s_MM(R) + E_MM * (-ds_MM/dR)
```

**What to check**:
- Are switching function gradients (`ds_ML/dR`, `ds_MM/dR`) being computed?
- Are energy terms multiplied by switching gradients being added?
- Verify switching functions are applied **before** force computation, not after

**Debugging**:
```python
# Check if switching gradients are computed
print("ML scale:", ml_scale)
print("MM scale:", mm_scale)
print("ML energy:", ml_energy)
print("MM energy:", mm_energy)
# Compute switching gradients manually
ml_switch_grad = jax.grad(lambda R: ml_switch_simple(...))(R)
mm_switch_grad = jax.grad(lambda R: mm_switch_simple(...))(R)
print("ML switch grad:", ml_switch_grad)
print("MM switch grad:", mm_switch_grad)
```

---

### 2. **Atom Index Mapping Errors** ⚠️ CRITICAL
**Location**: `mmml/pycharmmInterface/mmml_calculator.py:1262-1315`

**Problem**: ML forces are computed for a subset of atoms (monomers) but must be mapped to full system indices.

**What to check**:
- Are `monomer_atom_indices` correct? (lines 1262-1296)
- Do indices match between ML force computation and mapping?
- Are forces being added to correct positions or overwritten?
- Are there atoms outside monomer indices that should have forces?

**Debugging**:
```python
print("n_atoms:", n_atoms)
print("n_monomers:", n_monomers)
print("ATOMS_PER_MONOMER:", ATOMS_PER_MONOMER)
print("monomer_atom_indices:", monomer_atom_indices)
print("ML forces shape:", ml_forces.shape)
print("Expected ML atoms:", n_monomers * ATOMS_PER_MONOMER)
print("Atoms with zero forces:", np.where(np.all(computed_forces == 0, axis=1))[0])
print("Atoms that should have forces:", np.where(np.any(ref_forces != 0, axis=1))[0])
```

---

### 3. **Missing Force Contributions**
**Location**: Multiple locations

**Problem**: Some atoms have zero forces when they should have non-zero forces.

**Possible causes**:
- **MM forces not computed for all atoms**: Check if MM forces are computed for all atom pairs
- **ML forces only computed for monomers**: Dimer forces might not be properly mapped
- **Switching functions zeroing out contributions**: Check if switching scales are accidentally zero
- **Force masking**: Check if atom masks or batch masks are incorrectly filtering forces

**What to check**:
- Are MM forces computed for **all** atom pairs, not just monomer pairs?
- Are dimer ML forces being added correctly?
- Check switching function values for atoms with zero forces
- Verify no atom masks are applied incorrectly

**Debugging**:
```python
# Check individual contributions
print("ML forces (raw):", ml_forces)
print("MM forces (raw):", mm_forces)
print("ML scale per atom:", ml_scale)  # If per-atom
print("MM scale per atom:", mm_scale)  # If per-atom
print("Atoms in monomers:", monomer_atom_indices)
print("Atoms NOT in monomers:", [i for i in range(n_atoms) if i not in monomer_atom_indices])
```

---

### 4. **Shape Mismatches and Padding**
**Location**: `mmml/pycharmmInterface/mmml_calculator.py:1272-1296, 1337-1343`

**Problem**: Force arrays may have incorrect shapes, leading to padding/truncation errors.

**What to check**:
- Do ML forces have shape `(n_monomers * ATOMS_PER_MONOMER, 3)`?
- Do MM forces have shape `(n_atoms, 3)`?
- Is padding/truncation happening correctly?
- Are forces being added with `.at[...].add()` or overwritten?

**Debugging**:
```python
print("ML forces shape:", ml_forces.shape)
print("MM forces shape:", mm_forces.shape)
print("Output forces shape:", outputs["out_F"].shape)
print("n_atoms:", n_atoms)
# Check if padding is correct
if mm_F.shape[0] < n_atoms:
    print("MM forces padded from", mm_F.shape[0], "to", n_atoms)
```

---

### 5. **Switching Function Distance Calculation**
**Location**: `mmml/utils/hybrid_optimization.py:565-604`

**Problem**: Switching functions may use incorrect distance metrics (COM distance vs pair distances).

**What to check**:
- Is switching applied per-pair or globally?
- Are pair distances computed correctly?
- Is COM distance used when pair distances should be used (or vice versa)?
- Are cutoff parameters correct (`ml_cutoff`, `mm_switch_on`, `mm_cutoff`)?

**Debugging**:
```python
# Check distances used for switching
pair_distances = jnp.linalg.norm(R[pair_idx_atom_atom[:, 0]] - R[pair_idx_atom_atom[:, 1]], axis=1)
print("Pair distances:", pair_distances)
print("Min distance:", jnp.min(pair_distances))
print("Max distance:", jnp.max(pair_distances))
print("ML cutoff:", ml_cutoff_val)
print("MM switch on:", mm_switch_on_val)
print("MM cutoff:", mm_cutoff_val)
# Check switching values
ml_scales = ml_switch_simple(pair_distances, ml_cutoff_val, mm_switch_on_val)
mm_scales = mm_switch_simple(pair_distances, mm_switch_on_val, mm_cutoff_val)
print("ML scales:", ml_scales)
print("MM scales:", mm_scales)
```

---

### 6. **Force Sign Errors**
**Location**: Multiple locations

**Problem**: Forces may have incorrect signs (energy gradients vs forces).

**What to check**:
- Are forces computed as `-grad(E)` or `+grad(E)`?
- Are MM forces computed correctly: `mm_forces = -jax.grad(mm_energy_fn)(R)`?
- Are ML forces from model already negative gradients?
- Check conversion factors: `ml_force_conversion_factor`, `ev2kcalmol`

**Debugging**:
```python
# Check force signs
print("ML energy:", ml_energy)
print("MM energy:", mm_energy)
# Compute numerical gradients
def num_grad(fn, R, eps=1e-5):
    grad = np.zeros_like(R)
    for i in range(len(R)):
        for j in range(3):
            R_plus = R.copy()
            R_plus[i, j] += eps
            R_minus = R.copy()
            R_minus[i, j] -= eps
            grad[i, j] = (fn(R_plus) - fn(R_minus)) / (2 * eps)
    return -grad  # Negative gradient = force
```

---

### 7. **NaN/Inf Handling Masking Errors**
**Location**: `mmml/pycharmmInterface/mmml_calculator.py:1298-1302, 1333-1335, 1350-1352`

**Problem**: NaN/Inf checks may be zeroing out valid forces.

**What to check**:
- Are forces being set to zero incorrectly?
- Are there actual NaN/Inf values, or is the check too aggressive?
- Check if `jnp.isfinite()` is working correctly

**Debugging**:
```python
# Check for NaN/Inf before and after filtering
print("ML forces before filtering:", ml_forces)
print("Has NaN:", jnp.any(jnp.isnan(ml_forces)))
print("Has Inf:", jnp.any(jnp.isinf(ml_forces)))
ml_forces_filtered = jnp.where(jnp.isfinite(ml_forces), ml_forces, 0.0)
print("ML forces after filtering:", ml_forces_filtered)
print("Forces zeroed:", jnp.sum(ml_forces_filtered == 0) - jnp.sum(ml_forces == 0))
```

---

### 8. **MM Force Computation Issues**
**Location**: `mmml/pycharmmInterface/mmml_calculator.py:1147-1203`

**Problem**: MM forces may not be computed correctly, especially with switching.

**What to check**:
- Is `mm_forces = -jax.grad(mm_energy_fn)(R)` correct?
- Are switching functions applied in energy computation before gradient?
- Are pair indices (`pair_idx_atom_atom`) correct?
- Are charges and LJ parameters correct?

**Debugging**:
```python
# Check MM energy computation
mm_E = calculate_mm_energy(positions)
print("MM energy:", mm_E)
# Check MM forces
mm_F = -jax.grad(calculate_mm_energy)(positions)
print("MM forces:", mm_F)
# Check pair indices
print("Number of pairs:", len(pair_idx_atom_atom))
print("Pair indices sample:", pair_idx_atom_atom[:10])
```

---

### 9. **ML Force Extraction from Model**
**Location**: `mmml/utils/hybrid_optimization.py:494-515`, `mmml/pycharmmInterface/mmml_calculator.py:1248-1321`

**Problem**: ML forces may not be extracted correctly from model output.

**What to check**:
- Are forces extracted from correct model output keys?
- Are forces mapped correctly from batched format to system format?
- Are monomer vs dimer forces combined correctly?
- Check `ml_force_conversion_factor`

**Debugging**:
```python
# Check ML model output
print("ML output keys:", ml_output.keys())
print("ML forces raw shape:", ml_output["forces"].shape)
print("ML energy raw:", ml_output["energy"])
# Check force extraction
print("ML forces extracted:", ml_forces)
print("ML forces shape:", ml_forces.shape)
```

---

### 10. **Energy-Force Consistency**
**Location**: All force computation locations

**Problem**: Forces may not be consistent with energy (not negative gradients).

**What to check**:
- Verify forces are negative gradients of energy
- Check numerical vs analytical gradients
- Ensure energy and forces use same switching functions

**Debugging**:
```python
# Numerical gradient check
def energy_fn(R):
    E, F = compute_energy_forces(R, Z, params_dict)
    return E

R_perturbed = R.copy()
R_perturbed[0, 0] += 1e-5
E_plus = energy_fn(R_perturbed)
E_minus = energy_fn(R - np.array([1e-5, 0, 0]))
num_grad = (E_plus - E_minus) / (2 * 1e-5)
analytical_force = computed_forces[0, 0]
print("Numerical grad:", num_grad)
print("Analytical force:", analytical_force)
print("Difference:", abs(num_grad + analytical_force))  # Should be ~0
```

---

## Recommended Debugging Workflow

1. **Start with switching functions** (most likely culprit):
   - Check if switching gradients are included
   - Verify switching function values
   - Test with switching disabled (`ml_scale=1.0`, `mm_scale=1.0`)

2. **Check atom indexing**:
   - Verify which atoms should have forces
   - Check if ML forces are mapped correctly
   - Ensure all atoms are included in force computation

3. **Verify individual contributions**:
   - Compute ML-only forces
   - Compute MM-only forces
   - Compare each to reference

4. **Check force signs and magnitudes**:
   - Verify force signs are correct
   - Check conversion factors
   - Compare magnitudes to reference

5. **Test edge cases**:
   - Single monomer
   - Two monomers at different distances
   - Atoms outside monomers

---

## Quick Diagnostic Script

```python
import numpy as np
import jax.numpy as jnp

def diagnose_forces(computed_forces, ref_forces, n_atoms, monomer_indices=None):
    """Quick diagnostic for force issues."""
    
    print("=" * 60)
    print("FORCE DIAGNOSTICS")
    print("=" * 60)
    
    # Check shapes
    print(f"\nShapes: computed={computed_forces.shape}, ref={ref_forces.shape}")
    assert computed_forces.shape == ref_forces.shape, "Shape mismatch!"
    
    # Find atoms with zero forces
    computed_zero = np.all(computed_forces == 0, axis=1)
    ref_zero = np.all(ref_forces == 0, axis=1)
    
    print(f"\nAtoms with zero computed forces: {np.where(computed_zero)[0]}")
    print(f"Atoms with zero ref forces: {np.where(ref_zero)[0]}")
    print(f"Atoms with zero computed but non-zero ref: {np.where(computed_zero & ~ref_zero)[0]}")
    print(f"Atoms with non-zero computed but zero ref: {np.where(~computed_zero & ref_zero)[0]}")
    
    # Check magnitudes
    computed_mags = np.linalg.norm(computed_forces, axis=1)
    ref_mags = np.linalg.norm(ref_forces, axis=1)
    
    print(f"\nForce magnitudes:")
    print(f"  Computed: min={computed_mags.min():.4f}, max={computed_mags.max():.4f}, mean={computed_mags.mean():.4f}")
    print(f"  Reference: min={ref_mags.min():.4f}, max={ref_mags.max():.4f}, mean={ref_mags.mean():.4f}")
    
    # Check differences
    diff = computed_forces - ref_forces
    diff_mags = np.linalg.norm(diff, axis=1)
    
    print(f"\nForce differences:")
    print(f"  Max difference: {diff_mags.max():.4f}")
    print(f"  Mean difference: {diff_mags.mean():.4f}")
    print(f"  Atoms with largest differences: {np.argsort(diff_mags)[-5:][::-1]}")
    
    # Check monomer mapping if provided
    if monomer_indices is not None:
        print(f"\nMonomer atom indices: {monomer_indices}")
        monomer_mask = np.zeros(n_atoms, dtype=bool)
        monomer_mask[monomer_indices] = True
        
        print(f"Atoms in monomers: {np.where(monomer_mask)[0]}")
        print(f"Atoms NOT in monomers: {np.where(~monomer_mask)[0]}")
        print(f"Non-monomer atoms with non-zero computed forces: {np.where(~monomer_mask & ~computed_zero)[0]}")
        print(f"Non-monomer atoms with non-zero ref forces: {np.where(~monomer_mask & ~ref_zero)[0]}")
    
    print("=" * 60)

# Usage:
# diagnose_forces(computed_forces, ref_forces, n_atoms=20, monomer_indices=[0,1,4,5,6,7,8,9])
```

---

## Most Likely Issues (Prioritized)

1. **Switching function gradients missing** - Forces scaled instead of proper gradients
2. **Atom index mapping** - ML forces not mapped to correct atoms
3. **Missing MM contributions** - Some atoms not included in MM force computation
4. **Switching function distance** - Wrong distance metric (COM vs pair distances)

