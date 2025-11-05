# ZBL Repulsion Fix and NaN Debugging

## What Was Fixed

### 1. ZBL Call Signature

**Problem**: ZBL repulsion was called with wrong arguments
```python
# ❌ BROKEN
repulsion_per_pair = self.repulsion(
    atomic_numbers[dst_idx],
    atomic_numbers[src_idx],
    displacements,
)
```

**Solution**: Match working PhysNet's call signature
```python
# ✓ FIXED  
r, off_dist, eshift = self._calc_switches(displacements, batch_mask)

repulsion = self.repulsion(
    atomic_numbers,      # Full array, not indexed
    r,                   # Distance factors
    off_dist,            # Off-distance mask
    1 - eshift,          # Energy shift (inverted)
    dst_idx,             # Destination indices
    src_idx,             # Source indices
    atom_mask,           # Atom mask
    batch_mask,          # Batch mask
    batch_segments,      # Batch segments
    batch_size,          # Batch size
)
```

### 2. Switching Functions Added

Implemented `_calc_switches()` to calculate:
- **r**: Smoothly switched distance factors (short-range + long-range)
- **off_dist**: Distance cutoff mask
- **eshift**: Energy shift for smooth potential

These are needed for numerical stability at short distances.

### 3. Proper Energy Aggregation

```python
# Repulsion is per-atom, not per-pair
energy = jax.ops.segment_sum(
    energy_per_atom + repulsion,  # Both per-atom
    segment_ids=batch_segments,
    num_segments=batch_size,
)
```

## Testing for NaN Issues

### Run the Debug Test

```bash
python examples/test_zbl_nan_debug.py
```

This test:
1. Creates realistic molecular geometries (H₂O, CH₄)
2. Tests model with ZBL **disabled**
3. Tests model with ZBL **enabled**
4. Checks for NaN in energies, forces, and repulsion
5. Reports which component causes NaN (if any)

### Expected Output

```
================================================================================
ZBL Repulsion and NaN Debug Test
================================================================================

================================================================================
Testing with ZBL=DISABLED
================================================================================

Molecules:
  Water:   Z=[8 1 1], positions shape=(3, 3)
  Methane: Z=[6 1 1 1 1], positions shape=(5, 3)

Batch:
  Z shape: (2, 10)
  R shape: (2, 10, 3)

Initializing model (ZBL=False)...
✓ Model initialized: 52,XXX parameters

Running forward pass...
✓ Energy: [...]
✓ Forces shape: (20, 3)
  Forces min/max: [..., ...]
  Forces mean: ...

================================================================================
NO NaN - All values are finite
================================================================================

================================================================================
Testing with ZBL=ENABLED
================================================================================
...
```

### Interpreting Results

#### Case 1: Both Pass ✅
```
ZBL Disabled: ✓ No NaN
ZBL Enabled:  ✓ No NaN

✓ All tests passed!
Model is working correctly with and without ZBL.
```
**Action**: None needed, model is working!

---

#### Case 2: ZBL Causes NaN ⚠️
```
ZBL Disabled: ✓ No NaN
ZBL Enabled:  ❌ Has NaN

⚠️  ZBL IS CAUSING NaN!
```

**Possible Causes**:
1. **Short distances**: Atoms too close together
   - Check: Minimum distance in test geometries
   - Fix: Increase `min_dist` in `_calc_switches()`

2. **Switching function**: Parameters cause instability
   - Check: `switch_start`, `switch_end` values
   - Fix: Adjust switching ranges

3. **ZBL parameters**: Uninitialized or extreme values
   - Check: ZBL module initialization
   - Fix: Verify `trainable=True` and initial values

4. **Gradient explosion**: Autograd through ZBL unstable
   - Check: Force magnitudes
   - Fix: Add gradient clipping or scale ZBL contribution

**Debug Steps**:
1. Print intermediate values in `_calc_switches()`:
```python
print("distances:", distances)
print("r:", r)
print("off_dist:", off_dist)
print("eshift:", eshift)
```

2. Print ZBL output:
```python
repulsion = self.repulsion(...)
print("repulsion min/max:", jnp.min(repulsion), jnp.max(repulsion))
print("has NaN:", jnp.any(jnp.isnan(repulsion)))
```

3. Test with gentler parameters:
```python
min_dist = 0.1  # Larger minimum distance
switch_start = 2.0  # Start switching later
```

---

#### Case 3: Base Model Has NaN ❌
```
ZBL Disabled: ❌ Has NaN
ZBL Enabled:  ❌ Has NaN

⚠️  NaN exists even without ZBL!
```

**Possible Causes**:
1. **Charge/spin embedding**: Out of range values
2. **Message passing**: Numerical instability
3. **Energy head**: Extreme predictions
4. **e3x operations**: Incompatible tensor shapes

**Debug Steps**:
1. Check charge/spin inputs are in range
2. Test with simpler model (fewer iterations)
3. Check intermediate feature magnitudes
4. Disable charge/spin conditioning temporarily

## Manual Testing

### Test Different Molecules

```python
from examples.test_zbl_nan_debug import create_test_batch, test_model_with_zbl

# Create custom molecules
my_molecules = [
    (Z1, R1),  # Your molecule 1
    (Z2, R2),  # Your molecule 2
]

# Test
has_nan, outputs = test_model_with_zbl(zbl_enabled=True)
```

### Test at Different Distances

```python
# Test very short distances (should be handled gracefully)
Z = np.array([1, 1], dtype=np.int32)  # Two hydrogens
R = np.array([
    [0.0, 0.0, 0.0],
    [0.1, 0.0, 0.0],  # Very close! (0.1 Å)
], dtype=np.float32)

# Should not produce NaN even at short distance
```

### Test Different Charge/Spin States

```python
# Test charged molecules
outputs = model.apply(
    ...,
    total_charges=jnp.array([1.0, -1.0]),  # Cation, anion
    total_spins=jnp.array([2.0, 2.0]),     # Both doublets
)
```

## Common Issues and Solutions

### Issue 1: NaN in Forces Only

**Symptom**: Energy is fine, but forces contain NaN

**Cause**: Gradient computation unstable

**Fix**:
```python
# Add gradient clipping in training
grads = jax.grad(loss_fn)(params)
grads, grad_norm = optax.clip_by_global_norm(grads, max_norm=1.0)
```

### Issue 2: NaN at Initialization

**Symptom**: NaN appears during `model.init()`

**Cause**: Bad random initialization

**Fix**:
```python
# Use different random seed
key = jax.random.PRNGKey(123)  # Try different seeds
```

### Issue 3: NaN with Certain Molecules

**Symptom**: Some molecules work, others produce NaN

**Cause**: Specific geometric configurations problematic

**Fix**:
```python
# Check minimum distances in problematic molecules
dists = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=-1)
min_dist = dists[dists > 0].min()
print(f"Minimum distance: {min_dist:.3f} Å")

if min_dist < 0.5:
    print("⚠️  Atoms too close!")
```

### Issue 4: NaN During Training

**Symptom**: Model works initially but NaN appears during training

**Cause**: Learning rate too high or parameters diverging

**Fix**:
```python
# Reduce learning rate
learning_rate = 0.0001  # Lower than default

# Use EMA for stability
ema_decay = 0.999

# Monitor parameter magnitudes
param_norm = jnp.sqrt(sum(
    jnp.sum(p**2) for p in jax.tree_util.tree_leaves(params)
))
if param_norm > 1000:
    print("⚠️  Parameters growing too large!")
```

## Disabling ZBL Temporarily

If ZBL is causing issues, disable it temporarily:

```python
model = EF_ChargeSpinConditioned(
    ...,
    zbl=False,  # ← Disable ZBL
)
```

Once the rest of the model is working, re-enable ZBL and debug specifically.

## Summary

| Component | Status | Action |
|-----------|--------|--------|
| ZBL Call Signature | ✅ Fixed | Matches working PhysNet |
| Switching Functions | ✅ Added | `_calc_switches()` |
| Energy Aggregation | ✅ Fixed | Per-atom repulsion |
| Test Suite | ✅ Created | `test_zbl_nan_debug.py` |
| NaN Detection | ✅ Implemented | Checks E, F, repulsion |

## Next Steps

1. **Run the test**: `python examples/test_zbl_nan_debug.py`
2. **Check results**: Look for NaN indicators
3. **If NaN found**: Follow debug steps above
4. **If all clear**: Model is ready to use!

## Files Created

- `examples/test_zbl_nan_debug.py` - Comprehensive NaN debugging test
- `ZBL_FIX_AND_TEST.md` - This documentation

## References

- Working PhysNet: `mmml/physnetjax/physnetjax/models/model.py`
- ZBL Module: `mmml/physnetjax/physnetjax/models/zbl.py`
- Original issue: ZBL call signature mismatch

---

**Created**: November 2025  
**Status**: Ready for Testing  
**Location**: `/home/ericb/mmml/`

