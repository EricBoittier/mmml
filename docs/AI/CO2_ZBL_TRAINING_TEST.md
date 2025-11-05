# CO2 ZBL Training Stability Test

**Test ZBL repulsion during actual training to catch MD instabilities**

## Overview

This test trains a charge-spin conditioned PhysNet model on CO2 molecules with **ZBL repulsion enabled** across all prediction modes. Unlike inference-only tests, this catches instabilities that arise during gradient updates (which is when MD simulations fail).

## Why CO2?

- **Simple**: Only 3 atoms (C, O, O)
- **Linear**: Well-defined geometry
- **Realistic**: Real molecular system
- **Fast**: Quick training iterations
- **Sensitive**: Small molecule shows instabilities quickly

## Test Modes

The script tests 4 prediction modes:

| Mode | Predict E | Predict F | Predict D | Use Case |
|------|-----------|-----------|-----------|----------|
| **1. Energy Only** | ✓ | ✗ | ✗ | Screening, MC |
| **2. Energy + Forces** | ✓ | ✓ | ✗ | MD simulations |
| **3. Forces Only** | ✗ | ✓ | ✗ | Force matching |
| **4. E + F + Dipoles** | ✓ | ✓ | ✓ | Full QM |

Each mode is trained for 10 epochs with NaN detection at every step.

## Running the Test

```bash
python examples/train_co2_zbl_stability.py
```

### Expected Output

```
================================================================================
CO2 Training - ZBL Stability Test
================================================================================

Testing modes:
  1. Energy only
  2. Energy + Forces
  3. Forces only
  4. Energy + Forces + Dipoles

================================================================================
Training Mode: Energy Only
================================================================================
  Predict Energy:  True
  Predict Forces:  False
  Predict Dipoles: False
  Epochs: 10

Generating CO2 training data...
  Train batches: 5
  Valid batches: 1

Creating model (ZBL ENABLED)...
  Model initialized: 52,XXX parameters

Training...
  Epoch 2/10: Train Loss=0.123456, Valid Loss=0.234567
  Epoch 4/10: Train Loss=0.098765, Valid Loss=0.198765
  ...

✓ STABLE: Completed 10 epochs without NaN

[... repeats for other modes ...]

================================================================================
STABILITY SUMMARY
================================================================================
  Energy Only                   : ✓ STABLE
  Energy + Forces               : ✓ STABLE
  Forces Only                   : ✓ STABLE
  Energy + Forces + Dipoles     : ✓ STABLE
================================================================================

✅ SUCCESS: All modes are stable!
   ZBL is working correctly across all prediction modes.
```

### If Instability Detected

```
================================================================================
Training Mode: Energy + Forces
================================================================================
...
Training...
  Epoch 2/10: Train Loss=0.123456, Valid Loss=0.234567
  
  ❌ NaN DETECTED at epoch 3, batch 2!
     Loss NaN: False
     Grad NaN: True
     Energy NaN: False
     Forces NaN: True

❌ UNSTABLE: NaN detected at epoch 3

================================================================================
STABILITY SUMMARY
================================================================================
  Energy Only                   : ✓ STABLE
  Energy + Forces               : ❌ UNSTABLE (NaN @ epoch 3)
  Forces Only                   : ❌ UNSTABLE (NaN @ epoch 2)
  Energy + Forces + Dipoles     : ✓ STABLE
================================================================================

⚠️  INSTABILITY DETECTED!

Possible causes:
  1. ZBL parameters causing gradient explosion
  2. Learning rate too high for ZBL gradients
  3. Numerical instability in switching functions
  4. Issue with force computation through ZBL

Recommended fixes:
  • Reduce learning rate (try 0.0001)
  • Add gradient clipping
  • Adjust ZBL switching parameters
  • Scale ZBL contribution with learnable weight
```

## Understanding Results

### All Stable ✅

```
Energy Only                   : ✓ STABLE
Energy + Forces               : ✓ STABLE
Forces Only                   : ✓ STABLE
Energy + Forces + Dipoles     : ✓ STABLE
```

**Interpretation**: ZBL is working correctly!
- No gradient explosions
- No numerical instabilities
- Safe for MD simulations

**Action**: Model is ready for production use

---

### Forces Unstable ⚠️

```
Energy Only                   : ✓ STABLE
Energy + Forces               : ❌ UNSTABLE (NaN @ epoch 3)
Forces Only                   : ❌ UNSTABLE (NaN @ epoch 2)
Energy + Forces + Dipoles     : ❌ UNSTABLE (NaN @ epoch 5)
```

**Interpretation**: ZBL gradients causing instability
- Energy predictions fine
- Force gradients (∂E/∂R through ZBL) exploding
- Happens during backpropagation

**Root Cause**: 
- ZBL switching functions not smooth enough
- Short-range repulsion too steep
- Gradient magnitude too large

**Fixes**:

1. **Reduce Learning Rate**
```python
optimizer, _, _, _ = get_optimizer(
    learning_rate=0.0001,  # ← Lower from 0.001
)
```

2. **Add Gradient Clipping**
```python
# In train_step
grads, grad_norm = jax.tree_util.tree_map(
    lambda g: jnp.clip(g, -1.0, 1.0), grads
)
```

3. **Softer ZBL Switching**
```python
# In _calc_switches
switch_start = 2.0  # Start later (was 1.0)
switch_end = 12.0   # End later (was 10.0)
```

4. **Scale ZBL Contribution**
```python
# Add learnable scaling
zbl_scale = self.param("zbl_scale", lambda rng: jnp.array(0.1))
repulsion = zbl_scale * self.repulsion(...)
```

---

### Specific Mode Unstable ⚠️

```
Energy Only                   : ✓ STABLE
Energy + Forces               : ✓ STABLE
Forces Only                   : ✓ STABLE
Energy + Forces + Dipoles     : ❌ UNSTABLE (NaN @ epoch 4)
```

**Interpretation**: Issue with dipole prediction
- Not ZBL-related
- Problem in charge prediction head
- Dipole calculation unstable

**Fix**: Check charge prediction:
```python
# Ensure charges are bounded
charges = nn.tanh(charges_dense) * 2.0  # Limit to ±2e
```

---

### Early Epoch Failure ⚠️

```
Energy + Forces               : ❌ UNSTABLE (NaN @ epoch 1)
```

**Interpretation**: Immediate instability
- Bad initialization
- Learning rate too high from start
- ZBL parameters initialized poorly

**Fixes**:
1. Try different random seed
2. Reduce initial learning rate
3. Warm-up learning rate schedule

---

### Late Epoch Failure ⚠️

```
Energy + Forces               : ❌ UNSTABLE (NaN @ epoch 47)
```

**Interpretation**: Parameters diverging
- Training initially stable
- Parameters grow too large
- Numerical overflow

**Fixes**:
1. Add weight decay
2. Use EMA
3. Early stopping
4. Learning rate decay

## Generated Training Data

The script generates synthetic CO2 data:

```python
# 100 CO2 molecules with:
# - Perturbed C-O bond lengths (1.16 ± 0.05 Å)
# - Random rotations
# - Random translations
# - Harmonic forces towards equilibrium
# - Harmonic energies
```

This mimics MD trajectory data where:
- Geometries vary around equilibrium
- Forces restore toward minimum
- Energy correlates with displacement

## Monitoring During Training

Each training step checks for NaN in:
- **Loss**: Main training loss
- **Gradients**: All parameter gradients
- **Energy predictions**: Model output
- **Force predictions**: Model output

NaN is detected **immediately** and training stops.

## Common Failure Patterns

### Pattern 1: Gradual Divergence

```
Epoch 1: Loss=0.5
Epoch 2: Loss=0.3
Epoch 3: Loss=0.2
Epoch 4: Loss=0.5
Epoch 5: Loss=2.0
Epoch 6: Loss=50.0
Epoch 7: Loss=NaN
```

**Cause**: Learning rate too high
**Fix**: Reduce learning rate by 10x

---

### Pattern 2: Sudden Failure

```
Epoch 1: Loss=0.5
Epoch 2: Loss=0.3
Epoch 3: Loss=NaN
```

**Cause**: Bad gradient or bad initialization
**Fix**: Different seed or gradient clipping

---

### Pattern 3: Oscillation

```
Epoch 1: Loss=0.5
Epoch 2: Loss=2.0
Epoch 3: Loss=0.3
Epoch 4: Loss=5.0
Epoch 5: Loss=NaN
```

**Cause**: Optimization instability
**Fix**: Better optimizer (AdamW) or momentum

## Manual Debugging

### Test Specific Geometries

```python
# Add custom CO2 geometries to test
custom_data = [{
    'Z': np.array([6, 8, 8]),
    'R': np.array([...]),  # Your geometry
    'F': np.array([...]),
    'E': 0.0,
}]

batches = create_batches(custom_data, batch_size=1, num_atoms=10)
```

### Monitor Gradient Norms

```python
# In train_step, add:
grad_norms = {
    k: jnp.sqrt(jnp.sum(g**2))
    for k, g in jax.tree_util.tree_flatten(grads)[0]
}
print(f"Max grad norm: {max(grad_norms.values()):.6f}")
```

### Test Single Batch

```python
# Train on just one batch repeatedly
for epoch in range(100):
    params, opt_state, loss, metrics = train_step(
        model, params, opt_state, optimizer,
        train_batches[0],  # ← Same batch every time
        num_atoms,
        predict_energy=True,
        predict_forces=True,
    )
    print(f"Epoch {epoch}: Loss={loss:.6f}")
```

## Integration with Real Data

To test with real CO2 data:

```python
# Replace generate_co2_training_data with:
def load_real_co2_data(npz_file):
    data = np.load(npz_file)
    return [
        {
            'Z': data['Z'][i],
            'R': data['R'][i],
            'F': data['F'][i],
            'E': data['E'][i],
        }
        for i in range(len(data['Z']))
    ]

train_data = load_real_co2_data('co2_train.npz')
valid_data = load_real_co2_data('co2_valid.npz')
```

## Next Steps After Testing

### If All Stable ✅

1. Train on full dataset
2. Test on longer trajectories
3. Run actual MD simulations
4. Monitor for rare instabilities

### If Unstable ⚠️

1. Apply recommended fixes
2. Re-run stability test
3. Iterate until stable
4. Document final parameters

## Files

- `examples/train_co2_zbl_stability.py` - Training script
- `CO2_ZBL_TRAINING_TEST.md` - This documentation

## Related Tests

- `examples/test_zbl_nan_debug.py` - Inference-only NaN test
- `examples/predict_options_demo.py` - Prediction modes demo
- `ZBL_FIX_AND_TEST.md` - ZBL fix documentation

---

**Purpose**: Catch MD instabilities during training  
**Method**: Train on CO2 with all prediction modes  
**Detection**: NaN monitoring at every step  
**Result**: Pass/fail for each mode  
**Status**: Ready to run ✅

Run it now to test your ZBL implementation:
```bash
python examples/train_co2_zbl_stability.py
```

