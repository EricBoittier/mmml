# Batch Dimension Handling Fix

## 🐛 Problem

When `batch_size=1`, the model outputs predictions **without a batch dimension**:
- Expected: `(1, n_atoms, n_dcm, 3)` 
- Actual: `(n_atoms, n_dcm, 3)` ← Batch dimension squeezed!

This caused reshape errors in the loss function.

## Error Example

```
TypeError: cannot reshape array of shape (18, 3, 4) (size 216) 
into shape (1, 16, 3) (size 48)
```

**What happened:**
- Model output: `mono_pred` = `(18, 4)` (no batch dim)
- Code inferred `max_atoms` from `shape[1]` = `4` ❌ (wrong!)
- Tried to reshape: `(18, 4)` → `(1, 4*4, 3)` = `(1, 16, 3)` ❌ (wrong size!)

## ✅ Solution

Add intelligent batch dimension detection and handling:

```python
# Infer max_atoms from prediction shape
# Handle both batched and unbatched predictions
if batch_size == 1 and len(mono_prediction.shape) == 2:
    # Unbatched: (n_atoms, n_dcm)
    max_atoms = mono_prediction.shape[0]  # ✅ Correct: 18
    # Add batch dimension
    mono_prediction = mono_prediction[None, :, :]  # (1, 18, 4)
    dipo_prediction = dipo_prediction[None, :, :, :]  # (1, 18, 4, 3)
else:
    # Batched: (batch_size, n_atoms, n_dcm)
    max_atoms = mono_prediction.shape[1]

# Now reshaping works correctly!
d = jnp.moveaxis(dipo_prediction, -1, -2).reshape(batch_size, max_atoms * n_dcm, 3)
# (1, 18, 3, 4) → (1, 18*4, 3) = (1, 72, 3) ✅
```

## 🔧 Files Fixed

Applied to all loss functions:

1. **`esp_mono_loss()`** - Main ESP+monopole loss
2. **`dipo_esp_mono_loss()`** - Dipole-augmented loss
3. **`esp_mono_loss_pots()`** - ESP potential loss
4. **`esp_loss_pots()`** - ESP-only loss
5. **`mean_absolute_error()`** - MAE calculation

## 📊 Behavior

### Case 1: batch_size = 1 (Unbatched Predictions)

**Input Shapes:**
```python
mono_prediction: (18, 4)      # (n_atoms, n_dcm)
dipo_prediction: (18, 4, 3)   # (n_atoms, n_dcm, 3)
```

**Detection:**
```python
max_atoms = mono_prediction.shape[0]  # = 18 ✅
```

**Processing:**
```python
# Add batch dimension
mono_prediction = mono_prediction[None, :, :]  # → (1, 18, 4)
dipo_prediction = dipo_prediction[None, :, :, :]  # → (1, 18, 4, 3)

# Now reshape works
d = moveaxis(...).reshape(1, 18*4, 3)  # → (1, 72, 3) ✅
```

### Case 2: batch_size > 1 (Batched Predictions)

**Input Shapes:**
```python
mono_prediction: (4, 18, 4)      # (batch, n_atoms, n_dcm)
dipo_prediction: (4, 18, 4, 3)   # (batch, n_atoms, n_dcm, 3)
```

**Detection:**
```python
max_atoms = mono_prediction.shape[1]  # = 18 ✅
```

**Processing:**
```python
# Already batched, use as-is
d = moveaxis(...).reshape(4, 18*4, 3)  # → (4, 72, 3) ✅
```

## ✅ Testing

Verified with actual shapes:

```python
# Test unbatched (batch_size=1)
mono_pred = jnp.ones((18, 4))
dipo_pred = jnp.ones((18, 4, 3))

loss = esp_mono_loss(
    dipo_prediction=dipo_pred,
    mono_prediction=mono_pred,
    # ... other args ...
    batch_size=1,
    n_dcm=4
)
# ✅ Works! Loss computed successfully
```

## 💡 Why This Happens

JAX models often squeeze singleton dimensions when `batch_size=1`:

```python
# In model forward pass
if batch_size == 1:
    output = output.squeeze(0)  # Remove batch dimension
```

The loss function now handles both cases automatically!

## 🎯 Impact

✅ **Works with batch_size=1** - Most common use case  
✅ **Works with batch_size>1** - Multi-sample batches  
✅ **Automatic detection** - No manual specification needed  
✅ **Backward compatible** - Existing code still works  

## 📝 Pattern for Other Functions

If you have similar issues in other functions:

```python
def my_function(predictions, batch_size, ...):
    # Check if batch dimension is present
    if batch_size == 1 and len(predictions.shape) == expected_ndim - 1:
        # Missing batch dimension - add it
        predictions = predictions[None, ...]  # Add batch dim at front
        max_items = predictions.shape[0]  # First non-batch dimension
    else:
        # Batch dimension present
        max_items = predictions.shape[1]  # Second dimension
    
    # Continue processing...
```

## 🎉 Result

The loss functions now **just work™** regardless of whether the model outputs include a batch dimension or not!

**Harmony achieved through intelligent batch handling!** 🎵✨

