# Critical Fixes Applied

## 1. **Zero-Initialized Charges in PhysNet** ✅ FIXED
**File:** `mmml/physnetjax/physnetjax/models/model.py:568`

**Problem:** Charge prediction layer was initialized with zeros:
```python
kernel_init=jax.nn.initializers.zeros
```

**Impact:** 
- All charges started at 0
- Dipoles = Σ(q × r) = 0
- ESP predictions = 0
- No gradient signal to learn charges

**Fix:** Changed to small random initialization:
```python
kernel_init=jax.nn.initializers.normal(stddev=0.01)
```

---

## 2. **Broken COM Calculation in PhysNet** ✅ FIXED
**File:** `mmml/physnetjax/physnetjax/models/model.py:833`

**Problem:** COM was computed by summing over wrong axis:
```python
com = jnp.sum(dis_com, axis=1)  # WRONG: sums over xyz!
```

**Impact:**
- Nonsensical COM values
- Incorrect dipole calculations
- Dipoles not properly centered

**Fix:** Use `segment_sum` to sum over atoms per molecule:
```python
mass_weighted_pos = positions * masses[..., None]
total_mass_weighted_pos = jax.ops.segment_sum(
    mass_weighted_pos, 
    segment_ids=batch_segments, 
    num_segments=batch_size
)
total_mass = jax.ops.segment_sum(
    masses, 
    segment_ids=batch_segments, 
    num_segments=batch_size
)
com = total_mass_weighted_pos / total_mass[..., None]
```

---

## 3. **Inconsistent COM for DCMNet Dipole** ✅ FIXED
**File:** `examples/co2/dcmnet_physnet_train/trainer.py`

**Problem:** DCMNet dipole calculation used different COM method than PhysNet:
- PhysNet: `segment_sum` (proper batching)
- DCMNet: `jnp.sum` with reshape (incorrect for batching)

**Impact:**
- DCMNet dipoles calculated with wrong COM
- Inconsistent with PhysNet dipoles
- Poor correlation with true dipoles

**Fix:** Changed both places to use `segment_sum`:

**In `compute_loss` (line 625-642):**
```python
masses_flat = jnp.take(ase.data.atomic_masses, atomic_nums_flat)
mass_weighted_pos = positions_flat * masses_flat[..., None]
total_mass_weighted_pos = jax.ops.segment_sum(
    mass_weighted_pos,
    segment_ids=batch["batch_segments"],
    num_segments=batch_size
)
total_mass = jax.ops.segment_sum(
    masses_flat,
    segment_ids=batch["batch_segments"],
    num_segments=batch_size
)
com = total_mass_weighted_pos / total_mass[..., None]
```

**In `eval_step` (line 916-928):** Same approach.

---

## 4. **RMSE Not Using Distance Mask** ✅ FIXED
**File:** `examples/co2/dcmnet_physnet_train/trainer.py:eval_step`

**Problem:** ESP RMSE computed on all grid points, including those near atoms:
```python
rmse_esp_dcmnet = jnp.sqrt(jnp.mean((esp_pred_dcmnet - esp_target)**2))
```

**Impact:**
- RMSE inflated by high ESP values near atoms
- Validation metrics didn't reflect filtered training

**Fix:** Apply distance mask to RMSE calculation:
```python
esp_diff_dcmnet = (esp_pred_dcmnet - esp_target) * esp_mask
n_valid = esp_mask.sum()
rmse_esp_dcmnet = jnp.sqrt(jnp.sum(esp_diff_dcmnet ** 2) / (n_valid + 1e-10))
```

---

## 5. **Added Magnitude-Based ESP Filtering** ✅ NEW FEATURE
**File:** `examples/co2/dcmnet_physnet_train/trainer.py`

**Addition:** New `--esp-max-value` CLI argument to exclude high |ESP| values:
```python
# Distance-based mask
distance_mask = (min_distances >= esp_min_distance).astype(jnp.float32)

# Magnitude-based mask
if esp_max_value < 1e9:
    magnitude_mask = (jnp.abs(esp_target) <= esp_max_value).astype(jnp.float32)
    esp_mask = distance_mask * magnitude_mask
else:
    esp_mask = distance_mask
```

**Usage:**
```bash
python trainer.py \
  --esp-min-distance 1.0 \    # Distance filter (default)
  --esp-max-value 0.3 \        # Magnitude filter (new)
  ...
```

---

## Expected Results After Fixes

### Before Fixes:
- ❌ PhysNet dipoles: ~0 (flat, no correlation)
- ❌ DCMNet dipoles: ~0 (flat, no correlation)
- ❌ ESP RMSE: ~0.068 Ha/e (42 kcal/mol/e)
- ❌ Charges: all near zero, not learning

### After Fixes:
- ✅ PhysNet dipoles: Should correlate with true dipoles
- ✅ DCMNet dipoles: Should correlate with true dipoles
- ✅ ESP RMSE: Should drop to < 0.02 Ha/e (~12 kcal/mol/e)
- ✅ Charges: Should show proper distribution (±0.5e for CO2)
- ✅ Centered scatter plots: Should show strong positive correlation

---

## Action Required

**DELETE OLD CHECKPOINT** (has bugs baked in):
```bash
rm -rf /home/ericb/mmml/mmml/physnetjax/ckpts/co2_joint_physnet_dcmnet
```

**RETRAIN FROM SCRATCH** with the fixed code.

**MONITOR:**
- Charge statistics (should be non-zero from epoch 1)
- Dipole correlations (should improve rapidly)
- ESP RMSE (should decrease steadily)

---

## Technical Details

### COM Calculation Methods

**Wrong (old):**
```python
com = jnp.sum(dis_com, axis=1)  # Sums over xyz coordinates!
```

**Correct (new):**
```python
com = segment_sum(mass_weighted_pos) / segment_sum(masses)  # Sums over atoms
```

### Dipole Calculation

**Formula:** D = Σ q_i × (r_i - r_COM)

**Key Points:**
1. COM must be calculated from **atomic positions** weighted by **nuclear masses**
2. Both PhysNet and DCMNet must use **identical COM**
3. Use `segment_sum` for proper batch handling
4. DCMNet's `dipo_dist` contains **absolute positions** (atom_pos + offset)

---

## Files Modified

1. `mmml/physnetjax/physnetjax/models/model.py` - Charge init + COM calculation
2. `examples/co2/dcmnet_physnet_train/trainer.py` - DCMNet COM + ESP filtering

