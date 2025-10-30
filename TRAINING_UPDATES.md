# DCMNet Training Script Updates

## Summary
Updated the training script to use `lovely_jax` for enhanced array visualization and added comprehensive statistics display for training and validation sets.

## Changes Made

### 1. Added lovely_jax Support
- Added `lovely_jax` import with graceful fallback if not installed
- Enables better visualization of JAX arrays during debugging
- Added to `requirements.txt`

### 2. Enhanced Statistics Functions
- **`compute_statistics(predictions, targets=None)`**: Computes comprehensive statistics including:
  - Mean, standard deviation, min, max, median
  - MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error) when targets provided
  
- **`print_statistics_table(train_stats, valid_stats, epoch)`**: Displays formatted comparison table showing:
  - Loss metrics
  - Monopole prediction statistics (MAE, RMSE, mean, std)
  - Side-by-side train vs validation comparison
  - Differences between train and validation

### 3. Updated Training Functions
All training and evaluation functions now return predictions along with losses:
- `train_step`: Returns `params, opt_state, loss, mono, dipo`
- `eval_step`: Returns `loss, mono, dipo`
- `train_step_dipo`: Returns `params, opt_state, loss, esp_l, mono_l, dipo_l, mono, dipo`
- `eval_step_dipo`: Returns `loss, esp_l, mono_l, dipo_l, mono, dipo`

### 4. Enhanced Training Loops
Both `train_model` and `train_model_dipo` now:
- Collect predictions for all batches in each epoch
- Compute comprehensive statistics for training and validation sets
- Display detailed statistics table after each epoch
- Log additional metrics to TensorBoard:
  - Monopole MAE and RMSE (train and validation)
  - Monopole mean and std (train and validation)

### 5. Output Format
The training now displays a formatted table each epoch:
```
================================================================================
Epoch   1 Statistics
================================================================================
Metric               Train           Valid      Difference
--------------------------------------------------------------------------------
loss                 1.234e-03       1.456e-03   2.220e-04
mono_mae             1.000e-04       1.200e-04   2.000e-05
mono_rmse            1.500e-04       1.700e-04   2.000e-05
mono_mean            0.000e+00       5.000e-06   5.000e-06
mono_std             1.000e-01       1.050e-01   5.000e-03
--------------------------------------------------------------------------------
Monopole Prediction Statistics:
  Train: mean=0.000e+00, std=1.000e-01, min=-5.000e-01, max=5.000e-01
  Valid: mean=5.000e-06, std=1.050e-01, min=-5.200e-01, max=5.100e-01
================================================================================
```

## Installation
To use the enhanced features, install lovely_jax:
```bash
pip install lovely_jax
```

Or install all requirements:
```bash
pip install -r mmml/dcmnet/requirements.txt
```

## Benefits
1. **Better Debugging**: lovely_jax provides cleaner array printing
2. **Comprehensive Monitoring**: Track prediction quality beyond just loss
3. **Early Detection**: Spot overfitting or distribution shifts early
4. **Better Insights**: Understand model behavior through detailed statistics
5. **TensorBoard Integration**: All statistics logged for visualization

## Backward Compatibility
- All changes are backward compatible
- If lovely_jax is not installed, training continues with standard JAX printing
- Existing function signatures remain compatible through return value unpacking

