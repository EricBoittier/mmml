# SLURM Logging and Table Display Fixes

## Issues Fixed

### 1. ✅ **Truncated Values in Training Table**
**Problem**: Numbers were being cut off with "..." in the training progress table.

**Solution**: 
- Set explicit column widths in `pretty_printer.py`
- Added `overflow="fold"` to allow text wrapping
- Implemented smart number formatting:
  - Small numbers (<0.01) or large numbers (>=1000): scientific notation (`1.234e-03`)
  - Normal range: fixed decimal format (`.4f`)

**Changes**:
```python
# mmml/physnetjax/physnetjax/utils/pretty_printer.py

# Before:
table.add_column("Train Loss", style="medium_orchid3")

# After:
table.add_column("Train Loss", style="medium_orchid3", width=11, overflow="fold")
```

### 2. ✅ **Output Not Appearing in SLURM Log Files**
**Problem**: Training output wasn't always appearing in SLURM `.out` files or appeared with delays.

**Solutions Applied**:

#### a) **Python Unbuffered Mode**
SLURM scripts now use `python -u` flag:
```bash
# Before:
python trainer.py --train ...

# After:
python -u trainer.py --train ...
```
The `-u` flag forces unbuffered stdout/stderr, ensuring real-time output to logs.

#### b) **Explicit Stdout/Stderr Flushing**
Added flush calls after important operations in `training.py`:
```python
import sys
sys.stdout.flush()  # Force output to SLURM log file
sys.stderr.flush()  # Flush errors too
```

#### c) **Console Configuration for SLURM**
Updated Rich Console initialization:
```python
console = Console(
    width=250,              # Wide enough for all columns
    force_terminal=True,    # Force color output in SLURM
    force_interactive=False,  # Better for log files
)
```

## Files Modified

1. **`mmml/physnetjax/physnetjax/utils/pretty_printer.py`**
   - Updated `init_table()`: Added column widths and overflow handling
   - Updated `epoch_printer()`: Smart number formatting

2. **`mmml/physnetjax/physnetjax/training/training.py`**
   - Console initialization with SLURM-friendly settings
   - Added `sys.stdout.flush()` and `sys.stderr.flush()` calls

3. **`examples/co2/physnet_train/slurm_quick_scan.sh`**
   - Changed `python trainer.py` to `python -u trainer.py`

4. **`examples/co2/physnet_train/slurm_model_scan.sh`**
   - Changed `python trainer.py` to `python -u trainer.py`

## Expected Improvements

### Training Table Display
**Before**:
```
│  91   │ 4.73  │ Array │   14… │    4… │    1… │   15… │   14… │    … │    2… │
```

**After**:
```
│  91   │ 4.73s │ 1.000e-03 │  14.2341   │   4.5678   │   1.2345   │   15.6789    │   14.3456    │    0.8234    │    2.1234    │
```

### SLURM Log Files
- **Real-time output**: Progress appears immediately in log files
- **No missing epochs**: All training progress is captured
- **Complete tables**: Tables display fully without truncation
- **Flushed errors**: Error messages appear immediately

## Testing

To verify the fixes work:

```bash
# Submit a test job
sbatch slurm_quick_scan.sh

# Monitor in real-time (should see output immediately)
tail -f logs/quick_*.out

# Check that all values are visible (no "..." truncation)
grep "Valid Forces MAE" logs/quick_*.out
```

## Additional Tips for SLURM Logging

### 1. **Monitor Jobs in Real-Time**
```bash
# Watch log file as it updates
tail -f logs/quick_JOBID_0.out

# Follow multiple logs
tail -f logs/quick_*.out
```

### 2. **Check for Buffering Issues**
If output still seems delayed, you can also set:
```bash
export PYTHONUNBUFFERED=1
```
in your SLURM script before running Python.

### 3. **Increase Log File Update Frequency**
SLURM's default buffer can be adjusted:
```bash
#SBATCH --open-mode=append
```

### 4. **Force Immediate Output**
In Python code, you can also use:
```python
print("Message", flush=True)
```

## Performance Impact

These changes have **minimal performance impact**:
- Flush operations are cheap (microseconds)
- Only occur once per epoch (not in training loop)
- Console width and formatting don't affect computation

## Rollback

If issues occur, you can revert by:
```bash
# In SLURM scripts: remove -u flag
python trainer.py ...  # instead of python -u trainer.py ...

# In training.py: remove flush calls (optional)
# In pretty_printer.py: remove column widths (optional)
```

## Summary

✅ Full number values visible in training table  
✅ Real-time output in SLURM log files  
✅ No missing or delayed progress updates  
✅ Better debugging with immediate error messages  
✅ No performance impact on training

