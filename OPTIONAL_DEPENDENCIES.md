# Optional Dependencies - Safe Imports

## Overview

Some PhysNetJax modules have been updated with safe imports for optional dependencies. This means the core functionality works without these packages, but certain features (like plotting and TensorBoard logging) require them.

## Changes Made

### 1. **`mmml/physnetjax/physnetjax/utils/pretty_printer.py`**

**Optional Dependencies**:
- `asciichartpy` - ASCII chart plotting
- `polars` - DataFrame operations for plotting

**What Works Without Them**:
- ✅ All training table displays
- ✅ Model attribute printing
- ✅ Training progress tracking
- ✅ Optimizer/schedule information

**What Requires Them**:
- ⚠️ `get_panel()` - ASCII chart panels (gracefully degrades)
- ⚠️ `get_acp_plot()` - Polars-based plotting (shows warning)

**Error Handling**:
```python
# Before: ImportError on module load
import asciichartpy as acp

# After: Safe import with fallback
try:
    import asciichartpy as acp
    HAS_ASCIICHARTPY = True
except ImportError:
    HAS_ASCIICHARTPY = False
```

### 2. **`mmml/physnetjax/physnetjax/logger/tensorboard_interface.py`**

**Optional Dependencies**:
- `polars` - DataFrame operations
- `tensorboard` - TensorBoard event processing
- `tensorflow` - TensorFlow summary reading

**What Works Without Them**:
- ✅ All training functionality
- ✅ Model training without TensorBoard logging

**What Requires Them**:
- ⚠️ `tensorboard_to_polars()` - Converting TB logs to DataFrames
- ⚠️ `process_tensorboard_logs()` - Processing TB log directories

**Error Handling**:
Functions check dependencies and raise helpful errors:
```python
def _check_dependencies():
    missing = []
    if not HAS_POLARS:
        missing.append("polars")
    if not HAS_TENSORBOARD:
        missing.append("tensorboard")
    if not HAS_TENSORFLOW:
        missing.append("tensorflow")
    
    if missing:
        raise ImportError(
            f"TensorBoard logging requires: {', '.join(missing)}. "
            f"Install with: pip install {' '.join(missing)}"
        )
```

## Installation

See [INSTALLATION.md](INSTALLATION.md) for complete installation guide.

### Quick Install Options

**Minimal Installation (Core Training Only)**:
```bash
pip install -e .
```

**With Plotting Support**:
```bash
pip install -e ".[plotting]"
```

**With TensorBoard Support**:
```bash
pip install -e ".[tensorboard]"
```

**Training with Everything**:
```bash
pip install -e ".[plotting,tensorboard]"
```

### What Each Group Includes

**`plotting`** extras:
- `asciichartpy>=1.5.25` - ASCII chart visualization
- `polars` - DataFrame operations

**`tensorboard`** extras:
- `tensorboard>=2.11.0` - Event logging
- `tensorflow>=2.11.0` - Log file reading
- `polars` - Log processing

See `pyproject.toml` for complete dependency specifications.

## Usage Examples

### Training Without Optional Dependencies
```python
# This works fine without asciichartpy/polars
from mmml.physnetjax.physnetjax.restart.restart import get_params_model
from mmml.physnetjax.physnetjax.models.model import EF

params, model = get_params_model("checkpoint_path")
# ✅ Works!
```

### Using Plotting Features
```python
# Requires asciichartpy and polars
from mmml.physnetjax.physnetjax.utils.pretty_printer import get_acp_plot

try:
    get_acp_plot(data, keys=['loss', 'mae'])
except ImportError as e:
    print(f"Plotting unavailable: {e}")
    # Continue without plotting
```

### Using TensorBoard Features
```python
# Requires tensorboard, tensorflow, and polars
from mmml.physnetjax.physnetjax.logger.tensorboard_interface import process_tensorboard_logs

try:
    df = process_tensorboard_logs("logs/")
except ImportError as e:
    print(f"TensorBoard processing unavailable: {e}")
    # Use alternative logging method
```

## Benefits

✅ **Core functionality always works** - Training doesn't require optional packages  
✅ **Clear error messages** - Users know exactly what to install  
✅ **Graceful degradation** - Missing features don't break the code  
✅ **Easy debugging** - Import errors only occur when features are actually used  

## Troubleshooting

### "No module named 'asciichartpy'"
This is harmless unless you're using plotting features. To add plotting:
```bash
pip install asciichartpy
```

### "No module named 'polars'"
Only needed for DataFrame operations and plotting:
```bash
pip install polars
```

### "No module named 'tensorboard'" or "'tensorflow'"
Only needed for TensorBoard log processing:
```bash
pip install tensorboard tensorflow
```

### Check What's Available
```python
from mmml.physnetjax.physnetjax.utils.pretty_printer import (
    HAS_ASCIICHARTPY, 
    HAS_POLARS
)

print(f"ASCII charts: {HAS_ASCIICHARTPY}")
print(f"Polars: {HAS_POLARS}")

from mmml.physnetjax.physnetjax.logger.tensorboard_interface import (
    HAS_TENSORBOARD,
    HAS_TENSORFLOW
)

print(f"TensorBoard: {HAS_TENSORBOARD}")
print(f"TensorFlow: {HAS_TENSORFLOW}")
```

## Migration Guide

### For Existing Code

If you're using these modules, no changes needed! The code works the same way:

**Before and After - No Change Required**:
```python
from mmml.physnetjax.physnetjax.restart.restart import get_params_model

# Works exactly the same
params, model = get_params_model("checkpoint/")
```

### For New Code

If you want to use optional features, add error handling:

```python
from mmml.physnetjax.physnetjax.utils.pretty_printer import HAS_ASCIICHARTPY

if HAS_ASCIICHARTPY:
    # Use plotting features
    from mmml.physnetjax.physnetjax.utils.pretty_printer import get_acp_plot
    get_acp_plot(data, keys=['loss'])
else:
    # Fallback
    print("Plotting not available, skipping visualization")
```

## Summary

These changes make PhysNetJax more modular and easier to deploy:
- **Core training**: No extra dependencies needed
- **Visualization**: Optional plotting packages
- **Analysis**: Optional TensorBoard/DataFrame tools

The main workflow (training models, loading checkpoints, making predictions) works out of the box without any optional packages.

