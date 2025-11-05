# Documentation

## Charge-Spin Conditioned PhysNet

Multi-state energy and force predictions with charge and spin inputs.

**Location**: `docs/charge_spin/`

**Usage**:
```python
from mmml.physnetjax.physnetjax.models.model_charge_spin import EF_ChargeSpinConditioned

model = EF_ChargeSpinConditioned(features=128, charge_range=(-2,2), spin_range=(1,5))
outputs = model.apply(params, Z, R, dst_idx, src_idx, total_charges=Q, total_spins=S)
```

**Examples**: `examples/train_charge_spin_simple.py`, `examples/predict_options_demo.py`

---

## Packed Memmap Data Loader

Efficient training on large datasets via memory-mapped files.

**Location**: `docs/memmap_loader/`

**Usage**:
```python
from mmml.data.packed_memmap_loader import PackedMemmapLoader

loader = PackedMemmapLoader("data_path", batch_size=32)
for batch in loader.batches(num_atoms=60):
    # train...
```

**Examples**: `examples/train_memmap_simple.py`

---

## AI Development Notes

**Location**: `AI/`

Internal notes on fixes, debugging, and development decisions.
