# Energy Preprocessing Quick Reference

## Quick Start

### Train with atomic energy subtraction:
```bash
python trainer.py \
  --train train.npz --valid valid.npz \
  --subtract-atomic-energies
```

### Train with per-atom scaling:
```bash
python trainer.py \
  --train train.npz --valid valid.npz \
  --scale-by-atoms
```

### Convert from Hartree to eV:
```bash
python trainer.py \
  --train train.npz --valid valid.npz \
  --energy-unit hartree --convert-energy-to eV
```

## All Options

| Flag | Default | Description |
|------|---------|-------------|
| `--energy-unit` | `eV` | Input energy unit: `eV`, `hartree`, `kcal/mol`, `kJ/mol` |
| `--convert-energy-to` | `None` | Convert to target unit |
| `--subtract-atomic-energies` | `False` | Remove atomic energy contributions |
| `--atomic-energy-method` | `linear_regression` | Method: `linear_regression` or `mean` |
| `--scale-by-atoms` | `False` | Scale energies by number of atoms |
| `--normalize-energy` | `False` | Normalize to mean=0, std=1 |

## Processing Order

1. Unit conversion (if requested)
2. Atomic energy subtraction (if requested)  
3. Per-atom scaling (if requested)
4. Normalization (if requested)

## Common Combinations

### Learning binding energies:
```bash
--subtract-atomic-energies --normalize-energy
```

### Mixed-size molecules:
```bash
--scale-by-atoms --normalize-energy
```

### QM data in Hartree:
```bash
--energy-unit hartree --convert-energy-to eV --subtract-atomic-energies
```

## Testing

```bash
python test_energy_preprocessing.py
```

## Documentation

See `ENERGY_PREPROCESSING.md` for detailed documentation.

