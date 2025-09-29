# MMML Demo Scripts

This directory contains several demo scripts that showcase different aspects of the MMML (Mixed Machine Learning / Molecular Mechanics) package.

## Demo Scripts

### `demo.py` - Core Comparison Demo
The main demo script that compares the hybrid MM/ML calculator against the pure ML ASE calculator. This is the core functionality for validating that the calculator plumbing is consistent.

**Usage:**
```bash
python demo.py [options]
```

**Key features:**
- Loads configurations from acetone dataset
- Compares hybrid vs pure ML calculators
- Reports energy and force differences
- Saves detailed comparison reports

### `demo_test_minimize.py` - Minimization Testing
Tests the minimization capabilities using both hybrid and pure ML calculators.

**Usage:**
```bash
python demo_test_minimize.py [options]
```

**Key features:**
- Tests ASE minimization with both calculators
- Compares minimization results
- Validates calculator consistency during optimization

### `demo_pdbfile.py` - PDB Processing and MD Simulation
Processes PDB files and runs molecular dynamics simulations using the hybrid MM/ML calculator with PyCHARMM integration.

**Usage:**
```bash
python demo_pdbfile.py --pdbfile <path_to_pdb> [options]
```

**Key features:**
- Loads PDB files with correct atom names and types
- Runs MD simulations with energy monitoring
- Handles energy spikes with re-minimization
- Saves trajectories and energy plots

### `demo_packmol.py` - Packmol Box Creation
Creates boxes of molecules using Packmol and processes them with the hybrid MM/ML calculator.

**Usage:**
```bash
python demo_packmol.py --molecule-file <path_to_molecule> [options]
```

**Key features:**
- Creates molecular boxes using Packmol
- Analyzes packed box properties
- Minimizes box structures
- Optional MD simulation on packed boxes

### `demo_base.py` - Common Utilities
Contains shared functionality used across all demo scripts.

**Key features:**
- Common argument parsing
- Dataset and checkpoint path resolution
- Model parameter loading
- Unit conversion utilities

## Common Options

All demo scripts share common command-line options:

- `--dataset`: Path to dataset file (defaults to environment variable or default path)
- `--checkpoint`: Path to checkpoint directory (defaults to environment variable or default path)
- `--sample-index`: Index of configuration to evaluate
- `--n-monomers`: Number of monomers in system
- `--atoms-per-monomer`: Number of atoms per monomer
- `--ml-cutoff`: ML cutoff distance
- `--mm-switch-on`: MM switch-on distance
- `--mm-cutoff`: MM cutoff width
- `--include-mm`: Enable MM contributions
- `--skip-ml-dimers`: Skip ML dimer correction
- `--debug`: Enable debug output
- `--units`: Output units (eV or kcal/mol)
- `--output`: Output file path

## Environment Variables

- `MMML_DATA`: Path to the acetone dataset
- `MMML_CKPT`: Path to the checkpoint directory

## Dependencies

The demo scripts require the following optional dependencies:
- ASE (Atomic Simulation Environment)
- PyCHARMM
- JAX
- e3x
- Packmol (for demo_packmol.py)
- Matplotlib (for plotting)

## Examples

### Basic comparison:
```bash
python demo.py
```

### Test minimization:
```bash
python demo_test_minimize.py
```

### Process PDB file:
```bash
python demo_pdbfile.py --pdbfile molecule.pdb --temperature 300 --num-steps 10000
```

### Create molecular box:
```bash
python demo_packmol.py --molecule-file acetone.pdb --box-size 20 --n-molecules 10
```
