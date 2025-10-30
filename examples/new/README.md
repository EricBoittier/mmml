# Molpro XML Parser

A Python parser to read Molpro XML output files (following the schema at https://www.molpro.net/schema/molpro-output.xsd) and convert data into NumPy arrays for further analysis or machine learning applications.

## Features

- **Molecular Geometry**: Extracts atomic numbers and Cartesian coordinates
- **Energies**: Parses energies from various methods (RHF, MP2, CCSD, etc.)
- **Molecular Orbitals**: Reads orbital energies, occupancies, and coefficients
- **Vibrational Data**: Extracts frequencies, normal modes, and IR intensities
- **Properties**: Parses dipole moments, gradients, and Hessians
- **NumPy Arrays**: All data returned as convenient NumPy arrays
- **Namespace Handling**: Automatically handles XML namespaces (CML, Molpro)

## Installation

No special installation required. Just ensure you have:
```bash
pip install numpy
```

## Quick Start

### Basic Usage

```python
from read_molden import read_molpro_xml

# Parse a Molpro XML file
data = read_molpro_xml('molpro_output.xml')

# Access data as NumPy arrays
print(f"Atomic numbers: {data.atomic_numbers}")
print(f"Coordinates:\n{data.coordinates}")
print(f"RHF Energy: {data.energies['RHF']:.10f} Hartree")
print(f"Dipole moment: {data.dipole_moment}")
```

### Command Line

```bash
python read_molden.py molpro_output.xml
```

## Data Structure

The `MolproData` class contains:

| Attribute | Type | Description |
|-----------|------|-------------|
| `atomic_numbers` | `np.ndarray(n_atoms,)` | Atomic numbers |
| `coordinates` | `np.ndarray(n_atoms, 3)` | Cartesian coordinates (Bohr or Angstrom) |
| `energies` | `Dict[str, float]` | Dictionary of energies by method |
| `orbital_energies` | `np.ndarray(n_orbitals,)` | Molecular orbital energies |
| `orbital_occupancies` | `np.ndarray(n_orbitals,)` | Orbital occupation numbers |
| `mo_coefficients` | `np.ndarray(n_basis, n_orbitals)` | MO coefficient matrix |
| `frequencies` | `np.ndarray(n_modes,)` | Vibrational frequencies (cm⁻¹) |
| `normal_modes` | `np.ndarray(n_modes, n_atoms, 3)` | Normal mode displacements |
| `intensities` | `np.ndarray(n_modes,)` | IR intensities |
| `dipole_moment` | `np.ndarray(3,)` | Dipole moment vector (Debye) |
| `gradient` | `np.ndarray(n_atoms, 3)` | Energy gradient |
| `hessian` | `np.ndarray(3*n_atoms, 3*n_atoms)` | Hessian matrix |

## Examples

See `example_usage.py` for more detailed examples including:
- Basic data extraction
- Feature extraction for machine learning
- Batch processing multiple files

### Example: Extract ML Features

```python
from read_molden import read_molpro_xml
import numpy as np

data = read_molpro_xml('molpro_output.xml')

# Create feature vector for ML
features = {
    'energy': data.energies.get('RHF', 0.0),
    'dipole_magnitude': np.linalg.norm(data.dipole_moment),
    'coordinates': data.coordinates.flatten(),
    'atomic_numbers': data.atomic_numbers,
}
```

### Example: Orbital Analysis

```python
data = read_molpro_xml('molpro_output.xml')

# Find HOMO and LUMO
occupied = data.orbital_energies[data.orbital_occupancies > 0]
unoccupied = data.orbital_energies[data.orbital_occupancies == 0]

homo_energy = occupied[-1]
lumo_energy = unoccupied[0]
homo_lumo_gap = lumo_energy - homo_energy

print(f"HOMO-LUMO gap: {homo_lumo_gap:.4f} Ha")
```

## Supported XML Formats

The parser supports:
- CML (Chemical Markup Language) format used by Molpro
- Both namespaced and non-namespaced XML
- Multiple energy methods (RHF, MP2, CCSD, CCSD(T), CI, MCSCF)
- Various property types from Molpro output

## Notes

- The parser automatically detects and handles XML namespaces
- If multiple values exist for the same property (e.g., from a scan), the last one is typically returned
- All coordinates are returned as provided in the XML (usually Angstrom, but check your Molpro input)
- Missing data fields are set to `None`

## Testing

Run the examples:
```bash
python example_usage.py
```

## Schema Reference

Molpro XML schema: https://www.molpro.net/schema/molpro-output.xsd

