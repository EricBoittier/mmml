"""
Example usage of the Molpro XML parser.
"""

import numpy as np
from read_molden import read_molpro_xml

# Example 1: Basic usage
def basic_example():
    """Load and display data from a Molpro XML file."""
    print("=" * 70)
    print("Example 1: Basic Data Loading")
    print("=" * 70)
    
    # Load the XML file
    data = read_molpro_xml('co2.xml')
    
    # Access molecular geometry as NumPy arrays
    print(f"\nAtomic numbers: {data.atomic_numbers}")
    print(f"Coordinates shape: {data.coordinates.shape}")
    print(f"Coordinates (Angstrom):\n{data.coordinates}")
    
    # Access energies
    print(f"\nEnergies available: {list(data.energies.keys())}")
    if 'RHF' in data.energies:
        print(f"RHF energy: {data.energies['RHF']:.10f} Hartree")
    
    # Access orbital data
    if data.orbital_energies is not None:
        print(f"\nNumber of orbitals: {len(data.orbital_energies)}")
        occupied = data.orbital_energies[data.orbital_occupancies > 0]
        unoccupied = data.orbital_energies[data.orbital_occupancies == 0]
        if len(occupied) > 0:
            print(f"HOMO energy: {occupied[-1]:.6f} Ha")
        if len(unoccupied) > 0:
            print(f"LUMO energy: {unoccupied[0]:.6f} Ha")
        else:
            print(f"LUMO: Not available (all orbitals occupied)")
    
    # Access properties
    if data.dipole_moment is not None:
        print(f"\nDipole moment: {data.dipole_moment}")
        print(f"Dipole magnitude: {np.linalg.norm(data.dipole_moment):.6f} Debye")
    
    if data.gradient is not None:
        print(f"\nGradient shape: {data.gradient.shape}")
        print(f"Max force: {np.max(np.abs(data.gradient)):.6e} Ha/Bohr")


# Example 2: Extract features for machine learning
def ml_features_example():
    """Extract features suitable for machine learning."""
    print("\n" + "=" * 70)
    print("Example 2: Machine Learning Feature Extraction")
    print("=" * 70)
    
    data = read_molpro_xml('co2.xml')
    
    # Create a feature vector for ML
    homo_energy = 0.0
    lumo_energy = 0.0
    if data.orbital_energies is not None:
        occupied = data.orbital_energies[data.orbital_occupancies > 0]
        unoccupied = data.orbital_energies[data.orbital_occupancies == 0]
        if len(occupied) > 0:
            homo_energy = occupied[-1]
        if len(unoccupied) > 0:
            lumo_energy = unoccupied[0]
    
    features = {
        'n_atoms': len(data.atomic_numbers),
        'atomic_numbers': data.atomic_numbers,
        'coordinates': data.coordinates.flatten(),  # Flatten to 1D
        'energy': data.energies.get('RHF', 0.0),
        'dipole_magnitude': np.linalg.norm(data.dipole_moment) if data.dipole_moment is not None else 0.0,
        'homo_energy': homo_energy,
        'lumo_energy': lumo_energy,
    }
    
    print("\nExtracted features for ML:")
    for key, value in features.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: array with shape {value.shape}")
        else:
            print(f"  {key}: {value}")
    
    # Example: compute derived features
    if data.coordinates is not None and data.atomic_numbers is not None:
        # Center of mass
        masses = {'C': 12.0, 'O': 16.0}
        element_map = {6: 'C', 8: 'O'}
        atom_masses = np.array([masses.get(element_map.get(z, 'H'), 1.0) for z in data.atomic_numbers])
        com = np.average(data.coordinates, axis=0, weights=atom_masses)
        print(f"\nCenter of mass: {com}")
        
        # Bond distances (for diatomic or simple molecules)
        if len(data.atomic_numbers) == 3:  # CO2 has 3 atoms
            d1 = np.linalg.norm(data.coordinates[1] - data.coordinates[0])
            d2 = np.linalg.norm(data.coordinates[2] - data.coordinates[0])
            print(f"Bond distances: {d1:.4f}, {d2:.4f} Angstrom")


# Example 3: Batch processing
def batch_processing_example():
    """Process multiple XML files (if available)."""
    print("\n" + "=" * 70)
    print("Example 3: Batch Processing")
    print("=" * 70)
    
    # In practice, you would have multiple files
    import glob
    xml_files = glob.glob('*.xml')
    
    print(f"\nFound {len(xml_files)} XML file(s)")
    
    # Collect data from all files
    all_energies = []
    all_dipoles = []
    
    for xml_file in xml_files:
        try:
            data = read_molpro_xml(xml_file)
            if 'RHF' in data.energies:
                all_energies.append(data.energies['RHF'])
            if data.dipole_moment is not None:
                all_dipoles.append(np.linalg.norm(data.dipole_moment))
        except Exception as e:
            print(f"Error processing {xml_file}: {e}")
    
    if all_energies:
        all_energies = np.array(all_energies)
        print(f"\nEnergy statistics:")
        print(f"  Mean: {np.mean(all_energies):.6f} Ha")
        print(f"  Std:  {np.std(all_energies):.6f} Ha")
        print(f"  Min:  {np.min(all_energies):.6f} Ha")
        print(f"  Max:  {np.max(all_energies):.6f} Ha")
    
    if all_dipoles:
        all_dipoles = np.array(all_dipoles)
        print(f"\nDipole magnitude statistics:")
        print(f"  Mean: {np.mean(all_dipoles):.6f} Debye")
        print(f"  Std:  {np.std(all_dipoles):.6f} Debye")


if __name__ == '__main__':
    # Run examples
    basic_example()
    ml_features_example()
    batch_processing_example()
    
    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)

