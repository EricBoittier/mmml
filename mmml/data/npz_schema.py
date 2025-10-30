"""
Standardized NPZ data format schema for MMML.

Defines the canonical structure for all NPZ files used across DCMNet,
PhysNetJAX, and other models in the MMML ecosystem.

Units Convention:
    - Coordinates (R): Angstrom
    - Energies (E): Hartree
    - Forces (F): Hartree/Bohr
    - Dipoles (D): Debye
    - ESP: Hartree/e
"""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field


# Required keys that must be present in every NPZ file
REQUIRED_KEYS: Dict[str, str] = {
    'R': 'Coordinates (n_structures, n_atoms, 3) [Angstrom]',
    'Z': 'Atomic numbers (n_structures, n_atoms) [int]',
    'E': 'Energies (n_structures,) [Hartree]',
    'N': 'Number of atoms per structure (n_structures,) [int]',
}

# Optional keys for additional properties
OPTIONAL_KEYS: Dict[str, str] = {
    'F': 'Forces/gradients (n_structures, n_atoms, 3) [Hartree/Bohr]',
    'D': 'Dipole moments (n_structures, 3) [Debye]',
    'esp': 'Electrostatic potential values (n_structures, n_grid)',
    'esp_grid': 'ESP grid coordinates (n_structures, n_grid, 3) [Angstrom]',
    'vdw_surface': 'VDW surface points (n_structures, n_surface, 3) [Angstrom]',
    'mono': 'Atomic monopoles/charges (n_structures, n_atoms)',
    'polar': 'Polarizability tensor (n_structures, 3, 3)',
    'quadrupole': 'Quadrupole tensor (n_structures, 3, 3)',
    'n_grid': 'Number of grid points per structure (n_structures,) [int]',
    'com': 'Center of mass (n_structures, 3) [Angstrom]',
    'id': 'Structure identifiers (n_structures,) [str or int]',
    'Dxyz': 'Dipole components (n_structures, 3) [Debye]',
    'espMask': 'ESP mask for VDW exclusion (n_structures, n_grid) [bool]',
}

# Metadata keys (stored as pickled dict or JSON)
METADATA_KEYS: Dict[str, str] = {
    'molpro_variables': 'Dictionary of Molpro internal variables',
    'generation_date': 'ISO format timestamp of generation',
    'molpro_version': 'Molpro version string',
    'basis_set': 'Basis set used for calculation',
    'method': 'Calculation method (RHF, MP2, CCSD, etc.)',
    'units': 'Dictionary mapping property names to units',
    'source_files': 'List of source XML files',
    'conversion_info': 'Information about XML → NPZ conversion',
}


@dataclass
class NPZSchema:
    """
    Schema validator and metadata for NPZ files.
    
    Attributes
    ----------
    required_keys : Set[str]
        Keys that must be present
    optional_keys : Set[str]
        Keys that may be present
    metadata_keys : Set[str]
        Metadata keys
    strict : bool
        If True, raise error on unknown keys; if False, just warn
    """
    
    required_keys: Set[str] = field(default_factory=lambda: set(REQUIRED_KEYS.keys()))
    optional_keys: Set[str] = field(default_factory=lambda: set(OPTIONAL_KEYS.keys()))
    metadata_keys: Set[str] = field(default_factory=lambda: set(METADATA_KEYS.keys()))
    strict: bool = False
    
    def validate(self, data: Dict[str, np.ndarray]) -> Tuple[bool, List[str]]:
        """
        Validate NPZ data against schema.
        
        Parameters
        ----------
        data : dict
            Dictionary of arrays from NPZ file
            
        Returns
        -------
        tuple
            (is_valid, error_messages)
        """
        errors = []
        
        # Check required keys
        missing_required = self.required_keys - set(data.keys())
        if missing_required:
            errors.append(f"Missing required keys: {missing_required}")
        
        # Check for unknown keys
        all_known_keys = self.required_keys | self.optional_keys | self.metadata_keys
        unknown_keys = set(data.keys()) - all_known_keys
        if unknown_keys:
            msg = f"Unknown keys: {unknown_keys}"
            if self.strict:
                errors.append(msg)
            else:
                print(f"Warning: {msg}")
        
        # Validate shapes
        if 'R' in data and 'Z' in data:
            r_shape = data['R'].shape
            z_shape = data['Z'].shape
            if len(r_shape) != 3 or r_shape[-1] != 3:
                errors.append(f"'R' must have shape (n_structures, n_atoms, 3), got {r_shape}")
            if len(z_shape) != 2:
                errors.append(f"'Z' must have shape (n_structures, n_atoms), got {z_shape}")
            if r_shape[:2] != z_shape:
                errors.append(f"'R' and 'Z' shape mismatch: {r_shape[:2]} vs {z_shape}")
        
        # Validate N matches actual number of atoms
        if 'N' in data and 'Z' in data:
            n_atoms_from_z = data['Z'].shape[1] if len(data['Z'].shape) > 1 else data['Z'].shape[0]
            if not np.all(data['N'] <= n_atoms_from_z):
                errors.append(f"'N' values exceed array dimensions")
        
        # Validate energy shape
        if 'E' in data:
            if len(data['E'].shape) not in [1, 2]:
                errors.append(f"'E' must have shape (n_structures,) or (n_structures, 1), got {data['E'].shape}")
        
        # Validate forces shape
        if 'F' in data and 'R' in data:
            if data['F'].shape != data['R'].shape:
                errors.append(f"'F' and 'R' shape mismatch: {data['F'].shape} vs {data['R'].shape}")
        
        # Validate dipole shape
        if 'D' in data:
            if len(data['D'].shape) != 2 or data['D'].shape[1] != 3:
                errors.append(f"'D' must have shape (n_structures, 3), got {data['D'].shape}")
        
        # Validate ESP data
        if 'esp' in data and 'esp_grid' in data:
            esp_shape = data['esp'].shape
            grid_shape = data['esp_grid'].shape
            if len(grid_shape) != 3 or grid_shape[-1] != 3:
                errors.append(f"'esp_grid' must have shape (n_structures, n_grid, 3), got {grid_shape}")
            if len(esp_shape) == 2 and esp_shape[1] != grid_shape[1]:
                errors.append(f"'esp' and 'esp_grid' n_grid mismatch: {esp_shape[1]} vs {grid_shape[1]}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def get_info(self, data: Dict[str, np.ndarray]) -> Dict:
        """
        Get information summary about the dataset.
        
        Parameters
        ----------
        data : dict
            Dictionary of arrays from NPZ file
            
        Returns
        -------
        dict
            Summary information
        """
        info = {
            'n_structures': len(data.get('E', [])),
            'n_atoms': data['Z'].shape[1] if 'Z' in data else None,
            'properties': list(data.keys()),
            'required_keys_present': list(self.required_keys & set(data.keys())),
            'optional_keys_present': list(self.optional_keys & set(data.keys())),
        }
        
        # Add property-specific info
        if 'R' in data:
            info['coordinate_range'] = {
                'min': float(data['R'].min()),
                'max': float(data['R'].max()),
            }
        
        if 'E' in data:
            info['energy_range'] = {
                'min': float(data['E'].min()),
                'max': float(data['E'].max()),
                'mean': float(data['E'].mean()),
                'std': float(data['E'].std()),
            }
        
        if 'Z' in data:
            unique_elements = np.unique(data['Z'][data['Z'] > 0])
            info['unique_elements'] = unique_elements.tolist()
            info['element_counts'] = {
                int(z): int(np.sum(data['Z'] == z))
                for z in unique_elements
            }
        
        return info


def validate_npz(
    file_path: str,
    strict: bool = False,
    verbose: bool = True
) -> Tuple[bool, Optional[Dict]]:
    """
    Validate an NPZ file against the MMML schema.
    
    Parameters
    ----------
    file_path : str
        Path to NPZ file
    strict : bool, optional
        If True, fail on unknown keys, by default False
    verbose : bool, optional
        If True, print detailed information, by default True
        
    Returns
    -------
    tuple
        (is_valid, info_dict or None)
        
    Examples
    --------
    >>> is_valid, info = validate_npz('data.npz')
    >>> if is_valid:
    ...     print(f"Dataset contains {info['n_structures']} structures")
    """
    try:
        data = np.load(file_path, allow_pickle=True)
        schema = NPZSchema(strict=strict)
        
        # Validate
        is_valid, errors = schema.validate(data)
        
        if verbose:
            if is_valid:
                print(f"✓ NPZ file '{file_path}' is valid")
            else:
                print(f"✗ NPZ file '{file_path}' has errors:")
                for error in errors:
                    print(f"  - {error}")
        
        # Get info
        info = schema.get_info(data) if is_valid else None
        
        if verbose and info:
            print(f"\nDataset Summary:")
            print(f"  Structures: {info['n_structures']}")
            print(f"  Atoms per structure: {info['n_atoms']}")
            print(f"  Properties: {', '.join(info['properties'])}")
            if 'unique_elements' in info:
                print(f"  Elements: {info['unique_elements']}")
        
        return is_valid, info
        
    except Exception as e:
        if verbose:
            print(f"✗ Error loading NPZ file: {e}")
        return False, None


def create_empty_npz(
    n_structures: int,
    n_atoms: int,
    properties: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """
    Create an empty NPZ dataset with the correct structure.
    
    Parameters
    ----------
    n_structures : int
        Number of structures
    n_atoms : int
        Maximum number of atoms per structure
    properties : list, optional
        List of optional properties to include, by default None
        
    Returns
    -------
    dict
        Dictionary of empty arrays
    """
    data = {
        'R': np.zeros((n_structures, n_atoms, 3)),
        'Z': np.zeros((n_structures, n_atoms), dtype=np.int32),
        'E': np.zeros(n_structures),
        'N': np.zeros(n_structures, dtype=np.int32),
    }
    
    if properties:
        for prop in properties:
            if prop == 'F':
                data['F'] = np.zeros((n_structures, n_atoms, 3))
            elif prop == 'D' or prop == 'Dxyz':
                data[prop] = np.zeros((n_structures, 3))
            elif prop == 'mono':
                data['mono'] = np.zeros((n_structures, n_atoms))
            elif prop == 'polar' or prop == 'quadrupole':
                data[prop] = np.zeros((n_structures, 3, 3))
    
    return data


def main():
    """CLI entry point for validation."""
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        is_valid, info = validate_npz(file_path, verbose=True)
        sys.exit(0 if is_valid else 1)
    else:
        print("Usage: python npz_schema.py <npz_file>")
        print("   or: mmml validate <npz_file>")
        print("\nSchema Information:")
        print("\nRequired keys:")
        for key, desc in REQUIRED_KEYS.items():
            print(f"  {key}: {desc}")
        print("\nOptional keys:")
        for key, desc in OPTIONAL_KEYS.items():
            print(f"  {key}: {desc}")
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())

