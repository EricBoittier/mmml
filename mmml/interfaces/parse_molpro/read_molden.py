"""
Read Molpro XML output files and convert data to NumPy arrays.
Schema: https://www.molpro.net/schema/molpro-output.xsd
"""

import xml.etree.ElementTree as ET
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field


@dataclass
class MolproData:
    """Container for Molpro XML data as NumPy arrays."""
    
    # Molecular geometry
    atomic_numbers: Optional[np.ndarray] = None  # (n_atoms,)
    coordinates: Optional[np.ndarray] = None  # (n_atoms, 3) in Bohr or Angstrom
    
    # Energies
    energies: Dict[str, float] = field(default_factory=dict)  # e.g., 'scf', 'mp2', 'ccsd'
    
    # Molecular orbitals
    orbital_energies: Optional[np.ndarray] = None  # (n_orbitals,)
    orbital_occupancies: Optional[np.ndarray] = None  # (n_orbitals,)
    mo_coefficients: Optional[np.ndarray] = None  # (n_basis, n_orbitals)
    
    # Basis set information
    basis_functions: Optional[np.ndarray] = None
    
    # Vibrational frequencies
    frequencies: Optional[np.ndarray] = None  # (n_modes,)
    normal_modes: Optional[np.ndarray] = None  # (n_modes, n_atoms, 3)
    intensities: Optional[np.ndarray] = None  # (n_modes,)
    
    # Dipole moment
    dipole_moment: Optional[np.ndarray] = None  # (3,)
    
    # Gradient
    gradient: Optional[np.ndarray] = None  # (n_atoms, 3)
    
    # Hessian
    hessian: Optional[np.ndarray] = None  # (3*n_atoms, 3*n_atoms)
    
    # Molpro variables
    variables: Dict[str, Union[float, np.ndarray]] = field(default_factory=dict)
    
    # Cube data (ESP, density, etc.)
    cube_data: Dict[str, Dict[str, Union[np.ndarray, str]]] = field(default_factory=dict)
    # Format: {'esp': {'values': np.ndarray, 'origin': np.ndarray, 'axes': np.ndarray, 
    #                   'dimensions': np.ndarray, 'file': str}, ...}
    
    # Other properties
    properties: Dict[str, Union[float, np.ndarray]] = field(default_factory=dict)


class MolproXMLParser:
    """Parser for Molpro XML output files."""
    
    def __init__(self, xml_file: str):
        """
        Initialize parser with XML file.
        
        Args:
            xml_file: Path to Molpro XML output file
        """
        self.xml_file = xml_file
        self.tree = ET.parse(xml_file)
        self.root = self.tree.getroot()
        
        # Handle namespace if present
        self.ns = {}
        if '}' in self.root.tag:
            self.ns['molpro'] = self.root.tag.split('}')[0].strip('{')
    
    def _find_all(self, path: str, use_cml: bool = False) -> List[ET.Element]:
        """Find all elements matching path, handling namespace."""
        if self.ns:
            # Handle both molpro and cml namespaces
            if use_cml:
                # Replace path separators with cml namespace
                parts = path.split('/')
                namespaced_parts = []
                for part in parts:
                    if part and not part.startswith('.') and ':' not in part:
                        namespaced_parts.append(f'cml:{part}')
                    else:
                        namespaced_parts.append(part)
                path = '/'.join(namespaced_parts)
                ns = {**self.ns, 'cml': 'http://www.xml-cml.org/schema'}
                return self.root.findall(path, ns)
            else:
                # Replace path separators with molpro namespace
                parts = path.split('/')
                namespaced_parts = []
                for part in parts:
                    if part and not part.startswith('.') and ':' not in part:
                        namespaced_parts.append(f'molpro:{part}')
                    else:
                        namespaced_parts.append(part)
                path = '/'.join(namespaced_parts)
                return self.root.findall(path, self.ns)
        return self.root.findall(path, {})
    
    def _find(self, path: str, use_cml: bool = False) -> Optional[ET.Element]:
        """Find first element matching path, handling namespace."""
        if self.ns:
            # Handle both molpro and cml namespaces
            if use_cml:
                # Replace path separators with cml namespace
                parts = path.split('/')
                namespaced_parts = []
                for part in parts:
                    if part and not part.startswith('.') and ':' not in part:
                        namespaced_parts.append(f'cml:{part}')
                    else:
                        namespaced_parts.append(part)
                path = '/'.join(namespaced_parts)
                ns = {**self.ns, 'cml': 'http://www.xml-cml.org/schema'}
                return self.root.find(path, ns)
            else:
                # Replace path separators with molpro namespace
                parts = path.split('/')
                namespaced_parts = []
                for part in parts:
                    if part and not part.startswith('.') and ':' not in part:
                        namespaced_parts.append(f'molpro:{part}')
                    else:
                        namespaced_parts.append(part)
                path = '/'.join(namespaced_parts)
                return self.root.find(path, self.ns)
        return self.root.find(path, {})
    
    def parse_geometry(self, use_last: bool = True) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Parse molecular geometry from CML format.
        
        Args:
            use_last: If True, use the last molecule entry (final geometry).
                     If False, use the first molecule entry.
        
        Returns:
            Tuple of (atomic_numbers, coordinates)
        """
        # Periodic table for element symbol to atomic number conversion
        element_to_z = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
            'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
            'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
            'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
            'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
            'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43,
            'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
            'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54
        }
        
        # Try CML format first (most common in Molpro output)
        ns = {**self.ns, 'cml': 'http://www.xml-cml.org/schema'} if self.ns else {}
        
        # Find all molecules, then select first or last
        molecules = self._find_all('.//cml:molecule', use_cml=True)
        molecule = None
        if molecules:
            molecule = molecules[-1] if use_last else molecules[0]
        elif not molecules:
            # Fallback to find() for single molecule
            molecule = self._find('.//cml:molecule', use_cml=True)
        
        atoms = []
        coords = []
        
        if molecule is not None:
            atom_array = molecule.find('cml:atomArray' if self.ns else 'atomArray', ns)
            if atom_array is not None:
                for atom in atom_array.findall('cml:atom' if self.ns else 'atom', ns):
                    # Get element type
                    element = atom.get('elementType')
                    if element and element in element_to_z:
                        atoms.append(element_to_z[element])
                    elif atom.get('elementNumber'):
                        atoms.append(int(atom.get('elementNumber')))
                    
                    # Get coordinates
                    x = atom.get('x3')
                    y = atom.get('y3')
                    z_coord = atom.get('z3')
                    
                    if x and y and z_coord:
                        coords.append([float(x), float(y), float(z_coord)])
        
        # Fallback: try non-CML format
        if not atoms:
            molecule = self._find('.//molecule')
            if molecule is not None:
                for atom in molecule.findall('atom' if not self.ns else 'molpro:atom', self.ns):
                    z = atom.get('elementNumber')
                    if z:
                        atoms.append(int(z))
                    
                    x = atom.get('x3')
                    y = atom.get('y3')
                    z_coord = atom.get('z3')
                    
                    if x and y and z_coord:
                        coords.append([float(x), float(y), float(z_coord)])
        
        atomic_numbers = np.array(atoms) if atoms else None
        coordinates = np.array(coords) if coords else None
        
        return atomic_numbers, coordinates
    
    def parse_energies(self) -> Dict[str, float]:
        """
        Parse energies from various methods.
        
        Returns:
            Dictionary mapping method name to energy value
        """
        energies = {}
        
        # Look for property elements with name="Energy"
        for energy_elem in self._find_all('.//property'):
            name = energy_elem.get('name')
            if name == 'Energy':
                method = energy_elem.get('method', 'unknown')
                value = energy_elem.get('value')
                if value:
                    energies[method] = float(value)
        
        return energies
    
    def parse_orbitals(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Parse molecular orbital data.
        
        Returns:
            Tuple of (orbital_energies, occupancies, mo_coefficients)
        """
        orbitals = self._find('.//orbitals')
        if orbitals is None:
            return None, None, None
        
        energies = []
        occupancies = []
        coefficients = []
        
        for orbital in orbitals.findall('orbital' if not self.ns else 'molpro:orbital', self.ns):
            energy = orbital.get('energy')
            occ = orbital.get('occupation')
            
            if energy:
                energies.append(float(energy))
            if occ:
                occupancies.append(float(occ))
            
            # Parse coefficients if present
            coeff_text = orbital.text
            if coeff_text:
                coeff_list = [float(c) for c in coeff_text.split()]
                coefficients.append(coeff_list)
        
        orbital_energies = np.array(energies) if energies else None
        orbital_occupancies = np.array(occupancies) if occupancies else None
        
        # Convert coefficients to 2D array (n_basis x n_orbitals)
        if coefficients and all(len(c) == len(coefficients[0]) for c in coefficients):
            mo_coefficients = np.array(coefficients).T
        else:
            mo_coefficients = None
        
        return orbital_energies, orbital_occupancies, mo_coefficients
    
    def parse_frequencies(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Parse vibrational frequencies and normal modes according to XSD schema.
        
        Returns:
            Tuple of (frequencies, normal_modes, intensities)
        """
        vibrations = self._find('.//vibrations')
        if vibrations is None:
            return None, None, None
        
        freqs = []
        modes = []
        intens = []
        
        # Schema uses normalCoordinate elements with wavenumber and IRintensity attributes
        for normal_coord in vibrations.findall('normalCoordinate' if not self.ns else 'molpro:normalCoordinate', self.ns):
            # Get frequency from wavenumber attribute
            wavenumber = normal_coord.get('wavenumber')
            if wavenumber:
                freqs.append(float(wavenumber))
            
            # Get IR intensity
            ir_intensity = normal_coord.get('IRintensity')
            if ir_intensity:
                intens.append(float(ir_intensity))
            
            # Parse normal mode displacements from text content
            mode_text = normal_coord.text
            if mode_text:
                mode_vals = [float(v) for v in mode_text.split()]
                # Reshape to (n_atoms, 3)
                if len(mode_vals) >= 3:
                    n_atoms = len(mode_vals) // 3
                    mode_array = np.array(mode_vals[:n_atoms*3]).reshape(n_atoms, 3)
                    modes.append(mode_array)
        
        frequencies = np.array(freqs) if freqs else None
        intensities = np.array(intens) if intens else None
        normal_modes = np.array(modes) if modes else None
        
        return frequencies, normal_modes, intensities
    
    def parse_dipole(self) -> Optional[np.ndarray]:
        """
        Parse dipole moment.
        
        Returns:
            Dipole moment vector (3,)
        """
        # Look for property with name="Dipole moment"
        for dipole in self._find_all('.//property'):
            name = dipole.get('name')
            if name == 'Dipole moment':
                value = dipole.get('value')
                if value:
                    # Value is space-separated string of x, y, z components
                    values = [float(v) for v in value.split()]
                    if len(values) == 3:
                        return np.array(values)
        
        # Fallback: try other formats
        dipole = self._find('.//dipole')
        if dipole is not None:
            x = dipole.get('x')
            y = dipole.get('y')
            z = dipole.get('z')
            
            if x and y and z:
                return np.array([float(x), float(y), float(z)])
            
            # Try parsing from text
            text = dipole.text
            if text:
                values = [float(v) for v in text.split()]
                if len(values) == 3:
                    return np.array(values)
        
        return None
    
    def parse_gradient(self, use_last: bool = True) -> Optional[np.ndarray]:
        """
        Parse energy gradient.
        
        Args:
            use_last: If True, use the last gradient entry (final geometry).
                     If False, use the first gradient entry.
        
        Returns:
            Gradient array (n_atoms, 3)
        """
        gradients = self._find_all('.//gradient')
        gradient = None
        if gradients:
            gradient = gradients[-1] if use_last else gradients[0]
        elif not gradients:
            gradient = self._find('.//gradient')
        
        if gradient is None:
            return None
        
        grad_text = gradient.text
        if grad_text:
            values = [float(v) for v in grad_text.split()]
            n_atoms = len(values) // 3
            return np.array(values).reshape(n_atoms, 3)
        
        return None
    
    def parse_hessian(self) -> Optional[np.ndarray]:
        """
        Parse Hessian matrix.
        
        Returns:
            Hessian array (3*n_atoms, 3*n_atoms)
        """
        hessian = self._find('.//hessian')
        if hessian is None:
            return None
        
        hess_text = hessian.text
        if hess_text:
            values = [float(v) for v in hess_text.split()]
            n = int(np.sqrt(len(values)))
            if n * n == len(values):
                return np.array(values).reshape(n, n)
        
        return None
    
    def parse_variables(self) -> Dict[str, Union[float, np.ndarray]]:
        """
        Parse Molpro internal variables according to XSD schema.
        
        Returns:
            Dictionary mapping variable name to value
        """
        variables = {}
        
        # Find all variable elements within variables containers
        for variables_container in self._find_all('.//variables'):
            for var in variables_container.findall('variable' if not self.ns else 'molpro:variable', self.ns):
                name = var.get('name')
                var_type = var.get('type')
                length = var.get('length')
                
                # Get value from value element or text content
                value_elem = var.find('value' if not self.ns else 'molpro:value', self.ns)
                if value_elem is not None:
                    value_text = value_elem.text
                else:
                    value_text = var.text
                
                if name and value_text:
                    try:
                        # Parse the value based on length
                        if length and int(length) > 1:
                            # Array variable
                            values = [float(v) for v in value_text.split()]
                            variables[name] = np.array(values)
                        else:
                            # Scalar variable
                            variables[name] = float(value_text)
                    except (ValueError, AttributeError):
                        # If parsing fails, store as string
                        variables[name] = value_text.strip()
        
        return variables
    
    def parse_cubes(self, xml_dir: Optional[str] = None) -> Dict[str, Dict[str, Union[np.ndarray, str]]]:
        """
        Parse cube file metadata and optionally load cube data.
        
        Args:
            xml_dir: Directory containing the XML file (for locating cube files).
                    If None, cube files won't be loaded.
        
        Returns:
            Dictionary mapping cube quantity (e.g., 'esp', 'density') to cube data
        """
        from pathlib import Path
        
        cube_data = {}
        
        # Find all cube elements
        cubes = self._find_all('.//cube')
        
        for cube_elem in cubes:
            # Get field information
            field_elem = cube_elem.find('field' if not self.ns else 'molpro:field', self.ns)
            if field_elem is None:
                continue
            
            quantity = field_elem.get('quantity', 'unknown').lower()
            cube_file = field_elem.get('file')
            field_type = field_elem.get('type', '')
            
            # Parse grid metadata
            origin_elem = cube_elem.find('origin' if not self.ns else 'molpro:origin', self.ns)
            axes_elem = cube_elem.find('axes' if not self.ns else 'molpro:axes', self.ns)
            dimensions_elem = cube_elem.find('dimensions' if not self.ns else 'molpro:dimensions', self.ns)
            step_elem = cube_elem.find('step' if not self.ns else 'molpro:step', self.ns)
            
            cube_info = {
                'file': cube_file,
                'type': field_type,
                'method': cube_elem.get('method', ''),
            }

            # print(origin_elem.text)
            # print(axes_elem.text)
            # print(dimensions_elem.text)
            # print(step_elem.text)
            # print(cube_file)
            # print(field_type)
            # print(cube_elem.get('method', ''))
            # print(cube_elem.get('quantity', 'unknown'))
            # print(cube_elem.get('type', ''))
            # print(cube_elem.get('number', ''))
            # print(cube_elem.get('symmetry', ''))
            # print(cube_elem.get('occupancy', ''))
            # print(cube_elem.get('energy', ''))
            
            # Parse grid parameters
            if origin_elem is not None and origin_elem.text:
                cube_info['origin'] = np.array([float(v) for v in origin_elem.text.split()])
            
            if axes_elem is not None and axes_elem.text:
                axes_values = [float(v) for v in axes_elem.text.split()]
                # Reshape to 3x3 matrix
                if len(axes_values) == 9:
                    cube_info['axes'] = np.array(axes_values).reshape(3, 3)
                else:
                    cube_info['axes'] = np.array(axes_values)
            
            if dimensions_elem is not None and dimensions_elem.text:
                cube_info['dimensions'] = np.array([int(float(v)) for v in dimensions_elem.text.split()])
            
            if step_elem is not None and step_elem.text:
                cube_info['step'] = np.array([float(v) for v in step_elem.text.split()])
            
            # Try to load cube file if path provided
            if cube_file and xml_dir:
                cube_path = Path(xml_dir) / cube_file
                if cube_path.exists():
                    try:
                        cube_info['values'] = self._read_cube_file(cube_path)
                    except Exception as e:
                        print(f"Warning: Could not read cube file {cube_path}: {e}")
            
            # Store with descriptive key
            key = quantity if quantity != 'unknown' else field_type
            if key:
                cube_data[key] = cube_info
        
        return cube_data
    
    def _read_cube_file(self, cube_path: str) -> np.ndarray:
        """
        Read Gaussian cube file format.
        
        Args:
            cube_path: Path to cube file
            
        Returns:
            3D array of cube values
        """
        with open(cube_path, 'r') as f:
            lines = f.readlines()
        
        # Skip first two comment lines
        idx = 2
        
        # Read number of atoms and origin
        parts = lines[idx].split()
        n_atoms = abs(int(parts[0]))
        idx += 1
        
        # Read grid dimensions
        nx = int(lines[idx].split()[0])
        idx += 1
        ny = int(lines[idx].split()[0])
        idx += 1
        nz = int(lines[idx].split()[0])
        idx += 1
        
        # Skip atom lines
        idx += n_atoms
        
        # Read cube data
        values = []
        for line in lines[idx:]:
            values.extend([float(v) for v in line.split()])
        
        # Reshape to 3D grid
        cube_array = np.array(values).reshape(nx, ny, nz)
        
        return cube_array
    
    def parse_all(self, use_last_geometry: bool = True, load_cubes: bool = True) -> MolproData:
        """
        Parse all available data from XML file.
        
        Args:
            use_last_geometry: If True (default), use the last geometry/gradient from
                             the XML file (e.g., final optimized structure). If False,
                             use the first geometry.
            load_cubes: If True (default), load cube file data (ESP, density) if available.
        
        Returns:
            MolproData object containing all parsed arrays
        """
        from pathlib import Path
        
        data = MolproData()
        
        # Parse geometry (use last by default for optimized structures)
        data.atomic_numbers, data.coordinates = self.parse_geometry(use_last=use_last_geometry)
        
        # Parse energies
        data.energies = self.parse_energies()
        
        # Parse orbitals
        data.orbital_energies, data.orbital_occupancies, data.mo_coefficients = self.parse_orbitals()
        
        # Parse vibrational data
        data.frequencies, data.normal_modes, data.intensities = self.parse_frequencies()
        
        # Parse properties
        data.dipole_moment = self.parse_dipole()
        data.gradient = self.parse_gradient(use_last=use_last_geometry)
        data.hessian = self.parse_hessian()
        
        # Parse Molpro variables
        data.variables = self.parse_variables()
        
        # Parse cube data (ESP, density, etc.)
        if load_cubes:
            xml_dir = Path(self.xml_file).parent
            data.cube_data = self.parse_cubes(xml_dir=str(xml_dir))
        
        return data


def read_molpro_xml(xml_file: str, use_last_geometry: bool = True, load_cubes: bool = True) -> MolproData:
    """
    Read Molpro XML output file and return data as NumPy arrays.
    
    Args:
        xml_file: Path to Molpro XML output file
        use_last_geometry: If True (default), extract the last geometry from files
                         with multiple geometries (e.g., optimization trajectories).
                         If False, use the first geometry.
        load_cubes: If True (default), load cube file data (ESP, density, etc.) if available.
        
    Returns:
        MolproData object containing parsed arrays
        
    Example:
        >>> data = read_molpro_xml('molpro_output.xml')
        >>> print(f"Coordinates shape: {data.coordinates.shape}")
        >>> print(f"SCF energy: {data.energies.get('scf')}")
        >>> print(f"Orbital energies: {data.orbital_energies}")
        >>> if 'esp' in data.cube_data:
        ...     print(f"ESP cube shape: {data.cube_data['esp']['values'].shape}")
    """
    parser = MolproXMLParser(xml_file)
    return parser.parse_all(use_last_geometry=use_last_geometry, load_cubes=load_cubes)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python read_molden.py <molpro_xml_file>")
        sys.exit(1)
    
    xml_file = sys.argv[1]
    
    # Parse the XML file
    data = read_molpro_xml(xml_file)
    
    # Print summary of parsed data
    print(f"Parsed Molpro XML file: {xml_file}")
    print("\n" + "="*60)
    
    if data.atomic_numbers is not None:
        print(f"\nMolecular Geometry:")
        print(f"  Number of atoms: {len(data.atomic_numbers)}")
        print(f"  Atomic numbers: {data.atomic_numbers}")
        if data.coordinates is not None:
            print(f"  Coordinates shape: {data.coordinates.shape}")
            print(f"  Coordinates:\n{data.coordinates}")
    
    if data.energies:
        print(f"\nEnergies:")
        for method, energy in data.energies.items():
            print(f"  {method}: {energy:.10f} Hartree")
    
    if data.orbital_energies is not None:
        print(f"\nMolecular Orbitals:")
        print(f"  Number of orbitals: {len(data.orbital_energies)}")
        print(f"  Orbital energies (first 5): {data.orbital_energies[:5]}")
        if data.mo_coefficients is not None:
            print(f"  MO coefficients shape: {data.mo_coefficients.shape}")
    
    if data.frequencies is not None:
        print(f"\nVibrational Frequencies:")
        print(f"  Number of modes: {len(data.frequencies)}")
        print(f"  Frequencies (cm⁻¹): {data.frequencies}")
        if data.normal_modes is not None:
            print(f"  Normal modes shape: {data.normal_modes.shape}")
    
    if data.dipole_moment is not None:
        print(f"\nDipole Moment:")
        print(f"  {data.dipole_moment}")
        print(f"  Magnitude: {np.linalg.norm(data.dipole_moment):.6f} Debye")
    
    if data.gradient is not None:
        print(f"\nGradient:")
        print(f"  Shape: {data.gradient.shape}")
        print(f"  Max gradient component: {np.max(np.abs(data.gradient)):.6e}")
    
    if data.hessian is not None:
        print(f"\nHessian:")
        print(f"  Shape: {data.hessian.shape}")
    
    if data.variables:
        print(f"\nMolpro Variables:")
        print(f"  Number of variables: {len(data.variables)}")
        # Show first few variables
        for i, (name, value) in enumerate(data.variables.items()):
            if i >= 5:
                print(f"  ... and {len(data.variables) - 5} more")
                break
            if isinstance(value, np.ndarray):
                print(f"  {name}: array with shape {value.shape}")
            else:
                print(f"  {name}: {value}")
    
    print("\n" + "="*60)
