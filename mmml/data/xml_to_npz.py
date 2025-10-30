"""
Convert Molpro XML output to standardized NPZ format.

This module bridges the Molpro XML parser (parse_molpro) with the
standardized NPZ format used across all MMML models.
"""

import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
import json
from datetime import datetime
from tqdm import tqdm

# Import the excellent Molpro parser
sys.path.insert(0, str(Path(__file__).parent.parent / 'parse_molpro'))
from read_molden import read_molpro_xml, MolproData

from .npz_schema import validate_npz, NPZSchema


@dataclass
class ConversionStats:
    """Statistics from XML to NPZ conversion."""
    n_files: int = 0
    n_structures: int = 0
    n_failed: int = 0
    failed_files: List[str] = None
    properties_extracted: List[str] = None
    
    def __post_init__(self):
        if self.failed_files is None:
            self.failed_files = []
        if self.properties_extracted is None:
            self.properties_extracted = []


class MolproConverter:
    """
    Convert Molpro XML files to standardized NPZ format.
    
    Handles single files or batches of XML files, extracts all available
    properties, and creates NPZ files following the MMML schema.
    
    Parameters
    ----------
    padding_atoms : int, optional
        Number of atoms to pad to (for fixed-size arrays), by default 60
    include_variables : bool, optional
        Whether to include Molpro variables in metadata, by default True
    verbose : bool, optional
        Whether to print progress information, by default True
        
    Examples
    --------
    >>> converter = MolproConverter()
    >>> data = converter.convert_single('output.xml')
    >>> converter.save_npz(data, 'output.npz')
    """
    
    def __init__(
        self,
        padding_atoms: int = 60,
        include_variables: bool = True,
        verbose: bool = True
    ):
        self.padding_atoms = padding_atoms
        self.include_variables = include_variables
        self.verbose = verbose
        self.stats = ConversionStats()
    
    def convert_single(self, xml_file: Union[str, Path]) -> Dict[str, np.ndarray]:
        """
        Convert a single Molpro XML file to NPZ format.
        
        Parameters
        ----------
        xml_file : str or Path
            Path to Molpro XML file
            
        Returns
        -------
        dict
            Dictionary of arrays following NPZ schema
        """
        try:
            # Parse XML using our excellent parser
            molpro_data = read_molpro_xml(str(xml_file))
            
            # Convert to NPZ format
            npz_data = self._molpro_to_npz(molpro_data, source_file=str(xml_file))
            
            self.stats.n_files += 1
            self.stats.n_structures += 1
            
            return npz_data
            
        except Exception as e:
            if self.verbose:
                print(f"Error converting {xml_file}: {e}")
            self.stats.n_failed += 1
            self.stats.failed_files.append(str(xml_file))
            return None
    
    def _molpro_to_npz(
        self,
        data: MolproData,
        source_file: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Convert MolproData object to NPZ dictionary.
        
        Parameters
        ----------
        data : MolproData
            Parsed Molpro data
        source_file : str, optional
            Source XML filename for metadata
            
        Returns
        -------
        dict
            Dictionary ready for NPZ saving
        """
        npz_dict = {}
        
        # Required fields
        if data.coordinates is not None and data.atomic_numbers is not None:
            n_atoms_actual = len(data.atomic_numbers)
            
            # Pad to fixed size
            R = np.zeros((1, self.padding_atoms, 3))
            Z = np.zeros((1, self.padding_atoms), dtype=np.int32)
            
            R[0, :n_atoms_actual, :] = data.coordinates
            Z[0, :n_atoms_actual] = data.atomic_numbers
            
            npz_dict['R'] = R
            npz_dict['Z'] = Z
            npz_dict['N'] = np.array([n_atoms_actual], dtype=np.int32)
        
        # Energy (use first available method)
        if data.energies:
            # Priority: RHF, MP2, CCSD, first available
            energy = None
            for method in ['RHF', 'MP2', 'CCSD', 'CCSD(T)']:
                if method in data.energies:
                    energy = data.energies[method]
                    break
            if energy is None:
                energy = list(data.energies.values())[0]
            
            npz_dict['E'] = np.array([energy])
        
        # Optional: Forces/Gradient
        if data.gradient is not None:
            # Gradient is negative of forces
            F = np.zeros((1, self.padding_atoms, 3))
            F[0, :n_atoms_actual, :] = -data.gradient
            npz_dict['F'] = F
        
        # Optional: Dipole moment
        if data.dipole_moment is not None:
            npz_dict['D'] = data.dipole_moment.reshape(1, 3)
            npz_dict['Dxyz'] = data.dipole_moment.reshape(1, 3)
        
        # Optional: Orbital data
        if data.orbital_energies is not None:
            # Store orbital energies as additional property
            npz_dict['orbital_energies'] = data.orbital_energies.reshape(1, -1)
        
        if data.orbital_occupancies is not None:
            npz_dict['orbital_occupancies'] = data.orbital_occupancies.reshape(1, -1)
        
        # Optional: Vibrational data
        if data.frequencies is not None:
            npz_dict['frequencies'] = data.frequencies.reshape(1, -1)
        
        if data.intensities is not None:
            npz_dict['ir_intensities'] = data.intensities.reshape(1, -1)
        
        # Optional: Polarizability
        # Note: Not currently in Molpro parser, but placeholder for future
        
        # Metadata
        metadata = {
            'generation_date': datetime.now().isoformat(),
            'source_file': source_file,
            'units': {
                'R': 'Angstrom',
                'E': 'Hartree',
                'F': 'Hartree/Bohr',
                'D': 'Debye',
            },
            'energies_by_method': data.energies,
        }
        
        # Include Molpro variables if requested
        if self.include_variables and data.variables:
            # Convert variables to serializable format
            variables_serializable = {}
            for key, value in data.variables.items():
                if isinstance(value, np.ndarray):
                    variables_serializable[key] = value.tolist()
                else:
                    variables_serializable[key] = float(value) if isinstance(value, (int, float)) else str(value)
            metadata['molpro_variables'] = variables_serializable
        
        # Store metadata as pickled dict
        npz_dict['metadata'] = np.array([metadata], dtype=object)
        
        # Track what properties were extracted
        self.stats.properties_extracted = list(set(
            self.stats.properties_extracted + list(npz_dict.keys())
        ))
        
        return npz_dict
    
    def convert_batch(
        self,
        xml_files: List[Union[str, Path]],
        progress_bar: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Convert multiple Molpro XML files into a single NPZ dataset.
        
        Parameters
        ----------
        xml_files : list
            List of XML file paths
        progress_bar : bool, optional
            Whether to show progress bar, by default True
            
        Returns
        -------
        dict
            Combined NPZ dictionary
        """
        all_data = []
        
        iterator = tqdm(xml_files, desc="Converting XML files") if progress_bar else xml_files
        
        for xml_file in iterator:
            data = self.convert_single(xml_file)
            if data is not None:
                all_data.append(data)
        
        if not all_data:
            raise ValueError("No data was successfully converted!")
        
        # Combine all structures
        combined = self._combine_datasets(all_data)
        
        return combined
    
    def _combine_datasets(self, datasets: List[Dict]) -> Dict[str, np.ndarray]:
        """
        Combine multiple single-structure datasets into one.
        
        Parameters
        ----------
        datasets : list
            List of NPZ dictionaries
            
        Returns
        -------
        dict
            Combined NPZ dictionary
        """
        combined = {}
        
        # Get all unique keys (excluding metadata for now)
        all_keys = set()
        for ds in datasets:
            all_keys.update(k for k in ds.keys() if k != 'metadata')
        
        # Concatenate each property
        for key in all_keys:
            arrays = [ds[key] for ds in datasets if key in ds]
            if arrays:
                try:
                    combined[key] = np.concatenate(arrays, axis=0)
                except ValueError as e:
                    if self.verbose:
                        print(f"Warning: Could not concatenate '{key}': {e}")
        
        # Combine metadata
        all_metadata = [ds.get('metadata', [{}])[0] for ds in datasets]
        combined_metadata = {
            'generation_date': datetime.now().isoformat(),
            'n_sources': len(datasets),
            'source_files': [m.get('source_file') for m in all_metadata],
            'units': all_metadata[0].get('units', {}) if all_metadata else {},
        }
        
        # Aggregate Molpro variables (take first available)
        if any('molpro_variables' in m for m in all_metadata):
            for m in all_metadata:
                if 'molpro_variables' in m:
                    combined_metadata['molpro_variables'] = m['molpro_variables']
                    break
        
        combined['metadata'] = np.array([combined_metadata], dtype=object)
        
        return combined
    
    def save_npz(
        self,
        data: Dict[str, np.ndarray],
        output_file: Union[str, Path],
        validate: bool = True
    ):
        """
        Save NPZ data to file with optional validation.
        
        Parameters
        ----------
        data : dict
            NPZ dictionary
        output_file : str or Path
            Output file path
        validate : bool, optional
            Whether to validate before saving, by default True
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Validate if requested
        if validate:
            schema = NPZSchema()
            is_valid, errors = schema.validate(data)
            if not is_valid:
                print(f"Warning: Data validation failed:")
                for error in errors:
                    print(f"  - {error}")
                print("Saving anyway...")
        
        # Save
        np.savez_compressed(output_path, **data)
        
        if self.verbose:
            print(f"✓ Saved NPZ file: {output_path}")
            print(f"  Structures: {len(data.get('E', []))}")
            print(f"  Properties: {', '.join(k for k in data.keys() if k != 'metadata')}")
    
    def get_statistics(self) -> ConversionStats:
        """Get conversion statistics."""
        return self.stats
    
    def print_summary(self):
        """Print conversion summary."""
        print("\n" + "="*60)
        print("Conversion Summary")
        print("="*60)
        print(f"Files processed: {self.stats.n_files}")
        print(f"Structures extracted: {self.stats.n_structures}")
        print(f"Failed conversions: {self.stats.n_failed}")
        if self.stats.failed_files:
            print(f"Failed files: {', '.join(self.stats.failed_files[:5])}")
            if len(self.stats.failed_files) > 5:
                print(f"  ... and {len(self.stats.failed_files) - 5} more")
        print(f"Properties extracted: {', '.join(sorted(self.stats.properties_extracted))}")
        print("="*60)


def convert_xml_to_npz(
    xml_file: Union[str, Path],
    output_file: Union[str, Path],
    **kwargs
) -> bool:
    """
    Convenience function to convert single XML file to NPZ.
    
    Parameters
    ----------
    xml_file : str or Path
        Input XML file
    output_file : str or Path
        Output NPZ file
    **kwargs
        Additional arguments passed to MolproConverter
        
    Returns
    -------
    bool
        True if successful
        
    Examples
    --------
    >>> convert_xml_to_npz('output.xml', 'data.npz')
    """
    converter = MolproConverter(**kwargs)
    data = converter.convert_single(xml_file)
    
    if data is not None:
        converter.save_npz(data, output_file)
        return True
    return False


def batch_convert_xml(
    xml_files: List[Union[str, Path]],
    output_file: Union[str, Path],
    **kwargs
) -> bool:
    """
    Convenience function to convert multiple XML files to single NPZ.
    
    Parameters
    ----------
    xml_files : list
        List of input XML files
    output_file : str or Path
        Output NPZ file
    **kwargs
        Additional arguments passed to MolproConverter
        
    Returns
    -------
    bool
        True if successful
        
    Examples
    --------
    >>> batch_convert_xml(['file1.xml', 'file2.xml'], 'dataset.npz')
    """
    converter = MolproConverter(**kwargs)
    
    try:
        data = converter.convert_batch(xml_files)
        converter.save_npz(data, output_file)
        converter.print_summary()
        return True
    except Exception as e:
        print(f"Error in batch conversion: {e}")
        return False


if __name__ == '__main__':
    # Example CLI usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert Molpro XML to NPZ format')
    parser.add_argument('xml_files', nargs='+', help='Input XML file(s)')
    parser.add_argument('-o', '--output', required=True, help='Output NPZ file')
    parser.add_argument('--padding', type=int, default=60, help='Number of atoms to pad to')
    parser.add_argument('--no-variables', action='store_true', help='Exclude Molpro variables')
    parser.add_argument('--no-validate', action='store_true', help='Skip validation')
    
    args = parser.parse_args()
    
    success = batch_convert_xml(
        args.xml_files,
        args.output,
        padding_atoms=args.padding,
        include_variables=not args.no_variables,
        verbose=True
    )
    
    sys.exit(0 if success else 1)

