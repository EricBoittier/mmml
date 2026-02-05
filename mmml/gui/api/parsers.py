"""
File parsers for molecular data formats.

Supports:
- NPZ files (MMML format with R, Z, E, F, D, etc.)
- ASE trajectory files (.traj)
- PDB files
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from io import StringIO
import json

try:
    from ase import Atoms
    from ase.io import read as ase_read, write as ase_write
    from ase.io.trajectory import Trajectory
    HAS_ASE = True
except ImportError:
    HAS_ASE = False


# Periodic table for element symbols
ELEMENT_SYMBOLS = {
    1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O',
    9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
    16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 26: 'Fe', 29: 'Cu',
    30: 'Zn', 35: 'Br', 53: 'I'
}


@dataclass
class FrameData:
    """Data for a single molecular frame."""
    pdb_string: str
    n_atoms: int
    energy: Optional[float] = None
    forces: Optional[List[List[float]]] = None
    dipole: Optional[List[float]] = None
    charges: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'pdb_string': self.pdb_string,
            'n_atoms': self.n_atoms,
            'energy': self.energy,
            'forces': self.forces,
            'dipole': self.dipole,
            'charges': self.charges,
        }


@dataclass
class FileMetadata:
    """Metadata for a molecular file."""
    path: str
    filename: str
    file_type: str
    n_frames: int
    n_atoms: int
    available_properties: List[str]
    elements: List[str] = field(default_factory=list)
    energy_range: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'path': self.path,
            'filename': self.filename,
            'file_type': self.file_type,
            'n_frames': self.n_frames,
            'n_atoms': self.n_atoms,
            'available_properties': self.available_properties,
            'elements': self.elements,
            'energy_range': self.energy_range,
        }


def atoms_to_pdb(atoms: Atoms) -> str:
    """Convert ASE Atoms object to PDB string."""
    if not HAS_ASE:
        raise ImportError("ASE is required for molecular file parsing")
    
    buffer = StringIO()
    ase_write(buffer, atoms, format='proteindatabank')
    return buffer.getvalue()


def npz_frame_to_atoms(
    R: np.ndarray,
    Z: np.ndarray,
    N: Optional[int] = None
) -> Atoms:
    """
    Convert NPZ frame data to ASE Atoms object.
    
    Parameters
    ----------
    R : ndarray
        Positions (n_atoms, 3)
    Z : ndarray
        Atomic numbers (n_atoms,)
    N : int, optional
        Actual number of atoms (for padded arrays)
    
    Returns
    -------
    Atoms
        ASE Atoms object
    """
    if not HAS_ASE:
        raise ImportError("ASE is required for molecular file parsing")
    
    # Handle padding - remove zero atomic numbers
    mask = Z > 0
    if N is not None:
        # Use N to determine actual atoms
        mask = np.zeros_like(Z, dtype=bool)
        mask[:N] = True
        mask = mask & (Z > 0)
    
    positions = R[mask]
    atomic_numbers = Z[mask]
    
    return Atoms(numbers=atomic_numbers, positions=positions)


class MolecularFileParser:
    """
    Parser for molecular data files.
    
    Supports NPZ, ASE trajectory, and PDB formats.
    """
    
    SUPPORTED_EXTENSIONS = {'.npz', '.traj', '.pdb'}
    
    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        self.file_type = self._detect_file_type()
        self._data = None
        self._metadata = None
    
    def _detect_file_type(self) -> str:
        """Detect file type from extension."""
        ext = self.file_path.suffix.lower()
        if ext == '.npz':
            return 'npz'
        elif ext == '.traj':
            return 'ase_traj'
        elif ext == '.pdb':
            return 'pdb'
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    
    def _load_data(self):
        """Load file data into memory."""
        if self._data is not None:
            return
        
        if self.file_type == 'npz':
            self._data = np.load(self.file_path, allow_pickle=True)
        elif self.file_type == 'ase_traj':
            if not HAS_ASE:
                raise ImportError("ASE is required for trajectory files")
            self._data = ase_read(str(self.file_path), index=':')
        elif self.file_type == 'pdb':
            if not HAS_ASE:
                raise ImportError("ASE is required for PDB files")
            # Read all models from PDB
            try:
                self._data = ase_read(str(self.file_path), index=':')
            except Exception:
                # Single frame PDB
                self._data = [ase_read(str(self.file_path))]
    
    def get_metadata(self) -> FileMetadata:
        """Get file metadata."""
        if self._metadata is not None:
            return self._metadata
        
        self._load_data()
        
        if self.file_type == 'npz':
            self._metadata = self._get_npz_metadata()
        else:
            self._metadata = self._get_ase_metadata()
        
        return self._metadata
    
    def _get_npz_metadata(self) -> FileMetadata:
        """Get metadata from NPZ file."""
        data = self._data
        
        # Get basic info
        n_frames = len(data['E']) if 'E' in data else len(data['R'])
        Z_arr = data['Z']
        n_atoms = Z_arr.shape[1] if len(Z_arr.shape) > 1 else len(Z_arr)
        
        # Available properties
        properties = ['structure']
        if 'E' in data:
            properties.append('energy')
        if 'F' in data:
            properties.append('forces')
        if 'D' in data or 'Dxyz' in data:
            properties.append('dipole')
        if 'mono' in data:
            properties.append('charges')
        if 'esp' in data:
            properties.append('esp')
        
        # Get unique elements (handle object dtype)
        Z = np.asarray(Z_arr[0] if len(Z_arr.shape) > 1 else Z_arr)
        Z = Z.astype(np.int64)  # Ensure numeric type for comparison
        unique_Z = np.unique(Z[Z > 0])
        elements = [ELEMENT_SYMBOLS.get(int(z), f'X{z}') for z in unique_Z]
        
        # Energy range
        energy_range = None
        if 'E' in data:
            E = data['E']
            energy_range = {
                'min': float(np.min(E)),
                'max': float(np.max(E)),
                'mean': float(np.mean(E)),
            }
        
        return FileMetadata(
            path=str(self.file_path),
            filename=self.file_path.name,
            file_type='npz',
            n_frames=n_frames,
            n_atoms=n_atoms,
            available_properties=properties,
            elements=elements,
            energy_range=energy_range,
        )
    
    def _get_ase_metadata(self) -> FileMetadata:
        """Get metadata from ASE trajectory or PDB file."""
        frames = self._data
        
        n_frames = len(frames)
        n_atoms = len(frames[0]) if frames else 0
        
        # Available properties
        properties = ['structure']
        if frames and 'energy' in frames[0].info:
            properties.append('energy')
        if frames and 'forces' in frames[0].arrays:
            properties.append('forces')
        
        # Elements
        elements = []
        if frames:
            symbols = frames[0].get_chemical_symbols()
            elements = list(set(symbols))
        
        # Energy range
        energy_range = None
        if 'energy' in properties:
            energies = [f.info.get('energy', 0) for f in frames]
            energy_range = {
                'min': float(min(energies)),
                'max': float(max(energies)),
                'mean': float(sum(energies) / len(energies)),
            }
        
        return FileMetadata(
            path=str(self.file_path),
            filename=self.file_path.name,
            file_type=self.file_type,
            n_frames=n_frames,
            n_atoms=n_atoms,
            available_properties=properties,
            elements=elements,
            energy_range=energy_range,
        )
    
    def get_frame(self, index: int) -> FrameData:
        """
        Get data for a specific frame.
        
        Parameters
        ----------
        index : int
            Frame index
        
        Returns
        -------
        FrameData
            Frame data including PDB string and properties
        """
        self._load_data()
        
        if self.file_type == 'npz':
            return self._get_npz_frame(index)
        else:
            return self._get_ase_frame(index)
    
    def _get_npz_frame(self, index: int) -> FrameData:
        """Get frame from NPZ file."""
        data = self._data
        
        # Get coordinates and atomic numbers (handle object dtype)
        R = np.asarray(data['R'][index], dtype=np.float64)
        Z = np.asarray(data['Z'][index] if len(data['Z'].shape) > 1 else data['Z'], dtype=np.int64)
        N = int(data['N'][index]) if 'N' in data else None
        
        # Handle extra dimensions in R (e.g., shape (1, n_atoms, 3) -> (n_atoms, 3))
        while R.ndim > 2 and R.shape[0] == 1:
            R = R.squeeze(axis=0)
        
        # Convert to Atoms and PDB
        atoms = npz_frame_to_atoms(R, Z, N)
        pdb_string = atoms_to_pdb(atoms)
        
        # Get properties
        energy = None
        if 'E' in data:
            energy = float(data['E'][index])
        
        forces = None
        if 'F' in data:
            F = np.asarray(data['F'][index], dtype=np.float64)
            mask = Z > 0
            if N is not None:
                mask = np.zeros_like(Z, dtype=bool)
                mask[:N] = True
            forces = F[mask].tolist()
        
        dipole = None
        if 'D' in data:
            dipole = np.asarray(data['D'][index], dtype=np.float64).tolist()
        elif 'Dxyz' in data:
            dipole = np.asarray(data['Dxyz'][index], dtype=np.float64).tolist()
        
        charges = None
        if 'mono' in data:
            mono = np.asarray(data['mono'][index], dtype=np.float64)
            mask = Z > 0
            if N is not None:
                mask = np.zeros_like(Z, dtype=bool)
                mask[:N] = True
            charges = mono[mask].tolist()
        
        return FrameData(
            pdb_string=pdb_string,
            n_atoms=len(atoms),
            energy=energy,
            forces=forces,
            dipole=dipole,
            charges=charges,
        )
    
    def _get_ase_frame(self, index: int) -> FrameData:
        """Get frame from ASE trajectory or PDB file."""
        atoms = self._data[index]
        pdb_string = atoms_to_pdb(atoms)
        
        energy = atoms.info.get('energy')
        forces = atoms.arrays.get('forces')
        if forces is not None:
            forces = forces.tolist()
        
        return FrameData(
            pdb_string=pdb_string,
            n_atoms=len(atoms),
            energy=energy,
            forces=forces,
            dipole=None,
            charges=None,
        )
    
    def get_all_properties(self) -> Dict[str, List]:
        """
        Get all properties for all frames (for plotting).
        
        Returns
        -------
        dict
            Dictionary with property arrays
        """
        self._load_data()
        
        if self.file_type == 'npz':
            return self._get_npz_properties()
        else:
            return self._get_ase_properties()
    
    def _get_npz_properties(self) -> Dict[str, List]:
        """Get all properties from NPZ file."""
        data = self._data
        
        properties = {
            'frame_indices': list(range(len(data['E']) if 'E' in data else len(data['R']))),
        }
        
        if 'E' in data:
            E = np.asarray([float(e) for e in data['E']])
            properties['energy'] = E.tolist()
        
        if 'D' in data:
            D = np.stack([np.asarray(d, dtype=np.float64) for d in data['D']])
            properties['dipole_magnitude'] = np.linalg.norm(D, axis=1).tolist()
            properties['dipole_x'] = D[:, 0].tolist()
            properties['dipole_y'] = D[:, 1].tolist()
            properties['dipole_z'] = D[:, 2].tolist()
        elif 'Dxyz' in data:
            D = np.stack([np.asarray(d, dtype=np.float64) for d in data['Dxyz']])
            properties['dipole_magnitude'] = np.linalg.norm(D, axis=1).tolist()
            properties['dipole_x'] = D[:, 0].tolist()
            properties['dipole_y'] = D[:, 1].tolist()
            properties['dipole_z'] = D[:, 2].tolist()
        
        if 'F' in data:
            # Compute force magnitudes per frame
            F = data['F']
            Z = data['Z']
            N = data.get('N')
            
            max_forces = []
            mean_forces = []
            
            for i in range(len(F)):
                frame_F = np.asarray(F[i], dtype=np.float64)
                frame_Z = np.asarray(Z[i] if len(Z.shape) > 1 else Z, dtype=np.int64)
                mask = frame_Z > 0
                if N is not None:
                    mask = np.zeros_like(frame_Z, dtype=bool)
                    mask[:int(N[i])] = True
                
                frame_F = frame_F[mask]
                force_mags = np.linalg.norm(frame_F, axis=1)
                max_forces.append(float(np.max(force_mags)))
                mean_forces.append(float(np.mean(force_mags)))
            
            properties['force_max'] = max_forces
            properties['force_mean'] = mean_forces
        
        return properties
    
    def _get_ase_properties(self) -> Dict[str, List]:
        """Get all properties from ASE trajectory."""
        frames = self._data
        
        properties = {
            'frame_indices': list(range(len(frames))),
        }
        
        # Collect energies
        energies = []
        for f in frames:
            if 'energy' in f.info:
                energies.append(f.info['energy'])
        if energies:
            properties['energy'] = energies
        
        # Collect force magnitudes
        max_forces = []
        mean_forces = []
        for f in frames:
            if 'forces' in f.arrays:
                forces = f.arrays['forces']
                force_mags = np.linalg.norm(forces, axis=1)
                max_forces.append(float(np.max(force_mags)))
                mean_forces.append(float(np.mean(force_mags)))
        if max_forces:
            properties['force_max'] = max_forces
            properties['force_mean'] = mean_forces
        
        return properties


def list_molecular_files(directory: Union[str, Path]) -> List[Dict[str, str]]:
    """
    List all supported molecular files in a directory.
    
    Parameters
    ----------
    directory : str or Path
        Directory to search
    
    Returns
    -------
    list
        List of file info dictionaries
    """
    directory = Path(directory)
    files = []
    
    for ext in MolecularFileParser.SUPPORTED_EXTENSIONS:
        for file_path in directory.rglob(f'*{ext}'):
            files.append({
                'path': str(file_path),
                'filename': file_path.name,
                'relative_path': str(file_path.relative_to(directory)),
                'type': ext[1:],  # Remove leading dot
            })
    
    # Sort by filename
    files.sort(key=lambda x: x['filename'])
    
    return files
