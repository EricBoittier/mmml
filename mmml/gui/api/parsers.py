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
    electric_field: Optional[List[float]] = None
    positions: Optional[List[List[float]]] = None
    atomic_numbers: Optional[List[int]] = None
    replica_frames: Optional[List[Dict[str, Any]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'pdb_string': self.pdb_string,
            'n_atoms': self.n_atoms,
            'energy': self.energy,
            'forces': self.forces,
            'dipole': self.dipole,
            'charges': self.charges,
            'electric_field': self.electric_field,
            'positions': self.positions,
            'atomic_numbers': self.atomic_numbers,
            'replica_frames': self.replica_frames,
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
    n_replicas: int = 1
    
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
            'n_replicas': self.n_replicas,
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


def _get_npz_n_replicas(data: Any) -> int:
    """Infer replica count from NPZ arrays."""
    if 'n_replicas' in data:
        try:
            return max(1, int(np.asarray(data['n_replicas']).reshape(-1)[0]))
        except Exception:
            pass

    if 'R' in data:
        r_shape = np.asarray(data['R']).shape
        if len(r_shape) == 4:
            return int(r_shape[1])

    if 'E' in data:
        e_shape = np.asarray(data['E']).shape
        if len(e_shape) == 2:
            return int(e_shape[1])

    return 1


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
        self._frame_cache: Dict[tuple, FrameData] = {}  # Cache for computed frames
        self._max_cache_size = 100  # Maximum frames to cache
    
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
        n_replicas = _get_npz_n_replicas(data)
        
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
        elif 'Q' in data:
            properties.append('charges')
        if 'esp' in data:
            properties.append('esp')
        if 'Ef' in data:
            properties.append('electric_field')
        
        # Get unique elements (handle object dtype)
        Z = np.asarray(Z_arr[0] if len(Z_arr.shape) > 1 else Z_arr)
        Z = Z.astype(np.int64)  # Ensure numeric type for comparison
        unique_Z = np.unique(Z[Z > 0])
        elements = [ELEMENT_SYMBOLS.get(int(z), f'X{z}') for z in unique_Z]
        
        # Energy range
        energy_range = None
        if 'E' in data:
            E = np.asarray(data['E'], dtype=np.float64)
            if E.ndim > 1:
                E = E[:, 0]
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
            n_replicas=n_replicas,
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
    
    def get_frame(
        self,
        index: int,
        replica_index: int = 0,
        include_all_replicas: bool = False,
        include_pdb: bool = True,
    ) -> FrameData:
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
        # Check cache first
        cache_key = (index, replica_index, include_all_replicas, include_pdb)
        if cache_key in self._frame_cache:
            return self._frame_cache[cache_key]
        
        self._load_data()
        
        if self.file_type == 'npz':
            frame = self._get_npz_frame(
                index,
                replica_index=replica_index,
                include_all_replicas=include_all_replicas,
                include_pdb=include_pdb,
            )
        else:
            frame = self._get_ase_frame(index, include_pdb=include_pdb)
        
        # Add to cache (with simple LRU eviction)
        if len(self._frame_cache) >= self._max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._frame_cache))
            del self._frame_cache[oldest_key]
        self._frame_cache[cache_key] = frame
        
        return frame
    
    def _get_npz_frame(
        self,
        index: int,
        replica_index: int = 0,
        include_all_replicas: bool = False,
        include_pdb: bool = True,
    ) -> FrameData:
        """Get frame from NPZ file."""
        data = self._data

        n_replicas = _get_npz_n_replicas(data)
        replica_idx = min(max(replica_index, 0), max(n_replicas - 1, 0))

        # Get coordinates and atomic numbers (handle object dtype)
        R = np.asarray(data['R'][index], dtype=np.float64)
        Z = np.asarray(data['Z'][index] if len(data['Z'].shape) > 1 else data['Z'], dtype=np.int64)
        N = None
        raw_N = None
        if 'N' in data:
            arr_N = data['N']
            raw_N = np.asarray(arr_N[index] if arr_N.ndim > 0 else arr_N)
            if raw_N.ndim == 0:
                N = int(raw_N)
            elif raw_N.ndim == 1 and raw_N.size > 0:
                N = int(raw_N[min(replica_idx, raw_N.size - 1)])

        # Handle extra dimensions in R:
        # - (1, n_atoms, 3) -> (n_atoms, 3)
        # - (n_replicas, n_atoms, 3) -> select replica
        while R.ndim > 2 and R.shape[0] == 1:
            R = R.squeeze(axis=0)
        if R.ndim == 3 and n_replicas > 1:
            R = R[min(replica_idx, R.shape[0] - 1)]
        
        # Create mask for valid atoms
        mask = Z > 0
        if N is not None:
            mask = np.zeros_like(Z, dtype=bool)
            mask[:N] = True
            mask = mask & (Z > 0)
        
        # Convert to Atoms and PDB
        atoms = npz_frame_to_atoms(R, Z, N)
        pdb_string = atoms_to_pdb(atoms) if include_pdb else ""
        
        # Extract positions and atomic numbers for 3D visualization
        positions = R[mask].tolist()
        atomic_numbers = Z[mask].tolist()
        
        # Get properties
        energy = None
        if 'E' in data:
            frame_energy = np.asarray(data['E'][index], dtype=np.float64)
            if frame_energy.ndim == 0:
                energy = float(frame_energy)
            elif frame_energy.size > 0:
                energy = float(frame_energy[min(replica_idx, frame_energy.size - 1)])
        
        forces = None
        if 'F' in data:
            F = np.asarray(data['F'][index], dtype=np.float64)
            # Handle extra dimensions in F:
            # - (1, n_atoms, 3) -> (n_atoms, 3)
            # - (n_replicas, n_atoms, 3) -> select replica
            while F.ndim > 2 and F.shape[0] == 1:
                F = F.squeeze(axis=0)
            if F.ndim == 3 and n_replicas > 1:
                F = F[min(replica_idx, F.shape[0] - 1)]
            forces = F[mask].tolist()
        
        dipole = None
        if 'D' in data:
            frame_dipole = np.asarray(data['D'][index], dtype=np.float64)
            if frame_dipole.ndim == 1:
                dipole = frame_dipole.tolist()
            elif frame_dipole.ndim >= 2:
                dipole = frame_dipole[min(replica_idx, frame_dipole.shape[0] - 1)].tolist()
        elif 'Dxyz' in data:
            frame_dipole = np.asarray(data['Dxyz'][index], dtype=np.float64)
            if frame_dipole.ndim == 1:
                dipole = frame_dipole.tolist()
            elif frame_dipole.ndim >= 2:
                dipole = frame_dipole[min(replica_idx, frame_dipole.shape[0] - 1)].tolist()
        
        charges = None
        if 'mono' in data:
            mono = np.asarray(data['mono'][index], dtype=np.float64)
            # Handle extra dimensions in mono:
            # - (1, n_atoms) -> (n_atoms,)
            # - (n_replicas, n_atoms) -> select replica
            while mono.ndim > 1 and mono.shape[0] == 1:
                mono = mono.squeeze(axis=0)
            if mono.ndim == 2 and n_replicas > 1:
                mono = mono[min(replica_idx, mono.shape[0] - 1)]
            charges = mono[mask].tolist()
        elif 'Q' in data:
            Q = np.asarray(data['Q'][index], dtype=np.float64)
            if Q.ndim == 2 and n_replicas > 1:
                Q = Q[min(replica_idx, Q.shape[0] - 1)]
            charges = Q[mask].tolist()
        
        electric_field = None
        if 'Ef' in data:
            ef = np.asarray(data['Ef'], dtype=np.float64)
            if ef.ndim == 1:
                electric_field = ef.tolist()
            elif ef.ndim == 2:
                electric_field = ef[index].tolist()
            elif ef.ndim == 3:
                electric_field = ef[index, min(replica_idx, ef.shape[1] - 1)].tolist()

        replica_frames = None
        if include_all_replicas and n_replicas > 1 and data['R'][index].ndim >= 3:
            R_all = np.asarray(data['R'][index], dtype=np.float64)
            while R_all.ndim > 3 and R_all.shape[0] == 1:
                R_all = R_all.squeeze(axis=0)
            if R_all.ndim == 3:
                replica_frames = []
                for rep in range(min(n_replicas, R_all.shape[0])):
                    rep_mask = Z > 0
                    if raw_N is not None:
                        if np.asarray(raw_N).ndim == 0:
                            rep_n = int(raw_N)
                        else:
                            rep_arr = np.asarray(raw_N).reshape(-1)
                            rep_n = int(rep_arr[min(rep, rep_arr.size - 1)])
                        rep_mask = np.zeros_like(Z, dtype=bool)
                        rep_mask[:rep_n] = True
                        rep_mask = rep_mask & (Z > 0)
                    replica_frames.append({
                        'replica_index': rep,
                        'positions': R_all[rep][rep_mask].tolist(),
                        'atomic_numbers': Z[rep_mask].tolist(),
                    })
        
        return FrameData(
            pdb_string=pdb_string,
            n_atoms=len(atoms),
            energy=energy,
            forces=forces,
            dipole=dipole,
            charges=charges,
            electric_field=electric_field,
            positions=positions,
            atomic_numbers=atomic_numbers,
            replica_frames=replica_frames,
        )
    
    def _get_ase_frame(self, index: int, include_pdb: bool = True) -> FrameData:
        """Get frame from ASE trajectory or PDB file."""
        atoms = self._data[index]
        pdb_string = atoms_to_pdb(atoms) if include_pdb else ""
        
        energy = atoms.info.get('energy')
        forces = atoms.arrays.get('forces')
        if forces is not None:
            forces = forces.tolist()
        
        # Extract positions and atomic numbers for 3D visualization
        positions = atoms.get_positions().tolist()
        atomic_numbers = atoms.get_atomic_numbers().tolist()
        
        return FrameData(
            pdb_string=pdb_string,
            n_atoms=len(atoms),
            energy=energy,
            forces=forces,
            dipole=None,
            charges=None,
            positions=positions,
            atomic_numbers=atomic_numbers,
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
        n_replicas = _get_npz_n_replicas(data)
        replica_idx = 0
        
        properties = {
            'frame_indices': list(range(len(data['E']) if 'E' in data else len(data['R']))),
        }
        if n_replicas > 1:
            properties['replica_indices'] = list(range(n_replicas))
            properties['replica_series'] = {}
        
        if 'E' in data:
            E = np.asarray(data['E'], dtype=np.float64)
            if E.ndim == 1:
                properties['energy'] = E.tolist()
            elif E.ndim >= 2:
                properties['energy'] = E[:, min(replica_idx, E.shape[1] - 1)].tolist()
                if n_replicas > 1:
                    properties['replica_series']['energy'] = E.T.tolist()
        
        if 'D' in data:
            D = np.asarray(data['D'], dtype=np.float64)
            if D.ndim == 2 and D.shape[1] == 3:
                D_plot = D
            elif D.ndim == 3 and D.shape[2] == 3:
                D_plot = D[:, min(replica_idx, D.shape[1] - 1), :]
                if n_replicas > 1:
                    properties['replica_series']['dipole_magnitude'] = [
                        np.linalg.norm(D[:, rep, :], axis=1).tolist()
                        for rep in range(D.shape[1])
                    ]
            else:
                D_plot = np.zeros((len(properties['frame_indices']), 3), dtype=np.float64)
            properties['dipole_magnitude'] = np.linalg.norm(D_plot, axis=1).tolist()
            properties['dipole_x'] = D_plot[:, 0].tolist()
            properties['dipole_y'] = D_plot[:, 1].tolist()
            properties['dipole_z'] = D_plot[:, 2].tolist()
        elif 'Dxyz' in data:
            D = np.asarray(data['Dxyz'], dtype=np.float64)
            if D.ndim == 2 and D.shape[1] == 3:
                D_plot = D
            elif D.ndim == 3 and D.shape[2] == 3:
                D_plot = D[:, min(replica_idx, D.shape[1] - 1), :]
                if n_replicas > 1:
                    properties['replica_series']['dipole_magnitude'] = [
                        np.linalg.norm(D[:, rep, :], axis=1).tolist()
                        for rep in range(D.shape[1])
                    ]
            else:
                D_plot = np.zeros((len(properties['frame_indices']), 3), dtype=np.float64)
            properties['dipole_magnitude'] = np.linalg.norm(D_plot, axis=1).tolist()
            properties['dipole_x'] = D_plot[:, 0].tolist()
            properties['dipole_y'] = D_plot[:, 1].tolist()
            properties['dipole_z'] = D_plot[:, 2].tolist()
        
        if 'F' in data:
            # Compute force magnitudes per frame
            F = data['F']
            Z = data['Z']
            N = data.get('N')
            
            # Determine number of frames (use same logic as frame_indices)
            n_frames = len(data['E']) if 'E' in data else len(data['R'])
            
            max_forces = []
            mean_forces = []
            max_forces_by_replica = [[] for _ in range(n_replicas)] if n_replicas > 1 else None
            mean_forces_by_replica = [[] for _ in range(n_replicas)] if n_replicas > 1 else None
            
            for i in range(n_frames):
                frame_F = np.asarray(F[i], dtype=np.float64)
                # Handle extra dimensions in frame_F:
                # - (1, n_atoms, 3) -> (n_atoms, 3)
                # - (n_replicas, n_atoms, 3) -> select replica
                while frame_F.ndim > 2 and frame_F.shape[0] == 1:
                    frame_F = frame_F.squeeze(axis=0)
                
                frame_Z = np.asarray(Z[i] if len(Z.shape) > 1 else Z, dtype=np.int64)
                mask = frame_Z > 0
                n_atoms_i = None
                if N is not None:
                    frame_N = np.asarray(N[i] if N.ndim > 0 else N)
                    if frame_N.ndim == 0:
                        n_atoms_i = int(frame_N)
                    elif frame_N.ndim >= 1 and frame_N.size > 0:
                        n_atoms_i = int(frame_N[min(replica_idx, frame_N.size - 1)])

                if frame_F.ndim == 3 and n_replicas > 1:
                    for rep in range(frame_F.shape[0]):
                        rep_mask = frame_Z > 0
                        if N is not None:
                            if frame_N.ndim == 0:
                                n_atoms_rep = int(frame_N)
                            elif frame_N.ndim >= 1 and frame_N.size > 0:
                                n_atoms_rep = int(frame_N[min(rep, frame_N.size - 1)])
                            else:
                                n_atoms_rep = int(np.asarray(frame_N).reshape(-1)[0])
                            rep_mask = np.zeros_like(frame_Z, dtype=bool)
                            rep_mask[:n_atoms_rep] = True
                            rep_mask = rep_mask & (frame_Z > 0)
                        rep_force_mags = np.linalg.norm(frame_F[rep][rep_mask], axis=1)
                        max_forces_by_replica[rep].append(float(np.max(rep_force_mags)))
                        mean_forces_by_replica[rep].append(float(np.mean(rep_force_mags)))

                    if n_atoms_i is not None:
                        mask = np.zeros_like(frame_Z, dtype=bool)
                        mask[:n_atoms_i] = True
                        mask = mask & (frame_Z > 0)
                    force_mags = np.linalg.norm(frame_F[replica_idx][mask], axis=1)
                    max_forces.append(float(np.max(force_mags)))
                    mean_forces.append(float(np.mean(force_mags)))
                else:
                    if n_atoms_i is not None:
                        mask = np.zeros_like(frame_Z, dtype=bool)
                        mask[:n_atoms_i] = True
                        mask = mask & (frame_Z > 0)
                    frame_F_masked = frame_F[mask]
                    force_mags = np.linalg.norm(frame_F_masked, axis=1)
                    max_forces.append(float(np.max(force_mags)))
                    mean_forces.append(float(np.mean(force_mags)))
            
            properties['force_max'] = max_forces
            properties['force_mean'] = mean_forces
            if n_replicas > 1 and max_forces_by_replica is not None and mean_forces_by_replica is not None:
                properties['replica_series']['force_max'] = max_forces_by_replica
                properties['replica_series']['force_mean'] = mean_forces_by_replica
        
        if 'Ef' in data:
            Ef = np.asarray(data['Ef'], dtype=np.float64)
            n_frames = len(properties['frame_indices'])
            if Ef.ndim == 1 and Ef.shape[0] == 3:
                Ef_plot = np.tile(Ef[None, :], (n_frames, 1))
            elif Ef.ndim == 2 and Ef.shape[1] == 3:
                Ef_plot = Ef
            elif Ef.ndim == 3 and Ef.shape[2] == 3:
                Ef_plot = Ef[:, min(replica_idx, Ef.shape[1] - 1), :]
            else:
                Ef_plot = np.zeros((n_frames, 3), dtype=np.float64)
            properties['efield_magnitude'] = np.linalg.norm(Ef_plot, axis=1).tolist()
            properties['efield_x'] = Ef_plot[:, 0].tolist()
            properties['efield_y'] = Ef_plot[:, 1].tolist()
            properties['efield_z'] = Ef_plot[:, 2].tolist()
        
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
    
    def get_pca_projection(self, n_components: int = 2) -> Dict[str, Any]:
        """
        Compute PCA projection of molecular coordinates.
        
        Parameters
        ----------
        n_components : int
            Number of PCA components (2 or 3)
        
        Returns
        -------
        dict
            Dictionary with PCA projection data
        """
        self._load_data()
        
        if self.file_type == 'npz':
            return self._get_npz_pca(n_components)
        else:
            return self._get_ase_pca(n_components)

    def get_geometry_dataset(
        self,
        atoms: List[int],
        metric: Optional[str] = None,
        replica_index: int = 0,
        start: int = 0,
        end: Optional[int] = None,
        stride: int = 1,
    ) -> Dict[str, Any]:
        """
        Compute geometry metric dataset over a frame range.

        Parameters
        ----------
        atoms : list[int]
            Atom indices (2=bond, 3=angle, 4=dihedral)
        metric : str, optional
            Explicit metric kind ('bond', 'angle', 'dihedral')
        replica_index : int
            Replica index for NPZ data
        start : int
            Start frame (inclusive)
        end : int, optional
            End frame (exclusive), defaults to total number of frames
        stride : int
            Frame stride
        """
        if len(atoms) < 2 or len(atoms) > 4:
            raise ValueError("atoms must contain 2-4 indices")
        if len(set(atoms)) != len(atoms):
            raise ValueError("atoms must be unique")

        inferred = {2: 'bond', 3: 'angle', 4: 'dihedral'}[len(atoms)]
        metric_kind = metric or inferred
        if metric_kind != inferred:
            raise ValueError(f"metric '{metric_kind}' does not match {len(atoms)} selected atoms")

        metadata = self.get_metadata()
        total_frames = metadata.n_frames
        end = total_frames if end is None else min(end, total_frames)
        if start < 0 or start >= total_frames:
            raise ValueError(f"start index {start} out of range (0-{total_frames-1})")
        if end <= start:
            raise ValueError("end must be greater than start")
        if stride < 1:
            raise ValueError("stride must be >= 1")

        frame_indices = list(range(start, end, stride))
        props = self.get_all_properties()
        energies = props.get('energy')
        force_max = props.get('force_max')
        force_mean = props.get('force_mean')
        dipole_mag = props.get('dipole_magnitude')

        if self.file_type == 'npz':
            points = self._get_npz_geometry_dataset_points(frame_indices, atoms, metric_kind, replica_index)
        else:
            points = self._get_ase_geometry_dataset_points(frame_indices, atoms, metric_kind)

        def _pick(series: Optional[List[Any]], idx: int) -> Optional[float]:
            if series is None or idx >= len(series):
                return None
            val = series[idx]
            try:
                return float(val) if val is not None else None
            except Exception:
                return None

        return {
            'metric': metric_kind,
            'atoms': atoms,
            'start': start,
            'end': end,
            'stride': stride,
            'frame_indices': frame_indices,
            'points': [
                {
                    'frame': fi,
                    'value': points[i],
                    'energy': _pick(energies, fi),
                    'force_max': _pick(force_max, fi),
                    'force_mean': _pick(force_mean, fi),
                    'dipole_magnitude': _pick(dipole_mag, fi),
                }
                for i, fi in enumerate(frame_indices)
            ],
        }

    def _compute_metric(self, coords: np.ndarray, atoms: List[int], metric_kind: str) -> float:
        if np.max(atoms) >= coords.shape[0]:
            raise ValueError("atom index out of bounds for one or more frames")
        p = coords[atoms, :]

        def _norm(x: np.ndarray) -> float:
            return float(np.linalg.norm(x))

        if metric_kind == 'bond':
            return _norm(p[1] - p[0])
        if metric_kind == 'angle':
            ba = p[0] - p[1]
            bc = p[2] - p[1]
            denom = max(_norm(ba) * _norm(bc), 1e-12)
            c = float(np.dot(ba, bc) / denom)
            c = max(-1.0, min(1.0, c))
            return float(np.degrees(np.arccos(c)))
        if metric_kind == 'dihedral':
            # Signed torsion angle in degrees, using a standard [-180, 180] convention.
            b0 = p[0] - p[1]
            b1 = p[2] - p[1]
            b2 = p[3] - p[2]

            b1n = b1 / max(_norm(b1), 1e-12)
            v = b0 - np.dot(b0, b1n) * b1n
            w = b2 - np.dot(b2, b1n) * b1n

            x = float(np.dot(v, w))
            y = float(np.dot(np.cross(b1n, v), w))
            angle = float(np.degrees(np.arctan2(y, x)))
            if angle > 180.0:
                angle -= 360.0
            elif angle < -180.0:
                angle += 360.0
            return angle
        raise ValueError(f"Unknown metric '{metric_kind}'")

    def _compute_metric_batch(self, coords: np.ndarray, atoms: List[int], metric_kind: str) -> np.ndarray:
        """
        Vectorized metric computation over leading dimensions.

        Expected coords shape: (..., n_atoms, 3)
        Returns shape: (...)
        """
        if coords.ndim < 3 or coords.shape[-1] != 3:
            raise ValueError(f"Invalid coords shape for batched metric: {coords.shape}")
        if np.max(atoms) >= coords.shape[-2]:
            raise ValueError("atom index out of bounds for one or more frames")

        p = coords[..., atoms, :]  # (..., k, 3)

        if metric_kind == 'bond':
            return np.linalg.norm(p[..., 1, :] - p[..., 0, :], axis=-1)
        if metric_kind == 'angle':
            ba = p[..., 0, :] - p[..., 1, :]
            bc = p[..., 2, :] - p[..., 1, :]
            denom = np.maximum(np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1), 1e-12)
            c = np.sum(ba * bc, axis=-1) / denom
            c = np.clip(c, -1.0, 1.0)
            return np.degrees(np.arccos(c))
        if metric_kind == 'dihedral':
            # Signed torsion angle in degrees. Works for arbitrary leading dims
            # (e.g. [frame, replica, ...]) similar to a vmap over those axes.
            b0 = p[..., 0, :] - p[..., 1, :]
            b1 = p[..., 2, :] - p[..., 1, :]
            b2 = p[..., 3, :] - p[..., 2, :]

            b1_norm = np.maximum(np.linalg.norm(b1, axis=-1, keepdims=True), 1e-12)
            b1n = b1 / b1_norm

            v = b0 - np.sum(b0 * b1n, axis=-1, keepdims=True) * b1n
            w = b2 - np.sum(b2 * b1n, axis=-1, keepdims=True) * b1n

            x = np.sum(v * w, axis=-1)
            y = np.sum(np.cross(b1n, v) * w, axis=-1)
            ang = np.degrees(np.arctan2(y, x))
            return ((ang + 180.0) % 360.0) - 180.0
        raise ValueError(f"Unknown metric '{metric_kind}'")

    def _get_npz_geometry_dataset_points(
        self,
        frame_indices: List[int],
        atoms: List[int],
        metric_kind: str,
        replica_index: int,
    ) -> List[float]:
        data = self._data
        n_replicas = _get_npz_n_replicas(data)
        rep_idx = min(max(replica_index, 0), max(n_replicas - 1, 0))
        # Fast path: batch over frame (and replica axis when present), then select replica.
        R_batch = np.asarray(data['R'][frame_indices], dtype=np.float64)
        while R_batch.ndim > 4 and R_batch.shape[1] == 1:
            R_batch = np.squeeze(R_batch, axis=1)

        if R_batch.ndim == 4:
            # [frame, replica, atom, xyz] -> vmap-like over (frame, replica)
            rep_used = min(rep_idx, R_batch.shape[1] - 1)
            coords = R_batch[:, rep_used, :, :]
        elif R_batch.ndim == 3:
            # [frame, atom, xyz]
            coords = R_batch
        else:
            # Fallback to per-frame logic for unusual layouts.
            points: List[float] = []
            for fi in frame_indices:
                R = np.asarray(data['R'][fi], dtype=np.float64)
                Z = np.asarray(data['Z'][fi] if len(data['Z'].shape) > 1 else data['Z'], dtype=np.int64)
                N = None
                if 'N' in data:
                    arr_N = data['N']
                    raw_N = np.asarray(arr_N[fi] if arr_N.ndim > 0 else arr_N)
                    if raw_N.ndim == 0:
                        N = int(raw_N)
                    elif raw_N.ndim >= 1 and raw_N.size > 0:
                        N = int(raw_N[min(rep_idx, raw_N.size - 1)])
                while R.ndim > 2 and R.shape[0] == 1:
                    R = R.squeeze(axis=0)
                if R.ndim == 3 and n_replicas > 1:
                    R = R[min(rep_idx, R.shape[0] - 1)]
                mask = Z > 0
                if N is not None:
                    mask = np.zeros_like(Z, dtype=bool)
                    mask[:N] = True
                    mask = mask & (Z > 0)
                points.append(self._compute_metric(R[mask], atoms, metric_kind))
            return points

        # Validate selected atoms against per-frame valid counts (if N provided).
        if 'N' in data:
            arr_N = data['N']
            if arr_N.ndim == 0:
                N_sel = np.full(len(frame_indices), int(arr_N))
            else:
                N_sel = np.asarray(arr_N[frame_indices])
                if N_sel.ndim == 2:
                    N_sel = N_sel[:, min(rep_idx, N_sel.shape[1] - 1)]
                elif N_sel.ndim > 2:
                    N_sel = np.asarray([
                        np.asarray(arr_N[fi]).reshape(-1)[min(rep_idx, np.asarray(arr_N[fi]).size - 1)]
                        for fi in frame_indices
                    ])
            n_valid = N_sel.astype(np.int64).reshape(-1)
            if np.any(np.max(atoms) >= n_valid):
                raise ValueError("atom index out of bounds for one or more frames")
        elif np.max(atoms) >= coords.shape[1]:
            raise ValueError("atom index out of bounds for one or more frames")

        return self._compute_metric_batch(coords, atoms, metric_kind).astype(np.float64).tolist()

    def _get_ase_geometry_dataset_points(
        self,
        frame_indices: List[int],
        atoms: List[int],
        metric_kind: str,
    ) -> List[float]:
        frames = self._data
        points: List[float] = []
        for fi in frame_indices:
            coords = np.asarray(frames[fi].get_positions(), dtype=np.float64)
            points.append(self._compute_metric(coords, atoms, metric_kind))
        return points
    
    def _get_npz_pca(self, n_components: int) -> Dict[str, Any]:
        """Compute PCA for NPZ file."""
        data = self._data
        n_replicas = _get_npz_n_replicas(data)
        replica_idx = 0
        
        # Get coordinates
        R_raw = data['R']
        Z_raw = data['Z']
        n_frames = len(R_raw)
        
        # Flatten coordinates per frame
        coords_list = []
        for i in range(n_frames):
            R = np.asarray(R_raw[i], dtype=np.float64)
            Z = np.asarray(Z_raw[i] if len(Z_raw.shape) > 1 else Z_raw, dtype=np.int64)
            
            # Handle extra dimensions:
            # - (1, n_atoms, 3) -> (n_atoms, 3)
            # - (n_replicas, n_atoms, 3) -> select replica
            while R.ndim > 2 and R.shape[0] == 1:
                R = R.squeeze(axis=0)
            if R.ndim == 3 and n_replicas > 1:
                R = R[min(replica_idx, R.shape[0] - 1)]
            
            # Mask out padding
            mask = Z > 0
            R_masked = R[mask]
            
            # Flatten to 1D
            coords_list.append(R_masked.flatten())
        
        # Pad to same length if needed (in case of variable atom count)
        max_len = max(len(c) for c in coords_list)
        coords_padded = np.zeros((n_frames, max_len))
        for i, c in enumerate(coords_list):
            coords_padded[i, :len(c)] = c
        
        # Compute PCA using SVD
        coords_centered = coords_padded - coords_padded.mean(axis=0)
        U, S, Vt = np.linalg.svd(coords_centered, full_matrices=False)
        
        # Project onto principal components
        projections = coords_centered @ Vt.T[:, :n_components]
        
        # Compute explained variance
        explained_variance = (S[:n_components] ** 2) / (n_frames - 1)
        total_variance = np.sum(S ** 2) / (n_frames - 1)
        explained_variance_ratio = explained_variance / total_variance
        
        result = {
            'frame_indices': list(range(n_frames)),
            'pc1': projections[:, 0].tolist(),
            'pc2': projections[:, 1].tolist(),
            'explained_variance': explained_variance.tolist(),
            'explained_variance_ratio': explained_variance_ratio.tolist(),
        }
        
        if n_components >= 3:
            result['pc3'] = projections[:, 2].tolist()
        
        return result
    
    def _get_ase_pca(self, n_components: int) -> Dict[str, Any]:
        """Compute PCA for ASE trajectory."""
        frames = self._data
        n_frames = len(frames)
        
        # Flatten coordinates per frame
        coords_list = []
        for f in frames:
            coords_list.append(f.get_positions().flatten())
        
        # Pad to same length
        max_len = max(len(c) for c in coords_list)
        coords_padded = np.zeros((n_frames, max_len))
        for i, c in enumerate(coords_list):
            coords_padded[i, :len(c)] = c
        
        # Compute PCA using SVD
        coords_centered = coords_padded - coords_padded.mean(axis=0)
        U, S, Vt = np.linalg.svd(coords_centered, full_matrices=False)
        
        # Project onto principal components
        projections = coords_centered @ Vt.T[:, :n_components]
        
        # Compute explained variance
        explained_variance = (S[:n_components] ** 2) / (n_frames - 1)
        total_variance = np.sum(S ** 2) / (n_frames - 1)
        explained_variance_ratio = explained_variance / total_variance
        
        result = {
            'frame_indices': list(range(n_frames)),
            'pc1': projections[:, 0].tolist(),
            'pc2': projections[:, 1].tolist(),
            'explained_variance': explained_variance.tolist(),
            'explained_variance_ratio': explained_variance_ratio.tolist(),
        }
        
        if n_components >= 3:
            result['pc3'] = projections[:, 2].tolist()
        
        return result


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
