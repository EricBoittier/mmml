#!/usr/bin/env python3
"""
DCMNet calculator for ASE.

This calculator wraps a trained DCMNet model to compute:
- Distributed multipoles (monopoles and dipole positions)
- Electrostatic potential at arbitrary grid points
- Molecular dipole moment

Usage:
    from mmml.dcmnet.dcmnet_ase import DCMNetCalculator
    import pickle
    from ase import Atoms
    
    # Load trained model
    with open('checkpoint.pkl', 'rb') as f:
        checkpoint = pickle.load(f)
    
    # Create calculator
    calc = DCMNetCalculator(
        model=checkpoint['model'],
        params=checkpoint['params'],
        cutoff=10.0,
        n_dcm=6
    )
    
    # Use with ASE atoms
    atoms = Atoms('CO2', positions=[[0,0,0], [1.16,0,0], [-1.16,0,0]])
    atoms.calc = calc
    
    # Get ESP at grid points
    grid_points = np.array([[0, 0, 2.0], [0, 0, 3.0]])  # (n_points, 3)
    esp = calc.get_electrostatic_potential(grid_points)
"""

import numpy as np
import jax.numpy as jnp
import jax
from ase.calculators.calculator import Calculator, all_changes

try:
    import e3x
    HAVE_E3X = True
except ImportError:
    HAVE_E3X = False
    print("Warning: e3x not available. Message passing indices will be computed manually.")


class DCMNetCalculator(Calculator):
    """
    ASE calculator for DCMNet models.
    
    Computes distributed multipoles (monopoles and dipole positions) from a trained
    DCMNet model and provides methods to compute electrostatic potential at arbitrary
    grid points.
    
    Parameters
    ----------
    model : MessagePassingModel
        Trained DCMNet model instance
    params : Any
        Trained model parameters
    cutoff : float, optional
        Cutoff distance for neighbor list (Angstroms), by default 10.0
    n_dcm : int, optional
        Number of distributed multipoles per atom, by default 6
    """
    
    implemented_properties = ['charges', 'dipole', 'multipoles']
    
    def __init__(self, model, params, cutoff=10.0, n_dcm=6, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.params = params
        self.cutoff = cutoff
        self.n_dcm = n_dcm
        
        # Cache for last calculation
        self._last_atoms = None
        self._last_monopoles = None
        self._last_dipole_positions = None
        self._last_molecular_dipole = None
    
    def calculate(self, atoms=None, properties=['charges'], system_changes=all_changes):
        """
        Calculate distributed multipoles for the given atoms.
        
        Parameters
        ----------
        atoms : ase.Atoms
            Atomic structure
        properties : list
            Properties to calculate
        system_changes : list
            System changes since last calculation
        """
        Calculator.calculate(self, atoms, properties, system_changes)
        
        positions = atoms.get_positions()  # (n_atoms, 3)
        atomic_numbers = atoms.get_atomic_numbers()  # (n_atoms,)
        n_atoms = len(atoms)
        
        # Create message passing indices
        if HAVE_E3X:
            dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(n_atoms)
        else:
            # Manual computation if e3x not available
            dst_idx, src_idx = self._create_pairwise_indices(n_atoms)
        
        # Batch segments (single molecule)
        batch_segments = jnp.zeros(n_atoms, dtype=jnp.int32)
        
        # Run model forward pass
        mono_pred, dipo_pred = self.model.apply(
            self.params,
            atomic_numbers=jnp.array(atomic_numbers),
            positions=jnp.array(positions),
            dst_idx=jnp.array(dst_idx),
            src_idx=jnp.array(src_idx),
            batch_segments=jnp.array(batch_segments),
        )
        
        # Convert to numpy
        mono_pred = np.array(mono_pred)  # (n_atoms, n_dcm)
        dipo_pred = np.array(dipo_pred)  # (n_atoms, n_dcm, 3)
        
        # Store results
        self._last_atoms = atoms
        self._last_monopoles = mono_pred
        self._last_dipole_positions = dipo_pred
        
        # Compute atomic charges (sum of DCM monopoles per atom)
        atomic_charges = mono_pred.sum(axis=-1)  # (n_atoms,)
        self.results['charges'] = atomic_charges
        
        # Compute molecular dipole moment
        com = positions.mean(axis=0)  # Center of mass
        molecular_dipole = self._compute_molecular_dipole(
            dipo_pred, mono_pred, com
        )
        self._last_molecular_dipole = molecular_dipole
        self.results['dipole'] = molecular_dipole
        
        # Store multipoles for access
        self.results['multipoles'] = {
            'monopoles': mono_pred,
            'dipole_positions': dipo_pred,
            'atomic_charges': atomic_charges,
            'molecular_dipole': molecular_dipole,
        }
    
    def get_electrostatic_potential(self, grid_points):
        """
        Compute electrostatic potential at given grid points.
        
        Uses the distributed multipoles (monopoles and dipole positions) from
        the last calculation to compute ESP at arbitrary grid points.
        
        Parameters
        ----------
        grid_points : array_like
            Grid point positions, shape (n_points, 3) in Angstroms
            
        Returns
        -------
        array_like
            Electrostatic potential values, shape (n_points,) in Hartree/e
            
        Raises
        ------
        RuntimeError
            If no calculation has been performed yet
        """
        if self._last_monopoles is None or self._last_dipole_positions is None:
            raise RuntimeError(
                "No calculation performed yet. Call atoms.calc.calculate() or "
                "atoms.get_potential_energy() first."
            )
        
        # Flatten distributed multipoles
        # monopoles: (n_atoms, n_dcm) -> (n_atoms * n_dcm,)
        # dipole_positions: (n_atoms, n_dcm, 3) -> (n_atoms * n_dcm, 3)
        n_atoms = self._last_monopoles.shape[0]
        monopoles_flat = self._last_monopoles.reshape(-1)  # (n_atoms * n_dcm,)
        dipole_positions_flat = self._last_dipole_positions.reshape(-1, 3)  # (n_atoms * n_dcm, 3)
        
        # Compute ESP using calc_esp function
        from mmml.dcmnet.dcmnet.electrostatics import calc_esp
        
        grid_points_jnp = jnp.array(grid_points)
        if grid_points_jnp.ndim == 1:
            grid_points_jnp = grid_points_jnp.reshape(1, -1)
        
        esp = calc_esp(
            charge_positions=dipole_positions_flat,
            charge_values=monopoles_flat,
            grid_positions=grid_points_jnp
        )
        
        return np.array(esp)
    
    def get_distributed_multipoles(self):
        """
        Get distributed multipoles from last calculation.
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'monopoles': (n_atoms, n_dcm) array of monopole values
            - 'dipole_positions': (n_atoms, n_dcm, 3) array of dipole positions
            - 'atomic_charges': (n_atoms,) array of atomic charges (sum of DCM monopoles)
            - 'molecular_dipole': (3,) array of molecular dipole moment in Debye
        """
        if self._last_monopoles is None:
            raise RuntimeError("No calculation performed yet.")
        
        return {
            'monopoles': self._last_monopoles.copy(),
            'dipole_positions': self._last_dipole_positions.copy(),
            'atomic_charges': self._last_monopoles.sum(axis=-1),
            'molecular_dipole': self._last_molecular_dipole.copy() if self._last_molecular_dipole is not None else None,
        }
    
    def _create_pairwise_indices(self, n_atoms):
        """
        Create pairwise indices for message passing (fallback if e3x not available).
        
        Parameters
        ----------
        n_atoms : int
            Number of atoms
            
        Returns
        -------
        tuple
            (dst_idx, src_idx) arrays
        """
        dst_idx = []
        src_idx = []
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i != j:
                    dst_idx.append(i)
                    src_idx.append(j)
        return np.array(dst_idx, dtype=np.int32), np.array(src_idx, dtype=np.int32)
    
    def _compute_molecular_dipole(self, dipole_positions, monopoles, com):
        """
        Compute molecular dipole moment from distributed multipoles.
        
        Parameters
        ----------
        dipole_positions : array_like
            Dipole positions, shape (n_atoms, n_dcm, 3)
        monopoles : array_like
            Monopole values, shape (n_atoms, n_dcm)
        com : array_like
            Center of mass, shape (3,)
            
        Returns
        -------
        array_like
            Molecular dipole moment in Debye, shape (3,)
        """
        # Flatten to (n_atoms * n_dcm, 3) and (n_atoms * n_dcm,)
        dcm_positions = dipole_positions.reshape(-1, 3)
        dcm_charges = monopoles.reshape(-1)
        
        # Compute dipole: sum(q_i * (r_i - com))
        dipole = np.zeros(3)
        for i in range(len(dcm_charges)):
            dipole += dcm_charges[i] * (dcm_positions[i] - com)
        
        # Convert to Debye (atomic units to Debye)
        return dipole * 1.88873


# Example usage script
if __name__ == '__main__':
    import sys
    import pickle
    from pathlib import Path
    from ase import Atoms
    
    # Example: Load model and create calculator
    if len(sys.argv) < 2:
        print("Usage: python dcmnet_ase.py <checkpoint.pkl>")
        print("\nExample:")
        print("  python dcmnet_ase.py checkpoint.pkl")
        sys.exit(1)
    
    checkpoint_path = Path(sys.argv[1])
    
    print("Loading checkpoint...")
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    # Extract model and params
    model = checkpoint.get('model')
    params = checkpoint.get('params') or checkpoint.get('best_params')
    
    if model is None or params is None:
        raise ValueError(
            f"Checkpoint must contain 'model' and 'params' keys. "
            f"Found keys: {checkpoint.keys()}"
        )
    
    # Get n_dcm from model or checkpoint
    n_dcm = getattr(model, 'n_dcm', checkpoint.get('n_dcm', 6))
    cutoff = getattr(model, 'cutoff', checkpoint.get('cutoff', 10.0))
    
    print(f"Creating calculator (n_dcm={n_dcm}, cutoff={cutoff} Å)...")
    calc = DCMNetCalculator(
        model=model,
        params=params,
        cutoff=cutoff,
        n_dcm=n_dcm
    )
    
    # Test with CO2
    print("\nTesting with CO2 molecule...")
    atoms = Atoms('CO2', positions=[[0, 0, 0], [1.16, 0, 0], [-1.16, 0, 0]])
    atoms.calc = calc
    
    # Calculate properties
    charges = atoms.get_charges()
    dipole = atoms.get_dipole_moment()
    
    print(f"\nResults:")
    print(f"  Atomic charges: {charges}")
    print(f"  Total charge: {charges.sum():.6f} e")
    print(f"  Molecular dipole: {dipole} Debye")
    print(f"  Dipole magnitude: {np.linalg.norm(dipole):.6f} Debye")
    
    # Test ESP calculation
    print("\nTesting ESP calculation...")
    # Create a grid around the molecule
    grid_points = np.array([
        [0, 0, 2.0],   # Above center
        [0, 0, -2.0],  # Below center
        [2.0, 0, 0],   # Right
        [-2.0, 0, 0],  # Left
        [0, 2.0, 0],   # Up
        [0, -2.0, 0],  # Down
    ])
    
    esp = calc.get_electrostatic_potential(grid_points)
    print(f"\nESP at grid points (Hartree/e):")
    for i, (point, esp_val) in enumerate(zip(grid_points, esp)):
        print(f"  Point {i+1} [{point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}]: {esp_val:.6f}")
    
    # Get distributed multipoles
    multipoles = calc.get_distributed_multipoles()
    print(f"\nDistributed multipoles:")
    print(f"  Monopoles shape: {multipoles['monopoles'].shape}")
    print(f"  Dipole positions shape: {multipoles['dipole_positions'].shape}")
    print(f"  Atomic charges: {multipoles['atomic_charges']}")
    
    print("\n✅ All tests passed!")

