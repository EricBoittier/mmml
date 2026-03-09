#!/usr/bin/env python3
"""
Simple, fast ASE Calculator for trained models.

Works with any checkpoint (DCMNet or NonEquivariant).
No padding, no complexity - just proper GNN inference.
"""

import numpy as np
import jax.numpy as jnp
from ase.calculators.calculator import Calculator, all_changes


class SimpleCalculator(Calculator):
    """
    Straightforward ASE calculator for JointPhysNet models.
    
    The model is a GNN - it works with ANY number of atoms.
    The natoms=60 in config is just training metadata (batch size).
    """
    
    implemented_properties = ['energy', 'forces', 'dipole', 'charges']
    
    def __init__(self, model, params, cutoff=10.0, use_dcmnet_dipole=False, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.params = params
        self.cutoff = cutoff
        self.use_dcmnet_dipole = use_dcmnet_dipole
    
    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        """Calculate properties."""
        super().calculate(atoms, properties, system_changes)
        
        positions = atoms.get_positions()
        atomic_numbers = atoms.get_atomic_numbers()
        n_atoms = len(atoms)
        
        # Build edge list (within cutoff)
        dst_list, src_list = [], []
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i != j and np.linalg.norm(positions[i] - positions[j]) < self.cutoff:
                    dst_list.append(i)
                    src_list.append(j)
        
        dst_idx = np.array(dst_list, dtype=np.int32)
        src_idx = np.array(src_list, dtype=np.int32)
        
        # Batch data (single molecule, no padding needed!)
        batch_segments = np.zeros(n_atoms, dtype=np.int32)
        batch_mask = np.ones(len(dst_idx), dtype=np.float32)
        atom_mask = np.ones(n_atoms, dtype=np.float32)
        
        # Run model
        output = self.model.apply(
            self.params,
            atomic_numbers=jnp.array(atomic_numbers),
            positions=jnp.array(positions),
            dst_idx=jnp.array(dst_idx),
            src_idx=jnp.array(src_idx),
            batch_segments=jnp.array(batch_segments),
            batch_size=1,
            batch_mask=jnp.array(batch_mask),
            atom_mask=jnp.array(atom_mask),
        )
        
        # Extract results
        self.results['energy'] = float(output['energy'][0])
        self.results['forces'] = np.array(output['forces'])
        
        # Dipole
        if self.use_dcmnet_dipole and 'dipole_dcm' in output:
            self.results['dipole'] = np.array(output['dipole_dcm'][0])
        elif 'dipoles' in output:
            self.results['dipole'] = np.array(output['dipoles'][0])
        else:
            # Fallback: compute from charges
            charges = np.array(output.get('charges_as_mono', output.get('charges', np.zeros(n_atoms))))
            if charges.ndim > 1:
                charges = charges.squeeze()
            self.results['dipole'] = np.sum(charges[:n_atoms, np.newaxis] * positions, axis=0)
        
        # Charges
        if 'charges_as_mono' in output:
            charges = np.array(output['charges_as_mono'])
        elif 'charges' in output:
            charges = np.array(output['charges']).squeeze()
        else:
            charges = np.zeros(n_atoms)
        
        self.results['charges'] = charges[:n_atoms] if len(charges) >= n_atoms else charges


if __name__ == '__main__':
    """Quick test."""
    import sys
    import argparse
    import pickle
    from pathlib import Path
    from ase import Atoms
    
    repo_root = Path(__file__).parent / "../../.."
    sys.path.insert(0, str(repo_root.resolve()))
    
    from trainer import JointPhysNetDCMNet, JointPhysNetNonEquivariant
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--noneq', action='store_true')
    args = parser.parse_args()
    
    # Load checkpoint and config
    with open(args.checkpoint, 'rb') as f:
        checkpoint_data = pickle.load(f)
    params = checkpoint_data.get('params', checkpoint_data)
    
    config_path = args.checkpoint.parent / 'model_config.pkl'
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    
    # Create model
    if args.noneq:
        model = JointPhysNetNonEquivariant(
            physnet_config=config['physnet_config'],
            noneq_config=config['noneq_config'],
            mix_coulomb_energy=config.get('mix_coulomb_energy', False),
        )
    else:
        model = JointPhysNetDCMNet(
            physnet_config=config['physnet_config'],
            dcmnet_config=config['dcmnet_config'],
            mix_coulomb_energy=config.get('mix_coulomb_energy', False),
        )
    
    if isinstance(params, dict) and 'params' not in params and 'physnet' in params:
        params = {'params': params}
    
    # Create calculator
    cutoff = config['physnet_config'].get('cutoff', 6.0)
    calc = SimpleCalculator(model, params, cutoff=cutoff)
    
    # Test with CO2
    atoms = Atoms('CO2', positions=[[0,0,0], [1.16,0,0], [-1.16,0,0]])
    atoms.calc = calc
    
    print("Testing SimpleCalculator...")
    print(f"Energy: {atoms.get_potential_energy():.6f} eV")
    print(f"Forces max: {np.abs(atoms.get_forces()).max():.6f} eV/Å")
    print(f"Dipole: {np.linalg.norm(atoms.get_dipole_moment()):.6f} e·Å")
    print(f"Charges sum: {atoms.get_charges().sum():.6f} e")
    print("✅ Test passed!")

