#!/usr/bin/env python3
"""
Generic ASE Calculator for trained MMML models.

Works with any checkpoint - automatically detects model type and provides
a simple interface for molecular dynamics and property calculations.

Usage:
    from mmml.cli.calculator import MMMLCalculator
    from ase import Atoms
    
    calc = MMMLCalculator.from_checkpoint('path/to/checkpoint')
    atoms = Atoms('CO2', positions=[[0,0,0], [1.16,0,0], [-1.16,0,0]])
    atoms.calc = calc
    
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    
Or from command line:
    python -m mmml.cli.calculator --checkpoint path/to/ckpt --test-molecule CO2
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    print("❌ Error: JAX not installed")
    print("Install with: pip install jax jaxlib")
    sys.exit(1)

try:
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes
except ImportError:
    print("❌ Error: ASE not installed")
    print("Install with: pip install ase")
    sys.exit(1)


class MMMLCalculator(Calculator):
    """
    Generic ASE calculator for MMML models.
    
    This calculator automatically handles:
    - Edge list construction
    - Batch formatting
    - Property extraction (energy, forces, dipole, charges)
    
    Parameters
    ----------
    model : Any
        The trained model (e.g., JointPhysNetDCMNet, PhysNet, etc.)
    params : Any
        Trained model parameters
    cutoff : float
        Cutoff distance for neighbor list (Angstroms)
    use_dcmnet_dipole : bool
        If True and model has DCMNet, use DCMNet dipole; otherwise use atomic charges
    """
    
    implemented_properties = ['energy', 'forces', 'dipole', 'charges']
    
    def __init__(
        self,
        model: Any,
        params: Any,
        cutoff: float = 10.0,
        use_dcmnet_dipole: bool = False,
        **kwargs
    ):
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
                if i != j:
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist < self.cutoff:
                        dst_list.append(i)
                        src_list.append(j)
        
        dst_idx = np.array(dst_list, dtype=np.int32)
        src_idx = np.array(src_list, dtype=np.int32)
        
        # Batch data (single molecule)
        batch_segments = np.zeros(n_atoms, dtype=np.int32)
        batch_mask = np.ones(len(dst_idx), dtype=np.float32) if len(dst_idx) > 0 else np.array([], dtype=np.float32)
        atom_mask = np.ones(n_atoms, dtype=np.float32)
        
        # Run model
        try:
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
        except Exception as e:
            print(f"⚠️  Error during model.apply: {e}")
            print(f"Attempting alternative calling convention...")
            # Try alternative calling convention
            try:
                output = self.model.apply(
                    self.params,
                    jnp.array(atomic_numbers),
                    jnp.array(positions),
                    jnp.array(dst_idx),
                    jnp.array(src_idx),
                    jnp.array(batch_segments),
                    1,  # batch_size
                    jnp.array(batch_mask),
                    jnp.array(atom_mask),
                )
            except Exception as e2:
                print(f"❌ Error: Could not run model: {e2}")
                raise
        
        # Extract results
        self.results['energy'] = float(output['energy'][0]) if 'energy' in output else 0.0
        self.results['forces'] = np.array(output['forces']) if 'forces' in output else np.zeros((n_atoms, 3))
        
        # Dipole
        if self.use_dcmnet_dipole and 'dipole_dcm' in output:
            self.results['dipole'] = np.array(output['dipole_dcm'][0])
        elif 'dipoles' in output:
            self.results['dipole'] = np.array(output['dipoles'][0])
        elif 'dipole' in output:
            self.results['dipole'] = np.array(output['dipole'][0] if output['dipole'].ndim > 1 else output['dipole'])
        else:
            # Fallback: compute from charges
            charges = self._extract_charges(output, n_atoms)
            self.results['dipole'] = np.sum(charges[:, np.newaxis] * positions, axis=0)
        
        # Charges
        self.results['charges'] = self._extract_charges(output, n_atoms)
    
    def _extract_charges(self, output: Dict, n_atoms: int) -> np.ndarray:
        """Extract charges from model output."""
        if 'charges_as_mono' in output:
            charges = np.array(output['charges_as_mono'])
        elif 'charges' in output:
            charges = np.array(output['charges']).squeeze()
        elif 'atomic_charges' in output:
            charges = np.array(output['atomic_charges']).squeeze()
        else:
            charges = np.zeros(n_atoms)
        
        # Ensure correct shape
        if charges.ndim > 1:
            charges = charges.squeeze()
        
        return charges[:n_atoms] if len(charges) >= n_atoms else np.pad(charges, (0, n_atoms - len(charges)))
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        cutoff: float = 10.0,
        use_dcmnet_dipole: bool = False,
        model_type: Optional[str] = None,
    ) -> 'MMMLCalculator':
        """
        Create calculator from checkpoint file.
        
        Parameters
        ----------
        checkpoint_path : Path
            Path to checkpoint pickle file or directory containing checkpoint
        cutoff : float
            Cutoff distance for neighbor lists
        use_dcmnet_dipole : bool
            Use DCMNet dipole if available
        model_type : str, optional
            Model type hint ('dcmnet', 'noneq', 'physnet', etc.)
            If None, attempts auto-detection
        
        Returns
        -------
        MMMLCalculator
            Configured calculator ready to use
        """
        import pickle
        
        checkpoint_path = Path(checkpoint_path)
        
        # Find checkpoint file
        if checkpoint_path.is_dir():
            # Look for common checkpoint names
            candidates = [
                checkpoint_path / 'best_params.pkl',
                checkpoint_path / 'final_params.pkl',
                checkpoint_path / 'checkpoint.pkl',
            ]
            for cand in candidates:
                if cand.exists():
                    checkpoint_path = cand
                    break
            else:
                raise FileNotFoundError(f"No checkpoint found in {checkpoint_path}")
        
        # Load checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        # Extract parameters
        if isinstance(checkpoint_data, dict) and 'params' in checkpoint_data:
            params = checkpoint_data['params']
        else:
            params = checkpoint_data
        
        # Load config
        config_path = checkpoint_path.parent / 'model_config.pkl'
        if config_path.exists():
            print(f"Loading config: {config_path}")
            with open(config_path, 'rb') as f:
                config = pickle.load(f)
            
            # Try to create model from config
            model = cls._create_model_from_config(config, model_type)
            cutoff = config.get('physnet_config', {}).get('cutoff', cutoff)
        else:
            print("⚠️  Warning: No model_config.pkl found")
            print("You may need to provide the model manually")
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Wrap params if needed
        if isinstance(params, dict) and 'params' not in params and any(k in params for k in ['physnet', 'dcmnet', 'noneq']):
            params = {'params': params}
        
        return cls(model, params, cutoff=cutoff, use_dcmnet_dipole=use_dcmnet_dipole)
    
    @staticmethod
    def _create_model_from_config(config: Dict, model_type: Optional[str] = None) -> Any:
        """Create model instance from configuration."""
        # Determine model type
        if model_type is None:
            if 'dcmnet_config' in config:
                model_type = 'dcmnet'
            elif 'noneq_config' in config:
                model_type = 'noneq'
            else:
                model_type = 'physnet'
        
        # Import and create model
        if model_type == 'dcmnet':
            try:
                from mmml.physnetjax.physnetjax.models.joint_physnet_dcmnet import JointPhysNetDCMNet
                model = JointPhysNetDCMNet(
                    physnet_config=config['physnet_config'],
                    dcmnet_config=config['dcmnet_config'],
                    mix_coulomb_energy=config.get('mix_coulomb_energy', False),
                )
            except ImportError:
                print("⚠️  Could not import JointPhysNetDCMNet, trying alternative...")
                raise
        elif model_type == 'noneq':
            try:
                from mmml.physnetjax.physnetjax.models.joint_physnet_noneq import JointPhysNetNonEquivariant
                model = JointPhysNetNonEquivariant(
                    physnet_config=config['physnet_config'],
                    noneq_config=config['noneq_config'],
                    mix_coulomb_energy=config.get('mix_coulomb_energy', False),
                )
            except ImportError:
                print("⚠️  Could not import JointPhysNetNonEquivariant, trying alternative...")
                raise
        else:
            try:
                from mmml.physnetjax.physnetjax.models.model import EF
                model = EF(**config.get('physnet_config', config))
            except ImportError:
                print("⚠️  Could not import PhysNet models")
                raise
        
        return model


def main():
    """Command-line interface for testing the calculator."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test MMML calculator with a simple molecule",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with CO2
  python -m mmml.cli.calculator --checkpoint my_model/best_params.pkl --test-molecule CO2
  
  # Test with H2O
  python -m mmml.cli.calculator --checkpoint my_model/ --test-molecule H2O
  
  # Custom molecule
  python -m mmml.cli.calculator --checkpoint my_model/ --symbols C O O --positions "0,0,0" "1.16,0,0" "-1.16,0,0"
        """
    )
    
    parser.add_argument('--checkpoint', type=Path, required=True,
                       help='Path to checkpoint file or directory')
    parser.add_argument('--cutoff', type=float, default=10.0,
                       help='Cutoff distance (Angstroms)')
    parser.add_argument('--use-dcmnet-dipole', action='store_true',
                       help='Use DCMNet dipole if available')
    parser.add_argument('--model-type', type=str, choices=['dcmnet', 'noneq', 'physnet'],
                       help='Model type (auto-detected if not specified)')
    
    # Test molecule options
    parser.add_argument('--test-molecule', type=str,
                       help='Test with predefined molecule (CO2, H2O, CH4, etc.)')
    parser.add_argument('--symbols', type=str, nargs='+',
                       help='Atomic symbols for custom molecule')
    parser.add_argument('--positions', type=str, nargs='+',
                       help='Positions as comma-separated coords (e.g., "0,0,0" "1,0,0")')
    
    args = parser.parse_args()
    
    # Create calculator
    print("\n🔧 Creating calculator...")
    calc = MMMLCalculator.from_checkpoint(
        args.checkpoint,
        cutoff=args.cutoff,
        use_dcmnet_dipole=args.use_dcmnet_dipole,
        model_type=args.model_type,
    )
    print("✅ Calculator created successfully")
    
    # Create test molecule
    if args.test_molecule:
        atoms = create_test_molecule(args.test_molecule)
    elif args.symbols and args.positions:
        symbols = args.symbols
        positions = [list(map(float, pos.split(','))) for pos in args.positions]
        atoms = Atoms(symbols=symbols, positions=positions)
    else:
        print("\n⚠️  No molecule specified, using default CO2")
        atoms = create_test_molecule('CO2')
    
    atoms.calc = calc
    
    # Calculate properties
    print(f"\n🧪 Testing with {atoms.get_chemical_formula()}...")
    print(f"   Positions:\n{atoms.get_positions()}")
    
    try:
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        dipole = atoms.get_dipole_moment()
        charges = atoms.get_charges()
        
        print(f"\n📊 Results:")
        print(f"   Energy: {energy:.6f} eV")
        print(f"   Max force: {np.abs(forces).max():.6f} eV/Å")
        print(f"   Dipole magnitude: {np.linalg.norm(dipole):.6f} e·Å ({np.linalg.norm(dipole) * 4.8032:.4f} Debye)")
        print(f"   Total charge: {charges.sum():.6f} e")
        print(f"   Charge range: [{charges.min():.6f}, {charges.max():.6f}] e")
        
        print("\n✅ Test passed!")
        
    except Exception as e:
        print(f"\n❌ Error during calculation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def create_test_molecule(name: str) -> Atoms:
    """Create a test molecule by name."""
    molecules = {
        'CO2': Atoms('CO2', positions=[[0,0,0], [1.16,0,0], [-1.16,0,0]]),
        'H2O': Atoms('H2O', positions=[[0,0,0], [0.757,0.586,0], [-0.757,0.586,0]]),
        'CH4': Atoms('CH5', positions=[
            [0,0,0],  # C
            [0.629,0.629,0.629],  # H
            [-0.629,-0.629,0.629],  # H
            [-0.629,0.629,-0.629],  # H
            [0.629,-0.629,-0.629],  # H
        ]),
        'NH3': Atoms('NH3', positions=[
            [0,0,0],  # N
            [0.94,0,0.33],  # H
            [-0.47,0.82,0.33],  # H
            [-0.47,-0.82,0.33],  # H
        ]),
    }
    
    name = name.upper()
    if name not in molecules:
        raise ValueError(f"Unknown molecule: {name}. Available: {list(molecules.keys())}")
    
    return molecules[name]


if __name__ == '__main__':
    main()

