"""
ASE calculator for the electric field model
"""

import ase
import ase.calculators.calculator as ase_calc
import ase.io as ase_io
import e3x
import jax
import jax.numpy as jnp
import numpy as np
import json
from pathlib import Path
from flax import linen as nn

# Import model and functions from training script
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from training import MessagePassingModel
from model_functions import energy_and_forces


def load_params(params_path):
    """Load parameters from JSON file."""
    with open(params_path, 'r') as f:
        params_dict = json.load(f)
    
    # Convert numpy arrays back from lists
    def convert_to_jax(obj):
        if isinstance(obj, dict):
            return {k: convert_to_jax(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            arr = np.array(obj)
            if arr.dtype == np.float64:
                return jnp.array(arr, dtype=jnp.float32)
            elif arr.dtype == np.int64:
                return jnp.array(arr, dtype=jnp.int32)
            return jnp.array(arr)
        return obj
    
    params = convert_to_jax(params_dict)
    return params


def load_config(config_path):
    """Load model configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


class AseCalculatorEF(ase_calc.Calculator):
    """ASE calculator for electric field model."""
    
    implemented_properties = ["energy", "forces", "dipole"]
    
    def __init__(self, params_path, config_path=None, electric_field=None, **kwargs):
        """
        Initialize the calculator.
        
        Parameters
        ----------
        params_path : str or Path
            Path to parameters JSON file (e.g., params.json or params-UUID.json)
        config_path : str or Path, optional
            Path to config JSON file. If None, will try to auto-detect from params UUID.
        electric_field : array-like, shape (3,), optional
            Default electric field vector in eV/(e·Å). If None, must be provided in atoms.info['electric_field'].
        **kwargs
            Additional arguments passed to ase.calculators.calculator.Calculator
        """
        ase_calc.Calculator.__init__(self, **kwargs)
        
        # Load parameters
        params_path = Path(params_path)
        if not params_path.exists():
            raise FileNotFoundError(f"Parameters file not found: {params_path}")
        self.params = load_params(params_path)
        
        # Load config (try to auto-detect from UUID if not provided)
        if config_path is None:
            # Try to extract UUID from params filename
            if params_path.stem.startswith('params-') and len(params_path.stem) > 7:
                uuid_part = params_path.stem[7:]  # Remove 'params-' prefix
                config_candidate = params_path.parent / f'config-{uuid_part}.json'
                if config_candidate.exists():
                    config_path = config_candidate
                else:
                    # Try config.json symlink
                    config_candidate = params_path.parent / 'config.json'
                    if config_candidate.exists():
                        config_path = config_candidate
        
        if config_path is not None:
            config_path = Path(config_path)
            if config_path.exists():
                config = load_config(config_path)
                # Extract model config (may be nested under 'model_config' key)
                if 'model_config' in config:
                    model_config = config['model_config']
                else:
                    model_config = {k: v for k, v in config.items() 
                                  if k in ['features', 'max_degree', 'num_iterations', 
                                          'num_basis_functions', 'cutoff', 'max_atomic_number', 
                                          'include_pseudotensors']}
            else:
                raise FileNotFoundError(f"Config file not found: {config_path}")
        else:
            # Default model config (infer from params if possible, or use defaults)
            print("Warning: No config file found. Using default model configuration.")
            model_config = {
                'features': 64,
                'max_degree': 2,
                'num_iterations': 2,
                'num_basis_functions': 64,
                'cutoff': 10.0,
                'max_atomic_number': 55,
                'include_pseudotensors': True,
            }
        
        # Create model
        self.model = MessagePassingModel(**model_config)
        
        # Store default electric field
        if electric_field is not None:
            self.electric_field = jnp.asarray(electric_field, dtype=jnp.float32)
        else:
            self.electric_field = None
        
        # JIT compile the model apply function for efficiency
        @jax.jit
        def model_apply(params, atomic_numbers, positions, Ef, dst_idx_flat, src_idx_flat, 
                       batch_segments, batch_size, dst_idx=None, src_idx=None):
            return self.model.apply(params, atomic_numbers, positions, Ef,
                                  dst_idx_flat=dst_idx_flat, src_idx_flat=src_idx_flat,
                                  batch_segments=batch_segments, batch_size=batch_size,
                                  dst_idx=dst_idx, src_idx=src_idx)
        
        self.model_apply = model_apply
        
        # JIT compile energy_and_forces
        @jax.jit
        def compute_energy_forces_dipole(atomic_numbers, positions, Ef, dst_idx, src_idx,
                                        dst_idx_flat, src_idx_flat, batch_segments, batch_size):
            return energy_and_forces(
                self.model_apply, self.params,
                atomic_numbers=atomic_numbers,
                positions=positions,
                Ef=Ef,
                dst_idx_flat=dst_idx_flat,
                src_idx_flat=src_idx_flat,
                batch_segments=batch_segments,
                batch_size=batch_size,
                dst_idx=dst_idx,
                src_idx=src_idx,
            )
        
        self.compute_energy_forces_dipole = compute_energy_forces_dipole
    
    def calculate(self, atoms=None, properties=['energy'], system_changes=ase_calc.all_changes):
        """
        Calculate properties for the given atoms.
        
        Parameters
        ----------
        atoms : ase.Atoms
            Atomic structure
        properties : list of str
            Properties to calculate (e.g., ['energy', 'forces', 'dipole'])
        system_changes : list of str
            System changes since last calculation
        """
        ase_calc.Calculator.calculate(self, atoms, properties, system_changes)
        
        # Get atomic numbers and positions
        atomic_numbers = jnp.asarray(atoms.get_atomic_numbers(), dtype=jnp.int32)
        positions = jnp.asarray(atoms.get_positions(), dtype=jnp.float32)
        n_atoms = len(atoms)
        
        # Get electric field from atoms.info or use default
        if 'electric_field' in atoms.info:
            Ef = jnp.asarray(atoms.info['electric_field'], dtype=jnp.float32)
        elif self.electric_field is not None:
            Ef = self.electric_field
        else:
            raise ValueError(
                "Electric field not provided. Set atoms.info['electric_field'] or "
                "provide electric_field parameter to calculator."
            )
        
        # Ensure Ef has shape (3,)
        Ef = Ef.reshape(3)
        
        # Create indices for message passing
        dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(n_atoms)
        dst_idx = jnp.asarray(dst_idx, dtype=jnp.int32)
        src_idx = jnp.asarray(src_idx, dtype=jnp.int32)
        
        # Set up batch (single molecule, batch_size=1)
        batch_size = 1
        batch_segments = jnp.zeros(n_atoms, dtype=jnp.int32)  # All atoms in same batch
        
        # Compute flattened indices
        offsets = jnp.arange(batch_size, dtype=jnp.int32) * n_atoms
        dst_idx_flat = (dst_idx[None, :] + offsets[:, None]).reshape(-1)
        src_idx_flat = (src_idx[None, :] + offsets[:, None]).reshape(-1)
        
        # Reshape for model (add batch dimension)
        atomic_numbers_batched = atomic_numbers[None, :]  # (1, N)
        positions_batched = positions[None, :, :]  # (1, N, 3)
        Ef_batched = Ef[None, :]  # (1, 3)
        
        # Compute energy, forces, and dipole
        energy, forces, dipole = self.compute_energy_forces_dipole(
            atomic_numbers=atomic_numbers_batched,
            positions=positions_batched,
            Ef=Ef_batched,
            dst_idx=dst_idx,
            src_idx=src_idx,
            dst_idx_flat=dst_idx_flat,
            src_idx_flat=src_idx_flat,
            batch_segments=batch_segments,
            batch_size=batch_size,
        )
        
        # Convert to numpy and extract scalar values
        energy = float(np.asarray(energy)[0])  # Extract from batch
        forces = np.asarray(forces)  # Already shape (N, 3)
        dipole = np.asarray(dipole[0])  # Extract from batch, shape (3,)
        
        # Store results
        self.results = {
            'energy': energy,
            'forces': forces,
            'dipole': dipole,
        }
    
    def set_electric_field(self, electric_field):
        """Set the default electric field for calculations."""
        self.electric_field = jnp.asarray(electric_field, dtype=jnp.float32)

