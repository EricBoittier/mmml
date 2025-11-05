#!/usr/bin/env python3
"""
Minimal PhysNet Training Script for Glycol Dataset

Usage:
    python train_minimal.py
"""

import numpy as np
import jax
import jax.numpy as jnp
import e3x
from pathlib import Path

# Add mmml to path
import sys
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from mmml.physnetjax.physnetjax.models.model import EF
from mmml.physnetjax.physnetjax.training.trainstep import train_step
from mmml.physnetjax.physnetjax.training.evalstep import eval_step
from mmml.physnetjax.physnetjax.training.optimizer import get_optimizer


def load_glycol_data(data_path, train_size=0.8, valid_size=0.1):
    """Load and split glycol.npz data."""
    print(f"Loading data from {data_path}...")
    data = np.load(data_path)
    
    # Extract arrays
    Z = data['Z']  # (N, 60) atomic numbers
    R = data['R']  # (N, 60, 3) positions in Angstroms
    F = data['F']  # (N, 60, 3) forces in eV/Angstrom
    E = data['E']  # (N,) energies in eV
    N = data['N']  # (N,) number of atoms per molecule
    
    n_total = len(E)
    print(f"Total samples: {n_total}")
    print(f"Actual max atoms per molecule: {int(N.max())}")
    print(f"Data padded shape: {Z.shape[1]} (will use only first {int(N.max())} atoms)")
    
    # Create train/valid/test split
    indices = np.arange(n_total)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    n_train = int(n_total * train_size)
    n_valid = int(n_total * valid_size)
    
    train_idx = indices[:n_train]
    valid_idx = indices[n_train:n_train+n_valid]
    test_idx = indices[n_train+n_valid:]
    
    print(f"Train samples: {len(train_idx)}")
    print(f"Valid samples: {len(valid_idx)}")
    print(f"Test samples: {len(test_idx)}")
    
    return {
        'train': {'Z': Z[train_idx], 'R': R[train_idx], 'F': F[train_idx], 
                  'E': E[train_idx], 'N': N[train_idx]},
        'valid': {'Z': Z[valid_idx], 'R': R[valid_idx], 'F': F[valid_idx], 
                  'E': E[valid_idx], 'N': N[valid_idx]},
        'test': {'Z': Z[test_idx], 'R': R[test_idx], 'F': F[test_idx], 
                 'E': E[test_idx], 'N': N[test_idx]},
    }


def create_batch_generator(data, batch_size, num_atoms, shuffle=True):
    """Generator for batches from numpy arrays."""
    n_samples = len(data['E'])
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    for start in range(0, n_samples, batch_size):
        batch_idx = indices[start:start+batch_size]
        batch_size_actual = len(batch_idx)
        
        # Get batch data (only first num_atoms to avoid excessive padding)
        Z = data['Z'][batch_idx, :num_atoms]  # (B, A)
        R = data['R'][batch_idx, :num_atoms]  # (B, A, 3)
        F = data['F'][batch_idx, :num_atoms]  # (B, A, 3)
        E = data['E'][batch_idx]  # (B,)
        N = data['N'][batch_idx]  # (B,)
        
        # Flatten for PhysNet
        Z_flat = Z.reshape(-1)  # (B*A,)
        R_flat = R.reshape(-1, 3)  # (B*A, 3)
        F_flat = F.reshape(-1, 3)  # (B*A, 3)
        
        # Create masks
        atom_mask = (Z_flat > 0).astype(np.float32)
        
        # Generate graph indices
        dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)
        batch_mask = jnp.ones_like(dst_idx, dtype=jnp.float32)
        batch_segments = np.repeat(np.arange(batch_size_actual), num_atoms).astype(np.int32)
        
        yield {
            'Z': jnp.array(Z_flat, dtype=jnp.int32),
            'R': jnp.array(R_flat, dtype=jnp.float32),
            'F': jnp.array(F_flat, dtype=jnp.float32),
            'E': jnp.array(E, dtype=jnp.float64),
            'N': jnp.array(N, dtype=jnp.int32),
            'atom_mask': jnp.array(atom_mask, dtype=jnp.float32),
            'batch_mask': batch_mask,
            'dst_idx': dst_idx,
            'src_idx': src_idx,
            'batch_segments': batch_segments,
        }, batch_size_actual


def train_epoch(model, params, ema_params, opt_state, transform_state, optimizer, 
                data, batch_size, num_atoms, energy_weight, forces_weight):
    """Train for one epoch."""
    losses = []
    energy_maes = []
    forces_maes = []
    
    for batch, batch_size_actual in create_batch_generator(data, batch_size, num_atoms, shuffle=True):
        (params, ema_params, opt_state, transform_state, 
         loss, energy_mae, forces_mae, dipole_mae) = train_step(
            model_apply=model.apply,
            optimizer_update=optimizer.update,
            transform_state=transform_state,
            batch=batch,
            batch_size=batch_size_actual,
            energy_weight=energy_weight,
            forces_weight=forces_weight,
            dipole_weight=0.0,
            charges_weight=0.0,
            opt_state=opt_state,
            doCharges=False,
            params=params,
            ema_params=ema_params,
            debug=False,
        )
        
        losses.append(loss)
        energy_maes.append(energy_mae)
        forces_maes.append(forces_mae)
    
    return (params, ema_params, opt_state, transform_state,
            float(np.mean(losses)), float(np.mean(energy_maes)), float(np.mean(forces_maes)))


def validate(model, ema_params, data, batch_size, num_atoms, energy_weight, forces_weight):
    """Validate the model."""
    losses = []
    energy_maes = []
    forces_maes = []
    
    for batch, batch_size_actual in create_batch_generator(data, batch_size, num_atoms, shuffle=False):
        loss, energy_mae, forces_mae, dipole_mae = eval_step(
            model_apply=model.apply,
            batch=batch,
            batch_size=batch_size_actual,
            energy_weight=energy_weight,
            forces_weight=forces_weight,
            dipole_weight=0.0,
            charges_weight=0.0,
            charges=False,
            params=ema_params,
        )
        
        losses.append(loss)
        energy_maes.append(energy_mae)
        forces_maes.append(forces_mae)
    
    return float(np.mean(losses)), float(np.mean(energy_maes)), float(np.mean(forces_maes))


def main():
    # Configuration
    data_path = Path(__file__).parent / "glycol.npz"
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001
    num_atoms = 10  # Max atoms in glycol dataset
    
    # Model hyperparameters
    features = 128
    max_degree = 2
    num_iterations = 3
    num_basis_functions = 16
    cutoff = 5.0
    n_res = 3
    
    # Loss weights (forces weight converts eV/Ang to kcal/mol/Ang: 23.06)
    energy_weight = 1.0
    forces_weight = 23.06
    
    print("="*80)
    print("Minimal PhysNet Training - Glycol Dataset")
    print("="*80)
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print("="*80)
    
    # Load data
    data_splits = load_glycol_data(data_path)
    
    # Initialize model
    print("\nInitializing model...")
    model = EF(
        features=features,
        max_degree=max_degree,
        num_iterations=num_iterations,
        num_basis_functions=num_basis_functions,
        cutoff=cutoff,
        max_atomic_number=118,
        charges=False,
        natoms=num_atoms,
        total_charge=0.0,
        n_res=n_res,
        zbl=True,
        debug=False,
        efa=False,
    )
    
    # Initialize parameters
    key = jax.random.PRNGKey(42)
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)
    
    # Get sample data for initialization
    sample_Z = data_splits['train']['Z'][0, :num_atoms]  # (num_atoms,)
    sample_R = data_splits['train']['R'][0, :num_atoms]  # (num_atoms, 3)
    
    params = model.init(key, sample_Z, sample_R, dst_idx, src_idx)
    print(f"Model initialized with {sum(x.size for x in jax.tree_util.tree_leaves(params)):,} parameters")
    
    # Initialize optimizer
    optimizer, transform, _, _ = get_optimizer(
        learning_rate=learning_rate,
        schedule_fn=None,
        optimizer=None,
        transform=None,
    )
    
    ema_params = params
    opt_state = optimizer.init(params)
    transform_state = transform.init(params)
    
    # Training loop
    print("\nStarting training...")
    print("="*80)
    
    best_valid_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 80)
        
        # Train
        params, ema_params, opt_state, transform_state, train_loss, train_e_mae, train_f_mae = train_epoch(
            model, params, ema_params, opt_state, transform_state, optimizer,
            data_splits['train'], batch_size, num_atoms, energy_weight, forces_weight
        )
        
        # Validate
        valid_loss, valid_e_mae, valid_f_mae = validate(
            model, ema_params, data_splits['valid'], batch_size, num_atoms, 
            energy_weight, forces_weight
        )
        
        # Print results
        print(f"Train Loss: {train_loss:.6f} | E_MAE: {train_e_mae:.4f} eV | F_MAE: {train_f_mae:.4f} eV/Å")
        print(f"Valid Loss: {valid_loss:.6f} | E_MAE: {valid_e_mae:.4f} eV | F_MAE: {valid_f_mae:.4f} eV/Å")
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            print(f"  → New best validation loss!")
    
    # Test
    print("\n" + "="*80)
    print("Final Test Set Evaluation")
    print("="*80)
    test_loss, test_e_mae, test_f_mae = validate(
        model, ema_params, data_splits['test'], batch_size, num_atoms,
        energy_weight, forces_weight
    )
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Test Energy MAE: {test_e_mae:.4f} eV")
    print(f"Test Forces MAE: {test_f_mae:.4f} eV/Å")
    print("="*80)
    print("Training completed!")


if __name__ == "__main__":
    main()

