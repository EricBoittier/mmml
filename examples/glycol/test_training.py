#!/usr/bin/env python3
"""
Quick test to verify PhysNet training setup works.
Runs just 1 epoch with small batch size.
"""

import numpy as np
import jax
import jax.numpy as jnp
import e3x
from pathlib import Path
import sys

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from mmml.physnetjax.physnetjax.models.model import EF
from mmml.physnetjax.physnetjax.training.trainstep import train_step
from mmml.physnetjax.physnetjax.training.optimizer import get_optimizer


def main():
    print("="*80)
    print("Quick PhysNet Training Test")
    print("="*80)
    
    # Load data
    data_path = Path(__file__).parent / "glycol.npz"
    print(f"\nLoading data from {data_path}...")
    data = np.load(data_path)
    
    # Use just 10 samples for quick test
    n_samples = 10
    Z = data['Z'][:n_samples]
    R = data['R'][:n_samples]
    F = data['F'][:n_samples]
    E = data['E'][:n_samples]
    N = data['N'][:n_samples]
    
    print(f"Using {n_samples} samples for quick test")
    
    # Initialize model
    print("\nInitializing model...")
    model = EF(
        features=64,  # Smaller for faster test
        max_degree=2,
        num_iterations=2,  # Fewer iterations
        num_basis_functions=16,
        cutoff=5.0,
        max_atomic_number=118,
        charges=False,
        natoms=60,
        total_charge=0.0,
        n_res=2,  # Fewer residual blocks
        zbl=True,
        debug=False,
        efa=False,
    )
    
    key = jax.random.PRNGKey(42)
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(60)
    params = model.init(key, Z[0], R[0], dst_idx, src_idx)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Model initialized with {n_params:,} parameters")
    
    # Initialize optimizer
    optimizer, transform, _, _ = get_optimizer(0.001, None, None, None)
    ema_params = params
    opt_state = optimizer.init(params)
    transform_state = transform.init(params)
    
    # Run one training batch
    print("\nRunning one training batch...")
    batch_size = 4
    
    # Prepare batch
    Z_batch = Z[:batch_size]
    R_batch = R[:batch_size]
    F_batch = F[:batch_size]
    E_batch = E[:batch_size]
    N_batch = N[:batch_size]
    
    # Flatten
    Z_flat = Z_batch.reshape(-1)
    R_flat = R_batch.reshape(-1, 3)
    F_flat = F_batch.reshape(-1, 3)
    atom_mask = (Z_flat > 0).astype(np.float32)
    batch_mask = jnp.ones_like(dst_idx, dtype=jnp.float32)
    batch_segments = np.repeat(np.arange(batch_size), 60).astype(np.int32)
    
    batch = {
        'Z': jnp.array(Z_flat, dtype=jnp.int32),
        'R': jnp.array(R_flat, dtype=jnp.float32),
        'F': jnp.array(F_flat, dtype=jnp.float32),
        'E': jnp.array(E_batch, dtype=jnp.float64),
        'N': jnp.array(N_batch, dtype=jnp.int32),
        'atom_mask': jnp.array(atom_mask, dtype=jnp.float32),
        'batch_mask': batch_mask,
        'dst_idx': dst_idx,
        'src_idx': src_idx,
        'batch_segments': batch_segments,
    }
    
    (params, ema_params, opt_state, transform_state, 
     loss, energy_mae, forces_mae, dipole_mae) = train_step(
        model_apply=model.apply,
        optimizer_update=optimizer.update,
        transform_state=transform_state,
        batch=batch,
        batch_size=batch_size,
        energy_weight=1.0,
        forces_weight=23.06,
        dipole_weight=0.0,
        charges_weight=0.0,
        opt_state=opt_state,
        doCharges=False,
        params=params,
        ema_params=ema_params,
        debug=False,
    )
    
    print(f"\nResults:")
    print(f"  Loss: {loss:.6f}")
    print(f"  Energy MAE: {energy_mae:.4f} eV")
    print(f"  Forces MAE: {forces_mae:.4f} eV/Å")
    
    print("\n" + "="*80)
    print("✅ Training test successful!")
    print("You can now run the full training with: python train_minimal.py")
    print("="*80)


if __name__ == "__main__":
    main()

