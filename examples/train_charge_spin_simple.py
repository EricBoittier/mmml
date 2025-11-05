#!/usr/bin/env python3
"""
Simple example of training charge-spin conditioned PhysNet.

This demonstrates how to use the EF_ChargeSpinConditioned model which
accepts total molecular charge and spin multiplicity as inputs.
"""

import e3x
import jax
import jax.numpy as jnp
import numpy as np

from mmml.physnetjax.physnetjax.models.model_charge_spin import EF_ChargeSpinConditioned
from mmml.physnetjax.physnetjax.training.optimizer import get_optimizer


def create_dummy_batch(batch_size=4, num_atoms=10):
    """Create dummy molecular data for testing."""
    # Atomic numbers (e.g., water molecules: O, H, H)
    Z = np.array([
        [8, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # H2O
        [6, 1, 1, 1, 1, 0, 0, 0, 0, 0],  # CH4
        [7, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # NH3
        [6, 8, 1, 1, 1, 0, 0, 0, 0, 0],  # CH3OH (simplified)
    ], dtype=np.int32)
    
    # Random positions (Angstrom)
    R = np.random.randn(batch_size, num_atoms, 3).astype(np.float32) * 2.0
    
    # Random forces
    F = np.random.randn(batch_size, num_atoms, 3).astype(np.float32) * 0.1
    
    # Energies (kcal/mol)
    E = np.random.randn(batch_size).astype(np.float64) * 100.0
    
    # Total charges (neutral, cation, anion, neutral)
    total_charge = np.array([0, 1, -1, 0], dtype=np.float32)
    
    # Spin multiplicities (singlet, doublet, doublet, singlet)
    # Spin multiplicity = 2S + 1, where S is total spin
    # Singlet (S=0): 1, Doublet (S=1/2): 2, Triplet (S=1): 3, etc.
    total_spin = np.array([1, 2, 2, 1], dtype=np.float32)
    
    # Graph connectivity
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)
    
    # Batch segments (which molecule each atom belongs to)
    batch_segments = np.repeat(np.arange(batch_size), num_atoms).astype(np.int32)
    
    return {
        "Z": jnp.array(Z),
        "R": jnp.array(R),
        "F": jnp.array(F),
        "E": jnp.array(E),
        "total_charge": jnp.array(total_charge),
        "total_spin": jnp.array(total_spin),
        "dst_idx": dst_idx,
        "src_idx": src_idx,
        "batch_segments": batch_segments,
    }


def main():
    print("="*80)
    print("Charge-Spin Conditioned PhysNet - Simple Example")
    print("="*80)
    
    # Configuration
    batch_size = 4
    num_atoms = 10
    num_epochs = 5
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Max atoms: {num_atoms}")
    print(f"  Epochs: {num_epochs}")
    
    # Create model
    print("\n1. Creating model...")
    model = EF_ChargeSpinConditioned(
        features=64,
        max_degree=2,
        num_iterations=2,
        num_basis_functions=8,
        cutoff=5.0,
        natoms=num_atoms,
        n_res=2,
        charge_embed_dim=8,
        spin_embed_dim=8,
        charge_range=(-2, 2),    # Support charges from -2 to +2
        spin_range=(1, 4),       # Support singlet, doublet, triplet, quartet
    )
    print("  ✓ Model created")
    
    # Initialize model
    print("\n2. Initializing parameters...")
    key = jax.random.PRNGKey(42)
    key, init_key = jax.random.split(key)
    
    sample_batch = create_dummy_batch(batch_size, num_atoms)
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)
    
    params = model.init(
        init_key,
        atomic_numbers=sample_batch["Z"][0],
        positions=sample_batch["R"][0],
        dst_idx=dst_idx,
        src_idx=src_idx,
        total_charges=sample_batch["total_charge"][:1],
        total_spins=sample_batch["total_spin"][:1],
    )
    
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  ✓ Initialized {n_params:,} parameters")
    
    # Create optimizer
    print("\n3. Setting up optimizer...")
    optimizer, _, _, _ = get_optimizer(
        learning_rate=0.001,
        schedule_fn=None,
        optimizer=None,
        transform=None,
    )
    opt_state = optimizer.init(params)
    print("  ✓ Optimizer ready")
    
    # Training function
    def train_step(params, opt_state, batch):
        def loss_fn(params):
            outputs = model.apply(
                params,
                atomic_numbers=batch["Z"],
                positions=batch["R"],
                dst_idx=batch["dst_idx"],
                src_idx=batch["src_idx"],
                total_charges=batch["total_charge"],
                total_spins=batch["total_spin"],
                batch_segments=batch["batch_segments"],
                batch_size=batch_size,
                batch_mask=jnp.ones_like(batch["dst_idx"]),
                atom_mask=(batch["Z"] > 0).astype(jnp.float32),
            )
            
            # Simple MSE loss
            energy_loss = jnp.mean((outputs["energy"] - batch["E"]) ** 2)
            forces_loss = jnp.mean((outputs["forces"] - batch["F"]) ** 2)
            
            return energy_loss + forces_loss * 0.1
        
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = jax.tree_util.tree_map(lambda p, u: p + u, params, updates)
        
        return params, opt_state, loss
    
    # Training loop
    print("\n4. Training...")
    print("-" * 80)
    
    for epoch in range(1, num_epochs + 1):
        # Generate batch
        batch = create_dummy_batch(batch_size, num_atoms)
        
        # Train step
        params, opt_state, loss = train_step(params, opt_state, batch)
        
        print(f"Epoch {epoch}/{num_epochs}: Loss = {loss:.6f}")
    
    # Test inference
    print("\n5. Testing inference...")
    print("-" * 80)
    
    test_batch = create_dummy_batch(2, num_atoms)
    
    outputs = model.apply(
        params,
        atomic_numbers=test_batch["Z"],
        positions=test_batch["R"],
        dst_idx=test_batch["dst_idx"],
        src_idx=test_batch["src_idx"],
        total_charges=test_batch["total_charge"],
        total_spins=test_batch["total_spin"],
        batch_segments=test_batch["batch_segments"],
        batch_size=2,
        batch_mask=jnp.ones_like(test_batch["dst_idx"]),
        atom_mask=(test_batch["Z"] > 0).astype(jnp.float32),
    )
    
    print("\nModel outputs:")
    print(f"  Energies shape: {outputs['energy'].shape}")
    print(f"  Forces shape: {outputs['forces'].shape}")
    print(f"  Predicted energies: {outputs['energy']}")
    print(f"  True energies: {test_batch['E']}")
    
    print("\nMolecular properties:")
    print(f"  Total charges: {test_batch['total_charge']}")
    print(f"  Spin multiplicities: {test_batch['total_spin']}")
    
    print("\n" + "="*80)
    print("✓ Example completed successfully!")
    print("="*80)
    
    print("\nKey features demonstrated:")
    print("  ✓ Charge conditioning (neutral, cation, anion)")
    print("  ✓ Spin conditioning (singlet, doublet, triplet, quartet)")
    print("  ✓ Energy and force prediction")
    print("  ✓ Gradient-based training")
    
    print("\nNext steps:")
    print("  • Use real molecular data with charge/spin labels")
    print("  • Train on multiple charge/spin states")
    print("  • Compare predictions across different states")
    print("  • Use for excited state calculations")


if __name__ == "__main__":
    main()

