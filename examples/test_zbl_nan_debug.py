#!/usr/bin/env python3
"""
Test ZBL Repulsion and Debug NaN Issues

This test creates realistic molecular geometries and tests the charge-spin model
with and without ZBL repulsion to identify NaN issues.
"""

import e3x
import jax
import jax.numpy as jnp
import numpy as np

from mmml.physnetjax.physnetjax.models.model_charge_spin import EF_ChargeSpinConditioned


def create_water_molecule():
    """Create a realistic water molecule geometry."""
    # O-H bond length ~0.96 Å, H-O-H angle ~104.5°
    Z = np.array([8, 1, 1], dtype=np.int32)  # O, H, H
    R = np.array([
        [0.0, 0.0, 0.0],           # O at origin
        [0.96, 0.0, 0.0],          # H1
        [-0.24, 0.93, 0.0],        # H2
    ], dtype=np.float32)
    return Z, R


def create_methane_molecule():
    """Create a realistic methane molecule geometry."""
    # C-H bond length ~1.09 Å, tetrahedral
    Z = np.array([6, 1, 1, 1, 1], dtype=np.int32)  # C, H, H, H, H
    R = np.array([
        [0.0, 0.0, 0.0],           # C at origin
        [1.09, 0.0, 0.0],          # H1
        [-0.363, 1.027, 0.0],      # H2
        [-0.363, -0.513, 0.889],   # H3
        [-0.363, -0.513, -0.889],  # H4
    ], dtype=np.float32)
    return Z, R


def create_test_batch(molecules, num_atoms=10):
    """Create a padded batch from molecules."""
    batch_size = len(molecules)
    
    Z_batch = np.zeros((batch_size, num_atoms), dtype=np.int32)
    R_batch = np.zeros((batch_size, num_atoms, 3), dtype=np.float32)
    
    for i, (Z, R) in enumerate(molecules):
        n = len(Z)
        Z_batch[i, :n] = Z
        R_batch[i, :n] = R
    
    return Z_batch, R_batch


def test_model_with_zbl(zbl_enabled=True, verbose=True):
    """Test model with or without ZBL."""
    if verbose:
        print("\n" + "="*80)
        print(f"Testing with ZBL={'ENABLED' if zbl_enabled else 'DISABLED'}")
        print("="*80)
    
    # Create test molecules
    water = create_water_molecule()
    methane = create_methane_molecule()
    molecules = [water, methane]
    
    num_atoms = 10
    Z_batch, R_batch = create_test_batch(molecules, num_atoms)
    
    if verbose:
        print(f"\nMolecules:")
        print(f"  Water:   Z={water[0]}, positions shape={water[1].shape}")
        print(f"  Methane: Z={methane[0]}, positions shape={methane[1].shape}")
        print(f"\nBatch:")
        print(f"  Z shape: {Z_batch.shape}")
        print(f"  R shape: {R_batch.shape}")
    
    # Create model
    model = EF_ChargeSpinConditioned(
        features=32,
        max_degree=2,
        num_iterations=2,
        num_basis_functions=8,
        cutoff=5.0,
        natoms=num_atoms,
        n_res=1,
        zbl=zbl_enabled,  # Toggle ZBL
        charge_embed_dim=8,
        spin_embed_dim=8,
        charge_range=(-1, 1),
        spin_range=(1, 3),
    )
    
    # Initialize
    key = jax.random.PRNGKey(42)
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)
    
    if verbose:
        print(f"\nInitializing model (ZBL={zbl_enabled})...")
    
    params = model.init(
        key,
        atomic_numbers=jnp.array(Z_batch[0]),
        positions=jnp.array(R_batch[0]),
        dst_idx=dst_idx,
        src_idx=src_idx,
        total_charges=jnp.array([0.0]),
        total_spins=jnp.array([1.0]),
    )
    
    if verbose:
        n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
        print(f"✓ Model initialized: {n_params:,} parameters")
    
    # Forward pass
    if verbose:
        print(f"\nRunning forward pass...")
    
    batch_segments = np.repeat(np.arange(len(molecules)), num_atoms).astype(np.int32)
    
    outputs = model.apply(
        params,
        atomic_numbers=jnp.array(Z_batch.reshape(-1)),  # Flatten
        positions=jnp.array(R_batch.reshape(-1, 3)),     # Flatten
        dst_idx=dst_idx,
        src_idx=src_idx,
        total_charges=jnp.array([0.0, 0.0]),  # Neutral
        total_spins=jnp.array([1.0, 1.0]),    # Singlet
        batch_segments=batch_segments,
        batch_size=len(molecules),
        batch_mask=jnp.ones_like(dst_idx),
        atom_mask=(jnp.array(Z_batch.reshape(-1)) > 0).astype(jnp.float32),
        predict_energy=True,
        predict_forces=True,
    )
    
    # Check for NaN
    energy = outputs["energy"]
    forces = outputs["forces"]
    
    has_nan = False
    if jnp.any(jnp.isnan(energy)):
        if verbose:
            print("❌ ENERGY contains NaN!")
        has_nan = True
    else:
        if verbose:
            print(f"✓ Energy: {energy}")
    
    if jnp.any(jnp.isnan(forces)):
        if verbose:
            print("❌ FORCES contain NaN!")
        has_nan = True
    else:
        if verbose:
            print(f"✓ Forces shape: {forces.shape}")
            print(f"  Forces min/max: [{jnp.min(forces):.6f}, {jnp.max(forces):.6f}]")
            print(f"  Forces mean: {jnp.mean(forces):.6f}")
    
    if outputs["repulsion"] is not None:
        repulsion = outputs["repulsion"]
        if jnp.any(jnp.isnan(repulsion)):
            if verbose:
                print("❌ REPULSION contains NaN!")
            has_nan = True
        else:
            if verbose:
                print(f"✓ Repulsion shape: {repulsion.shape}")
                print(f"  Repulsion min/max: [{jnp.min(repulsion):.6f}, {jnp.max(repulsion):.6f}]")
    
    if verbose:
        if has_nan:
            print("\n" + "!"*80)
            print("NaN DETECTED!")
            print("!"*80)
        else:
            print("\n" + "✓"*80)
            print("NO NaN - All values are finite")
            print("✓"*80)
    
    return has_nan, outputs


def main():
    print("="*80)
    print("ZBL Repulsion and NaN Debug Test")
    print("="*80)
    print("\nThis test creates realistic molecular geometries and tests for NaN issues")
    print("with ZBL repulsion enabled and disabled.")
    
    # Test without ZBL
    has_nan_no_zbl, outputs_no_zbl = test_model_with_zbl(zbl_enabled=False, verbose=True)
    
    # Test with ZBL
    has_nan_with_zbl, outputs_with_zbl = test_model_with_zbl(zbl_enabled=True, verbose=True)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"ZBL Disabled: {'❌ Has NaN' if has_nan_no_zbl else '✓ No NaN'}")
    print(f"ZBL Enabled:  {'❌ Has NaN' if has_nan_with_zbl else '✓ No NaN'}")
    
    if has_nan_with_zbl and not has_nan_no_zbl:
        print("\n⚠️  ZBL IS CAUSING NaN!")
        print("Possible causes:")
        print("  1. Numerical instability at short distances")
        print("  2. Incorrect switching function parameters")
        print("  3. Issue with ZBL parameter initialization")
        print("  4. Gradient computation through ZBL")
    elif has_nan_no_zbl:
        print("\n⚠️  NaN exists even without ZBL!")
        print("Issue is likely in:")
        print("  1. Charge/spin embedding")
        print("  2. Message passing")
        print("  3. Energy prediction head")
    else:
        print("\n✓ All tests passed!")
        print("Model is working correctly with and without ZBL.")
    
    print("="*80)


if __name__ == "__main__":
    main()

