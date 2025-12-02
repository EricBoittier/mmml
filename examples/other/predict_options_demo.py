#!/usr/bin/env python3
"""
Demo: Selective Energy/Forces Prediction

This demonstrates the new predict_energy and predict_forces flags
for optimized inference.
"""

import time
import e3x
import jax
import jax.numpy as jnp
import numpy as np

from mmml.physnetjax.physnetjax.models.model_charge_spin import EF_ChargeSpinConditioned


def create_test_molecule(num_atoms=10):
    """Create a test molecule."""
    Z = np.array([8, 1, 1] + [0] * (num_atoms - 3), dtype=np.int32)  # Water
    R = np.random.randn(num_atoms, 3).astype(np.float32) * 2.0
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)
    return Z, R, dst_idx, src_idx


def main():
    print("="*80)
    print("Selective Prediction Demo")
    print("="*80)
    
    # Create model
    model = EF_ChargeSpinConditioned(
        features=64,
        num_iterations=2,
        natoms=10,
        charge_range=(-1, 1),
        spin_range=(1, 3),
    )
    
    # Initialize
    key = jax.random.PRNGKey(42)
    Z, R, dst_idx, src_idx = create_test_molecule()
    
    params = model.init(
        key,
        atomic_numbers=Z,
        positions=R,
        dst_idx=dst_idx,
        src_idx=src_idx,
        total_charges=jnp.array([0.0]),
        total_spins=jnp.array([1.0]),
    )
    
    print(f"✓ Model initialized\n")
    
    # Test different prediction modes
    print("-"*80)
    print("1. DEFAULT: Predict both energy and forces")
    print("-"*80)
    
    outputs = model.apply(
        params,
        atomic_numbers=Z,
        positions=R,
        dst_idx=dst_idx,
        src_idx=src_idx,
        total_charges=jnp.array([0.0]),
        total_spins=jnp.array([1.0]),
        predict_energy=True,   # Default
        predict_forces=True,   # Default
    )
    
    print(f"Energy: {outputs['energy']}")
    print(f"Forces shape: {outputs['forces'].shape}")
    print(f"Forces available: {outputs['forces'] is not None}")
    
    print("\n" + "-"*80)
    print("2. ENERGY ONLY: Skip force computation (faster!)")
    print("-"*80)
    
    start = time.time()
    outputs = model.apply(
        params,
        atomic_numbers=Z,
        positions=R,
        dst_idx=dst_idx,
        src_idx=src_idx,
        total_charges=jnp.array([0.0]),
        total_spins=jnp.array([1.0]),
        predict_energy=True,
        predict_forces=False,  # ← Skip forces (no autograd!)
    )
    energy_only_time = time.time() - start
    
    print(f"Energy: {outputs['energy']}")
    print(f"Forces: {outputs['forces']}")
    print(f"✓ No gradient computation needed!")
    print(f"Time: {energy_only_time*1000:.2f} ms")
    
    print("\n" + "-"*80)
    print("3. FORCES ONLY: Skip returning energy")
    print("-"*80)
    
    outputs = model.apply(
        params,
        atomic_numbers=Z,
        positions=R,
        dst_idx=dst_idx,
        src_idx=src_idx,
        total_charges=jnp.array([0.0]),
        total_spins=jnp.array([1.0]),
        predict_energy=False,  # ← Don't return energy
        predict_forces=True,
    )
    
    print(f"Energy: {outputs['energy']}")
    print(f"Forces shape: {outputs['forces'].shape}")
    print(f"✓ Energy still computed (needed for gradient), but not returned")
    
    print("\n" + "-"*80)
    print("4. DIFFERENT CHARGE STATES: Use defaults (neutral singlet)")
    print("-"*80)
    
    # Neutral singlet (default behavior)
    outputs_neutral = model.apply(
        params,
        atomic_numbers=Z,
        positions=R,
        dst_idx=dst_idx,
        src_idx=src_idx,
        total_charges=jnp.array([0.0]),  # ← Neutral
        total_spins=jnp.array([1.0]),    # ← Singlet
        predict_energy=True,
        predict_forces=False,
    )
    
    # Cation doublet
    outputs_cation = model.apply(
        params,
        atomic_numbers=Z,
        positions=R,
        dst_idx=dst_idx,
        src_idx=src_idx,
        total_charges=jnp.array([1.0]),  # ← +1 charge
        total_spins=jnp.array([2.0]),    # ← Doublet
        predict_energy=True,
        predict_forces=False,
    )
    
    # Anion doublet
    outputs_anion = model.apply(
        params,
        atomic_numbers=Z,
        positions=R,
        dst_idx=dst_idx,
        src_idx=src_idx,
        total_charges=jnp.array([-1.0]),  # ← -1 charge
        total_spins=jnp.array([2.0]),     # ← Doublet
        predict_energy=True,
        predict_forces=False,
    )
    
    print(f"Neutral (singlet):  E = {outputs_neutral['energy'][0]:.6f}")
    print(f"Cation  (doublet):  E = {outputs_cation['energy'][0]:.6f}")
    print(f"Anion   (doublet):  E = {outputs_anion['energy'][0]:.6f}")
    
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print("✓ predict_energy=True/False   → Control energy computation")
    print("✓ predict_forces=True/False   → Control force computation")
    print("✓ total_charges=0 (default)   → Neutral molecules")
    print("✓ total_spins=1 (default)     → Singlet (closed-shell)")
    print("\nUse Cases:")
    print("  • Energy only: Screening, Monte Carlo, optimization")
    print("  • Forces only: MD simulations (if E not needed)")
    print("  • Both: Full quantum chemistry, training")
    print("  • Multi-state: Ionization energies, S-T gaps, etc.")
    print("="*80)


if __name__ == "__main__":
    main()

