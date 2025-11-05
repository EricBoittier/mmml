#!/usr/bin/env python3
"""
CO2 Training Script - ZBL Stability Test

Tests training stability with ZBL repulsion enabled across different prediction modes:
1. Energy only
2. Energy + Forces
3. Forces only  
4. Energy + Forces + Dipoles

This helps identify MD instabilities that arise during training (gradient updates).
"""

import time
import e3x
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path

from mmml.physnetjax.physnetjax.models.model_charge_spin import EF_ChargeSpinConditioned
from mmml.physnetjax.physnetjax.training.optimizer import get_optimizer


def generate_co2_training_data(n_samples=100, seed=42):
    """
    Generate synthetic CO2 training data with perturbations.
    
    CO2: O=C=O (linear molecule)
    - C-O bond length: ~1.16 Å
    - O-C-O angle: 180° (linear)
    """
    rng = np.random.default_rng(seed)
    
    data = []
    for i in range(n_samples):
        # Base geometry (linear CO2)
        co_bond = 1.16 + rng.normal(0, 0.05)  # Perturb bond length
        
        # Add small random rotation and translation
        angle = rng.uniform(0, 2*np.pi)
        offset = rng.normal(0, 0.1, 3)
        
        Z = np.array([6, 8, 8], dtype=np.int32)  # C, O, O
        R = np.array([
            [0.0, 0.0, 0.0],
            [-co_bond, 0.0, 0.0],
            [co_bond, 0.0, 0.0],
        ], dtype=np.float32)
        
        # Apply rotation around z-axis
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
        R = (rot @ R.T).T + offset
        
        # Generate dummy forces (harmonic potential towards equilibrium)
        R_eq = np.array([
            [0.0, 0.0, 0.0],
            [-1.16, 0.0, 0.0],
            [1.16, 0.0, 0.0],
        ], dtype=np.float32)
        k = 50.0  # Force constant
        F = -k * (R - R_eq)
        
        # Energy (harmonic)
        E = 0.5 * k * np.sum((R - R_eq)**2)
        
        data.append({
            'Z': Z,
            'R': R,
            'F': F,
            'E': E,
        })
    
    return data


def create_batches(data, batch_size, num_atoms=10):
    """Create batches from data."""
    n_samples = len(data)
    n_batches = n_samples // batch_size
    
    batches = []
    for i in range(n_batches):
        batch_data = data[i*batch_size:(i+1)*batch_size]
        
        Z_batch = np.zeros((batch_size, num_atoms), dtype=np.int32)
        R_batch = np.zeros((batch_size, num_atoms, 3), dtype=np.float32)
        F_batch = np.zeros((batch_size, num_atoms, 3), dtype=np.float32)
        E_batch = np.zeros(batch_size, dtype=np.float64)
        
        for j, sample in enumerate(batch_data):
            n = len(sample['Z'])
            Z_batch[j, :n] = sample['Z']
            R_batch[j, :n] = sample['R']
            F_batch[j, :n] = sample['F']
            E_batch[j] = sample['E']
        
        batches.append({
            'Z': Z_batch,
            'R': R_batch,
            'F': F_batch,
            'E': E_batch,
        })
    
    return batches


def train_step(model, params, opt_state, optimizer, batch, num_atoms,
               predict_energy=True, predict_forces=True, predict_dipoles=False,
               energy_weight=1.0, forces_weight=1.0, dipoles_weight=1.0):
    """Single training step."""
    
    def loss_fn(params):
        # Flatten batch
        Z_flat = batch['Z'].reshape(-1)
        R_flat = batch['R'].reshape(-1, 3)
        batch_size = batch['Z'].shape[0]
        batch_segments = np.repeat(np.arange(batch_size), num_atoms).astype(np.int32)
        
        dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)
        
        outputs = model.apply(
            params,
            atomic_numbers=jnp.array(Z_flat),
            positions=jnp.array(R_flat),
            dst_idx=dst_idx,
            src_idx=src_idx,
            total_charges=jnp.zeros(batch_size),
            total_spins=jnp.ones(batch_size),
            batch_segments=batch_segments,
            batch_size=batch_size,
            batch_mask=jnp.ones_like(dst_idx),
            atom_mask=(jnp.array(Z_flat) > 0).astype(jnp.float32),
            predict_energy=predict_energy,
            predict_forces=predict_forces,
        )
        
        loss = 0.0
        metrics = {}
        
        # Energy loss
        if predict_energy and outputs['energy'] is not None:
            E_pred = outputs['energy']
            E_true = jnp.array(batch['E'])
            e_loss = jnp.mean((E_pred - E_true) ** 2)
            loss = loss + energy_weight * e_loss
            metrics['e_mae'] = jnp.mean(jnp.abs(E_pred - E_true))
        
        # Forces loss
        if predict_forces and outputs['forces'] is not None:
            F_pred = outputs['forces']
            F_true = jnp.array(batch['F'].reshape(-1, 3))
            mask = (jnp.array(Z_flat) > 0).astype(jnp.float32)
            F_diff = (F_pred - F_true) * mask[:, None]
            f_loss = jnp.mean(F_diff ** 2)
            loss = loss + forces_weight * f_loss
            metrics['f_mae'] = jnp.mean(jnp.abs(F_diff))
        
        # Check for NaN in predictions
        if outputs['energy'] is not None:
            metrics['e_has_nan'] = jnp.any(jnp.isnan(outputs['energy']))
        if outputs['forces'] is not None:
            metrics['f_has_nan'] = jnp.any(jnp.isnan(outputs['forces']))
        
        metrics['loss_has_nan'] = jnp.isnan(loss)
        
        return loss, metrics
    
    # Compute gradients
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    
    # Check for NaN in gradients
    grad_has_nan = any(
        jnp.any(jnp.isnan(g)) for g in jax.tree_util.tree_leaves(grads)
    )
    metrics['grad_has_nan'] = grad_has_nan
    
    # Update parameters
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = jax.tree_util.tree_map(lambda p, u: p + u, params, updates)
    
    return params, opt_state, loss, metrics


def train_mode(mode_name, predict_energy, predict_forces, predict_dipoles,
               n_epochs=10, batch_size=16, verbose=True):
    """Train in a specific mode and check stability."""
    
    if verbose:
        print("\n" + "="*80)
        print(f"Training Mode: {mode_name}")
        print("="*80)
        print(f"  Predict Energy:  {predict_energy}")
        print(f"  Predict Forces:  {predict_forces}")
        print(f"  Predict Dipoles: {predict_dipoles}")
        print(f"  Epochs: {n_epochs}")
    
    # Generate data
    if verbose:
        print("\nGenerating CO2 training data...")
    train_data = generate_co2_training_data(n_samples=80, seed=42)
    valid_data = generate_co2_training_data(n_samples=20, seed=123)
    
    num_atoms = 10
    train_batches = create_batches(train_data, batch_size, num_atoms)
    valid_batches = create_batches(valid_data, batch_size, num_atoms)
    
    if verbose:
        print(f"  Train batches: {len(train_batches)}")
        print(f"  Valid batches: {len(valid_batches)}")
    
    # Create model with ZBL enabled
    if verbose:
        print("\nCreating model (ZBL ENABLED)...")
    
    model = EF_ChargeSpinConditioned(
        features=32,
        max_degree=2,
        num_iterations=2,
        num_basis_functions=8,
        cutoff=5.0,
        natoms=num_atoms,
        n_res=1,
        zbl=True,  # ← ZBL ENABLED
        charges=predict_dipoles,
        charge_embed_dim=8,
        spin_embed_dim=8,
        charge_range=(0, 0),
        spin_range=(1, 1),
    )
    
    # Initialize
    key = jax.random.PRNGKey(42)
    sample_batch = train_batches[0]
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)
    
    params = model.init(
        key,
        atomic_numbers=jnp.array(sample_batch['Z'][0].reshape(-1)),
        positions=jnp.array(sample_batch['R'][0].reshape(-1, 3)),
        dst_idx=dst_idx,
        src_idx=src_idx,
        total_charges=jnp.zeros(1),
        total_spins=jnp.ones(1),
    )
    
    if verbose:
        n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
        print(f"  Model initialized: {n_params:,} parameters")
    
    # Create optimizer
    optimizer, _, _, _ = get_optimizer(
        learning_rate=0.001,
        schedule_fn=None,
        optimizer=None,
        transform=None,
    )
    opt_state = optimizer.init(params)
    
    # Training loop
    if verbose:
        print("\nTraining...")
    
    nan_detected = False
    nan_epoch = -1
    
    for epoch in range(1, n_epochs + 1):
        # Train
        train_loss = 0.0
        train_nan_count = 0
        
        for i, batch in enumerate(train_batches):
            params, opt_state, loss, metrics = train_step(
                model, params, opt_state, optimizer, batch, num_atoms,
                predict_energy=predict_energy,
                predict_forces=predict_forces,
                predict_dipoles=predict_dipoles,
            )
            
            train_loss += (loss - train_loss) / (i + 1)
            
            # Check for NaN
            if metrics.get('loss_has_nan', False) or \
               metrics.get('grad_has_nan', False) or \
               metrics.get('e_has_nan', False) or \
               metrics.get('f_has_nan', False):
                nan_detected = True
                nan_epoch = epoch
                if verbose:
                    print(f"\n  ❌ NaN DETECTED at epoch {epoch}, batch {i}!")
                    print(f"     Loss NaN: {metrics.get('loss_has_nan', False)}")
                    print(f"     Grad NaN: {metrics.get('grad_has_nan', False)}")
                    print(f"     Energy NaN: {metrics.get('e_has_nan', False)}")
                    print(f"     Forces NaN: {metrics.get('f_has_nan', False)}")
                break
        
        if nan_detected:
            break
        
        # Validate
        valid_loss = 0.0
        for i, batch in enumerate(valid_batches):
            _, _, loss, _ = train_step(
                model, params, opt_state, optimizer, batch, num_atoms,
                predict_energy=predict_energy,
                predict_forces=predict_forces,
                predict_dipoles=predict_dipoles,
            )
            valid_loss += (loss - valid_loss) / (i + 1)
        
        if verbose and epoch % 2 == 0:
            print(f"  Epoch {epoch}/{n_epochs}: Train Loss={train_loss:.6f}, Valid Loss={valid_loss:.6f}")
    
    # Results
    if nan_detected:
        if verbose:
            print(f"\n❌ UNSTABLE: NaN detected at epoch {nan_epoch}")
        return False, nan_epoch
    else:
        if verbose:
            print(f"\n✓ STABLE: Completed {n_epochs} epochs without NaN")
        return True, -1


def main():
    print("="*80)
    print("CO2 Training - ZBL Stability Test")
    print("="*80)
    print("\nThis script tests training stability with ZBL enabled across")
    print("different prediction modes to identify MD instabilities.")
    print("\nTesting modes:")
    print("  1. Energy only")
    print("  2. Energy + Forces")
    print("  3. Forces only")
    print("  4. Energy + Forces + Dipoles")
    
    results = {}
    
    # Test 1: Energy only
    stable, nan_epoch = train_mode(
        "Energy Only",
        predict_energy=True,
        predict_forces=False,
        predict_dipoles=False,
        n_epochs=10,
    )
    results['Energy Only'] = {'stable': stable, 'nan_epoch': nan_epoch}
    
    # Test 2: Energy + Forces
    stable, nan_epoch = train_mode(
        "Energy + Forces",
        predict_energy=True,
        predict_forces=True,
        predict_dipoles=False,
        n_epochs=10,
    )
    results['Energy + Forces'] = {'stable': stable, 'nan_epoch': nan_epoch}
    
    # Test 3: Forces only
    stable, nan_epoch = train_mode(
        "Forces Only",
        predict_energy=False,
        predict_forces=True,
        predict_dipoles=False,
        n_epochs=10,
    )
    results['Forces Only'] = {'stable': stable, 'nan_epoch': nan_epoch}
    
    # Test 4: Energy + Forces + Dipoles
    stable, nan_epoch = train_mode(
        "Energy + Forces + Dipoles",
        predict_energy=True,
        predict_forces=True,
        predict_dipoles=True,
        n_epochs=10,
    )
    results['Energy + Forces + Dipoles'] = {'stable': stable, 'nan_epoch': nan_epoch}
    
    # Summary
    print("\n" + "="*80)
    print("STABILITY SUMMARY")
    print("="*80)
    
    all_stable = True
    for mode, result in results.items():
        status = "✓ STABLE" if result['stable'] else f"❌ UNSTABLE (NaN @ epoch {result['nan_epoch']})"
        print(f"  {mode:30s}: {status}")
        if not result['stable']:
            all_stable = False
    
    print("="*80)
    
    if all_stable:
        print("\n✅ SUCCESS: All modes are stable!")
        print("   ZBL is working correctly across all prediction modes.")
    else:
        print("\n⚠️  INSTABILITY DETECTED!")
        print("\nPossible causes:")
        print("  1. ZBL parameters causing gradient explosion")
        print("  2. Learning rate too high for ZBL gradients")
        print("  3. Numerical instability in switching functions")
        print("  4. Issue with force computation through ZBL")
        print("\nRecommended fixes:")
        print("  • Reduce learning rate (try 0.0001)")
        print("  • Add gradient clipping")
        print("  • Adjust ZBL switching parameters")
        print("  • Scale ZBL contribution with learnable weight")
    
    print("="*80)


if __name__ == "__main__":
    main()

