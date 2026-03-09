#!/usr/bin/env python3
"""
Diagnostic script to analyze PhysNet and DCMNet charge predictions.

Usage:
    python diagnose_charges.py --checkpoint path/to/best_params.pkl
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from mmml.physnetjax.physnetjax.models.model import EF
from mmml.dcmnet.dcmnet.modules import MessagePassingModel


def load_model_and_data(checkpoint_path, valid_esp_path, valid_efd_path):
    """Load trained model and validation data."""
    checkpoint_path = Path(checkpoint_path)
    
    # Load checkpoint
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / 'best_params.pkl'
    
    with open(checkpoint_path, 'rb') as f:
        params = pickle.load(f)
    
    # Load config
    config_path = checkpoint_path.parent / 'model_config.pkl'
    if config_path.exists():
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        print(f"✅ Loaded model config from {config_path}")
    else:
        print("❌ No model config found - using defaults")
        config = {}
    
    # Default values
    defaults = {
        'physnet_features': 64,
        'physnet_iterations': 3,
        'physnet_basis': 64,
        'physnet_cutoff': 10.0,
        'dcmnet_features': 32,
        'dcmnet_iterations': 2,
        'dcmnet_basis': 32,
        'dcmnet_cutoff': 10.0,
        'n_dcm': 4,
    }
    
    # Merge with defaults
    for key, default_val in defaults.items():
        if key not in config:
            config[key] = default_val
            print(f"  Using default {key}: {default_val}")
    
    # Create model
    physnet = EF(
        features=config['physnet_features'],
        num_iterations=config['physnet_iterations'],
        num_basis_functions=config['physnet_basis'],
        cutoff=config['physnet_cutoff'],
        charges=True,
        max_atomic_number=17,
    )
    
    dcmnet = MessagePassingModel(
        features=config['dcmnet_features'],
        max_degree=2,
        num_iterations=config['dcmnet_iterations'],
        num_basis_functions=config['dcmnet_basis'],
        cutoff=config['dcmnet_cutoff'],
        max_atomic_number=17,
        n_dcm=config['n_dcm'],
    )
    
    # Load data
    esp_data = np.load(valid_esp_path)
    efd_data = np.load(valid_efd_path)
    
    return params, physnet, dcmnet, esp_data, efd_data, config


def analyze_charges(params, physnet, dcmnet, esp_data, efd_data, n_dcm, n_samples=10):
    """Analyze charge predictions for sample molecules."""
    
    print(f"\n{'='*70}")
    print("CHARGE ANALYSIS")
    print(f"{'='*70}\n")
    
    for idx in range(min(n_samples, len(esp_data['R']))):
        print(f"\n--- Sample {idx} ---")
        
        # Get molecule data
        R = esp_data['R'][idx]
        Z = esp_data['Z'][idx]
        N = int(esp_data['N'][idx])
        
        # Get only real atoms
        R_real = R[:N]
        Z_real = Z[:N]
        
        print(f"Molecule: {N} atoms")
        print(f"Atomic numbers: {Z_real}")
        
        # Create minimal batch for model
        from trainer import prepare_batch_data
        
        # Prepare single molecule as batch
        R_flat = R_real.reshape(-1, 3)
        Z_flat = Z_real.reshape(-1)
        batch_segments = np.zeros(N, dtype=np.int32)
        
        # Simple edge list (all pairs within cutoff)
        cutoff = 10.0
        dst_idx = []
        src_idx = []
        for i in range(N):
            for j in range(N):
                if i != j:
                    dist = np.linalg.norm(R_real[i] - R_real[j])
                    if dist < cutoff:
                        dst_idx.append(i)
                        src_idx.append(j)
        
        dst_idx = np.array(dst_idx, dtype=np.int32)
        src_idx = np.array(src_idx, dtype=np.int32)
        batch_mask = np.ones(len(dst_idx), dtype=np.float32)
        atom_mask = np.ones(N, dtype=np.float32)
        
        # PhysNet forward pass
        physnet_out = physnet.apply(
            params['physnet'],
            atomic_numbers=jnp.array(Z_flat),
            positions=jnp.array(R_flat),
            dst_idx=jnp.array(dst_idx),
            src_idx=jnp.array(src_idx),
            batch_segments=jnp.array(batch_segments),
            batch_size=1,
            batch_mask=jnp.array(batch_mask),
            atom_mask=jnp.array(atom_mask),
        )
        
        charges_physnet = np.array(physnet_out['charges'].squeeze())
        
        print(f"\nPhysNet Atomic Charges:")
        for i, (z, q) in enumerate(zip(Z_real, charges_physnet)):
            print(f"  Atom {i} (Z={z}): {q:+.6f} e")
        print(f"  Sum: {charges_physnet.sum():.6f} e")
        print(f"  Range: [{charges_physnet.min():.6f}, {charges_physnet.max():.6f}]")
        
        # DCMNet forward pass
        mono_dist, dipo_dist = dcmnet.apply(
            params['dcmnet'],
            atomic_numbers=jnp.array(Z_flat),
            positions=jnp.array(R_flat),
            dst_idx=jnp.array(dst_idx),
            src_idx=jnp.array(src_idx),
            batch_segments=jnp.array(batch_segments),
            batch_size=1,
        )
        
        mono_dist = np.array(mono_dist)  # (N, n_dcm)
        dipo_dist = np.array(dipo_dist)  # (N, n_dcm, 3)
        
        print(f"\nDCMNet Distributed Charges:")
        for i, (z, mono_atom) in enumerate(zip(Z_real, mono_dist)):
            print(f"  Atom {i} (Z={z}):")
            print(f"    Charges: {mono_atom}")
            print(f"    Sum: {mono_atom.sum():+.6f} e (target: {charges_physnet[i]:+.6f} e)")
            print(f"    Range: [{mono_atom.min():.6f}, {mono_atom.max():.6f}]")
        
        all_dcm_charges = mono_dist.flatten()
        print(f"\n  All DCMNet charges:")
        print(f"    Total sum: {all_dcm_charges.sum():.6f} e")
        print(f"    Range: [{all_dcm_charges.min():.6f}, {all_dcm_charges.max():.6f}]")
        print(f"    Mean: {all_dcm_charges.mean():.6f} e")
        print(f"    # Positive: {(all_dcm_charges > 0).sum()}/{len(all_dcm_charges)}")
        print(f"    # Negative: {(all_dcm_charges < 0).sum()}/{len(all_dcm_charges)}")
        
        # Compute ESP from both
        from mmml.dcmnet.dcmnet.electrostatics import calc_esp
        
        # Get ESP grid
        vdw_surface = esp_data['vdw_surface'][idx]
        esp_target = esp_data['esp'][idx]
        
        # DCMNet ESP
        mono_flat = mono_dist.reshape(-1)
        dipo_flat = np.moveaxis(dipo_dist, -1, -2).reshape(-1, 3)
        esp_dcmnet = np.array(calc_esp(jnp.array(dipo_flat), jnp.array(mono_flat), jnp.array(vdw_surface)))
        
        # PhysNet ESP
        distances = np.linalg.norm(vdw_surface[:, None, :] - R_real[None, :, :], axis=2)
        esp_physnet = np.sum(charges_physnet[None, :] / (distances + 1e-10), axis=1)
        
        print(f"\nESP Predictions:")
        print(f"  DCMNet ESP range: [{esp_dcmnet.min():.6f}, {esp_dcmnet.max():.6f}] Ha/e")
        print(f"  DCMNet ESP mean: {esp_dcmnet.mean():.6f} Ha/e")
        print(f"  PhysNet ESP range: [{esp_physnet.min():.6f}, {esp_physnet.max():.6f}] Ha/e")
        print(f"  PhysNet ESP mean: {esp_physnet.mean():.6f} Ha/e")
        print(f"  Target ESP range: [{esp_target.min():.6f}, {esp_target.max():.6f}] Ha/e")
        print(f"  Target ESP mean: {esp_target.mean():.6f} Ha/e")
        
        if idx >= 2:  # Just show first 3 molecules
            break


def main():
    parser = argparse.ArgumentParser(description='Diagnose charge predictions')
    parser.add_argument('--checkpoint', type=Path, required=True,
                       help='Path to checkpoint (best_params.pkl or directory)')
    parser.add_argument('--valid-esp', type=Path, 
                       default='../dcmnet_train/grids_esp_valid.npz',
                       help='Validation ESP file')
    parser.add_argument('--valid-efd', type=Path,
                       default='../physnet_train_charges/energies_forces_dipoles_valid.npz',
                       help='Validation EFD file')
    parser.add_argument('--n-samples', type=int, default=5,
                       help='Number of samples to analyze')
    
    args = parser.parse_args()
    
    print(f"Loading model from: {args.checkpoint}")
    params, physnet, dcmnet, esp_data, efd_data, config = load_model_and_data(
        args.checkpoint, args.valid_esp, args.valid_efd
    )
    
    analyze_charges(params, physnet, dcmnet, esp_data, efd_data, config['n_dcm'], args.n_samples)
    
    print(f"\n{'='*70}")
    print("DIAGNOSIS COMPLETE")
    print(f"{'='*70}")
    print("\nKey Questions:")
    print("1. Are PhysNet charges both positive and negative?")
    print("2. Are DCMNet distributed charges both positive and negative?")
    print("3. Do DCMNet charge sums match PhysNet atomic charges?")
    print("4. Why is DCMNet ESP only positive if charges have both signs?")


if __name__ == '__main__':
    main()

