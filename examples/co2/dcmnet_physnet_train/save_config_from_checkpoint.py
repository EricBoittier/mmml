#!/usr/bin/env python3
"""
Helper script to create model_config.pkl for old checkpoints.

If you have a checkpoint from before model_config.pkl was added,
you can use this script to save the configuration manually.

Usage:
    python save_config_from_checkpoint.py \
      --checkpoint-dir /path/to/checkpoint/dir \
      --natoms 60 \
      --max-atomic-number 28 \
      --physnet-features 64 \
      --physnet-iterations 5 \
      --physnet-n-res 3 \
      --dcmnet-features 32 \
      --n-dcm 3
"""

import argparse
import pickle
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Save model_config.pkl for existing checkpoint"
    )
    
    parser.add_argument('--checkpoint-dir', type=Path, required=True,
                       help='Checkpoint directory containing best_params.pkl')
    
    # PhysNet config
    parser.add_argument('--natoms', type=int, default=60)
    parser.add_argument('--max-atomic-number', type=int, default=28)
    parser.add_argument('--physnet-features', type=int, default=64)
    parser.add_argument('--physnet-iterations', type=int, default=5)
    parser.add_argument('--physnet-basis', type=int, default=64)
    parser.add_argument('--physnet-cutoff', type=float, default=6.0)
    parser.add_argument('--physnet-n-res', type=int, default=3)
    
    # DCMNet config
    parser.add_argument('--dcmnet-features', type=int, default=32)
    parser.add_argument('--dcmnet-iterations', type=int, default=2)
    parser.add_argument('--dcmnet-basis', type=int, default=32)
    parser.add_argument('--dcmnet-cutoff', type=float, default=10.0)
    parser.add_argument('--max-degree', type=int, default=2)
    parser.add_argument('--n-dcm', type=int, default=3)
    
    args = parser.parse_args()
    
    # Check checkpoint exists
    if not (args.checkpoint_dir / 'best_params.pkl').exists():
        print(f"❌ Error: best_params.pkl not found in {args.checkpoint_dir}")
        return
    
    # Create config
    model_config = {
        'physnet_config': {
            'features': args.physnet_features,
            'max_degree': 0,
            'num_iterations': args.physnet_iterations,
            'num_basis_functions': args.physnet_basis,
            'cutoff': args.physnet_cutoff,
            'max_atomic_number': args.max_atomic_number,
            'charges': True,
            'natoms': args.natoms,
            'total_charge': 0.0,
            'n_res': args.physnet_n_res,
            'zbl': False,
            'use_energy_bias': True,
            'debug': False,
            'efa': False,
        },
        'dcmnet_config': {
            'features': args.dcmnet_features,
            'max_degree': args.max_degree,
            'num_iterations': args.dcmnet_iterations,
            'num_basis_functions': args.dcmnet_basis,
            'cutoff': args.dcmnet_cutoff,
            'max_atomic_number': args.max_atomic_number,
            'n_dcm': args.n_dcm,
            'include_pseudotensors': False,
        }
    }
    
    # Save config
    config_path = args.checkpoint_dir / 'model_config.pkl'
    with open(config_path, 'wb') as f:
        pickle.dump(model_config, f)
    
    print(f"✅ Saved model config to: {config_path}")
    print(f"\nConfig:")
    print(f"  PhysNet: {args.physnet_features} features, {args.physnet_iterations} iterations, {args.physnet_n_res} residual blocks")
    print(f"  DCMNet: {args.dcmnet_features} features, {args.dcmnet_iterations} iterations, {args.n_dcm} DCM/atom")
    print(f"  Max atomic number: {args.max_atomic_number}")
    print(f"  Natoms: {args.natoms}")
    print(f"\nNow you can use ase_calculator.py without specifying these parameters!")

if __name__ == "__main__":
    main()

