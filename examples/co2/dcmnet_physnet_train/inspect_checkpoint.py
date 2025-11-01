#!/usr/bin/env python3
"""
Inspect a checkpoint and infer model configuration from parameter shapes.

Usage:
    python inspect_checkpoint.py --checkpoint path/to/best_params.pkl
"""

import argparse
import pickle
from pathlib import Path
import jax

def inspect_checkpoint(checkpoint_path: Path):
    """Inspect checkpoint and print parameter structure."""
    
    print("="*70)
    print("Checkpoint Inspector")
    print("="*70)
    
    print(f"\n📂 Loading: {checkpoint_path}")
    with open(checkpoint_path, 'rb') as f:
        params = pickle.load(f)
    
    print("\n✅ Parameters loaded")
    
    # Flatten and inspect
    flat_params = jax.tree_util.tree_flatten_with_path(params)[0]
    
    print(f"\nTotal parameters: {sum(x[1].size for x in flat_params):,}")
    
    print("\n" + "="*70)
    print("Parameter Structure:")
    print("="*70)
    
    # Group by model component
    physnet_params = []
    dcmnet_params = []
    
    for path, value in flat_params:
        path_str = '/'.join(str(k.key) for k in path)
        
        if 'physnet' in path_str.lower():
            physnet_params.append((path_str, value.shape, value.size))
        elif 'dcmnet' in path_str.lower():
            dcmnet_params.append((path_str, value.shape, value.size))
        else:
            # Print all to categorize
            print(f"  UNCATEGORIZED: {path_str}: {value.shape}")
    
    print("\n" + "-"*70)
    print("PhysNet Parameters:")
    print("-"*70)
    total_physnet = 0
    for path, shape, size in physnet_params:
        print(f"  {path:<60} {str(shape):<25} ({size:,} params)")
        total_physnet += size
    print(f"\nTotal PhysNet parameters: {total_physnet:,}")
    
    print("\n" + "-"*70)
    print("DCMNet Parameters:")
    print("-"*70)
    total_dcmnet = 0
    for path, shape, size in dcmnet_params:
        print(f"  {path:<60} {str(shape):<25} ({size:,} params)")
        total_dcmnet += size
    print(f"\nTotal DCMNet parameters: {total_dcmnet:,}")
    
    # Try to infer configuration
    print("\n" + "="*70)
    print("Inferred Configuration:")
    print("="*70)
    
    # Look for embedding layer to get max_atomic_number
    for path, shape, size in physnet_params:
        if 'Embed' in path and 'embedding' in path.lower():
            max_atomic_num = shape[0] - 1
            features = shape[3] if len(shape) > 3 else shape[1]
            print(f"\nPhysNet:")
            print(f"  max_atomic_number: {max_atomic_num}")
            print(f"  features: {features}")
            break
    
    # Count MessagePass iterations - find the highest numbered one
    physnet_mp_indices = set()
    for p, s, _ in physnet_params:
        if 'MessagePass_' in p:
            try:
                idx = int(p.split('MessagePass_')[1].split('/')[0])
                physnet_mp_indices.add(idx)
            except:
                pass
    
    if physnet_mp_indices:
        max_mp = max(physnet_mp_indices)
        num_iterations = max_mp + 1  # 0-indexed
        print(f"  num_iterations: {num_iterations} (MessagePass layers: {sorted(physnet_mp_indices)})")
    else:
        print(f"  num_iterations: Could not determine (no MessagePass layers found)")
    
    # Look for n_res (residual blocks)
    residual_blocks = sum(1 for p, s, _ in physnet_params if 'residual' in p.lower())
    
    # Look for DCMNet config
    for path, shape, size in dcmnet_params:
        if 'Embed' in path and 'embedding' in path.lower():
            max_atomic_num_dcm = shape[0] - 1
            features_dcm = shape[1] if len(shape) > 1 else 32
            print(f"\nDCMNet:")
            print(f"  max_atomic_number: {max_atomic_num_dcm}")
            print(f"  features: {features_dcm}")
            break
    
    # Count DCMNet MessagePass iterations - find the highest numbered one
    dcmnet_mp_indices = set()
    for p, s, _ in dcmnet_params:
        if 'MessagePass_' in p:
            try:
                idx = int(p.split('MessagePass_')[1].split('/')[0])
                dcmnet_mp_indices.add(idx)
            except:
                pass
    
    if dcmnet_mp_indices:
        max_mp_dcm = max(dcmnet_mp_indices)
        num_iterations_dcm = max_mp_dcm + 1
        print(f"  num_iterations: {num_iterations_dcm} (MessagePass layers: {sorted(dcmnet_mp_indices)})")
    else:
        print(f"  num_iterations: Could not determine (no MessagePass layers found)")
    
    # Look for TensorDense to find n_dcm
    for path, shape, size in dcmnet_params:
        if 'TensorDense' in path and 'kernel' in path:
            # Shape should reveal n_dcm
            if len(shape) >= 2:
                n_dcm = shape[1]
                print(f"  n_dcm: {n_dcm}")
                break
    
    print("\n" + "="*70)
    print("\nTo create model_config.pkl with these settings, run:")
    print("\npython save_config_from_checkpoint.py \\")
    print(f"  --checkpoint-dir {checkpoint_path.parent} \\")
    if 'max_atomic_num' in locals():
        print(f"  --max-atomic-number {max_atomic_num} \\")
    if 'features' in locals():
        print(f"  --physnet-features {features} \\")
    if 'num_iterations' in locals():
        print(f"  --physnet-iterations {num_iterations} \\")
    if 'features_dcm' in locals():
        print(f"  --dcmnet-features {features_dcm} \\")
    if 'num_iterations_dcm' in locals():
        print(f"  --dcmnet-iterations {num_iterations_dcm} \\")
    if 'n_dcm' in locals():
        print(f"  --n-dcm {n_dcm}")
    
    print("\n⚠️  Note: If the above inferred values look wrong, check the parameter tree above.")
    print("   The model architecture must EXACTLY match what was used during training.")
    print()

def main():
    parser = argparse.ArgumentParser(
        description="Inspect checkpoint and infer model configuration"
    )
    parser.add_argument('--checkpoint', type=Path, required=True,
                       help='Path to checkpoint file (best_params.pkl)')
    
    args = parser.parse_args()
    
    if not args.checkpoint.exists():
        print(f"❌ Error: Checkpoint not found: {args.checkpoint}")
        return
    
    inspect_checkpoint(args.checkpoint)

if __name__ == "__main__":
    main()

