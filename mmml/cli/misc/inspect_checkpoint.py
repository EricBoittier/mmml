#!/usr/bin/env python3
"""
Inspect a checkpoint and infer model configuration from parameter shapes.

This tool analyzes checkpoint files to:
- Count total parameters
- Show parameter structure
- Infer model configuration (features, iterations, etc.)
- Identify model type (PhysNet, DCMNet, etc.)

Usage:
    python -m mmml.cli.inspect_checkpoint --checkpoint path/to/best_params.pkl
    python -m mmml.cli.inspect_checkpoint --checkpoint checkpoints/my_model/
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    print("❌ Error: JAX not installed")
    print("Install with: pip install jax")
    sys.exit(1)


def find_checkpoint_file(path: Path) -> Path:
    """Find checkpoint file in directory or return path if it's a file."""
    if path.is_file():
        return path
    
    # Look for common checkpoint names
    candidates = [
        path / 'best_params.pkl',
        path / 'final_params.pkl',
        path / 'checkpoint.pkl',
        path / 'params.pkl',
    ]
    
    for cand in candidates:
        if cand.exists():
            return cand
    
    raise FileNotFoundError(f"No checkpoint file found in {path}")


def inspect_checkpoint(checkpoint_path: Path, verbose: bool = True):
    """Inspect checkpoint and print parameter structure."""
    
    if verbose:
        print("="*70)
        print("CHECKPOINT INSPECTOR")
        print("="*70)
        print(f"\n📂 Loading: {checkpoint_path}")
    
    with open(checkpoint_path, 'rb') as f:
        checkpoint_data = pickle.load(f)
    
    # Extract params
    if isinstance(checkpoint_data, dict) and 'params' in checkpoint_data:
        params = checkpoint_data['params']
    else:
        params = checkpoint_data
    
    if verbose:
        print("✅ Parameters loaded")
    
    # Flatten and inspect
    flat_params = jax.tree_util.tree_flatten_with_path(params)[0]
    
    total_params = sum(x[1].size for x in flat_params)
    if verbose:
        print(f"\n📊 Total parameters: {total_params:,} ({total_params/1e6:.2f} Million)")
    
    # Categorize parameters
    categorized = categorize_parameters(flat_params)
    
    if verbose:
        print("\n" + "="*70)
        print("PARAMETER STRUCTURE:")
        print("="*70)
        
        for category, params_list in categorized.items():
            if params_list:
                category_total = sum(size for _, _, size in params_list)
                print(f"\n{category} Parameters: {category_total:,}")
                print("-" * 70)
                for path, shape, size in params_list[:20]:  # Limit to first 20
                    print(f"  {path:<60} {str(shape):<25} ({size:,})")
                if len(params_list) > 20:
                    print(f"  ... and {len(params_list) - 20} more")
    
    # Infer configuration
    config = infer_configuration(flat_params, categorized, verbose=verbose)
    
    return config


def categorize_parameters(flat_params: List[Tuple]) -> Dict[str, List]:
    """Categorize parameters by model component."""
    categorized = {
        'PhysNet': [],
        'DCMNet': [],
        'NonEquivariant': [],
        'Other': [],
    }
    
    for path, value in flat_params:
        path_str = '/'.join(str(k.key) for k in path)
        item = (path_str, value.shape, value.size)
        
        if 'physnet' in path_str.lower():
            categorized['PhysNet'].append(item)
        elif 'dcmnet' in path_str.lower():
            categorized['DCMNet'].append(item)
        elif 'noneq' in path_str.lower():
            categorized['NonEquivariant'].append(item)
        else:
            categorized['Other'].append(item)
    
    return categorized


def infer_configuration(flat_params: List[Tuple], categorized: Dict, verbose: bool = True) -> Dict[str, Any]:
    """Infer model configuration from parameter structure."""
    
    config = {
        'model_type': None,
        'physnet_config': {},
        'dcmnet_config': {},
        'noneq_config': {},
    }
    
    if verbose:
        print("\n" + "="*70)
        print("INFERRED CONFIGURATION:")
        print("="*70)
    
    # Determine model type
    if categorized['DCMNet']:
        config['model_type'] = 'dcmnet'
        if verbose:
            print("\n🔍 Model Type: JointPhysNetDCMNet (Equivariant)")
    elif categorized['NonEquivariant']:
        config['model_type'] = 'noneq'
        if verbose:
            print("\n🔍 Model Type: JointPhysNetNonEquivariant")
    elif categorized['PhysNet']:
        config['model_type'] = 'physnet'
        if verbose:
            print("\n🔍 Model Type: PhysNet")
    else:
        config['model_type'] = 'unknown'
        if verbose:
            print("\n🔍 Model Type: Unknown")
    
    # Infer PhysNet config
    physnet_params = categorized['PhysNet'] + categorized['Other']
    
    for path, shape, size in physnet_params:
        # Embedding layer → max_atomic_number and features
        if 'Embed' in path and 'embedding' in path.lower():
            config['physnet_config']['max_atomic_number'] = shape[0] - 1
            config['physnet_config']['features'] = shape[3] if len(shape) > 3 else shape[1]
        
        # Radial basis → num_basis_functions
        if 'rbf' in path.lower() or 'basis' in path.lower():
            if 'num_basis_functions' not in config['physnet_config']:
                # Try to infer from shape
                pass
    
    # Count MessagePass iterations
    mp_indices = set()
    for path, shape, size in physnet_params:
        if 'MessagePass_' in path:
            try:
                idx = int(path.split('MessagePass_')[1].split('/')[0])
                mp_indices.add(idx)
            except:
                pass
    
    if mp_indices:
        config['physnet_config']['num_iterations'] = max(mp_indices) + 1
    
    # Look for cutoff in saved config
    for path, shape, size in physnet_params:
        if 'cutoff' in path.lower():
            # Might be stored as parameter
            pass
    
    # Infer DCMNet config if present
    if categorized['DCMNet']:
        dcmnet_params = categorized['DCMNet']
        
        for path, shape, size in dcmnet_params:
            if 'Embed' in path and 'embedding' in path.lower():
                config['dcmnet_config']['max_atomic_number'] = shape[0] - 1
                config['dcmnet_config']['features'] = shape[1] if len(shape) > 1 else 32
            
            if 'TensorDense' in path and 'kernel' in path:
                if len(shape) >= 2:
                    config['dcmnet_config']['n_dcm'] = shape[1]
        
        # Count DCMNet iterations
        dcm_mp_indices = set()
        for path, shape, size in dcmnet_params:
            if 'MessagePass_' in path:
                try:
                    idx = int(path.split('MessagePass_')[1].split('/')[0])
                    dcm_mp_indices.add(idx)
                except:
                    pass
        
        if dcm_mp_indices:
            config['dcmnet_config']['num_iterations'] = max(dcm_mp_indices) + 1
    
    # Print inferred config
    if verbose:
        if config['physnet_config']:
            print("\nPhysNet Configuration:")
            for key, value in config['physnet_config'].items():
                print(f"  {key}: {value}")
        
        if config['dcmnet_config']:
            print("\nDCMNet Configuration:")
            for key, value in config['dcmnet_config'].items():
                print(f"  {key}: {value}")
        
        if config['noneq_config']:
            print("\nNonEquivariant Configuration:")
            for key, value in config['noneq_config'].items():
                print(f"  {key}: {value}")
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Inspect checkpoint and infer model configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inspect checkpoint file
  python -m mmml.cli.inspect_checkpoint --checkpoint best_params.pkl
  
  # Inspect checkpoint directory (finds best_params.pkl automatically)
  python -m mmml.cli.inspect_checkpoint --checkpoint checkpoints/my_model/
  
  # Save configuration to JSON
  python -m mmml.cli.inspect_checkpoint --checkpoint model/best_params.pkl --save-config config.json
        """
    )
    
    parser.add_argument('--checkpoint', type=Path, required=True,
                       help='Path to checkpoint file or directory')
    parser.add_argument('--save-config', type=Path,
                       help='Save inferred configuration to JSON file')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress detailed output')
    
    args = parser.parse_args()
    
    # Find checkpoint
    try:
        checkpoint_file = find_checkpoint_file(args.checkpoint)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return 1
    
    # Inspect
    config = inspect_checkpoint(checkpoint_file, verbose=not args.quiet)
    
    # Save config if requested
    if args.save_config:
        import json
        with open(args.save_config, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"\n✅ Configuration saved to: {args.save_config}")
    
    if not args.quiet:
        print("\n" + "="*70)
        print("✅ Inspection complete!")
        print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

