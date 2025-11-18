#!/usr/bin/env python3
"""
Standalone script to save model parameters and configuration.

Usage:
    python scripts/save_model_checkpoint.py --checkpoint-dir <path> --output-dir <path>
    python scripts/save_model_checkpoint.py --checkpoint-dir checkpoints/epoch-100 --output-dir saved_model
"""

import argparse
from pathlib import Path
import sys

# Add parent directory to path to import mmml
sys.path.insert(0, str(Path(__file__).parent.parent))

from mmml.utils.model_checkpoint import save_model_checkpoint, load_model_checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="Save model parameters and config from existing checkpoint"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Directory containing existing checkpoint (e.g., checkpoints/epoch-100)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory to save params.pkl and model_config.json"
    )
    parser.add_argument(
        "--use-orbax",
        action="store_true",
        help="Use orbax format for parameters (default: pickle)"
    )
    
    args = parser.parse_args()
    
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    
    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)
    
    # Load existing checkpoint
    print(f"Loading checkpoint from: {checkpoint_dir}")
    checkpoint = load_model_checkpoint(
        checkpoint_dir,
        load_params=True,
        load_config=True,
        load_metadata=True,
        use_orbax=args.use_orbax
    )
    
    # Extract params and config
    params = checkpoint.get('params')
    config = checkpoint.get('config', {})
    metadata = checkpoint.get('metadata', {})
    
    if params is None:
        print("Error: No parameters found in checkpoint")
        sys.exit(1)
    
    # Create a dummy model object for config extraction
    # (In practice, you might want to reconstruct the actual model)
    class DummyModel:
        def __init__(self, config_dict):
            self.__dict__.update(config_dict)
    
    dummy_model = DummyModel(config)
    
    # Save in the new format
    print(f"\nSaving to: {output_dir}")
    saved_paths = save_model_checkpoint(
        params=params,
        model=dummy_model,
        save_dir=output_dir,
        config=config,
        metadata=metadata,
        use_orbax=args.use_orbax
    )
    
    print("\n✓ Checkpoint saved successfully!")
    print(f"\nTo load this checkpoint:")
    print(f"  from mmml.utils.model_checkpoint import load_model_checkpoint")
    print(f"  checkpoint = load_model_checkpoint('{output_dir}')")
    print(f"  params = checkpoint['params']")
    print(f"  config = checkpoint['config']")


if __name__ == "__main__":
    main()

