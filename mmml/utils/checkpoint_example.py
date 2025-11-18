"""
Example usage of the model checkpoint utilities.

This script demonstrates how to save and load model parameters and configurations.
"""

from pathlib import Path
import jax
import jax.numpy as jnp
from mmml.utils.model_checkpoint import (
    save_model_checkpoint,
    load_model_checkpoint,
    create_model_from_checkpoint,
    quick_save,
    quick_load,
)

# Example 1: Save a model checkpoint
def example_save():
    """Example of saving a model checkpoint."""
    from mmml.physnetjax.physnetjax.models.model import EF
    
    # Create a model
    model = EF(
        features=64,
        max_degree=2,
        num_iterations=2,
        cutoff=8.0,
        max_atomic_number=28,
    )
    
    # Initialize parameters (in real usage, these come from training)
    key = jax.random.PRNGKey(42)
    Z = jnp.array([6, 1, 1, 1, 8])  # Example: CH3OH
    R = jnp.array([[0.0, 0.0, 0.0],
                   [1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0],
                   [0.0, 0.0, 1.0],
                   [1.5, 0.0, 0.0]])
    
    params = model.init(key, R, Z)
    
    # Save checkpoint
    save_dir = Path("example_checkpoint")
    saved_paths = save_model_checkpoint(
        params=params,
        model=model,
        save_dir=save_dir,
        metadata={
            'epoch': 100,
            'train_loss': 0.001,
            'val_loss': 0.002,
            'notes': 'Example checkpoint'
        }
    )
    
    print(f"Saved checkpoint to: {save_dir}")
    return save_dir


# Example 2: Load a model checkpoint
def example_load(checkpoint_dir: Path):
    """Example of loading a model checkpoint."""
    checkpoint = load_model_checkpoint(checkpoint_dir)
    
    print("\nLoaded checkpoint:")
    print(f"  Config keys: {list(checkpoint['config'].keys())}")
    if 'metadata' in checkpoint:
        print(f"  Metadata: {checkpoint['metadata']}")
    
    return checkpoint


# Example 3: Create model from checkpoint
def example_create_model(checkpoint_dir: Path):
    """Example of creating a model instance from checkpoint."""
    from mmml.physnetjax.physnetjax.models.model import EF
    
    model, params, config = create_model_from_checkpoint(
        checkpoint_dir=checkpoint_dir,
        model_class=EF,
    )
    
    print(f"\nCreated model with config: {config}")
    print(f"Model features: {model.features}")
    print(f"Model cutoff: {model.cutoff}")
    
    return model, params, config


# Example 4: Quick save/load
def example_quick_save_load():
    """Example using quick save/load functions."""
    from mmml.physnetjax.physnetjax.models.model import EF
    
    model = EF(features=32, cutoff=6.0)
    key = jax.random.PRNGKey(0)
    Z = jnp.array([6, 1, 1, 1])
    R = jnp.array([[0.0, 0.0, 0.0],
                   [1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0],
                   [0.0, 0.0, 1.0]])
    params = model.init(key, R, Z)
    
    # Quick save
    save_path = Path("quick_checkpoint")
    quick_save(params, model, save_path)
    
    # Quick load
    checkpoint = quick_load(save_path)
    print(f"\nQuick loaded checkpoint with {len(checkpoint)} items")
    
    return checkpoint


if __name__ == "__main__":
    print("=" * 60)
    print("Model Checkpoint Utility Examples")
    print("=" * 60)
    
    # Run examples
    print("\n1. Saving checkpoint...")
    checkpoint_dir = example_save()
    
    print("\n2. Loading checkpoint...")
    checkpoint = example_load(checkpoint_dir)
    
    print("\n3. Creating model from checkpoint...")
    model, params, config = example_create_model(checkpoint_dir)
    
    print("\n4. Quick save/load...")
    quick_checkpoint = example_quick_save_load()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)

