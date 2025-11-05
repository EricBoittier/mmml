#!/usr/bin/env python3
"""
Simple example of training PhysNet with packed memmap data.

This is a minimal working example showing the basic usage of PackedMemmapLoader
with PhysNet training.
"""

import jax
import jax.numpy as jnp
from mmml.data.packed_memmap_loader import PackedMemmapLoader, split_loader
from mmml.physnetjax.physnetjax.models.model import EF
from mmml.physnetjax.physnetjax.training.trainstep import train_step
from mmml.physnetjax.physnetjax.training.evalstep import eval_step
from mmml.physnetjax.physnetjax.training.optimizer import get_optimizer
import e3x


def main():
    # Configuration
    DATA_PATH = "openqdc_packed_memmap"  # Path to your packed memmap data
    BATCH_SIZE = 32
    NUM_ATOMS = 60
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    
    print("=" * 80)
    print("Simple PhysNet Training Example")
    print("=" * 80)
    
    # 1. Create data loader
    print("\n1. Loading data...")
    loader = PackedMemmapLoader(
        path=DATA_PATH,
        batch_size=BATCH_SIZE,
        shuffle=True,
        bucket_size=8192,
        seed=42,
    )
    print(f"   Loaded {loader.N} molecules")
    
    # 2. Split into train/validation
    print("\n2. Splitting data...")
    train_loader, valid_loader = split_loader(loader, train_fraction=0.9, seed=42)
    print(f"   Training: {train_loader.N} molecules")
    print(f"   Validation: {valid_loader.N} molecules")
    
    # 3. Create model
    print("\n3. Creating model...")
    model = EF(
        features=128,
        max_degree=2,
        num_iterations=3,
        num_basis_functions=16,
        cutoff=5.0,
        max_atomic_number=118,
        charges=False,
        natoms=NUM_ATOMS,
        n_res=3,
        zbl=True,
    )
    print("   Model created")
    
    # 4. Initialize parameters
    print("\n4. Initializing model parameters...")
    key = jax.random.PRNGKey(42)
    key, init_key = jax.random.split(key)
    
    # Get a sample batch to initialize
    sample_batch = next(train_loader.batches(num_atoms=NUM_ATOMS))
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(NUM_ATOMS)
    
    params = model.init(
        init_key,
        atomic_numbers=sample_batch["Z"][0],
        positions=sample_batch["R"][0],
        dst_idx=dst_idx,
        src_idx=src_idx,
    )
    
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"   Initialized {n_params:,} parameters")
    
    # 5. Create optimizer
    print("\n5. Setting up optimizer...")
    optimizer, transform, schedule_fn, _ = get_optimizer(
        learning_rate=LEARNING_RATE,
        schedule_fn=None,
        optimizer=None,
        transform=None,
    )
    
    ema_params = params
    opt_state = optimizer.init(params)
    transform_state = transform.init(params)
    print("   Optimizer ready")
    
    # 6. Training loop
    print("\n6. Starting training...")
    print("=" * 80)
    
    energy_weight = 1.0
    forces_weight = 52.91  # kcal/mol conversion
    
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print("-" * 80)
        
        # Train
        train_loss = 0.0
        for i, batch in enumerate(train_loader.batches(num_atoms=NUM_ATOMS)):
            batch_size = int(batch["Z"].shape[0])
            
            (
                params,
                ema_params,
                opt_state,
                transform_state,
                loss,
                energy_mae,
                forces_mae,
                _,
            ) = train_step(
                model_apply=model.apply,
                optimizer_update=optimizer.update,
                transform_state=transform_state,
                batch=batch,
                batch_size=batch_size,
                energy_weight=energy_weight,
                forces_weight=forces_weight,
                dipole_weight=0.0,
                charges_weight=0.0,
                opt_state=opt_state,
                doCharges=False,
                params=params,
                ema_params=ema_params,
                debug=False,
            )
            
            train_loss += (loss - train_loss) / (i + 1)
        
        # Validate
        valid_loss = 0.0
        valid_energy_mae = 0.0
        valid_forces_mae = 0.0
        
        for i, batch in enumerate(valid_loader.batches(num_atoms=NUM_ATOMS)):
            batch_size = int(batch["Z"].shape[0])
            
            loss, energy_mae, forces_mae, _ = eval_step(
                model_apply=model.apply,
                batch=batch,
                batch_size=batch_size,
                energy_weight=energy_weight,
                forces_weight=forces_weight,
                dipole_weight=0.0,
                charges_weight=0.0,
                charges=False,
                params=ema_params,
            )
            
            valid_loss += (loss - valid_loss) / (i + 1)
            valid_energy_mae += (energy_mae - valid_energy_mae) / (i + 1)
            valid_forces_mae += (forces_mae - valid_forces_mae) / (i + 1)
        
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Valid Loss: {valid_loss:.6f}")
        print(f"Valid Energy MAE: {valid_energy_mae:.6f} kcal/mol")
        print(f"Valid Forces MAE: {valid_forces_mae:.6f} kcal/mol/Å")
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()

