#!/usr/bin/env python3
"""
PhysNet Training CLI Tool with Charge and Spin Conditioning

This CLI tool trains a PhysNet model that accepts total molecular charge and
total spin multiplicity as inputs, enabling multi-state predictions.

Usage:
    python -m mmml.cli.train_charge_spin \
        --data_path openqdc_packed_memmap \
        --batch_size 32 \
        --num_epochs 100 \
        --learning_rate 0.001
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict

import e3x
import jax
import jax.numpy as jnp
import numpy as np
from flax.training import orbax_utils, train_state

from mmml.physnetjax.physnetjax.models.model_charge_spin import EF_ChargeSpinConditioned
from mmml.physnetjax.physnetjax.training.optimizer import get_optimizer
from mmml.physnetjax.physnetjax.restart.restart import orbax_checkpointer
from mmml.physnetjax.physnetjax.directories import BASE_CKPT_DIR
from mmml.data.packed_memmap_loader import PackedMemmapLoader, split_loader


def train_step_charge_spin(
    model_apply,
    optimizer_update,
    batch: Dict,
    params,
    opt_state,
    ema_params,
    energy_weight: float = 1.0,
    forces_weight: float = 52.91,
):
    """
    Single training step for charge-spin conditioned model.
    
    Parameters
    ----------
    model_apply : callable
        Model apply function
    optimizer_update : callable
        Optimizer update function
    batch : dict
        Batch dictionary with molecular data
    params : Any
        Model parameters
    opt_state : Any
        Optimizer state
    ema_params : Any
        EMA parameters
    energy_weight : float
        Weight for energy loss
    forces_weight : float
        Weight for forces loss
        
    Returns
    -------
    tuple
        Updated (params, opt_state, ema_params, loss, energy_mae, forces_mae)
    """
    def loss_fn(params):
        outputs = model_apply(
            params,
            atomic_numbers=batch["Z"],
            positions=batch["R"],
            dst_idx=batch["dst_idx"],
            src_idx=batch["src_idx"],
            total_charges=batch["total_charge"],
            total_spins=batch["total_spin"],
            batch_segments=batch["batch_segments"],
            batch_size=int(batch["Z"].shape[0]),
            batch_mask=jnp.ones_like(batch["dst_idx"]),
            atom_mask=(batch["Z"] > 0).astype(jnp.float32),
        )
        
        # Energy loss
        energy_pred = outputs["energy"]
        energy_true = batch["E"]
        energy_loss = jnp.mean((energy_pred - energy_true) ** 2)
        energy_mae = jnp.mean(jnp.abs(energy_pred - energy_true))
        
        # Forces loss
        forces_pred = outputs["forces"]
        forces_true = batch["F"]
        mask = (batch["Z"] > 0).astype(jnp.float32)
        forces_diff = (forces_pred - forces_true) * mask[..., None]
        forces_loss = jnp.mean(forces_diff ** 2)
        forces_mae = jnp.mean(jnp.abs(forces_diff))
        
        # Combined loss
        total_loss = energy_weight * energy_loss + forces_weight * forces_loss
        
        return total_loss, (energy_mae, forces_mae)
    
    # Compute gradients
    (loss, (energy_mae, forces_mae)), grads = jax.value_and_grad(
        loss_fn, has_aux=True
    )(params)
    
    # Update parameters
    updates, opt_state = optimizer_update(grads, opt_state, params)
    params = jax.tree_util.tree_map(lambda p, u: p + u, params, updates)
    
    # Update EMA
    ema_decay = 0.999
    ema_params = jax.tree_util.tree_map(
        lambda ema, p: ema * ema_decay + p * (1 - ema_decay),
        ema_params,
        params,
    )
    
    return params, opt_state, ema_params, loss, energy_mae, forces_mae


def eval_step_charge_spin(
    model_apply,
    batch: Dict,
    params,
    energy_weight: float = 1.0,
    forces_weight: float = 52.91,
):
    """Evaluation step for charge-spin conditioned model."""
    outputs = model_apply(
        params,
        atomic_numbers=batch["Z"],
        positions=batch["R"],
        dst_idx=batch["dst_idx"],
        src_idx=batch["src_idx"],
        total_charges=batch["total_charge"],
        total_spins=batch["total_spin"],
        batch_segments=batch["batch_segments"],
        batch_size=int(batch["Z"].shape[0]),
        batch_mask=jnp.ones_like(batch["dst_idx"]),
        atom_mask=(batch["Z"] > 0).astype(jnp.float32),
    )
    
    # Energy metrics
    energy_pred = outputs["energy"]
    energy_true = batch["E"]
    energy_loss = jnp.mean((energy_pred - energy_true) ** 2)
    energy_mae = jnp.mean(jnp.abs(energy_pred - energy_true))
    
    # Forces metrics
    forces_pred = outputs["forces"]
    forces_true = batch["F"]
    mask = (batch["Z"] > 0).astype(jnp.float32)
    forces_diff = (forces_pred - forces_true) * mask[..., None]
    forces_loss = jnp.mean(forces_diff ** 2)
    forces_mae = jnp.mean(jnp.abs(forces_diff))
    
    # Combined loss
    total_loss = energy_weight * energy_loss + forces_weight * forces_loss
    
    return total_loss, energy_mae, forces_mae


def main():
    parser = argparse.ArgumentParser(
        description="Train PhysNet with charge and spin conditioning"
    )
    
    # Data arguments
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to packed memmap data")
    parser.add_argument("--valid_split", type=float, default=0.1,
                       help="Fraction for validation")
    
    # Model arguments
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--max_degree", type=int, default=2)
    parser.add_argument("--num_iterations", type=int, default=3)
    parser.add_argument("--num_basis_functions", type=int, default=16)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--num_atoms", type=int, default=60)
    parser.add_argument("--n_res", type=int, default=3)
    parser.add_argument("--charge_embed_dim", type=int, default=16,
                       help="Dimension of charge embedding")
    parser.add_argument("--spin_embed_dim", type=int, default=16,
                       help="Dimension of spin embedding")
    parser.add_argument("--charge_min", type=int, default=-5,
                       help="Minimum charge to support")
    parser.add_argument("--charge_max", type=int, default=5,
                       help="Maximum charge to support")
    parser.add_argument("--spin_min", type=int, default=1,
                       help="Minimum spin multiplicity (1=singlet)")
    parser.add_argument("--spin_max", type=int, default=7,
                       help="Maximum spin multiplicity (7=septet)")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--energy_weight", type=float, default=1.0)
    parser.add_argument("--forces_weight", type=float, default=52.91)
    parser.add_argument("--bucket_size", type=int, default=8192)
    
    # Other
    parser.add_argument("--name", type=str, default="physnet_charge_spin")
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    
    args = parser.parse_args()
    
    print("="*80)
    print("PhysNet Training with Charge and Spin Conditioning")
    print("="*80)
    print(f"Data path: {args.data_path}")
    print(f"Charge range: [{args.charge_min}, {args.charge_max}]")
    print(f"Spin range: [{args.spin_min}, {args.spin_max}] (multiplicity)")
    print("="*80)
    
    # Initialize random key
    key = jax.random.PRNGKey(args.seed)
    
    # Create data loaders
    print("\nLoading data...")
    loader = PackedMemmapLoader(
        args.data_path,
        batch_size=args.batch_size,
        shuffle=True,
        bucket_size=args.bucket_size,
        seed=args.seed,
    )
    
    # Split data
    train_loader, valid_loader = split_loader(
        loader, train_fraction=1.0 - args.valid_split, seed=args.seed
    )
    
    print(f"Training molecules: {train_loader.N}")
    print(f"Validation molecules: {valid_loader.N}")
    
    # Create model
    print("\nInitializing model...")
    model = EF_ChargeSpinConditioned(
        features=args.features,
        max_degree=args.max_degree,
        num_iterations=args.num_iterations,
        num_basis_functions=args.num_basis_functions,
        cutoff=args.cutoff,
        max_atomic_number=118,
        charges=False,
        natoms=args.num_atoms,
        n_res=args.n_res,
        zbl=True,
        charge_embed_dim=args.charge_embed_dim,
        spin_embed_dim=args.spin_embed_dim,
        charge_range=(args.charge_min, args.charge_max),
        spin_range=(args.spin_min, args.spin_max),
    )
    
    # Initialize parameters
    key, init_key = jax.random.split(key)
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(args.num_atoms)
    
    sample_batch = next(train_loader.batches(num_atoms=args.num_atoms))
    
    # Add default charge/spin if not in batch
    if "total_charge" not in sample_batch:
        sample_batch["total_charge"] = jnp.zeros((1,))
    if "total_spin" not in sample_batch:
        sample_batch["total_spin"] = jnp.ones((1,))
    
    params = model.init(
        init_key,
        atomic_numbers=sample_batch["Z"][0],
        positions=sample_batch["R"][0],
        dst_idx=dst_idx,
        src_idx=src_idx,
        total_charges=sample_batch["total_charge"][:1],
        total_spins=sample_batch["total_spin"][:1],
    )
    
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Model initialized with {n_params:,} parameters")
    
    # Create optimizer
    optimizer, transform, schedule_fn, optimizer_kwargs = get_optimizer(
        learning_rate=args.learning_rate,
        schedule_fn=None,
        optimizer=None,
        transform=None,
    )
    
    ema_params = params
    opt_state = optimizer.init(params)
    
    # Training loop
    print("\nStarting training...")
    print("="*80)
    
    best_valid_loss = float('inf')
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else BASE_CKPT_DIR
    ckpt_path = ckpt_dir / args.name
    ckpt_path.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, args.num_epochs + 1):
        epoch_start = time.time()
        
        print(f"\nEpoch {epoch}/{args.num_epochs}")
        print("-" * 80)
        
        # Train
        train_loss = 0.0
        train_energy_mae = 0.0
        train_forces_mae = 0.0
        
        for i, batch in enumerate(train_loader.batches(num_atoms=args.num_atoms)):
            # Add default charge/spin if not in data
            if "total_charge" not in batch:
                batch["total_charge"] = batch.get("Qtot", jnp.zeros(batch["Z"].shape[0]))
            if "total_spin" not in batch:
                batch["total_spin"] = jnp.ones(batch["Z"].shape[0])
            
            params, opt_state, ema_params, loss, e_mae, f_mae = train_step_charge_spin(
                model_apply=model.apply,
                optimizer_update=optimizer.update,
                batch=batch,
                params=params,
                opt_state=opt_state,
                ema_params=ema_params,
                energy_weight=args.energy_weight,
                forces_weight=args.forces_weight,
            )
            
            train_loss += (loss - train_loss) / (i + 1)
            train_energy_mae += (e_mae - train_energy_mae) / (i + 1)
            train_forces_mae += (f_mae - train_forces_mae) / (i + 1)
            
            if i % 50 == 0:
                print(f"  Batch {i}: Loss={loss:.6f}, E_MAE={e_mae:.6f}, F_MAE={f_mae:.6f}")
        
        # Validate
        valid_loss = 0.0
        valid_energy_mae = 0.0
        valid_forces_mae = 0.0
        
        for i, batch in enumerate(valid_loader.batches(num_atoms=args.num_atoms)):
            if "total_charge" not in batch:
                batch["total_charge"] = batch.get("Qtot", jnp.zeros(batch["Z"].shape[0]))
            if "total_spin" not in batch:
                batch["total_spin"] = jnp.ones(batch["Z"].shape[0])
            
            loss, e_mae, f_mae = eval_step_charge_spin(
                model_apply=model.apply,
                batch=batch,
                params=ema_params,
                energy_weight=args.energy_weight,
                forces_weight=args.forces_weight,
            )
            
            valid_loss += (loss - valid_loss) / (i + 1)
            valid_energy_mae += (e_mae - valid_energy_mae) / (i + 1)
            valid_forces_mae += (f_mae - valid_forces_mae) / (i + 1)
        
        epoch_time = time.time() - epoch_start
        
        # Print results
        print(f"\nEpoch {epoch} Results:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Train Energy MAE: {train_energy_mae:.6f} kcal/mol")
        print(f"  Train Forces MAE: {train_forces_mae:.6f} kcal/mol/Å")
        print(f"  Valid Loss: {valid_loss:.6f}")
        print(f"  Valid Energy MAE: {valid_energy_mae:.6f} kcal/mol")
        print(f"  Valid Forces MAE: {valid_forces_mae:.6f} kcal/mol/Å")
        print(f"  Time: {epoch_time:.2f} s")
        
        # Save best checkpoint
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            print(f"  → New best validation loss! Saving checkpoint...")
            
            state_obj = train_state.TrainState.create(
                apply_fn=model.apply, params=params, tx=optimizer
            )
            
            ckpt = {
                "model": state_obj,
                "model_attributes": model.return_attributes(),
                "ema_params": ema_params,
                "params": params,
                "epoch": epoch,
                "opt_state": opt_state,
                "best_loss": best_valid_loss,
            }
            
            save_args = orbax_utils.save_args_from_target(ckpt)
            orbax_checkpointer.save(
                ckpt_path / f"epoch-{epoch}", ckpt, save_args=save_args
            )
    
    print("\n" + "="*80)
    print("Training completed!")
    print(f"Best validation loss: {best_valid_loss:.6f}")
    print(f"Checkpoints saved to: {ckpt_path}")
    print("="*80)


if __name__ == "__main__":
    main()

