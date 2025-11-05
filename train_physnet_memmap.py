#!/usr/bin/env python3
"""
PhysNet Training Script with PackedMemmapLoader

This script trains a PhysNet model using memory-mapped data from OpenQDC or similar datasets.
It adapts the PackedMemmapLoader format to PhysNet's expected input format.

Usage:
    python train_physnet_memmap.py \
        --data_path openqdc_packed_memmap \
        --batch_size 32 \
        --num_epochs 100 \
        --learning_rate 0.001
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, Iterator

import e3x
import jax
import jax.numpy as jnp
import numpy as np
from flax.training import orbax_utils, train_state

from mmml.physnetjax.physnetjax.models.model import EF
from mmml.physnetjax.physnetjax.training.trainstep import train_step
from mmml.physnetjax.physnetjax.training.evalstep import eval_step
from mmml.physnetjax.physnetjax.training.optimizer import get_optimizer
from mmml.physnetjax.physnetjax.restart.restart import orbax_checkpointer
from mmml.physnetjax.physnetjax.directories import BASE_CKPT_DIR


class PackedMemmapLoader:
    """
    Memory-mapped data loader for packed molecular datasets.
    
    This loader efficiently handles variable-size molecules stored in packed format,
    with bucketed batching to minimize padding overhead.
    """
    
    def __init__(
        self,
        path: str,
        batch_size: int,
        shuffle: bool = True,
        bucket_size: int = 8192,
        seed: int = 0,
    ):
        self.path = path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.bucket_size = bucket_size
        self.rng = np.random.default_rng(seed)

        # Load metadata
        self.offsets = np.load(os.path.join(path, "offsets.npy"))
        self.n_atoms = np.load(os.path.join(path, "n_atoms.npy"))
        self.N = int(self.n_atoms.shape[0])
        sumA = int(self.offsets[-1])

        # Open read-only memmaps
        self.Z_pack = np.memmap(
            os.path.join(path, "Z_pack.int32"),
            dtype=np.int32,
            mode="r",
            shape=(sumA,),
        )
        self.R_pack = np.memmap(
            os.path.join(path, "R_pack.f32"),
            dtype=np.float32,
            mode="r",
            shape=(sumA, 3),
        )
        self.F_pack = np.memmap(
            os.path.join(path, "F_pack.f32"),
            dtype=np.float32,
            mode="r",
            shape=(sumA, 3),
        )
        self.E = np.memmap(
            os.path.join(path, "E.f64"),
            dtype=np.float64,
            mode="r",
            shape=(self.N,),
        )
        self.Qtot = np.memmap(
            os.path.join(path, "Qtot.f64"),
            dtype=np.float64,
            mode="r",
            shape=(self.N,),
        )

        self.indices = np.arange(self.N, dtype=np.int64)

    def _yield_indices_bucketed(self) -> Iterator[np.ndarray]:
        """Yield indices in buckets sorted by molecule size."""
        order = self.indices.copy()
        if self.shuffle:
            self.rng.shuffle(order)

        for start in range(0, self.N, self.bucket_size):
            chunk = order[start : start + self.bucket_size]
            # Sort by size within bucket to minimize padding
            chunk = chunk[np.argsort(self.n_atoms[chunk], kind="mergesort")]

            for bstart in range(0, len(chunk), self.batch_size):
                yield chunk[bstart : bstart + self.batch_size]

    def _slice_mol(self, k: int):
        """Extract a single molecule from packed arrays."""
        a0, a1 = int(self.offsets[k]), int(self.offsets[k + 1])
        return (
            self.Z_pack[a0:a1],
            self.R_pack[a0:a1],
            self.F_pack[a0:a1],
            self.E[k],
            self.Qtot[k],
        )

    def batches(self, num_atoms: int = None) -> Iterator[Dict[str, jnp.ndarray]]:
        """
        Generate batches in PhysNet-compatible format.
        
        Parameters
        ----------
        num_atoms : int, optional
            Maximum number of atoms. If None, uses max from batch.
            
        Yields
        ------
        dict
            Batch dictionary with keys: Z, R, F, E, N, Qtot, dst_idx, src_idx, batch_segments
        """
        for batch_idx in self._yield_indices_bucketed():
            if len(batch_idx) == 0:
                continue

            Amax = int(self.n_atoms[batch_idx].max())
            if num_atoms is not None:
                Amax = num_atoms
            B = len(batch_idx)

            # Preallocate arrays
            Z = np.zeros((B, Amax), dtype=np.int32)
            R = np.zeros((B, Amax, 3), dtype=np.float32)
            F = np.zeros((B, Amax, 3), dtype=np.float32)
            N = np.zeros((B,), dtype=np.int32)
            E = np.zeros((B,), dtype=np.float64)
            Qtot = np.zeros((B,), dtype=np.float64)

            # Fill batch
            for j, k in enumerate(batch_idx):
                z, r, f, e, q = self._slice_mol(int(k))
                a = z.shape[0]
                Z[j, :a] = z
                R[j, :a] = r
                F[j, :a] = f
                N[j] = a
                E[j] = e
                Qtot[j] = q

            # Generate graph indices for PhysNet
            dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(Amax)
            
            # Create batch segments (which molecule each atom belongs to)
            batch_segments = np.repeat(np.arange(B), Amax).astype(np.int32)

            yield {
                "Z": jnp.array(Z),
                "R": jnp.array(R),
                "F": jnp.array(F),
                "N": jnp.array(N),
                "E": jnp.array(E),
                "Qtot": jnp.array(Qtot),
                "dst_idx": dst_idx,
                "src_idx": src_idx,
                "batch_segments": batch_segments,
            }


def train_epoch(
    model,
    params,
    ema_params,
    opt_state,
    transform_state,
    optimizer,
    loader: PackedMemmapLoader,
    energy_weight: float,
    forces_weight: float,
    num_atoms: int,
):
    """Train for one epoch."""
    train_loss = 0.0
    train_energy_mae = 0.0
    train_forces_mae = 0.0
    
    for i, batch in enumerate(loader.batches(num_atoms=num_atoms)):
        batch_size = int(batch["Z"].shape[0])
        
        # Add masks that train_step expects
        batch["atom_mask"] = (batch["Z"] > 0).astype(jnp.float32).reshape(-1)
        batch["batch_mask"] = jnp.ones_like(batch["dst_idx"], dtype=jnp.float32)
        
        (
            params,
            ema_params,
            opt_state,
            transform_state,
            loss,
            energy_mae,
            forces_mae,
            dipole_mae,
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
        train_energy_mae += (energy_mae - train_energy_mae) / (i + 1)
        train_forces_mae += (forces_mae - train_forces_mae) / (i + 1)
        
        if i % 50 == 0:
            print(f"  Batch {i}: Loss={loss:.6f}, E_MAE={energy_mae:.6f}, F_MAE={forces_mae:.6f}")
    
    return params, ema_params, opt_state, transform_state, train_loss, train_energy_mae, train_forces_mae


def validate(
    model,
    ema_params,
    loader: PackedMemmapLoader,
    energy_weight: float,
    forces_weight: float,
    num_atoms: int,
):
    """Validate the model."""
    valid_loss = 0.0
    valid_energy_mae = 0.0
    valid_forces_mae = 0.0
    
    for i, batch in enumerate(loader.batches(num_atoms=num_atoms)):
        batch_size = int(batch["Z"].shape[0])
        
        # Add masks that eval_step expects
        batch["atom_mask"] = (batch["Z"] > 0).astype(jnp.float32).reshape(-1)
        batch["batch_mask"] = jnp.ones_like(batch["dst_idx"], dtype=jnp.float32)
        
        loss, energy_mae, forces_mae, dipole_mae = eval_step(
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
    
    return valid_loss, valid_energy_mae, valid_forces_mae


def main():
    parser = argparse.ArgumentParser(description="Train PhysNet on memory-mapped data")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to packed memmap data directory")
    parser.add_argument("--valid_split", type=float, default=0.1,
                       help="Fraction of data to use for validation")
    
    # Model arguments
    parser.add_argument("--features", type=int, default=128,
                       help="Number of features in hidden layers")
    parser.add_argument("--max_degree", type=int, default=2,
                       help="Maximum degree for spherical harmonics")
    parser.add_argument("--num_iterations", type=int, default=3,
                       help="Number of message passing iterations")
    parser.add_argument("--num_basis_functions", type=int, default=16,
                       help="Number of radial basis functions")
    parser.add_argument("--cutoff", type=float, default=5.0,
                       help="Cutoff distance in Angstroms")
    parser.add_argument("--num_atoms", type=int, default=60,
                       help="Maximum number of atoms per molecule")
    parser.add_argument("--n_res", type=int, default=3,
                       help="Number of residual blocks")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100,
                       help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--energy_weight", type=float, default=1.0,
                       help="Weight for energy loss")
    parser.add_argument("--forces_weight", type=float, default=52.91,
                       help="Weight for forces loss (kcal/mol conversion)")
    parser.add_argument("--bucket_size", type=int, default=8192,
                       help="Size of buckets for sorting by molecule size")
    
    # Checkpointing
    parser.add_argument("--name", type=str, default="physnet_memmap",
                       help="Experiment name for checkpointing")
    parser.add_argument("--ckpt_dir", type=str, default=None,
                       help="Checkpoint directory")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed")
    
    args = parser.parse_args()
    
    print("="*80)
    print("PhysNet Training with Packed Memmap Data")
    print("="*80)
    print(f"Data path: {args.data_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of epochs: {args.num_epochs}")
    print(f"Learning rate: {args.learning_rate}")
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
    
    # Split into train and validation
    n_total = loader.N
    n_valid = int(n_total * args.valid_split)
    n_train = n_total - n_valid
    
    print(f"Total molecules: {n_total}")
    print(f"Training molecules: {n_train}")
    print(f"Validation molecules: {n_valid}")
    
    # Create train and validation loaders
    train_loader = PackedMemmapLoader(
        args.data_path,
        batch_size=args.batch_size,
        shuffle=True,
        bucket_size=args.bucket_size,
        seed=args.seed,
    )
    train_loader.indices = train_loader.indices[:n_train]
    train_loader.N = n_train
    
    valid_loader = PackedMemmapLoader(
        args.data_path,
        batch_size=args.batch_size,
        shuffle=False,
        bucket_size=args.bucket_size,
        seed=args.seed,
    )
    valid_loader.indices = valid_loader.indices[n_train:]
    valid_loader.N = n_valid
    
    # Create model
    print("\nInitializing model...")
    model = EF(
        features=args.features,
        max_degree=args.max_degree,
        num_iterations=args.num_iterations,
        num_basis_functions=args.num_basis_functions,
        cutoff=args.cutoff,
        max_atomic_number=118,
        charges=False,
        natoms=args.num_atoms,
        total_charge=0.0,
        n_res=args.n_res,
        zbl=True,
        debug=False,
        efa=False,
    )
    
    # Initialize parameters
    key, init_key = jax.random.PRNGKey(args.seed), jax.random.PRNGKey(args.seed + 1)
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(args.num_atoms)
    
    # Get a sample batch to initialize
    sample_batch = next(train_loader.batches(num_atoms=args.num_atoms))
    params = model.init(
        init_key,
        atomic_numbers=sample_batch["Z"][0],
        positions=sample_batch["R"][0],
        dst_idx=dst_idx,
        src_idx=src_idx,
    )
    
    print(f"Model initialized with {sum(x.size for x in jax.tree_util.tree_leaves(params)):,} parameters")
    
    # Create optimizer
    optimizer, transform, schedule_fn, optimizer_kwargs = get_optimizer(
        learning_rate=args.learning_rate,
        schedule_fn=None,
        optimizer=None,
        transform=None,
    )
    
    ema_params = params
    opt_state = optimizer.init(params)
    transform_state = transform.init(params)
    
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
        print("Training...")
        params, ema_params, opt_state, transform_state, train_loss, train_energy_mae, train_forces_mae = train_epoch(
            model=model,
            params=params,
            ema_params=ema_params,
            opt_state=opt_state,
            transform_state=transform_state,
            optimizer=optimizer,
            loader=train_loader,
            energy_weight=args.energy_weight,
            forces_weight=args.forces_weight,
            num_atoms=args.num_atoms,
        )
        
        # Validate
        print("Validating...")
        valid_loss, valid_energy_mae, valid_forces_mae = validate(
            model=model,
            ema_params=ema_params,
            loader=valid_loader,
            energy_weight=args.energy_weight,
            forces_weight=args.forces_weight,
            num_atoms=args.num_atoms,
        )
        
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
        
        # Save checkpoint if best
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            print(f"  → New best validation loss! Saving checkpoint...")
            
            state = train_state.TrainState.create(
                apply_fn=model.apply, params=params, tx=optimizer
            )
            
            ckpt = {
                "model": state,
                "model_attributes": model.return_attributes(),
                "transform_state": transform_state,
                "ema_params": ema_params,
                "params": params,
                "epoch": epoch,
                "opt_state": opt_state,
                "best_loss": best_valid_loss,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
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

