# =========================
# FULL WORKING SCRIPT (FIXED)
# =========================
# Key fixes:
# 1) Set XLA/CUDA env flags BEFORE importing jax (restart kernel if in notebook).
# 2) Keep batch shapes consistent with the model:
#    atomic_numbers: (B, N), positions: (B, N, 3), Ef: (B, 3)
# 3) Do NOT pre-offset dst/src indices in prepare_batches; EFD() already offsets.
# 4) batch_segments must be length (B*N) with segment ids 0..B-1 repeated per atom.
# 5) Fix element_bias indexing to use flattened atomic numbers if enabled.

import os

# --- Environment (must be set before importing jax) ---
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".99"

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import functools
import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp

import optax
from optax import tree_utils as otu
from flax import linen as nn

import e3x

import matplotlib.pyplot as plt  # optional; kept because you had it
import ase  # optional; kept because you had it
from ase.visualize import view as view  # optional; kept because you had it

# Disable CUDA graph capture to avoid "library was not initialized" errors
# This makes training slower but more stable
# CUDA graph capture is incompatible with certain computation patterns
try:
    jax.config.update("jax_cuda_graph_level", 0)  # Disable CUDA graphs if supported
except:
    print("CUDA graph capture is not supported in this JAX version")
    pass  # Not available in this JAX version


import json
from mmml.utils.model_checkpoint import to_jsonable


print("JAX devices:", jax.devices())


import lovely_jax as lj
lj.monkey_patch()

# -------------------------
# Load dataset
# -------------------------
dataset = np.load("data-full.npz", allow_pickle=True)
print("Dataset keys:", dataset.files)


# -------------------------
# Loss helpers
# -------------------------
def mean_squared_loss(prediction, target):
    prediction = jnp.asarray(prediction)
    target = jnp.asarray(target)
    return jnp.mean(optax.l2_loss(prediction, target))


def mean_absolute_error(prediction, target):
    prediction = jnp.asarray(prediction)
    target = jnp.asarray(target)
    return jnp.mean(jnp.abs(prediction - target))


# -------------------------
# Model
# -------------------------
class MessagePassingModel(nn.Module):
    features: int = 32
    max_degree: int = 2
    num_iterations: int = 3
    num_basis_functions: int = 8
    cutoff: float = 5.0
    max_atomic_number: int = 55
    include_pseudotensors: bool = True

    def EFD(self, atomic_numbers, positions, Ef, dst_idx, src_idx, dst_idx_flat, src_idx_flat, batch_segments, batch_size):
        """
        Expected shapes:
          atomic_numbers: (B, N)
          positions:      (B, N, 3)
          Ef:             (B, 3)
          dst_idx/src_idx: (E,) indices for a single molecule of N atoms (no batching offsets)
          dst_idx_flat/src_idx_flat: (B*E,) pre-computed flattened indices (CUDA-graph-friendly)
          batch_segments: (B*N,) segment ids 0..B-1 repeated per atom
          batch_size:     int (static)
        Returns:
          proxy_energy: scalar (sum over batch)
          energy: (B,) per-molecule energy
        """
        # Basic dims: use static values (CUDA-graph-friendly)
        B = batch_size  # Static - known at compile time
        N = 29  # Static - constant number of atoms per molecule

        # Positions and atomic_numbers are already flattened in prepare_batches (like physnetjax)
        # Ensure they're properly shaped (1D for atomic_numbers, 2D for positions)
        positions_flat = positions.reshape(-1, 3)  # Ensure (B*N, 3)
        atomic_numbers_flat = atomic_numbers.reshape(-1)  # Ensure (B*N,) - 1D array

        # Compute displacements using e3x gather operations (CUDA-graph-friendly)
        # Must use flattened indices to get (B*E, 3) displacements matching MessagePass
        positions_dst = e3x.ops.gather_dst(positions_flat, dst_idx=dst_idx_flat)
        positions_src = e3x.ops.gather_src(positions_flat, src_idx=src_idx_flat)
        displacements = positions_src - positions_dst  # (B*E, 3)

        # Build an EF tensor of shape compatible with e3x.nn.Tensor()
        # e3x format is (num_atoms, parity, (lmax+1)^2, features)
        # Start with (B, 4) -> expand to (B*N, 2, 4, features) for parity=2 (pseudotensors)
        pad_ef = jnp.zeros((B, 1), dtype=positions_flat.dtype)

        xEF = jnp.concatenate((pad_ef, Ef), axis=-1)   # (B, 4) - [1, Ef_x, Ef_y, Ef_z]
        xEF = xEF[:, None, :, None]                  # (B, 1, 4, 1)
        xEF = jnp.tile(xEF, (1, 1, 1, self.features)) # (B, 1, 4, features)
        # Expand to per-atom level: (B, 1, 4, features) -> (B, N, 1, 4, features) -> (B*N, 1, 4, features)
        # Insert dimension for N, then repeat: (B, 1, 4, features) -> (B, 1, 1, 4, features) -> (B, N, 1, 4, features)
        xEF = xEF[:, None, :, :]  # (B, 1, 1, 4, features) - add dimension
        xEF = jnp.repeat(xEF, N, axis=1)  # (B, N, 1, 4, features) - repeat N times
        xEF = xEF.reshape(B * N, 1, 4, self.features)  # (B*N, 1, 4, features)
        # Expand parity dimension from 1 to 2 to match x (since include_pseudotensors=True)
        # xEF is (B*N, 1, 4, features), need (B*N, 2, 4, features) - broadcast the parity dimension
        xEF = jnp.broadcast_to(xEF, (B * N, 2, 4, self.features))

        # Use pre-computed flattened indices (passed from batch dict, computed outside JIT)
        # This avoids creating traced index arrays inside the JIT function
        basis = e3x.nn.basis(
            displacements,
            num=self.num_basis_functions,
            max_degree=self.max_degree,
            radial_fn=e3x.nn.reciprocal_bernstein,
            cutoff_fn=functools.partial(e3x.nn.smooth_cutoff, cutoff=self.cutoff)
        )

        # Embed atoms (flattened) - atomic_numbers_flat is already ensured to be 1D above
        x = e3x.nn.Embed(num_embeddings=self.max_atomic_number + 1, features=self.features)(atomic_numbers_flat)

        # Message passing loop
        for i in range(self.num_iterations):
            y = e3x.nn.MessagePass(
                include_pseudotensors=self.include_pseudotensors,
                max_degree=self.max_degree if i < self.num_iterations - 1 else 0
            )(x, basis, dst_idx=dst_idx_flat, src_idx=src_idx_flat)

            x = e3x.nn.add(x, y)
            x = e3x.nn.Dense(self.features)(x)
            x = e3x.nn.silu(x)
            x = e3x.nn.Dense(self.features, kernel_init=jax.nn.initializers.zeros)(x)
            x = e3x.nn.add(x, y)
            x = e3x.nn.silu(x)
            # Couple EF - xEF already has correct shape (B*N, 2, 4, features) matching x
            xEF = e3x.nn.Tensor()(x, xEF)
            x = e3x.nn.add(x, xEF)
            x = e3x.nn.TensorDense(max_degree=self.max_degree)(x)
            x = e3x.nn.add(x, xEF)
            x = e3x.nn.silu(x)

        x = e3x.nn.silu(x)

        # Reduce to scalars per atom
        x = e3x.nn.change_max_degree_or_type(x, max_degree=0, include_pseudotensors=False)

        # Predict atomic energies
        element_bias = self.param(
            "element_bias",
            lambda rng, shape: jnp.zeros(shape, dtype=positions.dtype),
            (self.max_atomic_number + 1,)
        )


        for i in range(3):
            x = e3x.nn.Dense(self.features)(x)
            x = e3x.nn.silu(x)
        atomic_energies = nn.Dense(1, use_bias=False, kernel_init=jax.nn.initializers.zeros)(x)
        atomic_energies = jnp.squeeze(atomic_energies, axis=(-1, -2, -3))  # (B*N,)

        # If you want element biases, do it in flattened space:
        atomic_energies = atomic_energies + element_bias[atomic_numbers_flat]

        # Sum per molecule without segment_sum/scatter (CUDA-graph-friendly).
        energy = atomic_energies.reshape(B, N).sum(axis=1)  # (B,)

        # Proxy energy for force differentiation
        return -jnp.sum(energy), energy

    @nn.compact
    def __call__(self, atomic_numbers, positions, Ef, dst_idx, src_idx, dst_idx_flat=None, src_idx_flat=None, batch_segments=None, batch_size=None):
        """Returns (proxy_energy, energy) - use energy_and_forces() wrapper for forces."""
        if batch_segments is None:
            # atomic_numbers expected (B,N); if B absent, treat as (N,)
            if atomic_numbers.ndim == 1:
                atomic_numbers = atomic_numbers[None, :]
                positions = positions[None, :, :]
                Ef = Ef[None, :]
            B, N = atomic_numbers.shape
            batch_size = B
            # Flatten positions and atomic_numbers to match batch prep pattern
            atomic_numbers = atomic_numbers.reshape(-1)  # (B*N,)
            positions = positions.reshape(-1, 3)          # (B*N, 3)
            batch_segments = jnp.repeat(jnp.arange(B, dtype=jnp.int32), N)
            # Compute flattened indices for this single-batch case
            offsets = jnp.arange(B, dtype=jnp.int32) * N
            dst_idx_flat = (dst_idx[None, :] + offsets[:, None]).reshape(-1)
            src_idx_flat = (src_idx[None, :] + offsets[:, None]).reshape(-1)

        proxy_energy, energy = self.EFD(
            atomic_numbers, positions, Ef, dst_idx, src_idx, dst_idx_flat, src_idx_flat, batch_segments, batch_size
        )
        return energy


def energy_and_forces(model_apply, params, atomic_numbers, positions, Ef, dst_idx, src_idx, batch_segments, batch_size):
    """Compute energy and forces (negative gradient of energy w.r.t. positions)."""
    def energy_fn(pos):
        energy = model_apply(
            params,
            atomic_numbers=atomic_numbers,
            positions=pos,
            Ef=Ef,
            dst_idx=dst_idx,
            src_idx=src_idx,
            batch_segments=batch_segments,
            batch_size=batch_size,
        )
        return -jnp.sum(energy), energy
    
    (_, energy), forces = jax.value_and_grad(energy_fn, has_aux=True)(positions)
    return energy, forces


# -------------------------
# Dataset prep
# -------------------------
def prepare_datasets(key, num_train, num_valid, dataset):
    num_data = len(dataset["R"])
    num_draw = num_train + num_valid
    if num_draw > num_data:
        raise RuntimeError(
            f"dataset only contains {num_data} points, requested num_train={num_train}, num_valid={num_valid}"
        )

    choice = np.asarray(jax.random.choice(key, num_data, shape=(num_draw,), replace=False))
    train_choice = choice[:num_train]
    valid_choice = choice[num_train:]

    # Note: R has shape (num_data, 1, N, 3) - squeeze out the extra dimension
    positions_raw = jnp.asarray(dataset["R"], dtype=jnp.float32)
    if positions_raw.ndim == 4 and positions_raw.shape[1] == 1:
        positions_raw = positions_raw.squeeze(axis=1)  # (num_data, N, 3)
    
    train_data = dict(
        atomic_numbers=jnp.asarray(dataset["Z"], dtype=jnp.int32)[train_choice],      # (num_train, N)
        positions=positions_raw[train_choice],                                         # (num_train, N, 3)
        electric_field=jnp.asarray(dataset["Ef"], dtype=jnp.float32)[train_choice],   # (num_train, 3)
        energies=jnp.asarray(dataset["E"], dtype=jnp.float32)[train_choice],          # (num_train,) or (num_train,1)
    )
    valid_data = dict(
        atomic_numbers=jnp.asarray(dataset["Z"], dtype=jnp.int32)[valid_choice],
        positions=positions_raw[valid_choice],                                         # (num_valid, N, 3)
        electric_field=jnp.asarray(dataset["Ef"], dtype=jnp.float32)[valid_choice],
        energies=jnp.asarray(dataset["E"], dtype=jnp.float32)[valid_choice],
    )
    return train_data, valid_data


def prepare_batches(key, data, batch_size):
    """
    Returns list of batch dicts with consistent shapes:
      atomic_numbers: (B,N)
      positions: (B,N,3)
      electric_field: (B,3)
      energies: (B,) or (B,1)
      dst_idx/src_idx: (E,) for single molecule (no offsets)
      batch_segments: (B*N,)
    """
    data_size = len(data["electric_field"])
    steps_per_epoch = data_size // batch_size

    perms = jax.random.permutation(key, data_size)
    perms = perms[: steps_per_epoch * batch_size]  # drop incomplete last batch
    perms = perms.reshape((steps_per_epoch, batch_size))

    num_atoms = 29 #data["positions"].shape[1]
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)
    # Ensure these are jax arrays with explicit dtype
    dst_idx = jnp.asarray(dst_idx, dtype=jnp.int32)
    src_idx = jnp.asarray(src_idx, dtype=jnp.int32)

    # Pre-compute flattened indices (CUDA-graph-friendly: computed outside JIT)
    offsets = jnp.arange(batch_size, dtype=jnp.int32) * num_atoms
    dst_idx_flat = (dst_idx[None, :] + offsets[:, None]).reshape(-1)
    src_idx_flat = (src_idx[None, :] + offsets[:, None]).reshape(-1)

    batch_segments = jnp.repeat(jnp.arange(batch_size, dtype=jnp.int32), num_atoms)

    batches = []
    for perm in perms:
        batches.append(
            dict(
                atomic_numbers=data["atomic_numbers"][perm].reshape(batch_size * num_atoms),  # (B*N,)
                positions=data["positions"][perm].reshape(batch_size * num_atoms, 3),          # (B*N, 3) - flattened like physnetjax
                energies=data["energies"][perm],                                               # (B,) or (B,1)
                electric_field=data["electric_field"][perm],                                   # (B,3)
                dst_idx=dst_idx,  # Keep original for reference
                src_idx=src_idx,  # Keep original for reference
                dst_idx_flat=dst_idx_flat,  # Pre-computed flattened indices
                src_idx_flat=src_idx_flat,  # Pre-computed flattened indices
                batch_segments=batch_segments,
            )
        )
    return batches


# -------------------------
# Train / Eval steps
# -------------------------
@functools.partial(jax.jit, static_argnames=("model_apply", "optimizer_update", "batch_size", "ema_decay"))
def train_step(model_apply, optimizer_update, batch, batch_size, opt_state, params, ema_params, transform_state, ema_decay=0.999):
    def loss_fn(params):
        energy = model_apply(
            params,
            atomic_numbers=batch["atomic_numbers"],
            positions=batch["positions"],
            Ef=batch["electric_field"],
            dst_idx=batch["dst_idx"],
            src_idx=batch["src_idx"],
            dst_idx_flat=batch["dst_idx_flat"],
            src_idx_flat=batch["src_idx_flat"],
            batch_segments=batch["batch_segments"],
            batch_size=batch_size,
        )
        loss = mean_squared_loss(energy.reshape(-1), batch["energies"].reshape(-1))
        return loss, energy

    (loss, energy), grad = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer_update(grad, opt_state, params)
    
    # Apply learning rate scaling from reduce_on_plateau transform
    updates = otu.tree_scalar_mul(transform_state.scale, updates)
    params = optax.apply_updates(params, updates)

    # Update EMA parameters
    ema_params = jax.tree_util.tree_map(
        lambda ema, new: ema_decay * ema + (1 - ema_decay) * new,
        ema_params,
        params,
    )

    mae = mean_absolute_error(energy, batch["energies"])
    return params, ema_params, opt_state, loss, mae


@functools.partial(jax.jit, static_argnames=("model_apply", "batch_size"))
def eval_step(model_apply, batch, batch_size, params):
    energy = model_apply(
        params,
        atomic_numbers=batch["atomic_numbers"],
        positions=batch["positions"],
        Ef=batch["electric_field"],
        dst_idx=batch["dst_idx"],
        src_idx=batch["src_idx"],
        dst_idx_flat=batch["dst_idx_flat"],
        src_idx_flat=batch["src_idx_flat"],
        batch_segments=batch["batch_segments"],
        batch_size=batch_size,
    )
    loss = mean_squared_loss(energy, batch["energies"])
    mae = mean_absolute_error(energy, batch["energies"])
    return loss, mae


def train_model(key, model, train_data, valid_data, num_epochs, learning_rate, batch_size, 
                clip_norm=10.0, ema_decay=0.999, early_stopping_patience=None, early_stopping_min_delta=0.0,
                reduce_on_plateau_patience=5, reduce_on_plateau_cooldown=5, reduce_on_plateau_factor=0.9,
                reduce_on_plateau_rtol=1e-4, reduce_on_plateau_accumulation_size=5, reduce_on_plateau_min_scale=0.01):
    """
    Train model with EMA, gradient clipping, early stopping, and learning rate reduction on plateau.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for initialization and shuffling
    model : MessagePassingModel
        Model instance
    train_data : dict
        Training data dictionary
    valid_data : dict
        Validation data dictionary
    num_epochs : int
        Maximum number of training epochs
    learning_rate : float
        Learning rate
    batch_size : int
        Batch size
    clip_norm : float, optional
        Gradient clipping norm (default: 10.0)
    ema_decay : float, optional
        EMA decay factor (default: 0.999)
    early_stopping_patience : int, optional
        Number of epochs to wait before early stopping (default: None, disabled)
    early_stopping_min_delta : float, optional
        Minimum change to qualify as an improvement (default: 0.0)
    reduce_on_plateau_patience : int, optional
        Patience for reduce on plateau (default: 5)
    reduce_on_plateau_cooldown : int, optional
        Cooldown for reduce on plateau (default: 5)
    reduce_on_plateau_factor : float, optional
        Factor to reduce LR by (default: 0.9)
    reduce_on_plateau_rtol : float, optional
        Relative tolerance for plateau detection (default: 1e-4)
    reduce_on_plateau_accumulation_size : int, optional
        Accumulation size for plateau detection (default: 5)
    reduce_on_plateau_min_scale : float, optional
        Minimum scale factor (default: 0.01)
    
    Returns
    -------
    params : dict
        Final EMA parameters (best validation loss)
    """
    # Create optimizer with gradient clipping
    optimizer = optax.chain(
        optax.clip_by_global_norm(clip_norm),
        optax.adam(learning_rate)
    )
    
    # Create reduce on plateau transform
    transform = optax.contrib.reduce_on_plateau(
        patience=reduce_on_plateau_patience,
        cooldown=reduce_on_plateau_cooldown,
        factor=reduce_on_plateau_factor,
        rtol=reduce_on_plateau_rtol,
        accumulation_size=reduce_on_plateau_accumulation_size,
        min_scale=reduce_on_plateau_min_scale,
    )

    # Initialize params with a single batch item (B=1) but correct ranks
    key, init_key = jax.random.split(key)
    num_atoms = train_data["positions"].shape[1]
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)
    # Ensure these are jax arrays with explicit dtype
    dst_idx = jnp.asarray(dst_idx, dtype=jnp.int32)
    src_idx = jnp.asarray(src_idx, dtype=jnp.int32)

    # Use slicing [0:1] to keep batch dimension, not [0] which removes it
    # __call__ will flatten these when batch_segments is None
    atomic_numbers0 = train_data["atomic_numbers"][0:1]    # (1, N)
    positions0 = train_data["positions"][0:1]                # (1, N, 3)
    ef0 = train_data["electric_field"][0:1]                # (1, 3)
    batch_segments0 = jnp.repeat(jnp.arange(1, dtype=jnp.int32), num_atoms)
    # Pre-compute flattened indices for initialization
    offsets0 = jnp.arange(1, dtype=jnp.int32) * num_atoms
    dst_idx_flat0 = (dst_idx[None, :] + offsets0[:, None]).reshape(-1)
    src_idx_flat0 = (src_idx[None, :] + offsets0[:, None]).reshape(-1)

    params = model.init(
        init_key,
        atomic_numbers=atomic_numbers0,
        positions=positions0,
        Ef=ef0,
        dst_idx=dst_idx,
        src_idx=src_idx,
        dst_idx_flat=dst_idx_flat0,
        src_idx_flat=src_idx_flat0,
        batch_segments=batch_segments0,
        batch_size=1,  # Init always uses batch_size=1
    )
    opt_state = optimizer.init(params)
    
    # Initialize transform state for reduce on plateau
    transform_state = transform.init(params)
    
    # Initialize EMA parameters
    ema_params = params

    # Validation batches prepared once
    key, valid_key = jax.random.split(key)
    valid_batches = prepare_batches(valid_key, valid_data, batch_size)

    print("Validation batches prepared")
    for k in valid_batches[0].keys():
        print(f"  {k}: {valid_batches[0][k]}")

    # Early stopping tracking
    best_valid_loss = float('inf')
    patience_counter = 0
    best_ema_params = ema_params

    for epoch in range(1, num_epochs + 1):
        key, shuffle_key = jax.random.split(key)
        train_batches = prepare_batches(shuffle_key, train_data, batch_size)

        train_loss = 0.0
        train_mae = 0.0
        for i, batch in enumerate(train_batches):
            params, ema_params, opt_state, loss, mae = train_step(
                model_apply=model.apply,
                optimizer_update=optimizer.update,
                batch=batch,
                batch_size=batch_size,
                opt_state=opt_state,
                params=params,
                ema_params=ema_params,
                transform_state=transform_state,
                ema_decay=ema_decay,
            )
            train_loss += (loss - train_loss) / (i + 1)
            train_mae += (mae - train_mae) / (i + 1)

        valid_loss = 0.0
        valid_mae = 0.0
        # Use EMA parameters for validation (as in physnetjax)
        for i, batch in enumerate(valid_batches):
            loss, mae = eval_step(
                model_apply=model.apply,
                batch=batch,
                batch_size=batch_size,
                params=ema_params,
            )
            valid_loss += (loss - valid_loss) / (i + 1)
            valid_mae += (mae - valid_mae) / (i + 1)

        # Update reduce on plateau transform state
        _, transform_state = transform.update(
            updates=params, state=transform_state, value=valid_loss
        )
        lr_scale = transform_state.scale

        # Early stopping logic
        improved = False
        if valid_loss < best_valid_loss - early_stopping_min_delta:
            best_valid_loss = valid_loss
            best_ema_params = ema_params
            patience_counter = 0
            improved = True
        else:
            patience_counter += 1

        train_mae = train_mae * 23.0605
        valid_mae = valid_mae * 23.0605

        print(f"epoch: {epoch:3d}                    train:   valid:")
        print(f"    loss [a.u.]             {train_loss: 8.6f} {valid_loss: 8.6f}")
        print(f"    energy mae [kcal/mol]   {train_mae: 8.6f} {valid_mae: 8.6f}")
        print(f"    LR scale: {lr_scale: 8.6f}, effective LR: {learning_rate * lr_scale: 8.6f}")
        if early_stopping_patience is not None:
            print(f"    best valid loss: {best_valid_loss: 8.6f}, patience: {patience_counter}/{early_stopping_patience}")
            if improved:
                print(f"    ✓ Improved! Saving best model.")
        
        # Early stopping check
        if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs.")
            print(f"Best validation loss: {best_valid_loss: 8.6f} at epoch {epoch - patience_counter}")
            break

    return best_ema_params


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data-full.npz")
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--max_degree", type=int, default=2)
    parser.add_argument("--num_iterations", type=int, default=2)
    parser.add_argument("--num_basis_functions", type=int, default=64)
    parser.add_argument("--cutoff", type=float, default=10.0)
    parser.add_argument("--num_train", type=int, default=8000)
    parser.add_argument("--num_valid", type=int, default=2000)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.0003)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--clip_norm", type=float, default=100.0)
    parser.add_argument("--ema_decay", type=float, default=0.5)
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0)
    parser.add_argument("--reduce_on_plateau_patience", type=int, default=5)
    parser.add_argument("--reduce_on_plateau_cooldown", type=int, default=5)
    parser.add_argument("--reduce_on_plateau_factor", type=float, default=0.9)
    parser.add_argument("--reduce_on_plateau_rtol", type=float, default=1e-4)
    parser.add_argument("--reduce_on_plateau_accumulation_size", type=int, default=5)
    parser.add_argument("--reduce_on_plateau_min_scale", type=float, default=0.01)
    args = parser.parse_args()

    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")


    # Training hyperparameters
    clip_norm = 10000.0  # Gradient clipping norm
    ema_decay = 0.5  # EMA decay factor
    early_stopping_patience = None  # Number of epochs to wait before early stopping (None to disable)
    early_stopping_min_delta = 0.0  # Minimum change to qualify as improvement

    # Reduce on plateau hyperparameters
    reduce_on_plateau_patience = 5  # Patience for reduce on plateau
    reduce_on_plateau_cooldown = 5  # Cooldown for reduce on plateau
    reduce_on_plateau_factor = 0.9  # Factor to reduce LR by
    reduce_on_plateau_rtol = 1e-4  # Relative tolerance for plateau detection
    reduce_on_plateau_accumulation_size = 5  # Accumulation size for plateau detection
    reduce_on_plateau_min_scale = 0.01  # Minimum scale factor

    # print the hyperparameters
    print("Hyperparameters:")
    print(f"  features: {args.features}")
    print(f"  max_degree: {args.max_degree}")
    print(f"  num_iterations: {args.num_iterations}")
    print(f"  num_basis_functions: {args.num_basis_functions}")
    print(f"  cutoff: {args.cutoff}")
    print(f"  num_train: {args. num_train}")
    print(f"  num_valid: {args.num_valid}")
    print(f"  num_epochs: {args.num_epochs}")
    print(f"  learning_rate: {args.learning_rate}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  clip_norm: {args.clip_norm}")
    print(f"  ema_decay: {args.ema_decay}")
    print(f"  early_stopping_patience: {args.early_stopping_patience}")
    print(f"  early_stopping_min_delta: {args.early_stopping_min_delta}")
    print(f"  reduce_on_plateau_patience: {args.reduce_on_plateau_patience}")
    print(f"  reduce_on_plateau_cooldown: {args.reduce_on_plateau_cooldown}")
    print(f"  reduce_on_plateau_factor: {args.reduce_on_plateau_factor}")
    print(f"  reduce_on_plateau_rtol: {args.reduce_on_plateau_rtol}")
    print(f"  reduce_on_plateau_accumulation_size: {args.reduce_on_plateau_accumulation_size}")
    print(f"  reduce_on_plateau_min_scale: {args.reduce_on_plateau_min_scale}")


    # -------------------------
    # Run
    # -------------------------
    data_key, train_key = jax.random.split(jax.random.PRNGKey(0), 2)

    # Load dataset
    dataset = np.load(args.data, allow_pickle=True)
    
    train_data, valid_data = prepare_datasets(
        data_key, num_train=args.num_train, num_valid=args.num_valid, dataset=dataset
    )


    print("\nPrepared data shapes:")
    print(f"  train atomic_numbers: {train_data['atomic_numbers'].shape}")
    print(f"  train positions:      {train_data['positions'].shape}")
    print(f"  train electric_field: {train_data['electric_field'].shape}")
    print(f"  train energies:       {train_data['energies'].shape}")

    message_passing_model = MessagePassingModel(
        features=args.features,
        max_degree=args.max_degree,
        num_iterations=args.num_iterations,
        num_basis_functions=args.num_basis_functions,
        cutoff=args.cutoff,
    )

    # Generate UUID for this training run
    run_uuid = str(uuid.uuid4())
    print(f"\n{'='*60}")
    print(f"Training Run UUID: {run_uuid}")
    print(f"{'='*60}\n")

    params = train_model(
        key=train_key,
        model=message_passing_model,
        train_data=train_data,
        valid_data=valid_data,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        clip_norm=args.clip_norm,
        ema_decay=args.ema_decay,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        reduce_on_plateau_patience=args.reduce_on_plateau_patience,
        reduce_on_plateau_cooldown=args.reduce_on_plateau_cooldown,
        reduce_on_plateau_factor=args.reduce_on_plateau_factor,
        reduce_on_plateau_rtol=args.reduce_on_plateau_rtol,
        reduce_on_plateau_accumulation_size=args.reduce_on_plateau_accumulation_size,
        reduce_on_plateau_min_scale=args.reduce_on_plateau_min_scale,
    )

    # Prepare model config
    model_config = {
        'uuid': run_uuid,
        'model': {
            'features': args.features,
            'max_degree': args.max_degree,
            'num_iterations': args.num_iterations,
            'num_basis_functions': args.num_basis_functions,
            'cutoff': args.cutoff,
            'max_atomic_number': 55,  # Fixed in model
            'include_pseudotensors': True,  # Fixed in model
        },
        'training': {
            'num_train': args.num_train,
            'num_valid': args.num_valid,
            'num_epochs': args.num_epochs,
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'clip_norm': args.clip_norm,
            'ema_decay': args.ema_decay,
            'early_stopping_patience': args.early_stopping_patience,
            'early_stopping_min_delta': args.early_stopping_min_delta,
            'reduce_on_plateau_patience': args.reduce_on_plateau_patience,
            'reduce_on_plateau_cooldown': args.reduce_on_plateau_cooldown,
            'reduce_on_plateau_factor': args.reduce_on_plateau_factor,
            'reduce_on_plateau_rtol': args.reduce_on_plateau_rtol,
            'reduce_on_plateau_accumulation_size': args.reduce_on_plateau_accumulation_size,
            'reduce_on_plateau_min_scale': args.reduce_on_plateau_min_scale,
        },
        'data': {
            'dataset': args.data,
        }
    }

    # Save config file
    config_filename = f'config-{run_uuid}.json'
    with open(config_filename, 'w') as f:
        json.dump(model_config, f, indent=2)
    print(f"\n✓ Model config saved to {config_filename}")

    # Save parameters with UUID
    params_filename = f'params-{run_uuid}.json'
    params_dict = to_jsonable(params)
    with open(params_filename, 'w') as f:
        json.dump(params_dict, f)
    print(f"✓ Parameters saved to {params_filename}")

    # Also save symlinks for convenience (params.json and config.json)
    try:
        if Path('params.json').exists():
            Path('params.json').unlink()
        if Path('config.json').exists():
            Path('config.json').unlink()
        Path('params.json').symlink_to(params_filename)
        Path('config.json').symlink_to(config_filename)
        print(f"✓ Created symlinks: params.json -> {params_filename}")
        print(f"✓ Created symlinks: config.json -> {config_filename}")
    except Exception as e:
        print(f"Note: Could not create symlinks: {e}")

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"UUID: {run_uuid}")
    print(f"Config: {config_filename}")
    print(f"Params: {params_filename}")
    print(f"{'='*60}")
