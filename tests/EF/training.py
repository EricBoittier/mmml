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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".99"
# Workaround for: "Failed to capture gpu graph: the library was not initialized"
# You can drop one of these flags if your setup only needs the first.
#os.environ["XLA_FLAGS"] = "--xla_gpu_enable_cuda_graphs=false"

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import functools
import json

import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp

import optax
from flax import linen as nn

import e3x

import matplotlib.pyplot as plt  # optional; kept because you had it
import ase  # optional; kept because you had it
from ase.visualize import view as view  # optional; kept because you had it


print("JAX devices:", jax.devices())

# -------------------------
# Load dataset
# -------------------------
dataset = np.load("dataset1.npz", allow_pickle=True)
# Expected keys (based on your code): 'Z', 'R', 'Ef', 'E'
# Z: (num_data, N) int
# R: (num_data, N, 3) float
# Ef: (num_data, 3) float (or (num_data, 1, 3) — we assume (num_data, 3))
# E: (num_data,) or (num_data, 1)
print("Dataset keys:", dataset.files)
print("Dataset shapes:")
print(f"  R:  {dataset['R'].shape}")
print(f"  Z:  {dataset['Z'].shape}")
print(f"  Ef: {dataset['Ef'].shape}")
print(f"  E:  {dataset['E'].shape}")


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

    def EFD(self, atomic_numbers, positions, Ef, dst_idx, src_idx, batch_segments, batch_size):
        """
        Expected shapes:
          atomic_numbers: (B, N)
          positions:      (B, N, 3)
          Ef:             (B, 3)
          dst_idx/src_idx: (E,) indices for a single molecule of N atoms (no batching offsets)
          batch_segments: (B*N,) segment ids 0..B-1 repeated per atom
          batch_size:     int (static)
        Returns:
          proxy_energy: scalar (sum over batch)
          energy: (B,) per-molecule energy
        """
        # Basic dims
        B = batch_size
        N = 29

        # Flatten batch for message passing
        positions_flat = positions.reshape(-1, 3)           # (B*N, 3)
        atomic_numbers_flat = atomic_numbers.reshape(-1)    # (B*N,)

        # Build an EF tensor of shape compatible with e3x.nn.Tensor()
        # e3x format is (num_atoms, 1, (lmax+1)^2, features)
        ones = jnp.ones((B, 1), dtype=positions.dtype)
        xEF = jnp.concatenate((ones, Ef), axis=-1)   # (B, 4) - [1, Ef_x, Ef_y, Ef_z]
        xEF = xEF[:, None, :, None]                  # (B, 1, 4, 1)
        xEF = jnp.tile(xEF, (1, 1, 1, self.features)) # (B, 1, 4, features)
        # Expand to per-atom level using batch_segments
        xEF = xEF[batch_segments]                    # (B*N, 1, 4, features)

        # Offsets for batched edges
        offsets = jnp.arange(B, dtype=jnp.int32) * N
        dst_idx_flat = (dst_idx[None, :] + offsets[:, None]).reshape(-1)
        src_idx_flat = (src_idx[None, :] + offsets[:, None]).reshape(-1)

        # Displacements on flattened positions
        displacements = positions_flat[src_idx_flat] - positions_flat[dst_idx_flat]
        
        # Basis
        basis = e3x.nn.basis(
            displacements,
            num=self.num_basis_functions,
            max_degree=self.max_degree,
            radial_fn=e3x.nn.reciprocal_bernstein,
            cutoff_fn=functools.partial(e3x.nn.smooth_cutoff, cutoff=self.cutoff)
        )

        # Embed atoms (flattened)
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

            # Couple EF (kept close to your original structure)
            xEF_t = e3x.nn.Tensor()(x, xEF)
            x = e3x.nn.add(x, xEF_t)

            x = e3x.nn.TensorDense(max_degree=self.max_degree)(x)

        # Reduce to scalars per atom
        x = e3x.nn.change_max_degree_or_type(x, max_degree=0, include_pseudotensors=False)

        # Predict atomic energies
        element_bias = self.param(
            "element_bias",
            lambda rng, shape: jnp.zeros(shape, dtype=positions.dtype),
            (self.max_atomic_number + 1,)
        )

        atomic_energies = nn.Dense(1, use_bias=False, kernel_init=jax.nn.initializers.zeros)(x)
        atomic_energies = jnp.squeeze(atomic_energies, axis=(-1, -2, -3))  # (B*N,)

        # If you want element biases, do it in flattened space:
        # atomic_energies = atomic_energies + element_bias[atomic_numbers_flat]

        # Sum per molecule using batch_segments (length B*N)
        energy = jax.ops.segment_sum(atomic_energies, segment_ids=batch_segments, num_segments=B)  # (B,)

        # Proxy energy for force differentiation
        return -jnp.sum(energy), energy

    @nn.compact
    def __call__(self, atomic_numbers, positions, Ef, dst_idx, src_idx, batch_segments=None, batch_size=None):
        if batch_segments is None:
            # atomic_numbers expected (B,N); if B absent, treat as (N,)
            if atomic_numbers.ndim == 1:
                atomic_numbers = atomic_numbers[None, :]
                positions = positions[None, :, :]
                Ef = Ef[None, :]
            B, N = atomic_numbers.shape
            batch_size = B
            batch_segments = jnp.repeat(jnp.arange(B, dtype=jnp.int32), N)

        energy_and_forces = jax.value_and_grad(self.EFD, argnums=1, has_aux=True)
        (_, energy), forces = energy_and_forces(
            atomic_numbers, positions, Ef, dst_idx, src_idx, batch_segments, batch_size
        )
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

    batch_segments = jnp.repeat(jnp.arange(batch_size, dtype=jnp.int32), num_atoms)

    batches = []
    for perm in perms:
        batches.append(
            dict(
                atomic_numbers=data["atomic_numbers"][perm],   # (B,N)
                positions=data["positions"][perm],             # (B,N,3)
                energies=data["energies"][perm],               # (B,) or (B,1)
                electric_field=data["electric_field"][perm],   # (B,3)
                dst_idx=dst_idx,
                src_idx=src_idx,
                batch_segments=batch_segments,
            )
        )
    return batches


# -------------------------
# Train / Eval steps
# -------------------------
@functools.partial(jax.jit, static_argnames=("model_apply", "optimizer_update", "batch_size"))
def train_step(model_apply, optimizer_update, batch, batch_size, opt_state, params):
    def loss_fn(params):
        energy, forces = model_apply(
            params,
            atomic_numbers=batch["atomic_numbers"],
            positions=batch["positions"],
            Ef=batch["electric_field"],
            dst_idx=batch["dst_idx"],
            src_idx=batch["src_idx"],
            batch_segments=batch["batch_segments"],
            batch_size=batch_size,
        )
        loss = mean_squared_loss(energy.reshape(-1), batch["energies"].reshape(-1))
        return loss, (energy, forces)

    (loss, (energy, forces)), grad = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer_update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)

    mae = mean_absolute_error(energy, batch["energies"])
    return params, opt_state, loss, mae


@functools.partial(jax.jit, static_argnames=("model_apply", "batch_size"))
def eval_step(model_apply, batch, batch_size, params):
    energy, forces = model_apply(
        params,
        atomic_numbers=batch["atomic_numbers"],
        positions=batch["positions"],
        Ef=batch["electric_field"],
        dst_idx=batch["dst_idx"],
        src_idx=batch["src_idx"],
        batch_segments=batch["batch_segments"],
        batch_size=batch_size,
    )
    loss = mean_squared_loss(energy, batch["energies"])
    mae = mean_absolute_error(energy, batch["energies"])
    return loss, mae


def train_model(key, model, train_data, valid_data, num_epochs, learning_rate, batch_size):
    optimizer = optax.adam(learning_rate)

    # Initialize params with a single batch item (B=1) but correct ranks
    key, init_key = jax.random.split(key)
    num_atoms = 29 #train_data["positions"].shape[1]
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)

    # Use slicing [0:1] to keep batch dimension, not [0] which removes it
    atomic_numbers0 = train_data["atomic_numbers"][0:1]    # (1, N)
    positions0 = train_data["positions"][0:1]              # (1, N, 3)
    ef0 = train_data["electric_field"][0:1]                # (1, 3)
    batch_segments0 = jnp.repeat(jnp.arange(1, dtype=jnp.int32), num_atoms)

    params = model.init(
        init_key,
        atomic_numbers=atomic_numbers0,
        positions=positions0,
        Ef=ef0,
        dst_idx=dst_idx,
        src_idx=src_idx,
        batch_segments=batch_segments0,
        batch_size=1,
    )
    opt_state = optimizer.init(params)

    # Validation batches prepared once
    key, valid_key = jax.random.split(key)
    valid_batches = prepare_batches(valid_key, valid_data, batch_size)

    for epoch in range(1, num_epochs + 1):
        key, shuffle_key = jax.random.split(key)
        train_batches = prepare_batches(shuffle_key, train_data, batch_size)

        train_loss = 0.0
        train_mae = 0.0
        for i, batch in enumerate(train_batches):
            params, opt_state, loss, mae = train_step(
                model_apply=model.apply,
                optimizer_update=optimizer.update,
                batch=batch,
                batch_size=batch_size,
                opt_state=opt_state,
                params=params,
            )
            train_loss += (loss - train_loss) / (i + 1)
            train_mae += (mae - train_mae) / (i + 1)

        valid_loss = 0.0
        valid_mae = 0.0
        for i, batch in enumerate(valid_batches):
            loss, mae = eval_step(
                model_apply=model.apply,
                batch=batch,
                batch_size=batch_size,
                params=params,
            )
            valid_loss += (loss - valid_loss) / (i + 1)
            valid_mae += (mae - valid_mae) / (i + 1)

        print(f"epoch: {epoch:3d}                    train:   valid:")
        print(f"    loss [a.u.]             {train_loss: 8.6f} {valid_loss: 8.6f}")
        print(f"    energy mae [kcal/mol]   {train_mae: 8.6f} {valid_mae: 8.6f}")

    return params


# -------------------------
# Hyperparameters
# -------------------------
features = 64
max_degree = 1
num_iterations = 2
num_basis_functions = 64
cutoff = 5.0

num_train = 500
num_valid = 20

num_epochs = 100
learning_rate = 0.003
batch_size = 1

# -------------------------
# Run
# -------------------------
data_key, train_key = jax.random.split(jax.random.PRNGKey(0), 2)

train_data, valid_data = prepare_datasets(
    data_key, num_train=num_train, num_valid=num_valid, dataset=dataset
)

print("\nPrepared data shapes:")
print(f"  train atomic_numbers: {train_data['atomic_numbers'].shape}")
print(f"  train positions:      {train_data['positions'].shape}")
print(f"  train electric_field: {train_data['electric_field'].shape}")
print(f"  train energies:       {train_data['energies'].shape}")

message_passing_model = MessagePassingModel(
    features=features,
    max_degree=max_degree,
    num_iterations=num_iterations,
    num_basis_functions=num_basis_functions,
    cutoff=cutoff,
)

params = train_model(
    key=train_key,
    model=message_passing_model,
    train_data=train_data,
    valid_data=valid_data,
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    batch_size=batch_size,
)

