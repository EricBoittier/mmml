import os

from pandas._testing import at

# --- Environment (must be set before importing jax) ---
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".99"

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import functools
import json
import uuid
from pathlib import Path
from types import SimpleNamespace

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


def load_params(params_path):
    """Load parameters from JSON file."""
    with open(params_path, 'r') as f:
        params_dict = json.load(f)
    
    # Convert numpy arrays back from lists
    def convert_to_jax(obj):
        if isinstance(obj, dict):
            return {k: convert_to_jax(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            arr = np.array(obj)
            if arr.dtype == np.float64:
                return jnp.array(arr, dtype=jnp.float32)
            elif arr.dtype == np.int64:
                return jnp.array(arr, dtype=jnp.int32)
            return jnp.array(arr)
        return obj
    
    params = convert_to_jax(params_dict)
    return params


print("JAX devices:", jax.devices())


import lovely_jax as lj
lj.monkey_patch()




def get_args(**overrides):
    """Parse command-line arguments, with optional keyword overrides (useful in notebooks).

    Usage:
        args = get_args()                          # CLI defaults
        args = get_args(features=32, cutoff=8.0)   # override specific params
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data-full.npz")
    parser.add_argument("--features", type=int, default=10)
    parser.add_argument("--max_degree", type=int, default=4)
    parser.add_argument("--num_iterations", type=int, default=2)
    parser.add_argument("--num_basis_functions", type=int, default=10)
    parser.add_argument("--cutoff", type=float, default=10.0)
    
    parser.add_argument("--num_train", type=int, default=8000)
    parser.add_argument("--num_valid", type=int, default=1000)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.0004)
    parser.add_argument("--batch_size", type=int, default=1000)

    parser.add_argument("--clip_norm", type=float, default=10000.0)
    parser.add_argument("--ema_decay", type=float, default=0.5)
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0)
    parser.add_argument("--reduce_on_plateau_patience", type=int, default=15)
    parser.add_argument("--reduce_on_plateau_cooldown", type=int, default=15)
    parser.add_argument("--reduce_on_plateau_factor", type=float, default=0.9)
    parser.add_argument("--reduce_on_plateau_rtol", type=float, default=1e-4)
    parser.add_argument("--reduce_on_plateau_accumulation_size", type=int, default=5)
    parser.add_argument("--reduce_on_plateau_min_scale", type=float, default=0.01)

    parser.add_argument("--restart", type=str, default=None)

    parser.add_argument("--energy_weight", type=float, default=1.0,
                       help="Weight for energy loss in total loss")
    parser.add_argument("--forces_weight", type=float, default=100.0,
                       help="Weight for forces loss in total loss")
    parser.add_argument("--dipole_weight", type=float, default=0.1,
                       help="Weight for dipole loss in total loss")
    parser.add_argument("--charge_weight", type=float, default=10.0,
                       help="Weight for charge neutrality loss (sum of charges per molecule squared)")
    parser.add_argument("--dipole_field_coupling", action="store_true",
                       help="Add explicit E_total = E_nn + mu·Ef coupling")
    parser.add_argument("--field_scale", type=float, default=0.001,
                       help="Ef_phys = Ef_input * field_scale (au)")
    args, _ = parser.parse_known_args()

    # Apply keyword overrides (for notebook usage)
    for key, value in overrides.items():
        if not hasattr(args, key):
            raise ValueError(f"Unknown argument: {key}")
        setattr(args, key, value)

    return args


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


def mean_absolute_error_forces(prediction, target, mask=None):
    """Compute MAE for forces, optionally masked."""
    prediction = jnp.asarray(prediction)
    target = jnp.asarray(target)
    errors = prediction - target
    if mask is not None:
        errors = errors * mask
        count = mask.sum()
    else:
        count = errors.size
    return jnp.where(count > 0, jnp.sum(jnp.abs(errors)) / count, 0.0)


# -------------------------
# Model
# -------------------------
HARTREE_TO_EV = 27.211386245988


class MessagePassingModel(nn.Module):
    features: int = 32
    max_degree: int = 2
    num_iterations: int = 3
    num_basis_functions: int = 8
    cutoff: float = 5.0
    max_atomic_number: int = 55
    include_pseudotensors: bool = True
    # Explicit dipole-field coupling:  E_total = E_nn + mu·Ef  (in eV)
    # When False (default) the existing implicit coupling through features is used.
    dipole_field_coupling: bool = False
    field_scale: float = 0.001  # Ef_phys [au] = Ef_input * field_scale

    def EFD(self, atomic_numbers, positions, Ef, dst_idx_flat, src_idx_flat, batch_segments, batch_size, dst_idx=None, src_idx=None):
        """
        Expected shapes:
          atomic_numbers: (B*N,) flattened
          positions:      (B*N, 3) flattened
          Ef:             (B, 3)
          dst_idx_flat/src_idx_flat: (B*E,) pre-computed flattened indices (CUDA-graph-friendly)
          batch_segments: (B*N,) segment ids 0..B-1 repeated per atom
          batch_size:     int (static)
          dst_idx/src_idx: (E,) optional - only needed when batch_segments is None
        Returns:
          proxy_energy: scalar (sum over batch)
          energy: (B,) per-molecule energy


        """
        # Positions and atomic_numbers are already flattened in prepare_batches (like physnetjax)
        # Ensure they're properly shaped (1D for atomic_numbers, 2D for positions)
        positions_flat = positions.reshape(-1, 3)  # Ensure (B*N, 3)
        atomic_numbers_flat = atomic_numbers.reshape(-1)  # Ensure (B*N,) - 1D array
                # Basic dims: use static values (CUDA-graph-friendly)
        B = batch_size  # Static - known at compile time
        N = atomic_numbers_flat.shape[0] // B   # Static - constant number of atoms per molecule
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


        xEF = e3x.nn.change_max_degree_or_type(xEF, max_degree=self.max_degree,
         include_pseudotensors=self.include_pseudotensors)

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
            x = e3x.nn.silu(x)
            x = e3x.nn.Dense(self.features)(x)
            x = e3x.nn.silu(x)
            # Couple EF - xEF already has correct shape (B*N, 2, 4, features) matching x
            xEF = e3x.nn.Tensor()(x, xEF)
            x = e3x.nn.add(x, xEF)
            x = e3x.nn.TensorDense(max_degree=self.max_degree)(x)
            x = e3x.nn.add(x, y)

        for i in range(4):
            x = e3x.nn.Dense(self.features)(x)
            x = e3x.nn.silu(x)
        x = e3x.nn.Dense(self.features)(x)

        # Save original x before reduction for dipole prediction
        x_orig = x  # (B*N, 2, (max_degree+1)^2, features)
        
        # Reduce to scalars per atom for energy prediction
        x = e3x.nn.change_max_degree_or_type(x, max_degree=0, include_pseudotensors=False)

        # Predict atomic charges (scalar per atom)
        # Use a separate branch from the same features
        x_charge = x  # (B*N, 1, 1, features)

        atomic_charges = nn.Dense(1, use_bias=False, kernel_init=jax.nn.initializers.zeros)(x_charge)
        atomic_charges = jnp.squeeze(atomic_charges, axis=(-1, -2, -3))  # (B*N,)

        # Predict atomic dipoles (3D vector per atom)
        # Use original x_orig and change max_degree to 1
        x_dipole = e3x.nn.change_max_degree_or_type(x_orig, max_degree=1, include_pseudotensors=False)
        # run through a tensor dense layer to get the dipole in the correct shape
        x_dipole = e3x.nn.TensorDense(max_degree=1)(x_dipole)
        # x_dipole shape: (B*N, parity, 4, features) where 4 = (lmax+1)^2 = (1+1)^2
        # Index 0: l=0 (scalar), indices 1-3: l=1 (dipole, 3 components)
        # Apply Dense to reduce features dimension: (B*N, parity, 4, features) -> (B*N, parity, 4, 1)
        x_dipole = e3x.nn.Dense(1, use_bias=False, kernel_init=jax.nn.initializers.zeros)(x_dipole)
        # Extract l=1 components (indices 1-3) and take real part (first parity dimension, index 0)
        # Shape: (B*N, parity, 3, 1) -> take parity=0 (real part) -> (B*N, 3, 1) -> squeeze -> (B*N, 3)
        atomic_dipoles = x_dipole[:, 0, 1:4, 0]  # (B*N, 3) - take first parity (real), l=1 components, squeeze features
        # Compute molecular dipole: μ = Σ(q_i * (r_i - COM)) + Σ(μ_i)
        # Reshape to (B, N, 3) for positions and dipoles, (B, N) for charges
        positions_batched = positions_flat.reshape(B, N, 3)  # (B, N, 3)
        charges_batched = atomic_charges.reshape(B, N)  # (B, N)
        dipoles_batched = atomic_dipoles.reshape(B, N, 3)  # (B, N, 3)

        # Center of mass (using atomic masses or uniform weighting)
        # For simplicity, use uniform weighting (geometric center)
        com = positions_batched.mean(axis=1, keepdims=True)  # (B, 1, 3)
        positions_centered = positions_batched - com  # (B, N, 3)
        # Charge contribution: Σ(q_i * (r_i - COM))
        charge_dipole = jnp.sum(charges_batched[:, :, None] * positions_centered, axis=1)  # (B, 3)
        # Atomic dipole contribution: Σ(μ_i)
        atomic_dipole_sum = jnp.sum(dipoles_batched, axis=1)  # (B, 3)
        # Total molecular dipole
        dipole = charge_dipole + atomic_dipole_sum  # (B, 3)
        #dipole = charge_dipole  # physnet 
        #dipole =  atomic_dipole_sum  # direct eqv. pred

        # Store atomic-level properties for downstream use (e.g. AAT)
        self.sow('intermediates', 'atomic_charges', charges_batched)      # (B, N)
        self.sow('intermediates', 'atomic_dipoles', dipoles_batched)      # (B, N, 3)

        # Predict atomic energies
        element_bias = self.param(
            "element_bias",
            lambda rng, shape: jnp.zeros(shape, dtype=positions.dtype),
            (self.max_atomic_number + 1,)
        )
        atomic_energies = nn.Dense(1, use_bias=False, kernel_init=jax.nn.initializers.zeros)(x)
        atomic_energies = jnp.squeeze(atomic_energies, axis=(-1, -2, -3))  # (B*N,)
        atomic_energies = atomic_energies + element_bias[atomic_numbers_flat]
        energy = atomic_energies.reshape(B, N).sum(axis=1)  # (B,)

        # Optional explicit dipole-field coupling:
        #   E_total = E_nn + mu · Ef_phys   (converted to eV)
        # mu [e·Bohr] * Ef_phys [Hartree/(e·Bohr)] = [Hartree] -> * 27.21 -> [eV]
        if self.dipole_field_coupling:

            coupling = jnp.sum(dipole * Ef , axis=-1)  # (B,)  mu·Ef_input
            coupling = coupling * self.field_scale * HARTREE_TO_EV  # -> eV
            
            energy = energy - coupling

        # add a Coulomb term to the energy
        # Pairwise Coulomb: E_coul = 0.5 * Σ_{i≠j} q_i * q_j / r_ij
        r_ij = jnp.linalg.norm(displacements, axis=-1)  # (B*E,)
        q_src = atomic_charges[src_idx_flat]  # (B*E,)
        q_dst = atomic_charges[dst_idx_flat]  # (B*E,)
        pair_coulomb = q_src * q_dst / (r_ij + 1e-10)  # (B*E,)
        # Sum per molecule (each pair counted twice in neighbor list, so divide by 2)
        edge_batch = batch_segments[dst_idx_flat]  # (B*E,) batch index per edge
        coulomb_energy = jax.ops.segment_sum(pair_coulomb, edge_batch, num_segments=B) / 2.0  # (B,)
        energy = energy - coulomb_energy * HARTREE_TO_EV * 14.399645  # Coulomb constant in eV·Å/e²

        # Proxy energy for force differentiation
        return -jnp.sum(energy), energy, dipole

    @nn.compact
    def __call__(self, atomic_numbers, positions, Ef, dst_idx_flat=None, src_idx_flat=None, batch_segments=None, batch_size=None, dst_idx=None, src_idx=None):
        """Returns (energy, dipole) - use energy_and_forces() wrapper for forces."""
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
            if dst_idx is None or src_idx is None:
                raise ValueError("dst_idx and src_idx are required when batch_segments is None")
            offsets = jnp.arange(B, dtype=jnp.int32) * N
            dst_idx_flat = (dst_idx[None, :] + offsets[:, None]).reshape(-1)
            src_idx_flat = (src_idx[None, :] + offsets[:, None]).reshape(-1)

        proxy_energy, energy, dipole = self.EFD(
            atomic_numbers, positions, Ef, dst_idx_flat, src_idx_flat, batch_segments, batch_size, dst_idx=dst_idx, src_idx=src_idx
        )
        return energy, dipole


def energy_and_forces(model_apply, params, atomic_numbers, positions, Ef, dst_idx_flat, src_idx_flat, batch_segments, batch_size, dst_idx=None, src_idx=None):
    """Compute energy, forces (negative gradient of energy w.r.t. positions), and dipole."""
    def energy_fn(pos):
        energy, dipole = model_apply(
            params,
            atomic_numbers=atomic_numbers,
            positions=pos,
            Ef=Ef,
            dst_idx_flat=dst_idx_flat,
            src_idx_flat=src_idx_flat,
            batch_segments=batch_segments,
            batch_size=batch_size,
            dst_idx=dst_idx,
            src_idx=src_idx,
        )
        return -jnp.sum(energy), (energy, dipole)
    
    (_, (energy, dipole)), forces = jax.value_and_grad(energy_fn, has_aux=True)(positions)
    return energy, forces, dipole


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
    
    # Handle forces: F has shape (num_data, 1, N, 3) - squeeze out the extra dimension
    forces_raw = jnp.asarray(dataset["F"], dtype=jnp.float32)
    if forces_raw.ndim == 4 and forces_raw.shape[1] == 1:
        forces_raw = forces_raw.squeeze(axis=1)  # (num_data, N, 3)
    
    # Load dipoles if available (key "D" in dataset)
    dipoles = None
    if "D" in dataset:
        dipoles_raw = jnp.asarray(dataset["D"], dtype=jnp.float32)
        if dipoles_raw.ndim == 2 and dipoles_raw.shape[1] == 3:
            # Shape is (num_data, 3) - already correct
            dipoles = dipoles_raw
        elif dipoles_raw.ndim == 3:
            # Shape might be (num_data, 1, 3) - squeeze
            dipoles = dipoles_raw.squeeze()
        else:
            dipoles = dipoles_raw
    
    train_data = dict(
        atomic_numbers=jnp.asarray(dataset["Z"], dtype=jnp.int32)[train_choice],      # (num_train, N)
        positions=positions_raw[train_choice],                                         # (num_train, N, 3)
        electric_field=jnp.asarray(dataset["Ef"], dtype=jnp.float32)[train_choice],   # (num_train, 3)
        energies=jnp.asarray(dataset["E"], dtype=jnp.float32)[train_choice],          # (num_train,) or (num_train,1)
        forces=forces_raw[train_choice],                                               # (num_train, N, 3)
    )
    if dipoles is not None:
        train_data["D"] = dipoles[train_choice]  # (num_train, 3)
    
    valid_data = dict(
        atomic_numbers=jnp.asarray(dataset["Z"], dtype=jnp.int32)[valid_choice],
        positions=positions_raw[valid_choice],                                         # (num_valid, N, 3)
        electric_field=jnp.asarray(dataset["Ef"], dtype=jnp.float32)[valid_choice],
        energies=jnp.asarray(dataset["E"], dtype=jnp.float32)[valid_choice],
        forces=forces_raw[valid_choice],                                               # (num_valid, N, 3)
    )
    if dipoles is not None:
        valid_data["D"] = dipoles[valid_choice]  # (num_valid, 3)
    return train_data, valid_data


def prepare_batches(key, data, batch_size, num_atoms=29, dst_idx_flat=None, src_idx_flat=None, batch_segments=None):
    """
    Returns list of batch dicts with consistent shapes:
      atomic_numbers: (B*N,) flattened
      positions: (B*N,3) flattened
      electric_field: (B,3)
      energies: (B,) or (B,1)
      forces: (B*N,3) flattened
      dst_idx_flat/src_idx_flat: (B*E,) pre-computed flattened indices
      batch_segments: (B*N,)
    
    Parameters
    ----------
    dst_idx_flat, src_idx_flat, batch_segments : optional pre-computed arrays
        If provided, these are reused instead of recomputing (performance optimization)
    """
    data_size = len(data["electric_field"])
    steps_per_epoch = data_size // batch_size

    perms = jax.random.permutation(key, data_size)
    perms = perms[: steps_per_epoch * batch_size]  # drop incomplete last batch
    perms = perms.reshape((steps_per_epoch, batch_size))

    # Pre-compute indices only if not provided (performance optimization)
    if dst_idx_flat is None or src_idx_flat is None or batch_segments is None:
        dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)
        # Ensure these are jax arrays with explicit dtype
        dst_idx = jnp.asarray(dst_idx, dtype=jnp.int32)
        src_idx = jnp.asarray(src_idx, dtype=jnp.int32)

        # Pre-compute flattened indices (CUDA-graph-friendly: computed outside JIT)
        offsets = jnp.arange(batch_size, dtype=jnp.int32) * num_atoms
        dst_idx_flat = (dst_idx[None, :] + offsets[:, None]).reshape(-1)
        src_idx_flat = (src_idx[None, :] + offsets[:, None]).reshape(-1)

        batch_segments = jnp.repeat(jnp.arange(batch_size, dtype=jnp.int32), num_atoms)

    # Pre-allocate list for better performance
    batches = [None] * steps_per_epoch
    has_dipoles = "D" in data and data["D"] is not None
    
    for idx, perm in enumerate(perms):
        # Create batch dict directly (avoid intermediate variables)
        batch_dict = {
            "atomic_numbers": data["atomic_numbers"][perm].reshape(batch_size * num_atoms),
            "positions": data["positions"][perm].reshape(batch_size * num_atoms, 3),
            "energies": data["energies"][perm],
            "forces": data["forces"][perm].reshape(batch_size * num_atoms, 3),
            "electric_field": data["electric_field"][perm],
            "dst_idx_flat": dst_idx_flat,
            "src_idx_flat": src_idx_flat,
            "batch_segments": batch_segments,
        }
        if has_dipoles:
            batch_dict["dipoles"] = data["D"][perm]
        batches[idx] = batch_dict
    return batches


# -------------------------
# Train / Eval steps
# -------------------------
@functools.partial(jax.jit, static_argnames=("model_apply", "optimizer_update", "batch_size", "ema_decay", "energy_weight", "forces_weight", "dipole_weight", "charge_weight"))
def train_step(model_apply, optimizer_update, batch, batch_size, opt_state, params, ema_params, transform_state, ema_decay=0.999, energy_weight=1.0, forces_weight=100.0, dipole_weight=10.0, charge_weight=1.0):
    def loss_fn(params):
        # Forward pass with mutable intermediates to capture atomic charges
        def energy_fn(pos):
            (energy, dipole), state = model_apply(
                params,
                atomic_numbers=batch["atomic_numbers"],
                positions=pos,
                Ef=batch["electric_field"],
                dst_idx_flat=batch["dst_idx_flat"],
                src_idx_flat=batch["src_idx_flat"],
                batch_segments=batch["batch_segments"],
                batch_size=batch_size,
                mutable=['intermediates'],
            )
            return -jnp.sum(energy), (energy, dipole, state)

        (_, (energy, dipole, state)), forces = jax.value_and_grad(
            energy_fn, has_aux=True
        )(batch["positions"])

        # Charge neutrality loss: penalize non-zero sum of charges per molecule
        charges = state['intermediates']['atomic_charges'][-1]  # (B, N)
        charge_sum_sq = jnp.mean(charges.sum(axis=1)**2)
        
        # Energy loss
        energy_loss = mean_squared_loss(energy.reshape(-1), batch["energies"].reshape(-1))
        
        # Force loss
        force_loss = mean_squared_loss(forces, batch["forces"])
        
        # Dipole loss (if targets available, key "D" in batch)
        dipole_loss = 0.0
        if "dipoles" in batch:
            dipole_loss = mean_squared_loss(dipole, batch["dipoles"])
        
        # Combined loss with charge neutrality
        total_loss = energy_weight * energy_loss + forces_weight * force_loss + charge_weight * charge_sum_sq
        if "dipoles" in batch:
            total_loss = total_loss + dipole_weight * dipole_loss
        
        return total_loss, (energy, forces, dipole, charge_sum_sq)

    (loss, (energy, forces, dipole, charge_sum_sq)), grad = jax.value_and_grad(loss_fn, has_aux=True)(params)
    
    # Check for NaN/Inf in loss
    loss_finite = jnp.isfinite(loss)
    
    # Replace NaN/Inf gradients with zeros to prevent NaN propagation
    grad = jax.tree_util.tree_map(
        lambda g: jnp.where(jnp.isfinite(g), g, 0.0),
        grad
    )
    
    # Compute updates
    updates, opt_state = optimizer_update(grad, opt_state, params)
    
    # Apply learning rate scaling from reduce_on_plateau transform
    updates = otu.tree_scale(transform_state.scale, updates)
    
    # Check updates for NaN/Inf and replace with zeros
    updates = jax.tree_util.tree_map(
        lambda u: jnp.where(jnp.isfinite(u), u, 0.0),
        updates
    )
    
    # Only apply updates if loss was finite
    params = jax.tree_util.tree_map(
        lambda p, u: jnp.where(loss_finite, p + u, p),
        params,
        updates
    )

    # Update EMA parameters (only if loss was finite)
    ema_params = jax.tree_util.tree_map(
        lambda ema, new: jnp.where(
            loss_finite,
            ema_decay * ema + (1 - ema_decay) * new,
            ema  # Keep old EMA if loss was NaN
        ),
        ema_params,
        params,
    )
    
    # Ensure loss is finite for return
    loss = jnp.where(loss_finite, loss, 1e6)
    
    # Ensure outputs are finite before computing metrics
    energy = jnp.where(jnp.isfinite(energy), energy, 0.0)
    forces = jnp.where(jnp.isfinite(forces), forces, 0.0)
    dipole = jnp.where(jnp.isfinite(dipole), dipole, 0.0)

    energy_mae = mean_absolute_error(energy, batch["energies"])
    forces_mae = mean_absolute_error_forces(forces, batch["forces"])
    dipole_mae = mean_absolute_error(dipole, batch["dipoles"]) if "dipoles" in batch else 0.0
    
    # R² computation skipped for training batches (only computed for validation)
    energy_r2 = 0.0
    forces_r2 = 0.0
    dipole_r2 = 0.0
    
    return params, ema_params, opt_state, loss, energy_mae, forces_mae, dipole_mae, charge_sum_sq, energy_r2, forces_r2, dipole_r2


@functools.partial(jax.jit, static_argnames=("model_apply", "batch_size", "energy_weight", "forces_weight", "dipole_weight", "charge_weight"))
def eval_step(model_apply, batch, batch_size, params, energy_weight=1.0, forces_weight=100.0, dipole_weight=10.0, charge_weight=1.0):
    # Compute energy, dipole, and capture atomic charges via mutable intermediates
    (energy, dipole), state = model_apply(
        params,
        atomic_numbers=batch["atomic_numbers"],
        positions=batch["positions"],
        Ef=batch["electric_field"],
        dst_idx_flat=batch["dst_idx_flat"],
        src_idx_flat=batch["src_idx_flat"],
        batch_segments=batch["batch_segments"],
        batch_size=batch_size,
        mutable=['intermediates'],
    )
    
    # Compute forces
    _, forces, _ = energy_and_forces(
        model_apply, params,
        atomic_numbers=batch["atomic_numbers"],
        positions=batch["positions"],
        Ef=batch["electric_field"],
        dst_idx=None,  # Not needed when batch_segments is provided
        src_idx=None,  # Not needed when batch_segments is provided
        dst_idx_flat=batch["dst_idx_flat"],
        src_idx_flat=batch["src_idx_flat"],
        batch_segments=batch["batch_segments"],
        batch_size=batch_size,
    )
    
    # Extract charge neutrality metric
    charges = state['intermediates']['atomic_charges'][-1]  # (B, N)
    charge_sum_sq = jnp.mean(charges.sum(axis=1)**2)
    
    # Check for NaN/Inf and replace with zeros to prevent evaluation crash
    energy = jnp.where(jnp.isfinite(energy), energy, 0.0)
    forces = jnp.where(jnp.isfinite(forces), forces, 0.0)
    dipole = jnp.where(jnp.isfinite(dipole), dipole, 0.0)
    
    # Compute losses
    energy_loss = mean_squared_loss(energy.reshape(-1), batch["energies"].reshape(-1))
    force_loss = mean_squared_loss(forces, batch["forces"])
    dipole_loss = mean_squared_loss(dipole, batch["dipoles"]) if "dipoles" in batch else 0.0
    total_loss = energy_weight * energy_loss + forces_weight * force_loss + charge_weight * charge_sum_sq
    if "dipoles" in batch:
        total_loss = total_loss + dipole_weight * dipole_loss
    
    # Ensure loss is finite
    total_loss = jnp.where(jnp.isfinite(total_loss), total_loss, 1e6)
    

    # Compute MAEs
    energy_mae = mean_absolute_error(energy, batch["energies"])
    forces_mae = mean_absolute_error_forces(forces, batch["forces"])
    dipole_mae = mean_absolute_error(dipole, batch["dipoles"]) if "dipoles" in batch else 0.0
    
    # Compute R² for energy
    energy_pred_flat = energy.reshape(-1)
    energy_target_flat = batch["energies"].reshape(-1)
    energy_errors = energy_pred_flat - energy_target_flat
    ss_res_energy = jnp.sum(energy_errors**2)
    ss_tot_energy = jnp.sum((energy_target_flat - jnp.mean(energy_target_flat))**2)
    # Add small epsilon to prevent division by zero
    eps = 1e-10
    energy_r2 = jnp.where(ss_tot_energy > eps, 1.0 - (ss_res_energy / (ss_tot_energy + eps)), 0.0)
    
    # Compute R² for forces (flattened)
    forces_errors = forces - batch["forces"]
    ss_res_forces = jnp.sum(forces_errors**2)
    ss_tot_forces = jnp.sum((batch["forces"] - jnp.mean(batch["forces"]))**2)
    forces_r2 = jnp.where(ss_tot_forces > eps, 1.0 - (ss_res_forces / (ss_tot_forces + eps)), 0.0)
    
    # Compute R² for dipoles (if available)
    dipole_r2 = 0.0
    if "dipoles" in batch:
        dipole_errors = dipole - batch["dipoles"]
        ss_res_dipole = jnp.sum(dipole_errors**2)
        ss_tot_dipole = jnp.sum((batch["dipoles"] - jnp.mean(batch["dipoles"]))**2)
        dipole_r2 = jnp.where(ss_tot_dipole > eps, 1.0 - (ss_res_dipole / (ss_tot_dipole + eps)), 0.0)
    
    return total_loss, energy_loss, force_loss, dipole_loss, charge_sum_sq, energy_mae, forces_mae, dipole_mae, energy_r2, forces_r2, dipole_r2


def train_model(key, model, train_data, valid_data, num_epochs, learning_rate, batch_size, 
                clip_norm=10.0, ema_decay=0.999, early_stopping_patience=None, early_stopping_min_delta=0.0,
                reduce_on_plateau_patience=5, reduce_on_plateau_cooldown=5, reduce_on_plateau_factor=0.9,
                reduce_on_plateau_rtol=1e-4, reduce_on_plateau_accumulation_size=5, reduce_on_plateau_min_scale=0.01,
                energy_weight=1.0, forces_weight=100.0, dipole_weight=10.0, charge_weight=1.0, initial_params=None):
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
    initial_params : dict, optional
        Initial parameters to start training from (for restart).
        If None, parameters are initialized from scratch (default: None)
    
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

    # Initialize params - either from restart file or from scratch
    if initial_params is not None:
        print("  Restarting from provided parameters...")
        params = initial_params
    else:
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
            dst_idx_flat=dst_idx_flat0,
            src_idx_flat=src_idx_flat0,
            batch_segments=batch_segments0,
            batch_size=1,  # Init always uses batch_size=1
            dst_idx=dst_idx,
            src_idx=src_idx,
        )
    opt_state = optimizer.init(params)
    
    # Initialize transform state for reduce on plateau
    transform_state = transform.init(params)
    
    # Initialize EMA parameters
    ema_params = params

    # Pre-compute indices once (performance optimization - reuse across epochs)
    num_atoms = train_data["positions"].shape[1]
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)
    dst_idx = jnp.asarray(dst_idx, dtype=jnp.int32)
    src_idx = jnp.asarray(src_idx, dtype=jnp.int32)
    offsets = jnp.arange(batch_size, dtype=jnp.int32) * num_atoms
    dst_idx_flat = (dst_idx[None, :] + offsets[:, None]).reshape(-1)
    src_idx_flat = (src_idx[None, :] + offsets[:, None]).reshape(-1)
    batch_segments = jnp.repeat(jnp.arange(batch_size, dtype=jnp.int32), num_atoms)

    # Validation batches prepared once
    key, valid_key = jax.random.split(key)
    valid_batches = prepare_batches(valid_key, valid_data, batch_size, 
                                    dst_idx_flat=dst_idx_flat, 
                                    src_idx_flat=src_idx_flat,
                                    batch_segments=batch_segments)

    for k in valid_batches[0].keys():
        print(f"    valid batch {k}: {valid_batches[0][k]}")

    # Early stopping tracking
    best_valid_loss = float('inf')
    patience_counter = 0
    best_ema_params = ema_params

    for epoch in range(1, num_epochs + 1):
        key, shuffle_key = jax.random.split(key)
        train_batches = prepare_batches(shuffle_key, train_data, batch_size,
                                       dst_idx_flat=dst_idx_flat,
                                       src_idx_flat=src_idx_flat,
                                       batch_segments=batch_segments)

        train_loss = 0.0
        train_energy_mae = 0.0
        train_forces_mae = 0.0
        train_dipole_mae = 0.0
        train_charge_loss = 0.0
        for i, batch in enumerate(train_batches):
            params, ema_params, opt_state, loss, energy_mae, forces_mae, dipole_mae, charge_loss_val, energy_r2, forces_r2, dipole_r2 = train_step(
                model_apply=model.apply,
                optimizer_update=optimizer.update,
                batch=batch,
                batch_size=batch_size,
                opt_state=opt_state,
                params=params,
                ema_params=ema_params,
                transform_state=transform_state,
                ema_decay=ema_decay,
                energy_weight=energy_weight,
                forces_weight=forces_weight,
                dipole_weight=dipole_weight,
                charge_weight=charge_weight,
            )
            # Don't block - let JAX execute asynchronously for maximum throughput
            # Only convert to Python float at the end for logging
            train_loss += (loss - train_loss) / (i + 1)
            train_energy_mae += (energy_mae - train_energy_mae) / (i + 1)
            train_forces_mae += (forces_mae - train_forces_mae) / (i + 1)
            train_dipole_mae += (dipole_mae - train_dipole_mae) / (i + 1)
            train_charge_loss += (charge_loss_val - train_charge_loss) / (i + 1)

        valid_loss = 0.0
        valid_energy_loss = 0.0
        valid_force_loss = 0.0
        valid_dipole_loss = 0.0
        valid_charge_loss = 0.0
        valid_energy_mae = 0.0
        valid_forces_mae = 0.0
        valid_dipole_mae = 0.0
        valid_energy_r2 = 0.0
        valid_forces_r2 = 0.0
        valid_dipole_r2 = 0.0
        # Use EMA parameters for validation (as in physnetjax)
        for i, batch in enumerate(valid_batches):
            loss, energy_loss, force_loss, dipole_loss_batch, charge_loss_batch, energy_mae, forces_mae, dipole_mae, energy_r2, forces_r2, dipole_r2 = eval_step(
                model_apply=model.apply,
                batch=batch,
                batch_size=batch_size,
                params=ema_params,
                energy_weight=energy_weight,
                forces_weight=forces_weight,
                dipole_weight=dipole_weight,
                charge_weight=charge_weight,
            )
            # Don't block - let JAX execute asynchronously
            valid_loss += (loss - valid_loss) / (i + 1)
            valid_energy_loss += (energy_loss - valid_energy_loss) / (i + 1)
            valid_force_loss += (force_loss - valid_force_loss) / (i + 1)
            valid_dipole_loss += (dipole_loss_batch - valid_dipole_loss) / (i + 1)
            valid_charge_loss += (charge_loss_batch - valid_charge_loss) / (i + 1)
            valid_energy_mae += (energy_mae - valid_energy_mae) / (i + 1)
            valid_forces_mae += (forces_mae - valid_forces_mae) / (i + 1)
            valid_dipole_mae += (dipole_mae - valid_dipole_mae) / (i + 1)
            valid_energy_r2 += (energy_r2 - valid_energy_r2) / (i + 1)
            valid_forces_r2 += (forces_r2 - valid_forces_r2) / (i + 1)
            valid_dipole_r2 += (dipole_r2 - valid_dipole_r2) / (i + 1)

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


        # Convert to Python floats only once for logging (performance optimization)
        valid_energy_mae_val = float(valid_energy_mae * 23.0609)
        valid_forces_mae_val = float(valid_forces_mae * 23.0609)
        train_energy_mae_val = float(train_energy_mae * 23.0609)
        train_forces_mae_val = float(train_forces_mae * 23.0609)
        
        # Block once at the end of epoch for logging (allows async execution during training)
        jax.block_until_ready(valid_loss)
        
        print(f"epoch: {epoch:3d}                    train:   valid:")
        print(f"    loss [a.u.]             {float(train_loss): 8.6f} {float(valid_loss): 8.6f}")
        print(f"    energy loss [a.u.]      {float(train_energy_mae**2): 8.6f} {float(valid_energy_loss): 8.6f}")
        print(f"    force loss [a.u.]       {float(train_forces_mae**2): 8.6f} {float(valid_force_loss): 8.6f}")
        print(f"    energy mae [kcal/mol]         {train_energy_mae_val: 8.6f} {valid_energy_mae_val: 8.6f}")
        print(f"    energy R²               {'N/A':>8s} {float(valid_energy_r2): 8.6f}")
        print(f"    forces mae [kcal/mol/Å]       {train_forces_mae_val: 8.6f} {valid_forces_mae_val: 8.6f}")
        print(f"    forces R²               {'N/A':>8s} {float(valid_forces_r2): 8.6f}")
        if train_dipole_mae > 0.0 or valid_dipole_mae > 0.0:
            print(f"    dipole mae [Debye]      {float(train_dipole_mae): 8.6f} {float(valid_dipole_mae): 8.6f}")
            print(f"    dipole R²               {'N/A':>8s} {float(valid_dipole_r2): 8.6f}")
            print(f"    dipole loss [a.u.]      {float(train_dipole_mae**2): 8.6f} {float(valid_dipole_loss): 8.6f}")
        print(f"    charge loss [e²]        {float(train_charge_loss): 8.6f} {float(valid_charge_loss): 8.6f}")
        print(f"    charge RMSE [e]         {float(jnp.sqrt(train_charge_loss)): 8.6f} {float(jnp.sqrt(valid_charge_loss)): 8.6f}")
        print(f"    LR scale: {float(lr_scale): 8.6f}, effective LR: {float(learning_rate * lr_scale): 8.6f}")
        if early_stopping_patience is not None:
            print(f"    best valid loss: {float(best_valid_loss): 8.6f}, patience: {patience_counter}/{early_stopping_patience}")
            if improved:
                print(f"    ✓ Improved! Saving best model.")
        
        # Early stopping check
        if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs.")
            print(f"Best validation loss: {best_valid_loss: 8.6f} at epoch {epoch - patience_counter}")
            break

    return best_ema_params


def main(args=None):
    if args is None:
        args = get_args()

    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    # print the hyperparameters
    print("Hyperparameters:")
    print(f"  restart: {args.restart}")
    print(f"  features: {args.features}")
    print(f"  max_degree: {args.max_degree}")
    print(f"  num_iterations: {args.num_iterations}")
    print(f"  num_basis_functions: {args.num_basis_functions}")
    print(f"  cutoff: {args.cutoff}")
    print(f"  num_train: {args.num_train}")
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
    print(f"  train forces:        {train_data['forces'].shape}")

    message_passing_model = MessagePassingModel(
        features=args.features,
        max_degree=args.max_degree,
        num_iterations=args.num_iterations,
        num_basis_functions=args.num_basis_functions,
        cutoff=args.cutoff,
        dipole_field_coupling=args.dipole_field_coupling,
        field_scale=args.field_scale,
    )

    # Load restart parameters if provided
    initial_params = None
    if args.restart is not None:
        print(f"\nLoading restart parameters from {args.restart}...")
        initial_params = load_params(args.restart)
        print("✓ Restart parameters loaded")

    # Generate UUID for this training run
    run_uuid = str(uuid.uuid4())
    print(f"\n{'='*60}")
    print(f"Training Run UUID: {run_uuid}")
    if args.restart is not None:
        print(f"Restarting from: {args.restart}")
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
        energy_weight=args.energy_weight,
        forces_weight=args.forces_weight,
        dipole_weight=args.dipole_weight,
        charge_weight=args.charge_weight,
        initial_params=initial_params,
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
            'dipole_field_coupling': args.dipole_field_coupling,
            'field_scale': args.field_scale,
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
            'energy_weight': args.energy_weight,
            'forces_weight': args.forces_weight,
            'dipole_weight': args.dipole_weight,
            'charge_weight': args.charge_weight,
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
    
    return params


if __name__ == "__main__":
    args = get_args()
    main(args)
