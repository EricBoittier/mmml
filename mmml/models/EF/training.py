from __future__ import annotations

import os

# --- Environment (must be set before importing jax) ---
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".99"

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import functools
import json
import sys
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

# ZBL repulsion (optional short-range nuclear repulsion)
try:
    from mmml.physnetjax.physnetjax.models.zbl import ZBLRepulsion
except ImportError:
    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from mmml.physnetjax.physnetjax.models.zbl import ZBLRepulsion

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

from mmml.data.units import ANGSTROM_TO_BOHR, EV_TO_KCAL_MOL, HARTREE_TO_EV
from mmml.utils.cli_args import exit_if_unknown_long_options
from mmml.utils.model_checkpoint import to_jsonable


def print_params_structure(params, label="params", max_depth=3, verbose=False):
    """Print the pytree structure of params for debugging ([STRUCT] lines)."""
    if not verbose:
        return

    def _print(obj, prefix="", depth=0):
        if depth > max_depth:
            print(f"{prefix}...")
            return
        if isinstance(obj, dict):
            for k, v in sorted(obj.items()):
                if isinstance(v, dict):
                    print(f"{prefix}{k}/ ({len(v)} keys)")
                    _print(v, prefix + "  ", depth + 1)
                elif hasattr(v, 'shape'):
                    print(f"{prefix}{k}: {type(v).__name__} shape={v.shape} dtype={v.dtype}")
                elif isinstance(v, (list, tuple)):
                    print(f"{prefix}{k}: {type(v).__name__} len={len(v)}")
                    for i, item in enumerate(v):
                        if hasattr(item, 'shape'):
                            print(f"{prefix}  [{i}]: {type(item).__name__} shape={item.shape} dtype={item.dtype}")
                        else:
                            print(f"{prefix}  [{i}]: {type(item).__name__}")
                else:
                    print(f"{prefix}{k}: {type(v).__name__} = {v}")
        elif hasattr(obj, 'shape'):
            print(f"{prefix}{type(obj).__name__} shape={obj.shape} dtype={obj.dtype}")
        else:
            print(f"{prefix}{type(obj).__name__}")
    print(f"\n[STRUCT] {label}:")
    print(f"[STRUCT]   top-level keys: {list(params.keys()) if isinstance(params, dict) else type(params)}")
    _print(params, prefix="[STRUCT]   ")


_FLAX_VARIABLE_COLLECTION_KEYS = frozenset({"params", "intermediates", "batch_stats"})


def sanitize_flax_variables_dict(params):
    """Keep only Flax variable collections; drop metadata (e.g. ``uuid``) from checkpoint JSON.

    Flax ``apply`` / ``jax.jit`` require variables to contain only known collections, not extra
    string keys at the top level.
    """
    if not isinstance(params, dict):
        return params
    out = {k: v for k, v in params.items() if k in _FLAX_VARIABLE_COLLECTION_KEYS}
    if "params" in out:
        return out
    if {"uuid", "model", "training", "data"}.intersection(params.keys()):
        raise ValueError(
            "This file looks like a CONFIG (keys uuid/model/training/data), not weights. "
            "Use a params JSON with top-level 'params' (e.g. params-<uuid>.json)."
        )
    leaves = jax.tree_util.tree_leaves(params)
    if not any(hasattr(leaf, "shape") for leaf in leaves):
        raise ValueError(
            "Params JSON has no top-level 'params' key and no array leaves. "
            f"Top-level keys: {list(params.keys())}"
        )
    return {"params": params}


def load_params(params_path, verbose=False):
    """Load parameters from JSON file.
    
    Handles the Flax sow() tuple-to-array conversion issue:
    Flax stores intermediates from sow() as tuples (array,), but JSON
    round-trip converts these to arrays with an extra dimension.
    This function strips the 'intermediates' key (which is not needed
    for inference or restart) to avoid pytree structure mismatches.
    """
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
    
    params = sanitize_flax_variables_dict(convert_to_jax(params_dict))
    
    # Strip 'intermediates' key if present — these are sow() artifacts that
    # get corrupted during JSON round-trip (tuples become arrays, changing
    # the pytree structure). They are recomputed during every forward pass
    # and are NOT needed for inference or restart.
    if isinstance(params, dict) and 'intermediates' in params:
        print(f"  Stripping 'intermediates' key from loaded params (sow artifacts)")
        params = {k: v for k, v in params.items() if k != 'intermediates'}
    
    print_params_structure(params, f"loaded from {params_path}", verbose=verbose)
    return params


def save_params_json(path: str | Path, params, *, verbose: bool = False) -> None:
    """Write params to JSON; strips ``intermediates`` (sow artifacts)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    params_to_save = params
    if isinstance(params, dict) and "intermediates" in params:
        params_to_save = {k: v for k, v in params.items() if k != "intermediates"}
        if verbose:
            print("  Stripped 'intermediates' from params before saving (sow artifacts)")
    if verbose:
        print_params_structure(params_to_save, f"params being saved -> {path}", verbose=True)
    jax.block_until_ready(params_to_save)
    with open(path, "w") as f:
        json.dump(to_jsonable(params_to_save), f)


print("JAX devices:", jax.devices())
# import lovely_jax as lj
# lj.monkey_patch()




def get_args(**overrides):
    """Parse command-line arguments, with optional keyword overrides (useful in notebooks).

    Usage:
        args = get_args()                          # CLI defaults
        args = get_args(features=32, cutoff=8.0)   # override specific params
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Single merged NPZ; random train/valid split via --num-train / --num-valid",
    )
    parser.add_argument(
        "--train-npz",
        type=str,
        default=None,
        help="Training split NPZ (R,Z,N,E,F,Ef[,Dxyz|D]) — use with --valid-npz instead of --data",
    )
    parser.add_argument(
        "--valid-npz",
        type=str,
        default=None,
        help="Validation split NPZ (same keys as train)",
    )
    parser.add_argument(
        "--test-npz",
        type=str,
        default=None,
        help="Optional test NPZ: only print shapes (not used for training)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory for params-*.json, config-*.json, and symlinks",
    )
    parser.add_argument("--features", type=int, default=10)
    parser.add_argument("--max_degree", type=int, default=4)
    parser.add_argument("--num_iterations", type=int, default=2)
    parser.add_argument("--num_basis_functions", type=int, default=10)
    parser.add_argument("--cutoff", type=float, default=10.0)
    
    parser.add_argument("--num_train", type=int, default=8000)
    parser.add_argument("--num_valid", type=int, default=1000)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.0004)
    parser.add_argument("--batch_size", type=int, default=256,
                       help="Batch size (default 256; use 128 or 64 if OOM)")

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
    parser.add_argument("--charge_weight", type=float, default=1000.0,
                       help="Weight for charge neutrality loss (sum of charges per molecule squared)")
    parser.add_argument("--dipole_field_coupling", action="store_true",
                       help="Add explicit E_total = E_nn + mu·Ef coupling")
    parser.add_argument("--field_scale", type=float, default=0.001,
                       help="Ef_phys = Ef_input * field_scale (au)")
    parser.add_argument("--zbl", action="store_true",
                       help="Add ZBL nuclear repulsion for short-range stability")
    parser.add_argument(
        "--include-pseudotensors",
        dest="include_pseudotensors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Equivariant parity dimension in e3x MessagePass / tensors (default: on)",
    )
    parser.add_argument("--gradient-checkpoint", action="store_true",
                       help="Use gradient checkpointing to reduce GPU memory (slower training)")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra debug output (e.g. [STRUCT] parameter tree dumps)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        metavar="N",
        help="Save EMA checkpoint every N epochs to params-epoch-NNNN-<uuid>.json (0 = no periodic saves)",
    )
    args, unknown = parser.parse_known_args()
    exit_if_unknown_long_options(unknown, prog="mmml ef-train")

    # Apply keyword overrides (for notebook usage)
    for key, value in overrides.items():
        if not hasattr(args, key):
            raise ValueError(f"Unknown argument: {key}")
        setattr(args, key, value)

    return args


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
    # ZBL: Ziegler-Biersack-Littmark nuclear repulsion for short-range stability
    zbl: bool = False

    def setup(self):
        if self.zbl:
            self.repulsion = ZBLRepulsion(cutoff=self.cutoff, trainable=True)

    def EFD(self, atomic_numbers, positions, Ef, dst_idx_flat, src_idx_flat, batch_segments, batch_size, dst_idx=None, src_idx=None):
        """
        Expected shapes:
          atomic_numbers: (B*N,) flattened
          positions:      (B*N, 3) flattened
          Ef:             (B, 3)
          dst_idx_flat/src_idx_flat: (B*E,) pre-computed flattened indices
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
        # Start with (B, 4) -> expand to (B*N, parity, 4, features) (parity 2 with pseudotensors, else 1)
        pad_ef = jnp.ones((B, 1), dtype=positions_flat.dtype)

        Ef = Ef * self.field_scale
        
        xEF = jnp.concatenate((pad_ef, Ef), axis=-1)   # (B, 4) - [1, Ef_x, Ef_y, Ef_z]
        xEF = xEF[:, None, :, None]                  # (B, 1, 4, 1)
        xEF = jnp.tile(xEF, (1, 1, 1, self.features)) # (B, 1, 4, features)
        # Expand to per-atom level: (B, 1, 4, features) -> (B, N, 1, 4, features) -> (B*N, 1, 4, features)
        # Insert dimension for N, then repeat: (B, 1, 4, features) -> (B, 1, 1, 4, features) -> (B, N, 1, 4, features)
        xEF = xEF[:, None, :, :]  # (B, 1, 1, 4, features) - add dimension
        xEF = jnp.repeat(xEF, N, axis=1)  # (B, N, 1, 4, features) - repeat N times
        xEF = xEF.reshape(B * N, 1, 4, self.features)  # (B*N, 1, 4, features)
        # Broadcast parity dim to match x (1 if no pseudotensors, 2 if include_pseudotensors)
        parity_dim = 2 if self.include_pseudotensors else 1
        xEF = jnp.broadcast_to(xEF, (B * N, parity_dim, 4, self.features))


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
                max_degree=self.max_degree
            )(x, basis, dst_idx=dst_idx_flat, src_idx=src_idx_flat)
            x = e3x.nn.add(x, y)
            x = e3x.nn.silu(x)
            # Couple EF - xEF shape matches x on the parity axis
            xEF = e3x.nn.Tensor()(x, xEF)
            x = e3x.nn.add(x, xEF)
            x = e3x.nn.TensorDense(max_degree=self.max_degree)(x)
            x = e3x.nn.add(x, y)
            
        # Save original x before reduction for dipole prediction
        x_orig = x  # (B*N, 2, (max_degree+1)^2, features)
        # Reduce to scalars per atom for energy prediction
        # Predict atomic charges (scalar per atom)
        # Use a separate branch from the same features
        x_charge = e3x.nn.change_max_degree_or_type(x, max_degree=0, include_pseudotensors=False)  # (B*N, 1, 1, features)

        atomic_charges = nn.Dense(1, use_bias=True, kernel_init=jax.nn.initializers.zeros)(x_charge)
        atomic_charges = e3x.nn.silu(atomic_charges)
        atomic_charges = jnp.squeeze(atomic_charges, axis=(-1, -2, -3))  # (B*N,)

        if self.max_degree > 0:
            # if max degree > 0
            # Predict atomic dipoles (3D vector per atom)
            # Use original x_orig and change max_degree to 1
            x_dipole = e3x.nn.change_max_degree_or_type(x_orig, max_degree=1, include_pseudotensors=False)
            # run through a tensor dense layer to get the dipole in the correct shape
            x_dipole = e3x.nn.TensorDense(max_degree=1)(x_dipole)
            x_dipole = e3x.nn.silu(x_dipole)
            # x_dipole shape: (B*N, parity, 4, features) where 4 = (lmax+1)^2 = (1+1)^2
            # Index 0: l=0 (scalar), indices 1-3: l=1 (dipole, 3 components)
            # Apply Dense to reduce features dimension: (B*N, parity, 4, features) -> (B*N, parity, 4, 1)
            x_dipole = e3x.nn.Dense(1, use_bias=False, kernel_init=jax.nn.initializers.zeros)(x_dipole)
            x_dipole = e3x.nn.silu(x_dipole)
            # Extract l=1 components (indices 1-3) and take real part (first parity dimension, index 0)
            # Shape: (B*N, parity, 3, 1) -> take parity=0 (real part) -> (B*N, 3, 1) -> squeeze -> (B*N, 3)
            atomic_dipoles = x_dipole[:, 0, 1:4, 0]  # (B*N, 3) - take first parity (real), l=1 components, squeeze features
            dipoles_batched = atomic_dipoles.reshape(B, N, 3)  # (B, N, 3)
        else:
            dipoles_batched = jnp.zeros((B, N, 3))



        # Compute molecular dipole: μ = Σ(q_i * (r_i - COM)) + Σ(μ_i)
        # Reshape to (B, N, 3) for positions and dipoles, (B, N) for charges
        positions_batched = positions_flat.reshape(B, N, 3)  # (B, N, 3)
        charges_batched = atomic_charges.reshape(B, N)  # (B, N)

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

        # Store atomic-level properties for downstream use (e.g. AAT)
        self.sow('intermediates', 'atomic_charges', charges_batched)      # (B, N)
        self.sow('intermediates', 'atomic_dipoles', dipoles_batched)      # (B, N, 3)

        # Predict atomic energies (reduce to scalar per atom like x_charge)
        x_energy = e3x.nn.change_max_degree_or_type(x, max_degree=0, include_pseudotensors=False)  # (B*N, 1, 1, features)
        element_bias = self.param(
            "element_bias",
            lambda rng, shape: jnp.zeros(shape, dtype=positions.dtype),
            (self.max_atomic_number + 1,)
        )
        atomic_energies = nn.Dense(1, use_bias=True, kernel_init=jax.nn.initializers.zeros)(x_energy)
        atomic_energies = jnp.squeeze(atomic_energies, axis=(-1, -2, -3))  # (B*N,)
        atomic_energies = atomic_energies + element_bias[atomic_numbers_flat]

        # ZBL nuclear repulsion (short-range, prevents atomic overlap)
        if self.zbl:
            distances = jnp.linalg.norm(displacements, axis=-1)  # (B*E,)
            distances = jnp.maximum(distances, 1e-8)
            atom_mask = jnp.ones(B * N, dtype=positions_flat.dtype)
            batch_mask = jnp.ones_like(distances, dtype=positions_flat.dtype)
            repulsion = self.repulsion(
                atomic_numbers_flat,
                distances,
                None,  # use ZBL internal switch
                None,
                dst_idx_flat,
                src_idx_flat,
                atom_mask,
                batch_mask,
                batch_segments,
                batch_size,
            )
            repulsion = jnp.squeeze(repulsion, axis=(-1, -2, -3))  # (B*N,)
            atomic_energies = atomic_energies + repulsion

        # Omit padding / inactive sites (Z<=0) from the total — same as ASE on
        # real atoms only; padded batches would otherwise double-count NN heads.
        valid_atom = (atomic_numbers_flat > 0).astype(atomic_energies.dtype)
        energy = (atomic_energies * valid_atom).reshape(B, N).sum(axis=1)  # (B,)

        # E_coul = (1/2) Σ q_i q_j / r_ij in Hartree with r_ij in Bohr; positions here are Å.
        r_ij_angstrom = jnp.linalg.norm(displacements, axis=-1)  # (B*E,)
        r_ij_bohr = r_ij_angstrom * ANGSTROM_TO_BOHR
        q_src = atomic_charges[src_idx_flat]  # (B*E,)
        q_dst = atomic_charges[dst_idx_flat]  # (B*E,)
        pair_coulomb_ha = (q_src * q_dst) / (r_ij_bohr + 1e-10)  # (B*E,) [Ha]
        # Neighbor list counts each undirected pair twice → divide by 2
        edge_batch = batch_segments[dst_idx_flat]  # (B*E,) batch index per edge
        coulomb_ha = jax.ops.segment_sum(pair_coulomb_ha, edge_batch, num_segments=B) / 2.0  # (B,)
        energy = energy - coulomb_ha * HARTREE_TO_EV  # Coulomb in eV to match NN / targets


        # Optional explicit dipole-field coupling:
        if self.dipole_field_coupling:
            coupling = jnp.sum( dipole * Ef , axis=-1)  # (B,)  mu·Ef_input
            coupling = coupling * self.field_scale * HARTREE_TO_EV
            energy = energy + coupling

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
def load_ef_npz(path: str | Path) -> dict:
    """
    Load one NPZ from fix-and-split / pyscf-evaluate style splits into the dict format
    expected by train_model / prepare_batches.

    Required keys: R, Z, E, F, Ef. Dipoles: D or Dxyz (same units as NPZ; often e·Å after fix-and-split), optional.

    Energy convention (E-field runs): default ``pyscf-evaluate`` adds the nuclear-field energy term
    after SCF (gpu4pyscf-style). Use ``--no-efield-include-nuclear-energy`` for legacy ``mf.kernel``-only
    energies. Match whatever you use when training.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"NPZ not found: {path}")
    raw = np.load(path, allow_pickle=True)

    def _need(k: str) -> np.ndarray:
        if k not in raw:
            raise KeyError(f"{path}: missing required array {k!r} (have {raw.files})")
        return np.asarray(raw[k])

    R = _need("R")
    Z = _need("Z")
    E = np.asarray(_need("E"), dtype=np.float64).ravel()
    F = _need("F")
    Ef = _need("Ef")

    R = np.asarray(R, dtype=np.float32)
    if R.ndim == 4 and R.shape[1] == 1:
        R = R[:, 0, :, :]
    if R.ndim != 3:
        raise ValueError(f"{path}: R must be (n, natoms, 3), got {R.shape}")

    F = np.asarray(F, dtype=np.float32)
    if F.ndim == 4 and F.shape[1] == 1:
        F = F[:, 0, :, :]
    if F.shape != R.shape:
        raise ValueError(f"{path}: F shape {F.shape} != R shape {R.shape}")

    Z = np.asarray(Z)
    if Z.ndim == 1:
        Z = np.broadcast_to(Z, (R.shape[0], Z.shape[0]))
    Z = np.asarray(Z, dtype=np.int32)
    if Z.shape[:2] != R.shape[:2]:
        raise ValueError(f"{path}: Z shape {Z.shape} incompatible with R {R.shape}")

    Ef = np.asarray(Ef, dtype=np.float32)
    if Ef.ndim == 1:
        Ef = np.broadcast_to(Ef, (R.shape[0], 3))
    if Ef.shape != (R.shape[0], 3):
        raise ValueError(f"{path}: Ef must be (n, 3), got {Ef.shape}")

    if E.shape[0] != R.shape[0]:
        raise ValueError(f"{path}: len(E)={E.shape[0]} != n_samples={R.shape[0]}")

    dip = None
    if "D" in raw:
        dip = np.asarray(raw["D"], dtype=np.float32)
    elif "Dxyz" in raw:
        dip = np.asarray(raw["Dxyz"], dtype=np.float32)
    if dip is not None:
        if dip.ndim == 3:
            dip = dip.squeeze(axis=1)
        if dip.shape != (R.shape[0], 3):
            raise ValueError(f"{path}: dipole shape {dip.shape}, expected ({R.shape[0]}, 3)")

    out = {
        "atomic_numbers": jnp.asarray(Z, dtype=jnp.int32),
        "positions": jnp.asarray(R, dtype=jnp.float32),
        "electric_field": jnp.asarray(Ef, dtype=jnp.float32),
        "energies": jnp.asarray(E, dtype=jnp.float32),
        "forces": jnp.asarray(F, dtype=jnp.float32),
    }
    if dip is not None:
        out["D"] = jnp.asarray(dip, dtype=jnp.float32)
    return out


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
    
    # Load dipoles if available (D or Dxyz from pyscf-evaluate / fix-and-split)
    dipoles = None
    dip_key = "D" if "D" in dataset else ("Dxyz" if "Dxyz" in dataset else None)
    if dip_key is not None:
        dipoles_raw = jnp.asarray(dataset[dip_key], dtype=jnp.float32)
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


def _flat_pairwise_indices(bs_b: int, num_atoms: int):
    """Flattened dst/src indices and batch_segments for a batch of ``bs_b`` systems."""
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)
    dst_idx = jnp.asarray(dst_idx, dtype=jnp.int32)
    src_idx = jnp.asarray(src_idx, dtype=jnp.int32)
    offsets = jnp.arange(bs_b, dtype=jnp.int32) * num_atoms
    dst_idx_flat = (dst_idx[None, :] + offsets[:, None]).reshape(-1)
    src_idx_flat = (src_idx[None, :] + offsets[:, None]).reshape(-1)
    batch_segments = jnp.repeat(jnp.arange(bs_b, dtype=jnp.int32), num_atoms)
    return dst_idx_flat, src_idx_flat, batch_segments


def prepare_batches(
    key,
    data,
    batch_size,
    num_atoms=None,
    dst_idx_flat=None,
    src_idx_flat=None,
    batch_segments=None,
    *,
    shuffle=True,
    drop_last=True,
):
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
    num_atoms : int, optional
        Atoms per system; defaults to the second axis of ``data['positions']``.
    dst_idx_flat, src_idx_flat, batch_segments : optional pre-computed arrays
        If provided, these are reused instead of recomputing (performance optimization).
        Must match ``batch_size`` and the atom count implied by ``data`` (see below).
    shuffle : bool, default True
        If True, shuffle sample order (training). If False, use frame order (evaluation).
    drop_last : bool, default True
        If True, drop the final partial batch (training default). If False, emit a smaller
        last batch so every sample is covered (evaluation / full-dataset inference).
    """
    n_atoms_data = int(data["positions"].shape[1])
    if num_atoms is None:
        num_atoms = n_atoms_data
    elif num_atoms != n_atoms_data:
        raise ValueError(
            f"prepare_batches: num_atoms={num_atoms} does not match data positions "
            f"(N={n_atoms_data} atoms per molecule)."
        )

    data_size = len(data["electric_field"])
    has_forces = data.get("forces") is not None
    if not has_forces:
        raise ValueError("prepare_batches: data['forces'] is required (use zeros if unused).")

    if shuffle:
        perm = jax.random.permutation(key, data_size)
    else:
        perm = jnp.arange(data_size, dtype=jnp.int32)

    if drop_last:
        n_used = (data_size // batch_size) * batch_size
        perm = perm[:n_used]
    else:
        n_used = data_size

    precomputed_ok = (
        dst_idx_flat is not None
        and src_idx_flat is not None
        and batch_segments is not None
    )
    if precomputed_ok:
        seg_atoms = int(batch_segments.shape[0]) // batch_size
        if seg_atoms != num_atoms:
            raise ValueError(
                "prepare_batches: precomputed batch_segments imply "
                f"{seg_atoms} atoms/molecule (batch_size={batch_size}), but data has "
                f"{num_atoms}. Build separate index tensors for train vs valid when "
                "atom counts differ."
            )

    has_dipoles = "D" in data and data["D"] is not None
    batches = []
    offset = 0
    while offset + batch_size <= n_used:
        idx = perm[offset : offset + batch_size]
        if precomputed_ok:
            d_flat, s_flat, b_seg = dst_idx_flat, src_idx_flat, batch_segments
        else:
            d_flat, s_flat, b_seg = _flat_pairwise_indices(batch_size, num_atoms)
        batch_dict = {
            "atomic_numbers": data["atomic_numbers"][idx].reshape(batch_size * num_atoms),
            "positions": data["positions"][idx].reshape(batch_size * num_atoms, 3),
            "energies": data["energies"][idx],
            "forces": data["forces"][idx].reshape(batch_size * num_atoms, 3),
            "electric_field": data["electric_field"][idx],
            "dst_idx_flat": d_flat,
            "src_idx_flat": s_flat,
            "batch_segments": b_seg,
        }
        if has_dipoles:
            batch_dict["dipoles"] = data["D"][idx]
        batches.append(batch_dict)
        offset += batch_size

    if offset < n_used:
        bs_rem = int(n_used - offset)
        idx = perm[offset:n_used]
        d_flat, s_flat, b_seg = _flat_pairwise_indices(bs_rem, num_atoms)
        batch_dict = {
            "atomic_numbers": data["atomic_numbers"][idx].reshape(bs_rem * num_atoms),
            "positions": data["positions"][idx].reshape(bs_rem * num_atoms, 3),
            "energies": data["energies"][idx],
            "forces": data["forces"][idx].reshape(bs_rem * num_atoms, 3),
            "electric_field": data["electric_field"][idx],
            "dst_idx_flat": d_flat,
            "src_idx_flat": s_flat,
            "batch_segments": b_seg,
        }
        if has_dipoles:
            batch_dict["dipoles"] = data["D"][idx]
        batches.append(batch_dict)

    return batches


# -------------------------
# Train / Eval steps
# -------------------------
@functools.partial(jax.jit, static_argnames=("model_apply", "optimizer_update", "batch_size", "ema_decay", "energy_weight", "forces_weight", "dipole_weight", "charge_weight", "gradient_checkpoint"))
def train_step(model_apply, optimizer_update, batch, batch_size, opt_state, params, ema_params, transform_state, ema_decay=0.999, energy_weight=1.0, forces_weight=100.0, dipole_weight=10.0, charge_weight=1.0, gradient_checkpoint=False):
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

        if gradient_checkpoint:
            try:
                policy = jax.checkpoint_policies.nothing_saveable
            except AttributeError:
                policy = None  # older JAX: default remat behavior
            energy_fn = jax.checkpoint(energy_fn, policy=policy)

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
        
        return total_loss, (energy, forces, dipole, charge_sum_sq, energy_loss, force_loss, dipole_loss)

    (loss, (energy, forces, dipole, charge_sum_sq, energy_loss, force_loss, dipole_loss)), grad = jax.value_and_grad(loss_fn, has_aux=True)(params)
    
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
    
    return (
        params,
        ema_params,
        opt_state,
        loss,
        energy_mae,
        forces_mae,
        dipole_mae,
        charge_sum_sq,
        energy_r2,
        forces_r2,
        dipole_r2,
        energy_loss,
        force_loss,
        dipole_loss,
    )


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
                energy_weight=1.0, forces_weight=100.0, dipole_weight=10.0, charge_weight=1.0, initial_params=None,
                gradient_checkpoint=False, verbose=False,
                checkpoint_dir: str | Path | None = None, run_uuid: str | None = None,
                save_every_n_epochs: int = 0, save_best: bool = True):
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
    verbose : bool, optional
        If True, print ``[STRUCT]`` parameter tree dumps (default: False).
    checkpoint_dir : str or Path, optional
        If set with ``run_uuid``, write checkpoints under this directory.
    run_uuid : str, optional
        Run id for checkpoint filenames (required if ``checkpoint_dir`` is set).
    save_every_n_epochs : int, optional
        If > 0, save current EMA params every N epochs (``params-epoch-NNNN-<uuid>.json``).
    save_best : bool, optional
        If True (default), when validation improves save ``params-best-<uuid>.json`` and
        ``best-valid-<uuid>.json`` (best weighted valid loss and epoch).
    
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
        
        # Detect if user accidentally loaded a config file instead of a params file.
        # Config files have keys like 'uuid', 'model', 'training', 'data'.
        # Params files have keys like 'params' (Flax format) or Dense/Embed layers (flat format).
        _config_keys = {'uuid', 'model', 'training', 'data'}
        if isinstance(initial_params, dict) and _config_keys.intersection(initial_params.keys()):
            raise ValueError(
                f"The loaded restart file appears to be a CONFIG file, not a PARAMS file.\n"
                f"  Top-level keys found: {list(initial_params.keys())}\n"
                f"  Config-like keys: {_config_keys.intersection(initial_params.keys())}\n"
                f"  Expected: a params file with top-level key 'params' (e.g., params-UUID.json)\n"
                f"  Hint: use --restart params-UUID.json, not config-UUID.json"
            )
        
        # Normalize initial_params to standard Flax format {'params': {...}}
        # The loaded params might be:
        #   (a) {'params': {...}}                          — standard Flax format
        #   (b) {'params': {...}, 'intermediates': {...}}   — full state (intermediates will be re-initialized)
        #   (c) {...flat weight dict...}                   — no 'params' wrapper (e.g., from a different pipeline)
        if isinstance(initial_params, dict) and 'params' not in initial_params:
            # Validate: check that leaves are array-like, not strings/dicts without arrays
            leaves = jax.tree_util.tree_leaves(initial_params)
            has_arrays = any(hasattr(leaf, 'shape') for leaf in leaves)
            if not has_arrays:
                raise ValueError(
                    f"Loaded restart file does not contain array data.\n"
                    f"  Top-level keys: {list(initial_params.keys())[:10]}\n"
                    f"  This doesn't look like a params file. Check the --restart path."
                )
            # Case (c): wrap flat weights in {'params': ...}
            print("  Note: loaded params have no 'params' key — wrapping as {'params': <loaded>}")
            initial_params = {'params': initial_params}
        
        # Ensure params have the correct structure for model.apply with mutable=['intermediates'].
        # If intermediates were stripped during load (correct behavior), we need to
        # re-initialize them by doing a dummy model.init and merging the weights.
        if isinstance(initial_params, dict) and 'intermediates' not in initial_params:
            print("  Re-initializing intermediates via model.init()...")
            key, init_key = jax.random.split(key)
            num_atoms = train_data["positions"].shape[1]
            dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)
            dst_idx = jnp.asarray(dst_idx, dtype=jnp.int32)
            src_idx = jnp.asarray(src_idx, dtype=jnp.int32)
            atomic_numbers0 = train_data["atomic_numbers"][0:1]
            positions0 = train_data["positions"][0:1]
            ef0 = train_data["electric_field"][0:1]
            batch_segments0 = jnp.repeat(jnp.arange(1, dtype=jnp.int32), num_atoms)
            offsets0 = jnp.arange(1, dtype=jnp.int32) * num_atoms
            dst_idx_flat0 = (dst_idx[None, :] + offsets0[:, None]).reshape(-1)
            src_idx_flat0 = (src_idx[None, :] + offsets0[:, None]).reshape(-1)
            ref_params = model.init(
                init_key, atomic_numbers=atomic_numbers0, positions=positions0, Ef=ef0,
                dst_idx_flat=dst_idx_flat0, src_idx_flat=src_idx_flat0,
                batch_segments=batch_segments0, batch_size=1, dst_idx=dst_idx, src_idx=src_idx,
            )
            # Use loaded weights but fresh intermediates structure
            params = dict(ref_params)  # copy structure from init
            params['params'] = initial_params['params']  # plug in loaded weights
            print("  ✓ Merged loaded weights with fresh intermediates")
            print_params_structure(params, "restart params (merged)", verbose=verbose)
        else:
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
        print_params_structure(params, "model.init() output", verbose=verbose)
    opt_state = optimizer.init(params)
    
    # Initialize transform state for reduce on plateau
    transform_state = transform.init(params)
    
    # Initialize EMA parameters
    ema_params = params

    # Pre-compute indices once per split (train vs valid can differ in N_atoms)
    def _batch_index_tensors(n_atoms: int):
        dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(n_atoms)
        dst_idx = jnp.asarray(dst_idx, dtype=jnp.int32)
        src_idx = jnp.asarray(src_idx, dtype=jnp.int32)
        offsets = jnp.arange(batch_size, dtype=jnp.int32) * n_atoms
        d_flat = (dst_idx[None, :] + offsets[:, None]).reshape(-1)
        s_flat = (src_idx[None, :] + offsets[:, None]).reshape(-1)
        b_seg = jnp.repeat(jnp.arange(batch_size, dtype=jnp.int32), n_atoms)
        return d_flat, s_flat, b_seg

    train_n = int(train_data["positions"].shape[1])
    valid_n = int(valid_data["positions"].shape[1])
    train_dst_idx_flat, train_src_idx_flat, train_batch_segments = _batch_index_tensors(train_n)
    valid_dst_idx_flat, valid_src_idx_flat, valid_batch_segments = _batch_index_tensors(valid_n)

    # Validation batches prepared once
    key, valid_key = jax.random.split(key)
    valid_batches = prepare_batches(
        valid_key,
        valid_data,
        batch_size,
        dst_idx_flat=valid_dst_idx_flat,
        src_idx_flat=valid_src_idx_flat,
        batch_segments=valid_batch_segments,
    )

    from mmml.physnetjax.physnetjax.data.data import print_shapes

    print_shapes(valid_batches[0], name="Validation Batch[0]")
    # print(
    #     "Loss terms: energy/force MSE use the same units as targets — typically E [eV], F [eV/Å]; "
    #     "dipole MSE is squared NPZ dipole units. Weighted total mixes terms via energy_weight, forces_weight, …"
    # )

    # Early stopping tracking
    best_valid_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_ema_params = ema_params

    _ckpt_dir = Path(checkpoint_dir).expanduser().resolve() if checkpoint_dir is not None else None
    _can_ckpt = _ckpt_dir is not None and run_uuid is not None
    if (save_every_n_epochs > 0 or save_best) and checkpoint_dir and not run_uuid:
        print("  Warning: checkpoint_dir set but run_uuid is None — skipping on-disk checkpoints.")

    for epoch in range(1, num_epochs + 1):
        key, shuffle_key = jax.random.split(key)
        train_batches = prepare_batches(
            shuffle_key,
            train_data,
            batch_size,
            dst_idx_flat=train_dst_idx_flat,
            src_idx_flat=train_src_idx_flat,
            batch_segments=train_batch_segments,
        )

        train_loss = 0.0
        train_energy_mae = 0.0
        train_forces_mae = 0.0
        train_dipole_mae = 0.0
        train_charge_loss = 0.0
        train_energy_loss = 0.0
        train_force_loss = 0.0
        train_dipole_loss = 0.0
        for i, batch in enumerate(train_batches):
            (
                params,
                ema_params,
                opt_state,
                loss,
                energy_mae,
                forces_mae,
                dipole_mae,
                charge_loss_val,
                energy_r2,
                forces_r2,
                dipole_r2,
                energy_loss_batch,
                force_loss_batch,
                dipole_loss_batch,
            ) = train_step(
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
                gradient_checkpoint=gradient_checkpoint,
            )
            # Don't block - let JAX execute asynchronously for maximum throughput
            # Only convert to Python float at the end for logging
            train_loss += (loss - train_loss) / (i + 1)
            train_energy_mae += (energy_mae - train_energy_mae) / (i + 1)
            train_forces_mae += (forces_mae - train_forces_mae) / (i + 1)
            train_dipole_mae += (dipole_mae - train_dipole_mae) / (i + 1)
            train_charge_loss += (charge_loss_val - train_charge_loss) / (i + 1)
            train_energy_loss += (energy_loss_batch - train_energy_loss) / (i + 1)
            train_force_loss += (force_loss_batch - train_force_loss) / (i + 1)
            train_dipole_loss += (dipole_loss_batch - train_dipole_loss) / (i + 1)

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
            best_epoch = epoch
            patience_counter = 0
            improved = True
        else:
            patience_counter += 1


        # Convert to Python floats only once for logging (performance optimization)
        valid_energy_mae_val = float(valid_energy_mae * EV_TO_KCAL_MOL)
        valid_forces_mae_val = float(valid_forces_mae * EV_TO_KCAL_MOL)
        train_energy_mae_val = float(train_energy_mae * EV_TO_KCAL_MOL)
        train_forces_mae_val = float(train_forces_mae * EV_TO_KCAL_MOL)
        
        # Block once at the end of epoch for logging (allows async execution during training)
        jax.block_until_ready(valid_loss)
        
        print(f"epoch: {epoch:3d}                    train:   valid:")
        print(f"    weighted total loss     {float(train_loss): 8.6f} {float(valid_loss): 8.6f}")
        print(f"    energy MSE [eV²]        {float(train_energy_loss): 8.6f} {float(valid_energy_loss): 8.6f}")
        print(f"    force MSE [(eV/Å)²]     {float(train_force_loss): 8.6f} {float(valid_force_loss): 8.6f}")
        print(f"    energy mae [kcal/mol]         {train_energy_mae_val: 8.6f} {valid_energy_mae_val: 8.6f}")
        print(f"    energy R²               {'N/A':>8s} {float(valid_energy_r2): 8.6f}")
        print(f"    forces mae [kcal/mol/Å]       {train_forces_mae_val: 8.6f} {valid_forces_mae_val: 8.6f}")
        print(f"    forces R²               {'N/A':>8s} {float(valid_forces_r2): 8.6f}")
        if train_dipole_mae > 0.0 or valid_dipole_mae > 0.0:
            print(f"    dipole mae [tgt units]  {float(train_dipole_mae): 8.6f} {float(valid_dipole_mae): 8.6f}")
            print(f"    dipole R²               {'N/A':>8s} {float(valid_dipole_r2): 8.6f}")
            print(f"    dipole MSE [tgt²]       {float(train_dipole_loss): 8.6f} {float(valid_dipole_loss): 8.6f}")
        print(f"    charge loss [e²]        {float(train_charge_loss): 8.6f} {float(valid_charge_loss): 8.6f}")
        print(f"    charge RMSE [e]         {float(jnp.sqrt(train_charge_loss)): 8.6f} {float(jnp.sqrt(valid_charge_loss)): 8.6f}")
        print(f"    LR scale: {float(lr_scale): 8.6f}, effective LR: {float(learning_rate * lr_scale): 8.6f}")
        if early_stopping_patience is not None:
            print(f"    best valid loss: {float(best_valid_loss): 8.6f}, patience: {patience_counter}/{early_stopping_patience}")
            if improved and not (_can_ckpt and save_best):
                print(f"    ✓ Improved!")

        # On-disk checkpoints (EMA weights + best validation metrics) after metrics log
        if _can_ckpt and save_every_n_epochs > 0 and epoch % save_every_n_epochs == 0:
            ck_path = _ckpt_dir / f"params-epoch-{epoch:04d}-{run_uuid}.json"
            save_params_json(ck_path, ema_params, verbose=False)
            print(f"    ✓ Periodic checkpoint: {ck_path.name}")
        if _can_ckpt and save_best and improved:
            best_path = _ckpt_dir / f"params-best-{run_uuid}.json"
            save_params_json(best_path, best_ema_params, verbose=False)
            metrics_path = _ckpt_dir / f"best-valid-{run_uuid}.json"
            with open(metrics_path, "w") as f:
                json.dump(
                    {
                        "best_valid_loss": float(best_valid_loss),
                        "best_epoch": int(best_epoch),
                        "uuid": run_uuid,
                    },
                    f,
                    indent=2,
                )
            print(
                f"    ✓ Best valid checkpoint: {best_path.name} "
                f"(weighted loss={float(best_valid_loss):.6f}, epoch {best_epoch})"
            )
        
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
    if args.gradient_checkpoint:
        print(f"  gradient_checkpoint: ON (reduces memory, ~2x slower)")
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
    print(f"  save_every: {args.save_every}")


    # -------------------------
    # Run
    # -------------------------
    data_key, train_key = jax.random.split(jax.random.PRNGKey(0), 2)

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if bool(args.train_npz) ^ bool(args.valid_npz):
        print(
            "Error: provide both --train-npz and --valid-npz (or use --data alone).",
            file=sys.stderr,
        )
        return None

    if args.train_npz and args.valid_npz:
        print(f"Loading splits:\n  train: {args.train_npz}\n  valid: {args.valid_npz}")
        train_data = load_ef_npz(args.train_npz)
        valid_data = load_ef_npz(args.valid_npz)
        args.num_train = int(train_data["positions"].shape[0])
        args.num_valid = int(valid_data["positions"].shape[0])
    elif args.data:
        dataset = np.load(args.data, allow_pickle=True)
        train_data, valid_data = prepare_datasets(
            data_key, num_train=args.num_train, num_valid=args.num_valid, dataset=dataset
        )
    else:
        print(
            "Error: provide either --data (single NPZ + random split) or both "
            "--train-npz and --valid-npz.",
            file=sys.stderr,
        )
        return None

    if args.test_npz:
        test_data = load_ef_npz(args.test_npz)
        print("\nTest split (informational only, not used in training):")
        for k, v in test_data.items():
            print(f"  {k}: {v.shape}")

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
        include_pseudotensors=args.include_pseudotensors,
        dipole_field_coupling=args.dipole_field_coupling,
        field_scale=args.field_scale,
        zbl=args.zbl,
    )


    print(f"Message passing model: {message_passing_model}")

    # Load restart parameters if provided
    initial_params = None
    if args.restart is not None:
        print(f"\nLoading restart parameters from {args.restart}...")
        initial_params = load_params(args.restart, verbose=args.verbose)
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
        gradient_checkpoint=args.gradient_checkpoint,
        verbose=args.verbose,
        checkpoint_dir=out_dir,
        run_uuid=run_uuid,
        save_every_n_epochs=args.save_every,
        save_best=True,
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
            'include_pseudotensors': args.include_pseudotensors,
            'dipole_field_coupling': args.dipole_field_coupling,
            'field_scale': args.field_scale,
            'zbl': args.zbl,
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
            'save_every': args.save_every,
        },
        'data': {
            'dataset': args.data,
            'train_npz': args.train_npz,
            'valid_npz': args.valid_npz,
            'test_npz': args.test_npz,
        }
    }

    # Save config file
    config_filename = str(out_dir / f"config-{run_uuid}.json")
    with open(config_filename, 'w') as f:
        json.dump(model_config, f, indent=2)
    print(f"\n✓ Model config saved to {config_filename}")

    # Save parameters with UUID (same stripping as save_params_json / checkpoints)
    params_filename = str(out_dir / f"params-{run_uuid}.json")
    save_params_json(params_filename, params, verbose=args.verbose)
    print(f"✓ Parameters saved to {params_filename}")
    
    # Verify round-trip: load back and check structure
    params_reloaded = load_params(params_filename, verbose=args.verbose)
    params_to_save = params
    if isinstance(params, dict) and "intermediates" in params:
        params_to_save = {k: v for k, v in params.items() if k != "intermediates"}
    # Quick sanity check: compare leaf values
    orig_leaves = jax.tree_util.tree_leaves(params_to_save)
    try:
        reload_leaves = jax.tree_util.tree_leaves(params_reloaded)
        if len(orig_leaves) != len(reload_leaves):
            print(f"⚠ WARNING: param tree leaf count mismatch! saved={len(orig_leaves)} reloaded={len(reload_leaves)}")
        else:
            max_diff = max(float(jnp.max(jnp.abs(jnp.asarray(a, dtype=jnp.float32) - jnp.asarray(b, dtype=jnp.float32)))) 
                         for a, b in zip(orig_leaves, reload_leaves) 
                         if hasattr(a, 'shape') and hasattr(b, 'shape') and a.shape == b.shape)
            print(f"✓ Round-trip check: {len(orig_leaves)} leaves, max diff = {max_diff:.2e}")
    except Exception as e:
        print(f"⚠ WARNING: round-trip structure mismatch: {e}")

    # Also save symlinks for convenience (params.json, config.json, best checkpoints)
    try:
        link_params = out_dir / "params.json"
        link_cfg = out_dir / "config.json"
        link_best = out_dir / "params-best.json"
        link_best_metrics = out_dir / "best-valid.json"
        if link_params.exists() or link_params.is_symlink():
            link_params.unlink()
        if link_cfg.exists() or link_cfg.is_symlink():
            link_cfg.unlink()
        if link_best.exists() or link_best.is_symlink():
            link_best.unlink()
        if link_best_metrics.exists() or link_best_metrics.is_symlink():
            link_best_metrics.unlink()
        link_params.symlink_to(Path(params_filename).name)
        link_cfg.symlink_to(Path(config_filename).name)
        best_params_name = f"params-best-{run_uuid}.json"
        best_metrics_name = f"best-valid-{run_uuid}.json"
        if (out_dir / best_params_name).is_file():
            link_best.symlink_to(best_params_name)
        if (out_dir / best_metrics_name).is_file():
            link_best_metrics.symlink_to(best_metrics_name)
        msg = f"✓ Created symlinks in {out_dir}: params.json, config.json"
        if (out_dir / best_params_name).is_file():
            msg += ", params-best.json"
        if (out_dir / best_metrics_name).is_file():
            msg += ", best-valid.json"
        print(msg)
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
    rc = 0 if main(args) is not None else 1
    sys.exit(rc)
