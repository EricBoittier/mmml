#!/usr/bin/env python3
"""
Train the EnergyForceModel on an HDF5 trajectory.

The HDF5 file is expected to come from mmml.utils.hdf5_reporter.HDF5Reporter
and contain at least:
  - 'positions'        : (T, N, 3)
  - 'forces'           : (T, N, 3)
  - 'potential_energy' : (T,)

Usage
-----
From the repository root:

    python tests/cuEq/train_from_hdf5.py \\
        --hdf5 trajectory.h5 \\
        --n-steps 10000 \\
        --batch-size 32 \\
        --learning-rate 1e-3 \\
        --force-weight 1.0 \\
        --output-dir ckpts_cuEq
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state, checkpoints

# Make project root importable so we can import tests.cuEq.model
import sys as _sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in _sys.path:
    _sys.path.insert(0, str(ROOT))

from tests.cuEq.model import EnergyForceModel, energy_fn, forces_fn
from mmml.utils.hdf5_reporter import load_hdf5_trajectory


class TrainState(train_state.TrainState):
    """Simple TrainState with optax optimizer."""


def load_dataset(path: Path, limit: int | None = None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Load positions, energies, and forces from an HDF5 trajectory."""
    data = load_hdf5_trajectory(
        path,
        datasets=["positions", "forces", "potential_energy"],
    )

    if "positions" not in data or "forces" not in data or "potential_energy" not in data:
        raise KeyError(
            "HDF5 file must contain 'positions', 'forces', and 'potential_energy' datasets.\n"
            f"Found keys: {list(data.keys())}"
        )

    positions = np.asarray(data["positions"])       # (T, N, 3)
    forces = np.asarray(data["forces"])             # (T, N, 3)
    energies = np.asarray(data["potential_energy"]) # (T,)

    if limit is not None:
        positions = positions[:limit]
        forces = forces[:limit]
        energies = energies[:limit]

    return jnp.asarray(positions), jnp.asarray(energies), jnp.asarray(forces)


def create_train_state(rng: jax.Array, positions: jnp.ndarray, learning_rate: float) -> TrainState:
    """Initialise model parameters and optimizer."""
    n_atoms = positions.shape[1]
    example_pos = positions[0]  # (N, 3)

    model = EnergyForceModel()
    params = model.init(rng, example_pos)

    tx = optax.adam(learning_rate)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def make_batched_energy_forces():
    """Return a vmapped energy/forces function over frames."""

    def single_eval(params, pos):
        e = energy_fn(params, pos)
        f = forces_fn(params, pos)
        return e, f

    return jax.vmap(single_eval, in_axes=(None, 0), out_axes=(0, 0))


def make_train_step(force_weight: float):
    batched_eval = make_batched_energy_forces()

    def loss_fn(params, positions, energies_ref, forces_ref):
        pred_E, pred_F = batched_eval(params, positions)
        e_loss = jnp.mean((pred_E - energies_ref) ** 2)
        f_loss = jnp.mean((pred_F - forces_ref) ** 2)
        total = e_loss + force_weight * f_loss
        return total, {"e_loss": e_loss, "f_loss": f_loss}

    @jax.jit
    def train_step(state: TrainState, batch_pos, batch_E, batch_F, rng):
        (loss_value, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, batch_pos, batch_E, batch_F
        )
        state = state.apply_gradients(grads=grads)
        return state, loss_value, metrics

    return train_step


def sample_batch(
    rng: jax.Array,
    positions: jnp.ndarray,
    energies: jnp.ndarray,
    forces: jnp.ndarray,
    batch_size: int,
):
    """Uniformly sample a batch of frames."""
    n_frames = positions.shape[0]
    idx = jax.random.choice(rng, n_frames, shape=(batch_size,), replace=False)
    return positions[idx], energies[idx], forces[idx]


def run_training(
    positions: jnp.ndarray,
    energies: jnp.ndarray,
    forces: jnp.ndarray,
    *,
    n_steps: int = 10000,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    force_weight: float = 1.0,
    seed: int = 0,
    log_interval: int = 100,
) -> Tuple[TrainState, Dict[str, List[float]]]:
    """Core training loop shared by CLI and notebook usage.

    Returns the final TrainState and a history dict with per-log-step losses.
    """
    rng = jax.random.PRNGKey(seed)
    rng, init_key = jax.random.split(rng)

    state = create_train_state(init_key, positions, learning_rate=learning_rate)
    train_step = make_train_step(force_weight=force_weight)

    history: Dict[str, List[float]] = {"step": [], "loss": [], "e_loss": [], "f_loss": []}

    print("Starting training ...")
    for step in range(1, n_steps + 1):
        rng, subkey = jax.random.split(rng)
        batch_pos, batch_E, batch_F = sample_batch(
            subkey, positions, energies, forces, batch_size=batch_size
        )

        rng, subkey2 = jax.random.split(rng)
        state, loss_value, metrics = train_step(state, batch_pos, batch_E, batch_F, subkey2)

        if step % log_interval == 0 or step == 1 or step == n_steps:
            e_loss = float(metrics["e_loss"])
            f_loss = float(metrics["f_loss"])
            loss_f = float(loss_value)
            print(
                f"Step {step:6d}  loss = {loss_f:.6f}  "
                f"E_loss = {e_loss:.6f}  F_loss = {f_loss:.6f}"
            )
            history["step"].append(step)
            history["loss"].append(loss_f)
            history["e_loss"].append(e_loss)
            history["f_loss"].append(f_loss)

    return state, history


def train_from_hdf5(
    hdf5: Path | str,
    *,
    n_steps: int = 10000,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    force_weight: float = 1.0,
    limit: int | None = None,
    seed: int = 0,
    log_interval: int = 100,
) -> Tuple[TrainState, Dict[str, List[float]]]:
    """Notebook-friendly helper to train directly from an HDF5 file.

    Example
    -------
    >>> from tests.cuEq.train_from_hdf5 import train_from_hdf5
    >>> state, history = train_from_hdf5("trajectory.h5", n_steps=1000)
    """
    hdf5_path = Path(hdf5)
    print(f"Loading HDF5 data from {hdf5_path} ...")
    positions, energies, forces = load_dataset(hdf5_path, limit=limit)
    n_frames, n_atoms, _ = positions.shape
    print(f"Loaded {n_frames} frames with {n_atoms} atoms.")

    state, history = run_training(
        positions,
        energies,
        forces,
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        force_weight=force_weight,
        seed=seed,
        log_interval=log_interval,
    )
    return state, history


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train cuEquivariance EnergyForceModel from HDF5.")
    p.add_argument("--hdf5", type=Path, required=True, help="HDF5 trajectory file.")
    p.add_argument("--n-steps", type=int, default=10000, help="Number of optimisation steps.")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size (frames per step).")
    p.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    p.add_argument("--force-weight", type=float, default=1.0, help="Relative weight of force MSE.")
    p.add_argument("--limit", type=int, default=None, help="Optional max number of frames to use.")
    p.add_argument("--output-dir", type=Path, default=Path("ckpts_cuEq"), help="Checkpoint directory.")
    p.add_argument("--seed", type=int, default=0, help="PRNG seed.")
    p.add_argument("--log-interval", type=int, default=100, help="Steps between logging.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    state, _ = train_from_hdf5(
        args.hdf5,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        force_weight=args.force_weight,
        limit=args.limit,
        seed=args.seed,
        log_interval=args.log_interval,
    )

    # Final checkpoint
    print(f"Saving checkpoint to {args.output_dir} ...")
    checkpoints.save_checkpoint(
        ckpt_dir=str(args.output_dir),
        target=state.params,
        step=args.n_steps,
        prefix="cuEq_",
        overwrite=True,
    )
    print("Done.")


if __name__ == "__main__":
    main()

