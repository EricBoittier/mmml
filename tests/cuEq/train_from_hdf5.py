#!/usr/bin/env python3
"""
Train the EnergyForceModel on HDF5 data.

Supports two workflows:

1. PhysNetJAX-style (prepare_h5_datasets + train_model_cueq):

    from mmml.physnetjax.physnetjax.data.read_h5 import prepare_h5_datasets
    from tests.cuEq.train_from_hdf5 import train_model_cueq

    train_data, valid_data, natoms = prepare_h5_datasets(
        key, filepath="qcell_dimers.h5",
        train_size=270_000, valid_size=30_706,
        charge_filter=0.0, natoms=34,
    )
    state, history = train_model_cueq(
        key, model, train_data, valid_data, natoms=natoms,
        num_epochs=10, batch_size=32,
    )

2. Single HDF5 trajectory (HDF5Reporter format):

    python tests/cuEq/train_from_hdf5.py --hdf5 trajectory.h5 ...
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
import h5py

# Make project root importable for mmml.* and tests.cuEq / cuEq
import sys as _sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in _sys.path:
    _sys.path.insert(0, str(ROOT))

try:
    from tests.cuEq.model import EnergyForceModel, energy_fn, forces_fn
except ModuleNotFoundError:
    from cuEq.model import EnergyForceModel, energy_fn, forces_fn

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


def create_train_state(
    rng: jax.Array,
    positions: jnp.ndarray,
    atomic_numbers: jnp.ndarray,
    total_charge: float,
    learning_rate: float,
) -> TrainState:
    """Initialise model parameters and optimizer."""
    example_pos = positions[0]  # (N, 3)

    model = EnergyForceModel()
    params = model.init(rng, example_pos, atomic_numbers, total_charge)

    tx = optax.adam(learning_rate)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def make_batched_energy_forces(atomic_numbers: jnp.ndarray, total_charge: float):
    """Return a vmapped energy/forces function over frames."""

    def single_eval(params, pos):
        e = energy_fn(params, pos, atomic_numbers, total_charge)
        f = forces_fn(params, pos, atomic_numbers, total_charge)
        return e, f

    return jax.vmap(single_eval, in_axes=(None, 0), out_axes=(0, 0))


def make_train_step(force_weight: float, atomic_numbers: jnp.ndarray, total_charge: float):
    batched_eval = make_batched_energy_forces(atomic_numbers, total_charge)

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
    atomic_numbers: jnp.ndarray,
    total_charge: float,
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
    # Atomic numbers and total charge are treated as fixed for the trajectory.
    # For now we assume they are stored as HDF5 attributes and pass them in
    # through the caller.
    rng = jax.random.PRNGKey(seed)
    rng, init_key = jax.random.split(rng)

    state = create_train_state(
        init_key,
        positions,
        atomic_numbers=atomic_numbers,
        total_charge=total_charge,
        learning_rate=learning_rate,
    )
    train_step = make_train_step(
        force_weight=force_weight,
        atomic_numbers=atomic_numbers,
        total_charge=total_charge,
    )

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
    atomic_numbers: jnp.ndarray | None = None,
    total_charge: float | None = None,
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
    >>> Z = jnp.array([1, 1, 8, 1, 1], dtype=jnp.int32)
    >>> state, history = train_from_hdf5("trajectory.h5", Z, total_charge=0.0, n_steps=1000)
    """
    hdf5_path = Path(hdf5)
    print(f"Loading HDF5 data from {hdf5_path} ...")
    positions, energies, forces = load_dataset(hdf5_path, limit=limit)
    n_frames, n_atoms, _ = positions.shape
    print(f"Loaded {n_frames} frames with {n_atoms} atoms.")

    # Try to infer atomic numbers and total charge from HDF5 attributes if not provided.
    with h5py.File(str(hdf5_path), "r") as f:
        if atomic_numbers is None:
            if "atomic_numbers" in f.attrs:
                atomic_numbers = jnp.asarray(f.attrs["atomic_numbers"], dtype=jnp.int32)
            else:
                raise ValueError(
                    "Atomic numbers must be provided either via the 'atomic_numbers' "
                    "HDF5 attribute or the atomic_numbers argument."
                )
        if total_charge is None:
            if "total_charge" in f.attrs:
                total_charge = float(f.attrs["total_charge"])
            else:
                total_charge = 0.0

    state, history = run_training(
        positions,
        energies,
        forces,
        atomic_numbers=atomic_numbers,
        total_charge=float(total_charge),
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        force_weight=force_weight,
        seed=seed,
        log_interval=log_interval,
    )
    return state, history


def train_model_cueq(
    key: jax.Array,
    model: EnergyForceModel,
    train_data: Dict[str, np.ndarray],
    valid_data: Dict[str, np.ndarray],
    *,
    natoms: int,
    num_epochs: int = 1,
    learning_rate: float = 1e-3,
    energy_weight: float = 1.0,
    forces_weight: float = 52.91,
    batch_size: int = 32,
    seed: int = 0,
    log_interval: int = 100,
) -> Tuple[TrainState, Dict[str, List[float]]]:
    """Train EnergyForceModel on data from prepare_h5_datasets.

    Consistent with PhysNetJAX training flow: use prepare_h5_datasets to load
    train_data/valid_data, then call this function.

    Parameters
    ----------
    key : jax.Array
        PRNG key for init and shuffling.
    model : EnergyForceModel
        cuEquivariance model instance.
    train_data, valid_data : dict
        Dicts with keys R, Z, F, E, N and optionally Q (from prepare_h5_datasets).
    natoms : int
        Max atoms per structure (padding size).
    num_epochs : int
        Number of training epochs.
    learning_rate : float
        Adam learning rate.
    energy_weight, forces_weight : float
        Loss weights (defaults match PhysNetJAX).
    batch_size : int
        Frames per step.
    seed : int
        PRNG seed (overrides key if key is from split).
    log_interval : int
        Steps between logging.

    Returns
    -------
    state : TrainState
        Final training state with trained params.
    history : dict
        Per-log-step losses.

    Example
    -------
    >>> key = jax.random.PRNGKey(40)
    >>> train_data, valid_data, natoms = prepare_h5_datasets(
    ...     key, filepath="qcell_dimers.h5",
    ...     train_size=270_000, valid_size=30_706,
    ...     charge_filter=0.0, natoms=34,
    ... )
    >>> model = EnergyForceModel()
    >>> state, history = train_model_cueq(
    ...     key, model, train_data, valid_data, natoms=natoms,
    ...     num_epochs=10, batch_size=32,
    ... )
    """
    R_train = jnp.asarray(train_data["R"])  # (n_train, natoms, 3)
    Z_train = jnp.asarray(train_data["Z"])  # (n_train, natoms)
    F_train = jnp.asarray(train_data["F"])  # (n_train, natoms, 3)
    E_train = jnp.asarray(train_data["E"])  # (n_train, 1)
    N_train = jnp.asarray(train_data["N"]).reshape(-1)  # (n_train,)
    Q_train = (
        jnp.asarray(train_data["Q"]).reshape(-1)
        if "Q" in train_data
        else jnp.zeros(R_train.shape[0])
    )

    n_train = R_train.shape[0]
    steps_per_epoch = max(1, n_train // batch_size)
    n_steps = num_epochs * steps_per_epoch

    # Atom mask: (natoms,) < N per sample -> (n_train, natoms)
    atom_masks_train = jnp.arange(natoms)[None, :] < N_train[:, None]

    # Build batched eval with per-sample Z, Q, mask
    def single_eval(params, pos, z, q, mask):
        e = energy_fn(params, pos, z, q, mask)
        f = forces_fn(params, pos, z, q, mask)
        return e, f

    batched_eval = jax.vmap(
        single_eval,
        in_axes=(None, 0, 0, 0, 0),
        out_axes=(0, 0),
    )

    def loss_fn(params, batch_R, batch_Z, batch_Q, batch_mask, batch_E, batch_F):
        pred_E, pred_F = batched_eval(params, batch_R, batch_Z, batch_Q, batch_mask)
        e_loss = jnp.mean((pred_E - batch_E.reshape(-1)) ** 2)
        # Force loss only over real atoms
        f_diff = (pred_F - batch_F) ** 2
        mask_3d = batch_mask[:, :, None]
        f_loss = jnp.sum(f_diff * mask_3d) / (jnp.sum(mask_3d) + 1e-9)
        total = energy_weight * e_loss + forces_weight * f_loss
        return total, {"e_loss": e_loss, "f_loss": f_loss}

    batch_size_actual = min(batch_size, n_train)

    @jax.jit
    def train_step(state, rng):
        replace = batch_size_actual >= n_train
        idx = jax.random.choice(rng, n_train, shape=(batch_size_actual,), replace=replace)
        batch_R = R_train[idx]
        batch_Z = Z_train[idx]
        batch_Q = Q_train[idx]
        batch_mask = atom_masks_train[idx]
        batch_E = E_train[idx]
        batch_F = F_train[idx]

        (loss_value, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, batch_R, batch_Z, batch_Q, batch_mask, batch_E, batch_F
        )
        state = state.apply_gradients(grads=grads)
        return state, loss_value, metrics

    rng = jax.random.PRNGKey(seed)
    rng, init_key = jax.random.split(rng)
    example_R = R_train[0]
    example_Z = Z_train[0]
    example_Q = float(Q_train[0])

    params = model.init(init_key, example_R, example_Z, example_Q)
    tx = optax.adam(learning_rate)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    history: Dict[str, List[float]] = {"step": [], "loss": [], "e_loss": [], "f_loss": []}
    print(f"Training cuEq model: {n_train} samples, {steps_per_epoch} steps/epoch, {n_steps} total steps")
    print("Starting training ...")

    for step in range(1, n_steps + 1):
        rng, subkey = jax.random.split(rng)
        state, loss_value, metrics = train_step(state, subkey)

        if step % log_interval == 0 or step == 1 or step == n_steps:
            loss_f = float(loss_value)
            e_loss = float(metrics["e_loss"])
            f_loss = float(metrics["f_loss"])
            print(
                f"Step {step:6d}  loss = {loss_f:.6f}  "
                f"E_loss = {e_loss:.6f}  F_loss = {f_loss:.6f}"
            )
            history["step"].append(step)
            history["loss"].append(loss_f)
            history["e_loss"].append(e_loss)
            history["f_loss"].append(f_loss)

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

