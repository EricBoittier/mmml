"""
Utilities for training the spooky PhysNetJAX model directly from TF-style datasets.

This module is intentionally self-contained and only depends on the spooky model
implementation, not on the standard EF model or cuEq code.

The expected single-example dictionary (e.g. from a TF dataset) looks like:

    {
        "atomic_numbers":  (N,),        # uint8 / int
        "positions":       (N, 3),      # float32
        "pbe0_energy":     (),          # float
        "pbe0_forces":     (N, 3),      # float32
        "charge":          (),          # int (total system charge Q)
        "multiplicity":    (),          # int (spin multiplicity S)
        ...
    }

We map this to the spooky model inputs:

  - atomic_numbers -> Z
  - positions      -> R
  - pbe0_energy    -> scalar energy target
  - pbe0_forces    -> forces target
  - charge         -> broadcast per-atom scalar feature (Q_atoms)
  - multiplicity   -> broadcast per-atom scalar feature (S_atoms)
"""

from __future__ import annotations

from typing import Any, Dict

import e3x
import jax
import jax.numpy as jnp


def _to_array(x: Any) -> Any:
    """
    Convert TF tensors or similar objects to plain NumPy/JAX arrays.

    Uses `.numpy()` when available (e.g. tf.Tensor), otherwise returns `x` as-is.
    """
    try:
        return x.numpy()
    except AttributeError:
        return x


def build_spooky_batch_from_example(
    example: Dict[str, Any],
    *,
    energy_key: str = "pbe0_energy",
    forces_key: str = "pbe0_forces",
) -> Dict[str, Any]:
    """
    Map a single dataset example dict to a spooky-model batch (batch_size = 1).

    Parameters
    ----------
    example
        A dict with at least the keys
        'atomic_numbers', 'positions', 'charge', 'multiplicity',
        and the keys specified by energy_key and forces_key.
        Values may be TF tensors or NumPy arrays.
    energy_key
        Key in example for the scalar energy target (default: "pbe0_energy").
        QCML dft_force_field uses "pbe0_energy"; other datasets may use "energy".
    forces_key
        Key in example for the forces target (default: "pbe0_forces").
        QCML uses "pbe0_forces"; other datasets may use "forces".

    Returns
    -------
    Dict[str, Any]
        Dictionary containing all tensors needed by the spooky EF model.
        batch_size is stored as a plain int for model.init compatibility.
    """
    Z_raw = _to_array(example["atomic_numbers"])
    R_raw = _to_array(example["positions"])
    E_raw = _to_array(example[energy_key])
    F_raw = _to_array(example[forces_key])

    Z = jnp.asarray(Z_raw, dtype=jnp.int32)          # (N,)
    R = jnp.asarray(R_raw, dtype=jnp.float32)        # (N, 3)
    E_ref = jnp.asarray(E_raw, dtype=jnp.float32)    # ()
    F_ref = jnp.asarray(F_raw, dtype=jnp.float32)    # (N, 3)

    N = Z.shape[0]
    batch_size = 1

    # System charge Q and spin multiplicity S -> broadcast per-atom scalar features
    Q = float(_to_array(example["charge"]))
    S = float(_to_array(example["multiplicity"]))

    # Per-atom conditioning as column vectors to keep shape explicit.
    Q_atoms = jnp.full((N, 1), Q, dtype=jnp.float32)
    S_atoms = jnp.full((N, 1), S, dtype=jnp.float32)

    # Graph indices for this single structure
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(N)  # (n_pairs,)

    batch_segments = jnp.zeros((N,), dtype=jnp.int32)      # all atoms belong to batch 0
    atom_mask = jnp.ones((N,), dtype=jnp.float32)          # no padding
    batch_mask = jnp.ones_like(dst_idx, dtype=jnp.float32) # all pairs valid

    batch = {
        "Z": Z,
        "R": R,
        "Q_atoms": Q_atoms,
        "S_atoms": S_atoms,
        "E": E_ref,
        "F": F_ref,
        "dst_idx": dst_idx,
        "src_idx": src_idx,
        "batch_segments": batch_segments,
        "batch_mask": batch_mask,
        "atom_mask": atom_mask,
        "batch_size": batch_size,  # plain int for model.init
    }
    return batch


def make_spooky_train_step(model, forces_weight: float = 52.91, energy_weight: float = 1.0):
    """
    Create a single optimisation step function for the spooky EF model.

    This does not depend on the standard EF model or cuEq code; it just calls
    ``model.apply`` with the fields produced by :func:`build_spooky_batch_from_example`.

    Parameters
    ----------
    model
        An instance of ``spooky_model.EF``.
    forces_weight
        Relative weight of the force MSE term in the loss.
    energy_weight
        Relative weight of the energy MSE term in the loss.
    """

    def loss_fn(params, batch):
        out = model.apply(
            params,
            atomic_numbers=batch["Z"],
            charges=batch["Q_atoms"],
            spins=batch["S_atoms"],
            positions=batch["R"],
            dst_idx=batch["dst_idx"],
            src_idx=batch["src_idx"],
            batch_segments=batch["batch_segments"],
            batch_size=batch["batch_size"],
            batch_mask=batch["batch_mask"],
            atom_mask=batch["atom_mask"],
        )

        # The spooky model's `energy` helper returns a negative total; its `__call__`
        # already takes care of that convention and exposes `out["energy"]` as the
        # quantity we should train against.
        E_pred = out["energy"].reshape(())               # scalar
        F_pred = out["forces"].reshape(batch["F"].shape) # (N, 3)

        e_loss = (E_pred - batch["E"]) ** 2
        f_loss = jnp.mean((F_pred - batch["F"]) ** 2)
        total = energy_weight * e_loss + forces_weight * f_loss
        return total, {"e_loss": e_loss, "f_loss": f_loss}

    def train_step(state, batch):
        (loss_val, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, batch
        )
        state = state.apply_gradients(grads=grads)
        return state, loss_val, metrics

    return train_step

