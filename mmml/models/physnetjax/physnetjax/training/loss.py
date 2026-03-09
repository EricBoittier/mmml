import sys

# Add custom path
sys.path.append("/home/boittier/jaxeq/dcmnet")

import functools

import ase
import e3x
import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from jax.random import randint
from optax import contrib
from optax import tree_utils as otu
from tqdm import tqdm

# from jax import config
# config.update('jax_enable_x64', True)


DTYPE = jnp.float32


def mean_squared_loss(
    energy_prediction: jnp.ndarray,
    energy_target: jnp.ndarray,
    forces_prediction: jnp.ndarray,
    energy_weight: float,
    forces_target: jnp.ndarray,
    forces_weight: float,
    atomic_mask: jnp.ndarray,
) -> float:
    """
    Calculate the mean squared loss for energy and forces predictions.

    Computes weighted sum of energy and force losses using L2 loss.
    Forces loss is normalized by the number of valid atoms.

    Parameters
    ----------
    energy_prediction : jnp.ndarray
        Predicted energy values
    energy_target : jnp.ndarray
        Target energy values
    forces_prediction : jnp.ndarray
        Predicted force values
    energy_weight : float
        Weight for energy loss term
    forces_target : jnp.ndarray
        Target force values
    forces_weight : float
        Weight for forces loss term
    atomic_mask : jnp.ndarray
        Mask indicating valid atoms

    Returns
    -------
    float
        Combined mean squared loss for energy and forces
    """
    energy_loss = jnp.mean(
        optax.l2_loss(energy_prediction.flatten(), energy_target.flatten())
    )
    forces_loss = (
        jnp.sum(optax.l2_loss(forces_prediction.flatten(), forces_target.flatten()))
        / atomic_mask.sum()
    )
    return energy_weight * energy_loss + forces_weight * forces_loss


def mean_squared_loss_D(
    energy_prediction: jnp.ndarray,
    energy_target: jnp.ndarray,
    forces_prediction: jnp.ndarray,
    forces_target: jnp.ndarray,
    forces_weight: float,
    dipole_prediction: jnp.ndarray,
    dipole_target: jnp.ndarray,
    dipole_weight: float,
) -> float:
    """
    Calculate the mean squared loss for energy, forces, and dipole predictions.

    Computes weighted sum of energy, force, and dipole losses using L2 loss.
    All losses are averaged over the batch.

    Parameters
    ----------
    energy_prediction : jnp.ndarray
        Predicted energy values
    energy_target : jnp.ndarray
        Target energy values
    forces_prediction : jnp.ndarray
        Predicted force values
    forces_target : jnp.ndarray
        Target force values
    forces_weight : float
        Weight for forces loss term
    dipole_prediction : jnp.ndarray
        Predicted dipole values
    dipole_target : jnp.ndarray
        Target dipole values
    dipole_weight : float
        Weight for dipole loss term

    Returns
    -------
    float
        Combined mean squared loss for energy, forces, and dipole
    """
    energy_loss = jnp.mean(
        optax.l2_loss(energy_prediction.squeeze(), energy_target.squeeze())
    )
    forces_loss = jnp.mean(
        optax.l2_loss(forces_prediction.squeeze(), forces_target.squeeze())
    )
    dipole_loss = jnp.mean(
        optax.l2_loss(dipole_prediction.squeeze(), dipole_target.squeeze())
    )
    return energy_loss + forces_weight * forces_loss + dipole_weight * dipole_loss


def mean_squared_loss_QD(
    energy_prediction: jnp.ndarray,
    energy_target: jnp.ndarray,
    energy_weight: float,
    forces_prediction: jnp.ndarray,
    forces_target: jnp.ndarray,
    forces_weight: float,
    dipole_prediction: jnp.ndarray,
    dipole_target: jnp.ndarray,
    dipole_weight: float,
    total_charges_prediction: jnp.ndarray,
    total_charge_target: jnp.ndarray,
    total_charge_weight: float,
    atomic_mask: jnp.ndarray,
) -> float:
    """
    Calculate the mean squared loss for energy, forces, dipole, and total charges predictions.

    Computes weighted sum of energy, force, dipole, and charge losses using L2 loss.
    Forces and charges are normalized by the number of valid atoms.
    Dipole loss is normalized by the number of components (3).

    Parameters
    ----------
    energy_prediction : jnp.ndarray
        Predicted energy values
    energy_target : jnp.ndarray
        Target energy values
    energy_weight : float
        Weight for energy loss term
    forces_prediction : jnp.ndarray
        Predicted force values
    forces_target : jnp.ndarray
        Target force values
    forces_weight : float
        Weight for forces loss term
    dipole_prediction : jnp.ndarray
        Predicted dipole values
    dipole_target : jnp.ndarray
        Target dipole values
    dipole_weight : float
        Weight for dipole loss term
    total_charges_prediction : jnp.ndarray
        Predicted total charges
    total_charge_target : jnp.ndarray
        Target total charges
    total_charge_weight : float
        Weight for total charges loss term
    atomic_mask : jnp.ndarray
        Mask for atomic positions

    Returns
    -------
    float
        Combined mean squared loss for energy, forces, dipole, and total charges
    """
    forces_prediction = forces_prediction * atomic_mask[..., None]
    forces_target = forces_target * atomic_mask[..., None]

    energy_loss = (
        jnp.sum(optax.l2_loss(energy_prediction.flatten(), energy_target.flatten()))
        / atomic_mask.sum()
    )
    forces_loss = (
        jnp.sum(optax.l2_loss(forces_prediction.flatten(), forces_target.flatten()))
        / atomic_mask.sum()
        * 3
    )
    dipole_loss = (
        jnp.sum(optax.l2_loss(dipole_prediction.flatten(), dipole_target.flatten())) / 3
    )
    charges_loss = (
        jnp.sum(
            optax.l2_loss(
                total_charges_prediction.flatten(), total_charge_target.flatten()
            )
        )
        / atomic_mask.sum()
    )
    return (
        energy_weight * energy_loss
        + forces_weight * forces_loss
        + dipole_weight * dipole_loss
        + total_charge_weight * charges_loss
    )


def mean_absolute_error(
    prediction: jnp.ndarray, target: jnp.ndarray, nsamples: int
) -> float:
    """
    Calculate the mean absolute error between prediction and target.

    Parameters
    ----------
    prediction : jnp.ndarray
        Predicted values
    target : jnp.ndarray
        Target values
    nsamples : int
        Number of samples for normalization

    Returns
    -------
    float
        Mean absolute error
    """
    return jnp.sum(jnp.abs(prediction.squeeze() - target.squeeze())) / nsamples


@functools.partial(jax.jit, static_argnames=("batch_size"))
def dipole_calc(
    positions: jnp.ndarray,
    atomic_numbers: jnp.ndarray,
    charges: jnp.ndarray,
    batch_segments: jnp.ndarray,
    batch_size: int,
) -> jnp.ndarray:
    """
    Calculate dipoles for a batch of molecules.

    Computes molecular dipole moments from atomic charges and positions
    relative to the center of mass of each molecule.

    Parameters
    ----------
    positions : jnp.ndarray
        Atomic positions, shape (N, 3)
    atomic_numbers : jnp.ndarray
        Atomic numbers, shape (N,)
    charges : jnp.ndarray
        Atomic charges, shape (N,)
    batch_segments : jnp.ndarray
        Batch segment indices, shape (N,)
    batch_size : int
        Number of molecules in the batch

    Returns
    -------
    jnp.ndarray
        Calculated dipoles for each molecule in the batch, shape (batch_size, 3)
    """
    charges = charges.squeeze()
    positions = positions.squeeze()
    atomic_numbers = atomic_numbers.squeeze()
    masses = jnp.take(ase.data.atomic_masses, atomic_numbers)
    bs_masses = jax.ops.segment_sum(
        masses, segment_ids=batch_segments, num_segments=batch_size
    )
    masses_per_atom = jnp.take(bs_masses, batch_segments)
    dis_com = positions * masses[..., None] / masses_per_atom[..., None]
    com = jnp.sum(dis_com, axis=1)
    pos_com = positions - com[..., None]
    dipoles = jax.ops.segment_sum(
        pos_com * charges[..., None],
        segment_ids=batch_segments,
        num_segments=batch_size,
    )
    return dipoles
