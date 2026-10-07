import sys

# Add custom path
sys.path.append("/home/boittier/jaxeq/dcmnet")

import functools

import ase
import jax
import jax.numpy as jnp
import optax

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


def blend_regression_loss(
    gt_loss: jnp.ndarray,
    teacher_loss: jnp.ndarray,
    alpha: float,
) -> jnp.ndarray:
    """Blend ground-truth and teacher regression losses.

    ``alpha=1`` uses ground truth only; ``alpha=0`` uses teacher only.
    """
    return alpha * gt_loss + (1.0 - alpha) * teacher_loss


def blend_component_loss(
    gt_loss: jnp.ndarray,
    teacher_loss: jnp.ndarray,
    alpha: float,
    distill: bool,
) -> jnp.ndarray:
    """Blend one loss component when distillation is enabled for that target."""
    if not distill:
        return gt_loss
    return blend_regression_loss(gt_loss, teacher_loss, alpha)


def mean_squared_loss_distill(
    energy_prediction: jnp.ndarray,
    forces_prediction: jnp.ndarray,
    energy_target_gt: jnp.ndarray,
    forces_target_gt: jnp.ndarray,
    energy_target_teacher: jnp.ndarray,
    forces_target_teacher: jnp.ndarray,
    energy_weight: float,
    forces_weight: float,
    atomic_mask: jnp.ndarray,
    distill_alpha: float,
    distill_energy: bool,
    distill_forces: bool,
) -> jnp.ndarray:
    """Energy/forces loss with optional per-target teacher distillation."""
    energy_gt = jnp.mean(
        optax.l2_loss(energy_prediction.flatten(), energy_target_gt.flatten())
    )
    energy_teacher = jnp.mean(
        optax.l2_loss(energy_prediction.flatten(), energy_target_teacher.flatten())
    )
    forces_gt = (
        jnp.sum(optax.l2_loss(forces_prediction.flatten(), forces_target_gt.flatten()))
        / atomic_mask.sum()
    )
    forces_teacher = (
        jnp.sum(
            optax.l2_loss(forces_prediction.flatten(), forces_target_teacher.flatten())
        )
        / atomic_mask.sum()
    )
    energy_loss = blend_component_loss(
        energy_gt, energy_teacher, distill_alpha, distill_energy
    )
    forces_loss = blend_component_loss(
        forces_gt, forces_teacher, distill_alpha, distill_forces
    )
    return energy_weight * energy_loss + forces_weight * forces_loss


def mean_squared_loss_QD_distill(
    energy_prediction: jnp.ndarray,
    forces_prediction: jnp.ndarray,
    dipole_prediction: jnp.ndarray,
    total_charges_prediction: jnp.ndarray,
    energy_target_gt: jnp.ndarray,
    forces_target_gt: jnp.ndarray,
    dipole_target_gt: jnp.ndarray,
    total_charge_target: jnp.ndarray,
    energy_target_teacher: jnp.ndarray,
    forces_target_teacher: jnp.ndarray,
    dipole_target_teacher: jnp.ndarray,
    energy_weight: float,
    forces_weight: float,
    dipole_weight: float,
    total_charge_weight: float,
    atomic_mask: jnp.ndarray,
    distill_alpha: float,
    distill_energy: bool,
    distill_forces: bool,
    distill_dipole: bool,
) -> jnp.ndarray:
    """Charge/dipole loss with optional per-target teacher distillation."""
    forces_prediction = forces_prediction * atomic_mask[..., None]
    forces_target_gt = forces_target_gt * atomic_mask[..., None]
    forces_target_teacher = forces_target_teacher * atomic_mask[..., None]

    energy_gt = (
        jnp.sum(optax.l2_loss(energy_prediction.flatten(), energy_target_gt.flatten()))
        / atomic_mask.sum()
    )
    energy_teacher = (
        jnp.sum(
            optax.l2_loss(energy_prediction.flatten(), energy_target_teacher.flatten())
        )
        / atomic_mask.sum()
    )
    forces_gt = (
        jnp.sum(optax.l2_loss(forces_prediction.flatten(), forces_target_gt.flatten()))
        / atomic_mask.sum()
        * 3
    )
    forces_teacher = (
        jnp.sum(
            optax.l2_loss(
                forces_prediction.flatten(), forces_target_teacher.flatten()
            )
        )
        / atomic_mask.sum()
        * 3
    )
    dipole_gt = (
        jnp.sum(optax.l2_loss(dipole_prediction.flatten(), dipole_target_gt.flatten()))
        / 3
    )
    dipole_teacher = (
        jnp.sum(
            optax.l2_loss(dipole_prediction.flatten(), dipole_target_teacher.flatten())
        )
        / 3
    )
    charges_loss = (
        jnp.sum(
            optax.l2_loss(
                total_charges_prediction.flatten(), total_charge_target.flatten()
            )
        )
        / atomic_mask.sum()
    )

    energy_loss = blend_component_loss(
        energy_gt, energy_teacher, distill_alpha, distill_energy
    )
    forces_loss = blend_component_loss(
        forces_gt, forces_teacher, distill_alpha, distill_forces
    )
    dipole_loss = blend_component_loss(
        dipole_gt, dipole_teacher, distill_alpha, distill_dipole
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
