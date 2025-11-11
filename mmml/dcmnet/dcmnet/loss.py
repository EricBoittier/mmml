import functools

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.random import randint

from .electrostatics import batched_electrostatic_potential, calc_esp

from .utils import reshape_dipole


def pred_dipole(dcm, com, q):
    """
    Calculate molecular dipole moment from distributed multipoles.
    
    Parameters
    ----------
    dcm : array_like
        Distributed multipole positions, shape (N, 3)
    com : array_like
        Center of mass coordinates, shape (3,)
    q : array_like
        Charges/monopoles, shape (N,)
        
    Returns
    -------
    array_like
        Molecular dipole moment in Debye, shape (3,)
    """
    dipole_out = jnp.zeros(3)
    for i, _ in enumerate(dcm):
        dipole_out += q[i] * (_ - com)
    return dipole_out * 1.88873
    # return jnp.linalg.norm(dipole_out)* 4.80320


@functools.partial(jax.jit, static_argnames=("batch_size", "esp_w", "chg_w", "n_dcm"))
def esp_mono_loss(
    dipo_prediction,
    mono_prediction,
    esp_target,
    vdw_surface,
    mono,
    ngrid,
    n_atoms,
    batch_size,
    esp_w,
    chg_w,
    n_dcm,
):
    """
    Combined ESP and monopole loss function for DCMNet training.
    
    Computes loss as weighted sum of ESP fitting error and monopole
    constraint violation. Handles dummy atoms and charge neutrality.
    
    Parameters
    ----------
    dipo_prediction : array_like
        Predicted distributed dipole positions, shape (batch_size, natoms, n_dcm, 3)
    mono_prediction : array_like
        Predicted monopoles, shape (batch_size, natoms, n_dcm)
    esp_target : array_like
        Target ESP values, shape (batch_size, ngrid)
    vdw_surface : array_like
        VDW surface grid points, shape (batch_size, ngrid, 3)
    mono : array_like
        Reference monopoles, shape (batch_size, natoms)
    ngrid : array_like
        Number of grid points per system, shape (batch_size,)
    n_atoms : array_like
        Number of atoms per system, shape (batch_size,)
    batch_size : int
        Batch size
    esp_w : float
        Weight for ESP loss term
    n_dcm : int
        Number of distributed multipoles per atom
        
    Returns
    -------
    tuple
        (loss, esp_pred, esp_target, esp_errors) where:
        - loss: scalar total loss value
        - esp_pred: predicted ESP values at grid points
        - esp_target: target ESP values at grid points
        - esp_errors: per-grid-point errors
    """
    # sum_of_dc_monopoles = mono_prediction.sum(axis=-1)
    # l2_loss_mono = optax.l2_loss(sum_of_dc_monopoles, mono)
    # mono_loss = jnp.mean(l2_loss_mono)
    # d = jnp.moveaxis(dipo_prediction, -1, -2).reshape(batch_size, max_atoms * n_dcm, 3)
    # m = mono_prediction.reshape(batch_size, max_atoms * n_dcm)
    # batched_pred = batched_electrostatic_potential(d, m, vdw_surface)
    # l2_loss = optax.l2_loss(batched_pred, esp_target)
    # esp_loss = jnp.mean(l2_loss) * esp_w
    # return esp_loss + mono_loss
    # Ensure scalar n_atoms even if shaped (1,) or (1,1)
    n_atoms = jnp.ravel(n_atoms)[0]
    
    # Infer max_atoms from prediction shape
    # Handle both batched and unbatched/flattened predictions
    # Model can return either:
    # - Flattened: (batch_size * num_atoms, n_dcm) and (batch_size * num_atoms, n_dcm, 3)
    # - Batched: (batch_size, num_atoms, n_dcm) and (batch_size, num_atoms, n_dcm, 3)
    # - Unbatched (batch_size=1): (num_atoms, n_dcm) and (num_atoms, n_dcm, 3)
    
    mono_ndim = len(mono_prediction.shape)
    dipo_ndim = len(dipo_prediction.shape)
    
    if batch_size == 1 and mono_ndim == 2:
        # Unbatched: (n_atoms, n_dcm)
        max_atoms = mono_prediction.shape[0]
        # Add batch dimension
        mono_prediction = mono_prediction[None, :, :]  # (1, n_atoms, n_dcm)
        dipo_prediction = dipo_prediction[None, :, :, :]  # (1, n_atoms, n_dcm, 3)
    elif mono_ndim == 2 and dipo_ndim == 3:
        # Check if flattened: (batch_size * num_atoms, n_dcm) and (batch_size * num_atoms, n_dcm, 3)
        total_atoms_mono = mono_prediction.shape[0]
        total_atoms_dipo = dipo_prediction.shape[0]
        
        # If both are divisible by batch_size and match, they're flattened
        if (total_atoms_mono % batch_size == 0 and 
            total_atoms_dipo % batch_size == 0 and 
            total_atoms_mono == total_atoms_dipo):
            # Flattened format - reshape to batched
            max_atoms = total_atoms_mono // batch_size
            mono_prediction = mono_prediction.reshape(batch_size, max_atoms, n_dcm)
            dipo_prediction = dipo_prediction.reshape(batch_size, max_atoms, n_dcm, 3)
        else:
            # Batched: (batch_size, n_atoms, n_dcm)
            max_atoms = mono_prediction.shape[1]
    else:
        # Batched: (batch_size, n_atoms, n_dcm)
        max_atoms = mono_prediction.shape[1]
    
    d = jnp.moveaxis(dipo_prediction, -1, -2).reshape(batch_size, max_atoms * n_dcm, 3)
    m = mono_prediction.reshape(batch_size, max_atoms * n_dcm)

    # # 0 the charges for dummy atoms
    # NDC = n_atoms * n_dcm
    # valid_atoms = jnp.where(jnp.arange(max_atoms * n_dcm) < NDC, 1, 0)
    # d = d[0]
    # m = m[0] * valid_atoms
    # # constrain the net charge to 0.0
    # avg_chg = m.sum() / NDC
    # m = (m - avg_chg) * valid_atoms

    # monopole loss
    mono_prediction = m.reshape(max_atoms, n_dcm)
    sum_of_dc_monopoles = mono_prediction.sum(axis=-1)
    l2_loss_mono = optax.l2_loss(sum_of_dc_monopoles, mono)
    mono_loss_corrected = l2_loss_mono.sum() / jnp.maximum(n_atoms, 1.0)

    # esp_loss
    # batched_pred = batched_electrostatic_potential(d, m, vdw_surface)
    batched_pred = batched_electrostatic_potential(d, m, vdw_surface)
    # Ensure both arrays have same shape
    if batched_pred.ndim > 1:
        batched_pred = batched_pred.squeeze()
    if esp_target.ndim > 1:
        esp_target = esp_target.squeeze()
    
    # Compute per-grid-point errors (difference, not squared)
    esp_errors = batched_pred - esp_target
    
    l2_loss = optax.l2_loss(batched_pred, esp_target)
    # remove dummy grid points using actual grid length
    # n_points = l2_loss.shape[0]
    # ngrid_scalar = jnp.ravel(ngrid)[0]
    # valid = jnp.arange(n_points) < ngrid_scalar
    # valid_grids = jnp.where(valid, l2_loss, 0)
    esp_loss_corrected = l2_loss.sum() / jnp.ravel(ngrid)[0]
    total_loss = esp_loss_corrected * esp_w + mono_loss_corrected * chg_w
    
    return total_loss, batched_pred, esp_target, esp_errors


@functools.partial(jax.jit, static_argnames=("batch_size", "esp_w", "chg_w", "n_dcm"))
def dipo_esp_mono_loss(
    dipo_prediction,
    mono_prediction,
    esp_target,
    vdw_surface,
    mono,
    Dxyz,
    com,
    espMask,
    n_atoms,
    batch_size,
    esp_w,
    chg_w,
    n_dcm,
):
    """
    Dipole-augmented ESP and monopole loss function.
    
    Computes loss as weighted sum of ESP fitting error, monopole constraint
    violation, and dipole moment error. Includes charge neutrality and
    dummy atom handling.
    
    Parameters
    ----------
    dipo_prediction : array_like
        Predicted distributed dipole positions, shape (batch_size, natoms, n_dcm, 3)
    mono_prediction : array_like
        Predicted monopoles, shape (batch_size, natoms, n_dcm)
    esp_target : array_like
        Target ESP values, shape (batch_size, ngrid)
    vdw_surface : array_like
        VDW surface grid points, shape (batch_size, ngrid, 3)
    mono : array_like
        Reference monopoles, shape (batch_size, natoms)
    Dxyz : array_like
        Reference dipole moments, shape (batch_size, 3)
    com : array_like
        Center of mass coordinates, shape (batch_size, 3)
    espMask : array_like
        ESP evaluation masks, shape (batch_size, ngrid)
    n_atoms : array_like
        Number of atoms per system, shape (batch_size,)
    batch_size : int
        Batch size
    esp_w : float
        Weight for ESP loss term
    n_dcm : int
        Number of distributed multipoles per atom
        
    Returns
    -------
    tuple
        (esp_loss, mono_loss, dipole_loss)
    """
    # Infer max_atoms from prediction shape
    # Handle both batched and unbatched predictions
    if batch_size == 1 and len(mono_prediction.shape) == 2:
        # Unbatched: (n_atoms, n_dcm)
        max_atoms = mono_prediction.shape[0]
        # Add batch dimension
        mono_prediction = mono_prediction[None, :, :]  # (1, n_atoms, n_dcm)
        dipo_prediction = dipo_prediction[None, :, :, :]  # (1, n_atoms, n_dcm, 3)
    else:
        # Batched: (batch_size, n_atoms, n_dcm)
        max_atoms = mono_prediction.shape[1]
    
    d = jnp.moveaxis(dipo_prediction, -1, -2).reshape(batch_size, max_atoms * n_dcm, 3)
    m = mono_prediction.reshape(batch_size, max_atoms * n_dcm)

    # 0 the charges for dummy atoms
    # Ensure scalar n_atoms even if shaped (1,) or (1,1)
    n_atoms = jnp.ravel(n_atoms)[0]
    NDC = n_atoms * n_dcm
    valid_atoms = jnp.where(jnp.arange(max_atoms * n_dcm) < NDC, 1, 0)
    d = d[0]
    m = m[0] * valid_atoms
    # constrain the net charge to 0.0
    avg_chg = m.sum() / jnp.maximum(NDC, 1.0)
    m = (m - avg_chg) * valid_atoms

    # monopole loss
    mono_prediction = m.reshape(max_atoms, n_dcm)
    sum_of_dc_monopoles = mono_prediction.sum(axis=-1)
    l2_loss_mono = optax.l2_loss(sum_of_dc_monopoles, mono)
    mono_loss_corrected = l2_loss_mono.sum() / jnp.maximum(n_atoms, 1.0)

    # dipole loss
    molecular_dipole = pred_dipole(d, com[0], m)
    # jax.debug.print("{x} {y}", x=molecular_dipole, y=Dxyz[0])
    dipo_loss = optax.l2_loss(molecular_dipole, Dxyz[0]).sum()

    # esp_loss
    # batched_pred = batched_electrostatic_potential(d, m, vdw_surface)
    batched_pred = calc_esp(d, m, vdw_surface[0])
    l2_loss = optax.l2_loss(batched_pred, esp_target[0])
    # remove dummy grid points
    valid_grids = jnp.where(espMask[0], l2_loss, 0)
    esp_loss_corrected = valid_grids.sum() / espMask[0].sum()
    # jax.debug.print("{x} {y} {z}", x=esp_loss_corrected * esp_w, y=mono_loss_corrected, z=dipo_loss * 10)
    return esp_loss_corrected * esp_w * 0.0 , mono_loss_corrected*0.0 , dipo_loss * chg_w


def esp_mono_loss_pots(
    dipo_prediction, mono_prediction, vdw_surface, mono, batch_size, n_dcm
):
    """
    Compute ESP from distributed multipoles for loss calculation.
    
    Parameters
    ----------
    dipo_prediction : array_like
        Predicted distributed dipole positions
    mono_prediction : array_like
        Predicted monopoles
    vdw_surface : array_like
        VDW surface grid points
    mono : array_like
        Reference monopoles
    batch_size : int
        Batch size
    n_dcm : int
        Number of distributed multipoles per atom
        
    Returns
    -------
    array_like
        Predicted ESP values
    """
    # Infer max_atoms from prediction shape
    # Handle both batched and unbatched predictions
    if batch_size == 1 and len(mono_prediction.shape) == 2:
        # Unbatched: (n_atoms, n_dcm)
        max_atoms = mono_prediction.shape[0]
    elif len(mono_prediction.shape) > 1:
        max_atoms = mono_prediction.shape[1]
    else:
        max_atoms = mono_prediction.size // n_dcm
    
    return calc_esp(
        dipo_prediction, mono_prediction.reshape(batch_size, n_dcm * max_atoms), vdw_surface
    )


def esp_loss_pots(dipo_prediction, mono_prediction, vdw_surface, mono, batch_size):
    """
    Compute ESP from atomic monopoles for comparison.
    
    Parameters
    ----------
    dipo_prediction : array_like
        Predicted distributed dipole positions
    mono_prediction : array_like
        Predicted monopoles
    vdw_surface : array_like
        VDW surface grid points
    mono : array_like
        Reference monopoles
    batch_size : int
        Batch size
        
    Returns
    -------
    array_like
        Predicted ESP values from atomic monopoles
    """
    # Infer max_atoms from prediction shape
    # Handle both batched and unbatched predictions
    if batch_size == 1 and len(mono_prediction.shape) == 1:
        # Unbatched: (n_atoms,)
        max_atoms = mono_prediction.size
    elif len(mono_prediction.shape) > 1:
        max_atoms = mono_prediction.shape[1] if batch_size > 1 else mono_prediction.shape[0]
    else:
        max_atoms = mono_prediction.size // batch_size
    
    d = dipo_prediction.reshape(batch_size, max_atoms, 3)
    mono = mono.reshape(batch_size, max_atoms)
    m = mono_prediction.reshape(batch_size, max_atoms)
    batched_pred = batched_electrostatic_potential(d, m, vdw_surface)

    return batched_pred


def mean_absolute_error(prediction, target, batch_size):
    """
    Calculate mean absolute error for non-zero target values.
    
    Parameters
    ----------
    prediction : array_like
        Predicted values
    target : array_like
        Target values
    batch_size : int
        Batch size
        
    Returns
    -------
    float
        Mean absolute error
    """
    # Infer max_atoms from target shape
    if len(target.shape) > 1:
        max_atoms = target.shape[1] if batch_size > 1 else target.shape[0]
    else:
        max_atoms = target.size // batch_size if batch_size > 0 else target.size
    
    nonzero = jnp.nonzero(target, size=batch_size * max_atoms)
    return jnp.mean(jnp.abs(prediction[nonzero] - target[nonzero]))


def esp_loss_eval(pred, target, ngrid):
    """
    Evaluate ESP loss for non-zero target values.
    
    Parameters
    ----------
    pred : array_like
        Predicted ESP values
    target : array_like
        Target ESP values
    ngrid : int
        Number of grid points
        
    Returns
    -------
    float
        Root mean square error in kcal/mol
    """
    target = target.flatten()
    esp_non_zero = np.nonzero(target)
    l2_loss = optax.l2_loss(pred[esp_non_zero], target[esp_non_zero]) * 2
    esp_loss = np.mean(l2_loss) ** 0.5
    return esp_loss


def get_predictions(mono_dc2, dipo_dc2, batch, batch_size, n_dcm):
    """
    Get ESP predictions from both distributed and atomic monopoles.
    
    Parameters
    ----------
    mono_dc2 : array_like
        Distributed monopole predictions
    dipo_dc2 : array_like
        Distributed dipole predictions
    batch : dict
        Batch dictionary
    batch_size : int
        Batch size
    n_dcm : int
        Number of distributed multipoles per atom
        
    Returns
    -------
    tuple
        (esp_dc_pred, mono_pred) - ESP from distributed multipoles and atomic monopoles
    """
    mono = mono_dc2
    dipo = dipo_dc2

    esp_dc_pred = esp_mono_loss_pots(
        dipo, mono, batch["vdw_surface"], batch["mono"], batch_size, n_dcm
    )

    mono_pred = esp_loss_pots(
        batch["positions"],
        batch["mono"],
        batch["vdw_surface"],
        batch["mono"],
        batch_size,
    )
    return esp_dc_pred, mono_pred
