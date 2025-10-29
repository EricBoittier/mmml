import functools

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.random import randint

from dcmnet.electrostatics import batched_electrostatic_potential, calc_esp
from dcmnet.modules import NATOMS
from dcmnet.utils import reshape_dipole


def pred_dipole(dcm, com, q):
    dipole_out = jnp.zeros(3)
    for i, _ in enumerate(dcm):
        dipole_out += q[i] * (_ - com)
    return dipole_out * 1.88873
    # return jnp.linalg.norm(dipole_out)* 4.80320


@functools.partial(jax.jit, static_argnames=("batch_size", "esp_w", "n_dcm"))
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
    n_dcm,
):
    """ """
    # sum_of_dc_monopoles = mono_prediction.sum(axis=-1)
    # l2_loss_mono = optax.l2_loss(sum_of_dc_monopoles, mono)
    # mono_loss = jnp.mean(l2_loss_mono)
    # d = jnp.moveaxis(dipo_prediction, -1, -2).reshape(batch_size, NATOMS * n_dcm, 3)
    # m = mono_prediction.reshape(batch_size, NATOMS * n_dcm)
    # batched_pred = batched_electrostatic_potential(d, m, vdw_surface)
    # l2_loss = optax.l2_loss(batched_pred, esp_target)
    # esp_loss = jnp.mean(l2_loss) * esp_w
    # return esp_loss + mono_loss
    # Ensure scalar n_atoms even if shaped (1,) or (1,1)
    n_atoms = jnp.ravel(n_atoms)[0]
    d = jnp.moveaxis(dipo_prediction, -1, -2).reshape(batch_size, NATOMS * n_dcm, 3)
    m = mono_prediction.reshape(batch_size, NATOMS * n_dcm)

    # 0 the charges for dummy atoms
    NDC = n_atoms * n_dcm
    valid_atoms = jnp.where(jnp.arange(60 * n_dcm) < NDC, 1, 0)
    d = d[0]
    m = m[0] * valid_atoms
    # constrain the net charge to 0.0
    avg_chg = m.sum() / NDC
    m = (m - avg_chg) * valid_atoms

    # monopole loss
    mono_prediction = m.reshape(NATOMS, n_dcm)
    sum_of_dc_monopoles = mono_prediction.sum(axis=-1)
    l2_loss_mono = optax.l2_loss(sum_of_dc_monopoles, mono)
    mono_loss_corrected = l2_loss_mono.sum() / n_atoms

    # esp_loss
    # batched_pred = batched_electrostatic_potential(d, m, vdw_surface)
    batched_pred = calc_esp(d, m, vdw_surface[0])
    l2_loss = optax.l2_loss(batched_pred, esp_target[0])
    # remove dummy grid points using actual grid length
    n_points = l2_loss.shape[0]
    valid = jnp.arange(n_points) < ngrid[0]
    valid_grids = jnp.where(valid, l2_loss, 0)
    esp_loss_corrected = valid_grids.sum() / ngrid[0]
    return esp_loss_corrected * esp_w + mono_loss_corrected


@functools.partial(jax.jit, static_argnames=("batch_size", "esp_w", "n_dcm"))
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
    n_dcm,
):
    """ """
    d = jnp.moveaxis(dipo_prediction, -1, -2).reshape(batch_size, NATOMS * n_dcm, 3)
    m = mono_prediction.reshape(batch_size, NATOMS * n_dcm)

    # 0 the charges for dummy atoms
    # Ensure scalar n_atoms even if shaped (1,) or (1,1)
    n_atoms = jnp.ravel(n_atoms)[0]
    NDC = n_atoms * n_dcm
    valid_atoms = jnp.where(jnp.arange(60 * n_dcm) < NDC, 1, 0)
    d = d[0]
    m = m[0] * valid_atoms
    # constrain the net charge to 0.0
    avg_chg = m.sum() / NDC
    m = (m - avg_chg) * valid_atoms

    # monopole loss
    mono_prediction = m.reshape(NATOMS, n_dcm)
    sum_of_dc_monopoles = mono_prediction.sum(axis=-1)
    l2_loss_mono = optax.l2_loss(sum_of_dc_monopoles, mono)
    mono_loss_corrected = l2_loss_mono.sum() / n_atoms

    # dipole loss
    molecular_dipole = pred_dipole(d, com, m)
    # jax.debug.print("{x} {y}", x=molecular_dipole[0], y=Dxyz[0])
    dipo_loss = optax.l2_loss(molecular_dipole[0], Dxyz[0]).sum()

    # esp_loss
    # batched_pred = batched_electrostatic_potential(d, m, vdw_surface)
    batched_pred = calc_esp(d, m, vdw_surface[0])
    l2_loss = optax.l2_loss(batched_pred, esp_target[0])
    # remove dummy grid points
    valid_grids = jnp.where(espMask[0], l2_loss, 0)
    esp_loss_corrected = valid_grids.sum() / espMask[0].sum()
    # jax.debug.print("{x} {y} {z}", x=esp_loss_corrected * esp_w, y=mono_loss_corrected, z=dipo_loss * 10)
    return esp_loss_corrected * esp_w * 0.0 , mono_loss_corrected*0.0 , dipo_loss 


def esp_mono_loss_pots(
    dipo_prediction, mono_prediction, vdw_surface, mono, batch_size, n_dcm
):
    """ """
    return calc_esp(
        dipo_prediction, mono_prediction.reshape(batch_size, n_dcm * 60), vdw_surface
    )


def esp_loss_pots(dipo_prediction, mono_prediction, vdw_surface, mono, batch_size):
    d = dipo_prediction.reshape(batch_size, NATOMS, 3)
    mono = mono.reshape(batch_size, NATOMS)
    m = mono_prediction.reshape(batch_size, NATOMS)
    batched_pred = batched_electrostatic_potential(d, m, vdw_surface)

    return batched_pred


def mean_absolute_error(prediction, target, batch_size):
    nonzero = jnp.nonzero(target, size=batch_size * NATOMS)
    return jnp.mean(jnp.abs(prediction[nonzero] - target[nonzero]))


def esp_loss_eval(pred, target, ngrid):
    target = target.flatten()
    esp_non_zero = np.nonzero(target)
    l2_loss = optax.l2_loss(pred[esp_non_zero], target[esp_non_zero]) * 2
    esp_loss = np.mean(l2_loss) ** 0.5
    return esp_loss


def get_predictions(mono_dc2, dipo_dc2, batch, batch_size, n_dcm):
    mono = mono_dc2
    dipo = dipo_dc2

    esp_dc_pred = esp_mono_loss_pots(
        dipo, mono, batch["vdw_surface"], batch["mono"], batch_size, n_dcm
    )

    mono_pred = esp_loss_pots(
        batch["positions"],
        batch["mono"],
        # batch["esp"],
        batch["vdw_surface"],
        batch["mono"],
        batch_size,
    )
    return esp_dc_pred, mono_pred
