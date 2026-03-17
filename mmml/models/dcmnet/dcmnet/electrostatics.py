import functools
import jax
import jax.numpy as jnp
from jax import vmap

from mmml.data.units import ANGSTROM_TO_BOHR


@functools.partial(jax.jit)
def coulomb_potential(q, r):
    """V = q/r [Hartree/e] when r is in Angstrom. Converts r to Bohr internally."""
    r_bohr = r * ANGSTROM_TO_BOHR
    return q / (r_bohr + 1e-10)


# @functools.partial(jax.jit, static_argnames=('grid_positions'))
def calc_esp(charge_positions, charge_values, grid_positions):
    if grid_positions.ndim != 2:
        grid_positions = grid_positions.reshape(-1, grid_positions.shape[-1])
    if charge_positions.ndim != 2:
        charge_positions = charge_positions.reshape(-1, charge_positions.shape[-1])
    if charge_values.ndim != 1:
        charge_values = charge_values.reshape(-1)
    # Expand the grid positions and charge positions to compute all pairwise differences
    diff = grid_positions[:, jnp.newaxis, :] - charge_positions[jnp.newaxis, :, :]
    # Compute the Euclidean distance between each grid point and each charge
    r = jnp.linalg.norm(diff, axis=-1)
    C = coulomb_potential(charge_values[None, :], r)
    V = jnp.sum(C, axis=-1)
    return V


batched_electrostatic_potential = vmap(calc_esp, in_axes=(0, 0, 0), out_axes=0)
