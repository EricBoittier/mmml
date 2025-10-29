import functools
import jax
import jax.numpy as jnp
from jax import vmap


@functools.partial(jax.jit)
def coulomb_potential(q, r):
    return q / (r * 1.88973)


# # @functools.partial(jax.jit, static_argnames=('grid_positions'))
# def calc_esp(charge_positions, charge_values, grid_positions):
#     # Expand the grid positions and charge positions to compute all pairwise differences
#     diff = grid_positions[:, jnp.newaxis, :] - charge_positions[jnp.newaxis, :, :]
#     # Compute the Euclidean distance between each grid point and each charge
#     r = jnp.linalg.norm(diff, axis=-1)
#     C = coulomb_potential(charge_values[None, :], r)
#     V = jnp.sum(C, axis=-1)
#     return V

def calc_esp(charge_positions, charges, grid_positions):
    grid_positions = jnp.asarray(grid_positions)
    charge_positions = jnp.asarray(charge_positions)
    charges = jnp.asarray(charges)

    # Fix flattened/1D inputs
    if grid_positions.ndim == 1:
        assert grid_positions.size % 3 == 0, "grid_positions must be multiple of 3"
        grid_positions = grid_positions.reshape(-1, 3)
    if charge_positions.ndim == 1:
        assert charge_positions.size % 3 == 0, "charge_positions must be multiple of 3"
        charge_positions = charge_positions.reshape(-1, 3)
    if charges.ndim > 1:
        charges = charges.reshape(-1)  # e.g. (1, M) -> (M,)

    # Compute pairwise diffs and ESP
    diff = grid_positions[:, None, :] - charge_positions[None, :, :]   # (n_grid, n_charges, 3)
    r = jnp.linalg.norm(diff, axis=-1)                                 # (n_grid, n_charges)
    esp = jnp.sum(charges[None, :] / r, axis=1)                        # (n_grid,)
    return esp


batched_electrostatic_potential = vmap(calc_esp, in_axes=(0, 0, 0), out_axes=0)
