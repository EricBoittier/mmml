"""E3x surrogate for QCML MBD energy, forces, C6, and polarizabilities."""

from __future__ import annotations

import functools

import e3x
import flax.linen as nn
import jax
import jax.numpy as jnp


class E3xMBDModel(nn.Module):
    """Predict positive atomic C6/alpha values and molecular MBD energy."""

    features: int = 64
    max_degree: int = 2
    num_iterations: int = 3
    num_basis_functions: int = 16
    cutoff: float = 12.0
    max_atomic_number: int = 118

    @nn.compact
    def __call__(
        self,
        positions: jnp.ndarray,
        atomic_numbers: jnp.ndarray,
        charge: jnp.ndarray,
        spin: jnp.ndarray,
        dst_idx: jnp.ndarray,
        src_idx: jnp.ndarray,
        batch_segments: jnp.ndarray,
        batch_size: int,
        atom_mask: jnp.ndarray,
        edge_mask: jnp.ndarray,
    ) -> dict[str, jnp.ndarray]:
        positions = jnp.asarray(positions, dtype=jnp.float32)
        atomic_numbers = jnp.asarray(atomic_numbers).reshape(-1)
        displacement = positions[src_idx] - positions[dst_idx]
        basis = e3x.nn.basis(
            displacement,
            max_degree=self.max_degree,
            num=self.num_basis_functions,
            radial_fn=e3x.nn.reciprocal_bernstein,
            cutoff_fn=functools.partial(e3x.nn.smooth_cutoff, cutoff=self.cutoff),
        )
        basis *= jnp.asarray(edge_mask, dtype=basis.dtype)[:, None, None, None]

        x = e3x.nn.Embed(
            num_embeddings=self.max_atomic_number + 1,
            features=self.features,
        )(atomic_numbers)
        state = jnp.stack((charge, spin), axis=-1).astype(jnp.float32)
        state = nn.silu(nn.Dense(self.features)(state))
        x = e3x.nn.add(x, state[batch_segments, None, None, :].astype(x.dtype))

        for _ in range(self.num_iterations):
            message = e3x.nn.MessagePass(
                max_degree=self.max_degree,
                include_pseudotensors=False,
            )(x, basis, dst_idx=dst_idx, src_idx=src_idx)
            x = e3x.nn.add(x, message.astype(x.dtype))
            x = e3x.nn.silu(x)
            x = e3x.nn.Dense(self.features)(x)

        scalars = e3x.nn.change_max_degree_or_type(
            x,
            max_degree=0,
            include_pseudotensors=False,
        )
        scalars = nn.silu(e3x.nn.Dense(self.features)(scalars))
        raw_properties = e3x.nn.Dense(3)(scalars)[:, 0, 0, :]
        mask = jnp.asarray(atom_mask, dtype=raw_properties.dtype)
        atomic_energy = raw_properties[:, 0] * mask
        polarizabilities = jax.nn.softplus(raw_properties[:, 1]) * mask
        c6_coefficients = jax.nn.softplus(raw_properties[:, 2]) * mask
        energy = jax.ops.segment_sum(
            atomic_energy,
            batch_segments,
            num_segments=batch_size,
        )
        return {
            "energy": energy,
            "atomic_energy": atomic_energy,
            "polarizabilities": polarizabilities,
            "c6_coefficients": c6_coefficients,
        }


def mbd_energy_and_forces(
    model: E3xMBDModel,
    params: dict,
    **inputs,
) -> tuple[dict[str, jnp.ndarray], jnp.ndarray]:
    """Return predictions and conservative forces from the MBD energy gradient."""
    positions = inputs.pop("positions")

    def energy_fn(current_positions):
        output = model.apply(
            {"params": params},
            positions=current_positions,
            **inputs,
        )
        return jnp.sum(output["energy"]), output

    (_, output), gradient = jax.value_and_grad(energy_fn, has_aux=True)(positions)
    return output, -gradient
