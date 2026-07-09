"""E3x message-passing model for molecular spherical multipoles."""

from __future__ import annotations

import functools

import e3x
import flax.linen as nn
import jax
import jax.numpy as jnp

class E3xMultipoleModel(nn.Module):
    """Predict molecular traceless multipoles through ``l=max_degree``.

    ``positions`` and ``atomic_numbers`` use a flattened atom layout. ``charge``
    and ``spin`` contain one scalar per molecule. Targets must use the same
    centered molecular origin as the input coordinates.
    """

    features: int = 64
    max_degree: int = 3
    num_iterations: int = 3
    num_basis_functions: int = 16
    cutoff: float = 6.0
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
        batch_segments: jnp.ndarray | None = None,
        batch_size: int | None = None,
        atom_mask: jnp.ndarray | None = None,
        edge_mask: jnp.ndarray | None = None,
    ) -> dict[str, jnp.ndarray]:
        positions = jnp.asarray(positions, dtype=jnp.float32)
        atomic_numbers = jnp.asarray(atomic_numbers).reshape(-1)
        charge = jnp.atleast_1d(charge).astype(positions.dtype)
        spin = jnp.atleast_1d(spin).astype(positions.dtype)

        if batch_segments is None:
            batch_segments = jnp.zeros(atomic_numbers.shape[0], dtype=jnp.int32)
        if batch_size is None:
            batch_size = charge.shape[0]
        if charge.shape[0] != batch_size or spin.shape[0] != batch_size:
            raise ValueError("charge and spin must contain one value per molecule")

        displacements = (
            e3x.ops.gather_src(positions, src_idx=src_idx)
            - e3x.ops.gather_dst(positions, dst_idx=dst_idx)
        )
        basis = e3x.nn.basis(
            displacements,
            max_degree=self.max_degree,
            num=self.num_basis_functions,
            radial_fn=e3x.nn.reciprocal_bernstein,
            cutoff_fn=functools.partial(e3x.nn.smooth_cutoff, cutoff=self.cutoff),
        )
        if edge_mask is not None:
            basis = basis * jnp.asarray(edge_mask, dtype=basis.dtype)[:, None, None, None]

        x = e3x.nn.Embed(
            num_embeddings=self.max_atomic_number + 1,
            features=self.features,
        )(atomic_numbers)

        molecular_state = jnp.stack((charge, spin), axis=-1)
        molecular_state = nn.Dense(self.features)(molecular_state)
        molecular_state = nn.silu(molecular_state)
        molecular_state = molecular_state[batch_segments, None, None, :].astype(x.dtype)
        x = e3x.nn.add(x, molecular_state)

        for _ in range(self.num_iterations):
            message = e3x.nn.MessagePass(
                max_degree=self.max_degree,
                include_pseudotensors=False,
            )(x, basis, dst_idx=dst_idx, src_idx=src_idx)
            message = message.astype(x.dtype)
            x = e3x.nn.add(x, message)
            x = e3x.nn.silu(x)
            x = e3x.nn.Dense(self.features)(x)

        x = e3x.nn.TensorDense(
            features=self.features,
            max_degree=self.max_degree,
            include_pseudotensors=False,
        )(x)
        x = e3x.nn.silu(x)
        atomic_irreps = e3x.nn.Dense(
            1,
            use_bias=False,
            kernel_init=jax.nn.initializers.zeros,
        )(x)[..., 0]

        if atom_mask is not None:
            atomic_irreps = atomic_irreps * jnp.asarray(atom_mask)[:, None, None]

        molecular_irreps = jax.ops.segment_sum(
            atomic_irreps,
            batch_segments,
            num_segments=batch_size,
        )
        packed_irreps = molecular_irreps[:, 0, :]
        return {"multipoles": packed_irreps}
