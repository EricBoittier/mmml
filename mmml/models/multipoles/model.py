"""E3x message-passing model for molecular spherical multipoles."""

from __future__ import annotations

import functools

import e3x
import flax.linen as nn
import jax
import jax.numpy as jnp


def _degree_slice(degree: int) -> tuple[int, int]:
    start = degree * degree
    return start, start + 2 * degree + 1


class _E3xMultipoleBackbone(nn.Module):
    """Shared E3x atom-wise equivariant encoder/readout."""

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
        batch_segments: jnp.ndarray,
        atom_mask: jnp.ndarray,
        edge_mask: jnp.ndarray | None,
    ) -> jnp.ndarray:
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
            x = e3x.nn.add(x, message.astype(x.dtype))
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
        return atomic_irreps * atom_mask[:, None, None]


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
    compose_dipole_from_atomic: bool = False
    enforce_total_charge: bool = True

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

        if atom_mask is not None:
            atom_mask = jnp.asarray(atom_mask, dtype=positions.dtype)
        else:
            atom_mask = jnp.ones(atomic_numbers.shape[0], dtype=positions.dtype)
        atomic_irreps = _E3xMultipoleBackbone(
            features=self.features,
            max_degree=self.max_degree,
            num_iterations=self.num_iterations,
            num_basis_functions=self.num_basis_functions,
            cutoff=self.cutoff,
            max_atomic_number=self.max_atomic_number,
        )(
            positions,
            atomic_numbers,
            charge,
            spin,
            dst_idx,
            src_idx,
            batch_segments,
            atom_mask,
            edge_mask,
        )

        molecular_irreps = jax.ops.segment_sum(
            atomic_irreps,
            batch_segments,
            num_segments=batch_size,
        )
        packed_irreps = molecular_irreps[:, 0, :]
        output = {"multipoles": packed_irreps}
        if not self.compose_dipole_from_atomic:
            return output

        packed_atomic_irreps = atomic_irreps[:, 0, :]
        atomic_charges = packed_atomic_irreps[:, 0]
        raw_atomic_charges = atomic_charges
        atom_counts = jax.ops.segment_sum(
            atom_mask,
            batch_segments,
            num_segments=batch_size,
        )
        if self.enforce_total_charge:
            charge_residual = charge - jax.ops.segment_sum(
                atomic_charges,
                batch_segments,
                num_segments=batch_size,
            )
            atomic_charges = atomic_charges + (
                charge_residual[batch_segments] / jnp.maximum(atom_counts[batch_segments], 1.0)
            ) * atom_mask

        position_irreps = e3x.so3.tensor_to_irreps(positions, degree=1)
        atomic_dipoles = packed_atomic_irreps[:, 1:4]
        charge_dipoles = atomic_charges[:, None] * position_irreps
        atomic_total_dipoles = (charge_dipoles + atomic_dipoles) * atom_mask[:, None]
        molecular_charge = jax.ops.segment_sum(
            atomic_charges,
            batch_segments,
            num_segments=batch_size,
        )[:, None]
        molecular_dipole = jax.ops.segment_sum(
            atomic_total_dipoles,
            batch_segments,
            num_segments=batch_size,
        )
        higher_multipoles = packed_irreps[:, 4:]
        composed_multipoles = jnp.concatenate(
            (molecular_charge, molecular_dipole, higher_multipoles),
            axis=-1,
        )
        output.update(
            {
                "multipoles": composed_multipoles,
                "raw_multipoles": packed_irreps,
                "atomic_charges": atomic_charges,
                "raw_atomic_charges": raw_atomic_charges,
                "atomic_dipoles": atomic_dipoles,
                "charge_dipoles": charge_dipoles,
                "atomic_total_dipoles": atomic_total_dipoles,
            }
        )
        return output


class E3xDegreeMultipoleModel(nn.Module):
    """Predict one molecular multipole degree with an independent E3x model.

    For ``target_degree=1`` with ``compose_from_atomic=True``, the model predicts
    latent atom-centered charges ``q_i`` and atom-centered dipoles ``mu_i`` and
    composes the molecular dipole as ``sum_i(q_i r_i + mu_i)``. For
    ``target_degree=2`` or ``3``, the model directly sums the atom-centered
    irreps for that degree.
    """

    target_degree: int
    features: int = 64
    num_iterations: int = 3
    num_basis_functions: int = 16
    cutoff: float = 6.0
    max_atomic_number: int = 118
    compose_from_atomic: bool = True
    enforce_total_charge: bool = True

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
        if self.target_degree not in (1, 2, 3):
            raise ValueError("target_degree must be 1, 2, or 3")
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
        if atom_mask is None:
            atom_mask = jnp.ones(atomic_numbers.shape[0], dtype=positions.dtype)
        else:
            atom_mask = jnp.asarray(atom_mask, dtype=positions.dtype)

        atomic_irreps = _E3xMultipoleBackbone(
            features=self.features,
            max_degree=self.target_degree,
            num_iterations=self.num_iterations,
            num_basis_functions=self.num_basis_functions,
            cutoff=self.cutoff,
            max_atomic_number=self.max_atomic_number,
        )(
            positions,
            atomic_numbers,
            charge,
            spin,
            dst_idx,
            src_idx,
            batch_segments,
            atom_mask,
            edge_mask,
        )
        packed_atomic_irreps = atomic_irreps[:, 0, :]

        if self.target_degree == 1 and self.compose_from_atomic:
            atomic_charges = packed_atomic_irreps[:, 0]
            raw_atomic_charges = atomic_charges
            atom_counts = jax.ops.segment_sum(
                atom_mask,
                batch_segments,
                num_segments=batch_size,
            )
            if self.enforce_total_charge:
                charge_residual = charge - jax.ops.segment_sum(
                    atomic_charges,
                    batch_segments,
                    num_segments=batch_size,
                )
                atomic_charges = atomic_charges + (
                    charge_residual[batch_segments]
                    / jnp.maximum(atom_counts[batch_segments], 1.0)
                ) * atom_mask
            atomic_dipoles = packed_atomic_irreps[:, 1:4]
            charge_dipoles = atomic_charges[:, None] * e3x.so3.tensor_to_irreps(
                positions,
                degree=1,
            )
            atomic_total_dipoles = (charge_dipoles + atomic_dipoles) * atom_mask[:, None]
            degree_irrep = jax.ops.segment_sum(
                atomic_total_dipoles,
                batch_segments,
                num_segments=batch_size,
            )
            return {
                "degree": degree_irrep,
                "multipole": degree_irrep,
                "atomic_charges": atomic_charges,
                "raw_atomic_charges": raw_atomic_charges,
                "atomic_dipoles": atomic_dipoles,
                "charge_dipoles": charge_dipoles,
                "atomic_total_dipoles": atomic_total_dipoles,
            }

        start, stop = _degree_slice(self.target_degree)
        degree_irrep = jax.ops.segment_sum(
            packed_atomic_irreps[:, start:stop] * atom_mask[:, None],
            batch_segments,
            num_segments=batch_size,
        )
        return {
            "degree": degree_irrep,
            "multipole": degree_irrep,
            "atomic_degree_irreps": packed_atomic_irreps[:, start:stop],
        }


class E3xDipoleModel(E3xDegreeMultipoleModel):
    target_degree: int = 1


class E3xQuadrupoleModel(E3xDegreeMultipoleModel):
    target_degree: int = 2


class E3xOctupoleModel(E3xDegreeMultipoleModel):
    target_degree: int = 3
