import jax
import jax.numpy as jnp
import flax.linen as nn
import cuequivariance as cue
import cuequivariance_jax as cuex
from cuequivariance_jax import (
    spherical_harmonics,
    triangle_multiplicative_update,
    equivariant_polynomial,
)


class TriangleMultiplicativeLayer(nn.Module):
    """Flax wrapper around cuequivariance_jax.triangle_multiplicative_update."""

    direction: str = "outgoing"
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x, mask=None):
        """Apply triangle multiplicative update.

        Args:
            x: array of shape (..., N, N, D_in)
            mask: optional array broadcastable to (..., N, N)
        """
        d_in = x.shape[-1]

        # Input norm/gating parameters
        norm_in_weight = self.param(
            "norm_in_weight", lambda key: jnp.ones((d_in,))
        )
        norm_in_bias = self.param(
            "norm_in_bias", lambda key: jnp.zeros((d_in,))
        )

        p_in_weight = self.param(
            "p_in_weight",
            lambda key: nn.initializers.lecun_normal()(key, (2 * d_in, d_in)),
        )
        p_in_bias = self.param(
            "p_in_bias", lambda key: jnp.zeros((2 * d_in,))
        )

        g_in_weight = self.param(
            "g_in_weight",
            lambda key: nn.initializers.lecun_normal()(key, (2 * d_in, d_in)),
        )
        g_in_bias = self.param(
            "g_in_bias", lambda key: jnp.zeros((2 * d_in,))
        )

        # Output norm/gating parameters (keep D_out = D_in for simplicity)
        norm_out_weight = self.param(
            "norm_out_weight", lambda key: jnp.ones((d_in,))
        )
        norm_out_bias = self.param(
            "norm_out_bias", lambda key: jnp.zeros((d_in,))
        )

        p_out_weight = self.param(
            "p_out_weight",
            lambda key: nn.initializers.lecun_normal()(key, (d_in, d_in)),
        )
        p_out_bias = self.param(
            "p_out_bias", lambda key: jnp.zeros((d_in,))
        )

        g_out_weight = self.param(
            "g_out_weight",
            lambda key: nn.initializers.lecun_normal()(key, (d_in, d_in)),
        )
        g_out_bias = self.param(
            "g_out_bias", lambda key: jnp.zeros((d_in,))
        )

        return triangle_multiplicative_update(
            x=x,
            direction=self.direction,
            key=None,  # all weights are provided explicitly
            mask=mask,
            norm_in_weight=norm_in_weight,
            norm_in_bias=norm_in_bias,
            p_in_weight=p_in_weight,
            p_in_bias=p_in_bias,
            g_in_weight=g_in_weight,
            g_in_bias=g_in_bias,
            norm_out_weight=norm_out_weight,
            norm_out_bias=norm_out_bias,
            p_out_weight=p_out_weight,
            p_out_bias=p_out_bias,
            g_out_weight=g_out_weight,
            g_out_bias=g_out_bias,
            eps=self.eps,
            # Use pure-JAX fallback implementation so gradients are defined
            fallback=True,
        )


class EnergyForceModel(nn.Module):
    """cuEquivariance-based energy/force model.

    Inputs:
      positions (R): (n_atoms, 3)
      atomic_numbers (Z): (n_atoms,)
      total_charge (Q): scalar
    """

    hidden_dim: int = 248
    num_layers: int = 3
    ls: tuple = (0, 1, 2, 3, 4)

    @nn.compact
    def __call__(self, positions, atomic_numbers, total_charge, atom_mask=None):
        """Predict a scalar potential energy from atomic positions and species.

        Args:
            positions: array of shape (n_atoms, 3) with Cartesian coordinates.
            atomic_numbers: array of shape (n_atoms,) with integer atomic numbers.
            total_charge: scalar total molecular charge.
            atom_mask: optional (n_atoms,) mask; 1 for real atoms, 0 for padding.
                If provided, energy is summed only over masked atoms.
        """
        n_atoms = positions.shape[-2]

        # Pairwise displacements and distances
        disp = positions[:, None, :] - positions[None, :, :]  # (n_atoms, n_atoms, 3)
        distances = jnp.linalg.norm(disp + 1e-9, axis=-1, keepdims=True)

        # Angular features via spherical harmonics (SO(3)-aware features)
        unit_vectors = disp / (distances + 1e-9)
        # Wrap vectors as an SO(3) irreps array before calling spherical_harmonics
        with cue.assume(cue.SO3, cue.ir_mul):
            vec_rep = cuex.RepArray("1", unit_vectors)
            sh_rep = spherical_harmonics(self.ls, vec_rep, normalize=True)
            sh = sh_rep.array

            # Additional angular features via equivariant_polynomial on RepArray inputs
            poly = cue.descriptors.spherical_harmonics(cue.SO3(1), list(self.ls))
            rep_in = cuex.RepArray("1", unit_vectors.reshape(-1, 3))
            sh_poly_rep = equivariant_polynomial(poly, [rep_in], method="naive")
            sh_poly = sh_poly_rep.array.reshape(n_atoms, n_atoms, -1)

        # Encode atomic numbers and total charge as pairwise scalar features
        Z = jnp.asarray(atomic_numbers, dtype=jnp.float32).reshape(n_atoms, 1)
        Z_i = jnp.broadcast_to(Z[:, None, :], (n_atoms, n_atoms, 1))
        Z_j = jnp.broadcast_to(Z[None, :, :], (n_atoms, n_atoms, 1))
        Q = jnp.broadcast_to(
            jnp.asarray(total_charge, dtype=jnp.float32).reshape(1, 1, 1),
            (n_atoms, n_atoms, 1),
        )

        # Combine radial and angular information into per-pair features
        pair_features = jnp.concatenate([distances, sh, sh_poly, Z_i, Z_j, Q], axis=-1)

        # Initial per-atom features from raw pairwise representation
        base_atom_features = pair_features.reshape(n_atoms, -1)
        x = nn.Dense(self.hidden_dim, name="in_proj")(base_atom_features)

        # Stack of triangle + per-atom residual blocks
        for layer in range(self.num_layers):
            # Triangle multiplicative update over pairwise features (AlphaFold2-style).
            # We stop gradients through this branch to avoid unsupported JAX primitives.
            pair_features = TriangleMultiplicativeLayer(name=f"triangle_{layer}")(
                pair_features
            )
            pair_features = jax.lax.stop_gradient(pair_features)
            atom_features_l = pair_features.reshape(n_atoms, -1)

            # Project triangle-updated atom features into hidden_dim
            h = nn.Dense(self.hidden_dim, name=f"tri_proj_{layer}")(atom_features_l)
            h = nn.LayerNorm(name=f"ln_{layer}")(h)
            h = nn.silu(h)

            # Residual connection
            x = x + h

        # Predict per-atom energies and forces directly
        per_atom_energy = nn.Dense(1)(x)   # (n_atoms, 1)
        per_atom_forces = nn.Dense(3)(x)   # (n_atoms, 3)

        if atom_mask is not None:
            mask = jnp.asarray(atom_mask, dtype=jnp.float32).reshape(-1, 1)
            energy = jnp.sum(per_atom_energy * mask)
            forces = per_atom_forces * mask
        else:
            energy = jnp.sum(per_atom_energy)
            forces = per_atom_forces

        return energy, forces


def energy_fn(params, positions, atomic_numbers, total_charge, atom_mask=None):
    """Convenience wrapper: energy(params, R, Z, Q)."""
    model = EnergyForceModel()
    energy, _ = model.apply(params, positions, atomic_numbers, total_charge, atom_mask)
    return energy


def forces_fn(params, positions, atomic_numbers, total_charge, atom_mask=None):
    """Predict forces directly from the model (no energy gradient)."""
    model = EnergyForceModel()
    _, forces = model.apply(params, positions, atomic_numbers, total_charge, atom_mask)
    return forces


if __name__ == "__main__":
    # Small demo: initialize the model and compute energy and forces
    key = jax.random.key(0)
    n_atoms = 4
    positions = jax.random.normal(key, (n_atoms, 3))
    atomic_numbers = jnp.array([1, 1, 1, 1], dtype=jnp.int32)
    total_charge = 0.0

    model = EnergyForceModel()
    params = model.init(key, positions, atomic_numbers, total_charge)

    energy, forces = model.apply(params, positions, atomic_numbers, total_charge)

    print("Energy:", energy)
    print("Forces shape:", forces.shape)