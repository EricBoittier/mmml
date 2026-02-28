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


class TriangleAttentionBlock(nn.Module):
    """Triangle-style attention over pair features using cuequivariance_jax.triangle_attention."""

    @nn.compact
    def __call__(self, pair_features, atom_mask=None):
        """Apply triangle attention to pairwise features.

        Args:
            pair_features: array of shape (n_atoms, n_atoms, D_in)
            atom_mask: optional (n_atoms,) mask; 1 for real atoms, 0 for padding.
        """
        n_atoms = pair_features.shape[0]
        d_in = pair_features.shape[-1]
        S = n_atoms * n_atoms
        H = self.num_heads
        D = self.head_dim

        # Flatten pair indices into a single sequence dimension S_qo = S_kv.
        x = pair_features.reshape(1, 1, S, d_in)  # (B=1, N=1, S, D_in)

        # Linear projections to queries/keys/values
        q = nn.Dense(H * D, name="q_proj")(x).reshape(1, 1, H, S, D)
        k = nn.Dense(H * D, name="k_proj")(x).reshape(1, 1, H, S, D)
        v = nn.Dense(H * D, name="v_proj")(x).reshape(1, 1, H, S, D)

        # Bias: no extra bias beyond mask, so use zeros
        bias = jnp.zeros((1, 1, H, S, S), dtype=x.dtype)

        # Mask: propagate atom_mask to pair-wise validity if provided
        if atom_mask is not None:
            atom_valid = jnp.asarray(atom_mask, dtype=bool).reshape(n_atoms)
            pair_valid = (atom_valid[:, None] & atom_valid[None, :]).reshape(S)
            mask = pair_valid.reshape(1, 1, 1, 1, S)
        else:
            mask = jnp.ones((1, 1, 1, 1, S), dtype=bool)

        # scale must be a Python float (hashable attribute for the primitive),
        # not a JAX array scalar
        scale = 1.0 / (float(D) ** 0.5)

        attn_out, _, _ = triangle_attention(q, k, v, bias, mask, scale)  # (1, 1, H, S, D)
        attn_out = attn_out.reshape(n_atoms, n_atoms, H * D)

        # Reduce over neighbor dimension to obtain per-atom features
        atom_features = attn_out.sum(axis=1)  # (n_atoms, H * D)
        return atom_features


class EnergyForceModel(nn.Module):
    """cuEquivariance-based energy/force model.

    Inputs:
      positions (R): (n_atoms, 3)
      atomic_numbers (Z): (n_atoms,)
      total_charge (Q): scalar
    """

    hidden_dim: int = 64
    num_layers: int = 32
    ls: tuple = (0, 1, 2)
    num_heads: int = 4
    head_dim: int = 32

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
        pair_features = jnp.concatenate([distances, sh, sh_poly, Z_i* Z_j, Q], axis=-1)

        # Simple per-atom features and residual MLP (no attention for speed/stability).
        atom_features = pair_features.reshape(n_atoms, -1)
        x = nn.Dense(self.hidden_dim, name="in_proj")(atom_features)

        for layer in range(self.num_layers):
            h = nn.Dense(self.hidden_dim, name=f"dense_{layer}")(x)
            h = nn.LayerNorm(name=f"ln_{layer}")(h)
            h = nn.silu(h)
            x = x + h  # residual

        # Predict per-atom energies; forces come from autodiff of energy
        per_atom_energy = nn.Dense(1)(x)   # (n_atoms, 1)

        if atom_mask is not None:
            mask = jnp.asarray(atom_mask, dtype=jnp.float32).reshape(-1, 1)
            energy = jnp.sum(per_atom_energy * mask)
        else:
            energy = jnp.sum(per_atom_energy)

        return energy


class PairSelfAttentionBlock(nn.Module):
    """Pure-JAX multi-head self-attention over pairwise features.

    This block operates on pair features of shape (n_atoms, n_atoms, D_in) and
    returns per-atom features of shape (n_atoms, num_heads * head_dim). It is
    implemented only with standard JAX/Flax ops, so it is fully compatible with
    JAX autodiff for forces.
    """

    num_heads: int = 4
    head_dim: int = 32

    @nn.compact
    def __call__(self, pair_features, atom_mask=None):
        """Apply self-attention to pairwise features.

        Args:
            pair_features: array of shape (n_atoms, n_atoms, D_in)
            atom_mask: optional (n_atoms,) mask; 1 for real atoms, 0 for padding.
        """
        n_atoms = pair_features.shape[0]
        d_in = pair_features.shape[-1]
        S = n_atoms * n_atoms
        H = self.num_heads
        D = self.head_dim

        # Flatten (i, j) pairs into a single sequence dimension S.
        x = pair_features.reshape(S, d_in)  # (S, D_in)

        # Linear projections to queries/keys/values
        q = nn.Dense(H * D, name="q_proj")(x).reshape(H, S, D)  # (H, S, D)
        k = nn.Dense(H * D, name="k_proj")(x).reshape(H, S, D)  # (H, S, D)
        v = nn.Dense(H * D, name="v_proj")(x).reshape(H, S, D)  # (H, S, D)

        # Scaled dot-product attention: scores shape (H, S_q, S_k)
        scale = 1.0 / jnp.sqrt(jnp.asarray(D, dtype=x.dtype))
        scores = jnp.einsum("hqd,hkd->hqk", q, k) * scale  # (H, S, S)

        if atom_mask is not None:
            # Build a mask over key positions based on valid atoms.
            atom_valid = jnp.asarray(atom_mask, dtype=bool).reshape(n_atoms)
            pair_valid = (atom_valid[:, None] & atom_valid[None, :]).reshape(S)  # (S,)
            key_mask = pair_valid[None, None, :]  # (1, 1, S_k)
            large_neg = jnp.asarray(-1e9, dtype=scores.dtype)
            scores = jnp.where(key_mask, scores, large_neg)

        # Attention weights and output
        attn_weights = nn.softmax(scores, axis=-1)  # (H, S_q, S_k)
        attn_out = jnp.einsum("hqk,hkd->hqd", attn_weights, v)  # (H, S, D)

        attn_out = attn_out.reshape(S, H * D)
        attn_out = attn_out.reshape(n_atoms, n_atoms, H * D)

        # Reduce over neighbor dimension to obtain per-atom features
        atom_features = attn_out.sum(axis=1)  # (n_atoms, H * D)
        return atom_features


def energy_fn(params, positions, atomic_numbers, total_charge, atom_mask=None):
    """Convenience wrapper: energy(params, R, Z, Q)."""
    model = EnergyForceModel()
    return model.apply(params, positions, atomic_numbers, total_charge, atom_mask)


def forces_fn(params, positions, atomic_numbers, total_charge, atom_mask=None):
    """Compute forces as negative gradient of energy w.r.t. positions."""
    grad_energy_wrt_positions = jax.grad(energy_fn, argnums=1)(
        params, positions, atomic_numbers, total_charge, atom_mask
    )
    return -grad_energy_wrt_positions


if __name__ == "__main__":
    # Small demo: initialize the model and compute energy and forces
    key = jax.random.key(0)
    n_atoms = 4
    positions = jax.random.normal(key, (n_atoms, 3))
    atomic_numbers = jnp.array([1, 1, 1, 1], dtype=jnp.int32)
    total_charge = 0.0

    model = EnergyForceModel()
    params = model.init(key, positions, atomic_numbers, total_charge)

    energy = energy_fn(params, positions, atomic_numbers, total_charge)
    forces = forces_fn(params, positions, atomic_numbers, total_charge)

    print("Energy:", energy)
    print("Forces shape:", forces.shape)