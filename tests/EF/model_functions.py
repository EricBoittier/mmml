"""
JAX functions to calculate partial derivatives of energy and dipole
with respect to positions and electric field.

Units convention (matching the training data):
  - Positions: Angstrom
  - Electric field (Ef_input): in units of 0.001 au  (i.e. Ef_phys = Ef_input * 0.001 Hartree/(e*Bohr))
  - Energy: eV
  - Dipole: atomic units (e*Bohr)
  - Forces: eV/Angstrom

To convert raw Jacobians to physical units, apply field_scale = 0.001:
  - polarizability (physical) = d(mu)/d(Ef_input) / field_scale   [au]
  - APT is directly in au_dipole / Angstrom  (no field_scale needed)
"""

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Energy derivatives
# ---------------------------------------------------------------------------

def energy_and_forces(model_apply, params, atomic_numbers, positions, Ef,
                      dst_idx_flat, src_idx_flat, batch_segments, batch_size,
                      dst_idx=None, src_idx=None):
    """Compute energy, forces = -dE/dR, and predicted dipole.

    Returns
    -------
    energy : (B,)
    forces : same shape as positions  (gradient of -sum(E) w.r.t. positions)
    dipole : (B, 3)
    """
    def energy_fn(pos):
        energy, dipole = model_apply(
            params,
            atomic_numbers=atomic_numbers, positions=pos, Ef=Ef,
            dst_idx_flat=dst_idx_flat, src_idx_flat=src_idx_flat,
            batch_segments=batch_segments, batch_size=batch_size,
            dst_idx=dst_idx, src_idx=src_idx,
        )
        return -jnp.sum(energy), (energy, dipole)

    (_, (energy, dipole)), forces = jax.value_and_grad(
        energy_fn, has_aux=True
    )(positions)
    return energy, forces, dipole


def hessian_matrix(model_apply, params, atomic_numbers, positions, Ef,
                   dst_idx_flat, src_idx_flat, batch_segments, batch_size,
                   dst_idx=None, src_idx=None):
    """Compute Hessian d²E/dR² for normal-mode analysis.

    Returns
    -------
    hess : shape (*positions.shape, *positions.shape)
        For single molecule with positions (1, N, 3) this is (1, N, 3, 1, N, 3).
        Units: eV / Angstrom².
    """
    def energy_fn(pos):
        energy, _dipole = model_apply(
            params,
            atomic_numbers=atomic_numbers, positions=pos, Ef=Ef,
            dst_idx_flat=dst_idx_flat, src_idx_flat=src_idx_flat,
            batch_segments=batch_segments, batch_size=batch_size,
            dst_idx=dst_idx, src_idx=src_idx,
        )
        return jnp.sum(energy)

    return jax.hessian(energy_fn)(positions)


# ---------------------------------------------------------------------------
# Energy derivatives w.r.t. electric field  (dE/dEf, d²E/dEf²)
#
# NOTE: These are energy-based field derivatives. Because this model does NOT
# have an explicit  E = E0 - mu·Ef  coupling, dE/dEf is NOT equal to the
# physical dipole moment. Use the dipole-based derivatives below instead for
# physical polarizability and related tensors.
# ---------------------------------------------------------------------------

def energy_and_dipole_from_field_derivative(
    model_apply, params, atomic_numbers, positions, Ef,
    dst_idx_flat, src_idx_flat, batch_segments, batch_size,
    dst_idx=None, src_idx=None,
):
    """Compute -dE/dEf (energy derivative w.r.t. field).

    WARNING: Because the model couples Ef through features (not via -mu·Ef),
    this does NOT give the physical dipole moment.  Use
    ``dipole_derivative_field`` for the physical polarizability.

    Returns
    -------
    total_energy : scalar  (sum over batch)
    dE_dEf : same shape as Ef
    """
    def energy_fn(ef):
        energy, _dipole = model_apply(
            params,
            atomic_numbers=atomic_numbers, positions=positions, Ef=ef,
            dst_idx_flat=dst_idx_flat, src_idx_flat=src_idx_flat,
            batch_segments=batch_segments, batch_size=batch_size,
            dst_idx=dst_idx, src_idx=src_idx,
        )
        return jnp.sum(energy)

    total_energy, dE_dEf = jax.value_and_grad(energy_fn)(Ef)
    return total_energy, -dE_dEf


def polarizability_from_energy_hessian(
    model_apply, params, atomic_numbers, positions, Ef,
    dst_idx_flat, src_idx_flat, batch_segments, batch_size,
    dst_idx=None, src_idx=None,
):
    """Compute -d²E/dEf² (energy Hessian w.r.t. field).

    WARNING: Same caveat as ``energy_and_dipole_from_field_derivative``.
    Prefer ``dipole_derivative_field`` for the physical polarizability.

    Returns
    -------
    hessian : shape (*Ef.shape, *Ef.shape)
        For Ef (1,3) this is (1, 3, 1, 3).
    """
    def energy_fn(ef):
        energy, _dipole = model_apply(
            params,
            atomic_numbers=atomic_numbers, positions=positions, Ef=ef,
            dst_idx_flat=dst_idx_flat, src_idx_flat=src_idx_flat,
            batch_segments=batch_segments, batch_size=batch_size,
            dst_idx=dst_idx, src_idx=src_idx,
        )
        return jnp.sum(energy)

    return -jax.hessian(energy_fn)(Ef)


# ---------------------------------------------------------------------------
# Dipole derivatives  (the physically meaningful response properties)
# ---------------------------------------------------------------------------

def dipole_derivative_field(
    model_apply, params, atomic_numbers, positions, Ef,
    dst_idx_flat, src_idx_flat, batch_segments, batch_size,
    dst_idx=None, src_idx=None,
):
    """Polarizability from predicted dipole:  alpha_ab = d(mu_a)/d(Ef_b).

    This is the CORRECT polarizability because the model's predicted dipole
    is trained on physical dipole targets.

    To convert to physical (au) polarizability, divide by ``field_scale``
    (typically 0.001):
        alpha_phys [au] = alpha_raw / field_scale

    Returns
    -------
    alpha : shape (*dipole.shape, *Ef.shape)
        For single molecule: (1, 3, 1, 3).
        Extract [0, :, 0, :] for the (3, 3) polarizability tensor.
    """
    def dipole_fn(ef):
        _energy, dipole = model_apply(
            params,
            atomic_numbers=atomic_numbers, positions=positions, Ef=ef,
            dst_idx_flat=dst_idx_flat, src_idx_flat=src_idx_flat,
            batch_segments=batch_segments, batch_size=batch_size,
            dst_idx=dst_idx, src_idx=src_idx,
        )
        return dipole  # (B, 3)

    # jacrev: 3 reverse passes (efficient for 3 outputs vs many inputs)
    return jax.jacrev(dipole_fn)(Ef)


def dipole_derivative_positions(
    model_apply, params, atomic_numbers, positions, Ef,
    dst_idx_flat, src_idx_flat, batch_segments, batch_size,
    dst_idx=None, src_idx=None,
):
    """Atomic Polar Tensor (APT):  P_{a,s,b} = d(mu_a)/d(R_{s,b}).

    Used for:
      - IR intensities:  I_k ~ |sum_s P^s . L_k^s|²   (L = normal-mode eigenvectors)
      - Distributed-origin-gauge approximation to the AAT

    Returns
    -------
    apt : shape (*dipole.shape, *positions.shape)
        For single molecule with positions (1, N, 3):
          apt shape is (1, 3, 1, N, 3).
        Extract [0, :, 0, :, :] for (3, N, 3)  where  apt[a, s, b] = d(mu_a)/d(R_{s,b}).
    """
    def dipole_fn(pos):
        _energy, dipole = model_apply(
            params,
            atomic_numbers=atomic_numbers, positions=pos, Ef=Ef,
            dst_idx_flat=dst_idx_flat, src_idx_flat=src_idx_flat,
            batch_segments=batch_segments, batch_size=batch_size,
            dst_idx=dst_idx, src_idx=src_idx,
        )
        return dipole  # (B, 3)

    # jacrev: 3 reverse passes (efficient: 3 outputs, N*3 inputs)
    return jax.jacrev(dipole_fn)(positions)


# ---------------------------------------------------------------------------
# Approximate Atomic Axial Tensor (AAT) — distributed origin gauge
# ---------------------------------------------------------------------------

# Speed of light in atomic units
_C_AU = 137.035999084

# Levi-Civita tensor
_LEVI_CIVITA = jnp.array([
    [[0, 0, 0], [0, 0, 1], [0, -1, 0]],
    [[0, 0, -1], [0, 0, 0], [1, 0, 0]],
    [[0, 1, 0], [-1, 0, 0], [0, 0, 0]],
], dtype=jnp.float32)


def _aat_from_charges(charges, positions):
    """Core AAT computation from per-atom charges and positions.

    M^s_{a,b} = -(q_s / 4c) * sum_g  eps_{a,b,g} * R_{s,g}

    Parameters
    ----------
    charges : (N,) — effective charges per atom (Z_s, q_eff, or ML charges).
    positions : (N, 3)

    Returns
    -------
    aat : (N, 3, 3)   aat[s, a, b].
    """
    q = jnp.asarray(charges, dtype=jnp.float32)
    eps_R = jnp.einsum('abg,sg->sab', _LEVI_CIVITA, positions)  # (N, 3, 3)
    return -(q[:, None, None] / (4.0 * _C_AU)) * eps_R


def aat_nuclear(positions, atomic_numbers):
    """AAT using bare nuclear charges Z_s (distributed-origin gauge).

    M^{nuc,s}_{a,b} = -(Z_s / 4c) * sum_g  eps_{a,b,g} * R_{s,g}

    This is the pure nuclear contribution from the Lorentz transformation:
    a nucleus with charge Z moving at velocity v in field E sees
    B_eff = -(v × E) / c².

    Returns
    -------
    aat : (N, 3, 3)
    """
    return _aat_from_charges(jnp.asarray(atomic_numbers, dtype=jnp.float32),
                             positions)


def born_effective_charges(apt):
    """Extract Born effective charges from the Atomic Polar Tensor.

    q_eff,s = (1/3) * Tr(P^s)  = (1/3) * sum_a  dmu_a / dR_{s,a}

    These charges include electronic screening (unlike bare Z) and are
    the correct "dressed" charges for the Lorentz/DO-gauge AAT.

    Parameters
    ----------
    apt : (3, N, 3)  — apt[a, s, b] = dmu_a / dR_{s,b}

    Returns
    -------
    q_eff : (N,)
    """
    # Trace over the two Cartesian indices (a == b diagonal)
    return jnp.einsum('isi->s', apt) / 3.0


def aat_born(apt, positions):
    """AAT using Born effective charges (from APT).

    Combines the Lorentz relation  B_eff = -(v × E)/c²  with
    electronically screened charges  q_eff = (1/3) Tr(APT).

    This is significantly better than bare-nuclear AAT because q_eff
    captures the electronic charge redistribution.

    Returns
    -------
    aat : (N, 3, 3)
    q_eff : (N,)
    """
    q_eff = born_effective_charges(apt)
    return _aat_from_charges(q_eff, positions), q_eff


def aat_ml_charges(apt, positions, model_charges):
    """AAT using the ML-predicted atomic charges from the model.

    The model internally predicts atomic partial charges q_i that are
    used in the dipole:  mu = sum_i q_i * (r_i - COM) + sum_i mu_i^atomic.

    These ML charges can be extracted via ``get_atomic_charges`` and give
    the most faithful representation of the model's charge distribution.

    Parameters
    ----------
    apt : (3, N, 3) — not used in the formula but kept for API consistency.
    positions : (N, 3)
    model_charges : (N,) — ML-predicted atomic partial charges.

    Returns
    -------
    aat : (N, 3, 3)
    """
    return _aat_from_charges(model_charges, positions)


# ---------------------------------------------------------------------------
# Extract atomic charges / dipoles from the model
# ---------------------------------------------------------------------------

def get_atomic_properties(model, params, atomic_numbers, positions, Ef,
                          dst_idx_flat, src_idx_flat, batch_segments, batch_size,
                          dst_idx=None, src_idx=None):
    """Run model forward pass and capture intermediate atomic charges & dipoles.

    Uses Flax's ``sow`` / ``mutable`` mechanism to extract the values stored
    by ``self.sow('intermediates', ...)`` inside the model's ``EFD`` method.

    Returns
    -------
    energy : (B,)
    dipole : (B, 3)
    atomic_charges : (B, N)
    atomic_dipoles : (B, N, 3)
    """
    (energy, dipole), state = model.apply(
        params,
        atomic_numbers, positions, Ef,
        dst_idx_flat=dst_idx_flat, src_idx_flat=src_idx_flat,
        batch_segments=batch_segments, batch_size=batch_size,
        dst_idx=dst_idx, src_idx=src_idx,
        mutable=['intermediates'],
    )
    intermediates = state.get('intermediates', {})
    # sow stores values in tuples; take the last one
    atomic_charges = intermediates.get('atomic_charges', (None,))[-1]
    atomic_dipoles = intermediates.get('atomic_dipoles', (None,))[-1]
    return energy, dipole, atomic_charges, atomic_dipoles


# Keep old name as alias for backward compatibility
def aat_distributed_origin(apt, positions, atomic_numbers=None):
    """Backward-compatible alias.  Prefer ``aat_nuclear`` or ``aat_born``."""
    if atomic_numbers is None:
        N = positions.shape[0]
        atomic_numbers = jnp.ones(N, dtype=jnp.float32)
    return aat_nuclear(positions, atomic_numbers)
