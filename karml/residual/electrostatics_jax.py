"""Explicit electrostatics in JAX — energy and *autodiff-consistent* forces.

The forces here are exact analytic derivatives of the same energy expression
(via `jax.grad`), so the residual target
    E_short = E_ref - E_expl_elec
    F_short = F_ref - F_expl_elec           (== F_ref + dE_expl_elec/dR)
stays strictly conservative. Do NOT hand-code a separate force formula.

Convention (must match training AND inference):
  * point charges (extend to multipoles below if needed)
  * pairwise 1/r Coulomb with optional exclusions and a smooth cutoff switch
  * units controlled by KE (Coulomb constant); charges in e, R in the same
    length unit KE is defined for.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

# Coulomb constant. Pick ONE unit system and keep it identical everywhere.
#   kcal/mol, Angstrom, e  -> 332.0637128
#   eV,       Angstrom, e  -> 14.399645
KE_KCAL_ANG = 332.0637128


def _switch(r, r_on, r_off):
    """CHARMM-style C^1 smooth switch: 1 below r_on, 0 above r_off."""
    x = (r_off**2 - r**2) / (r_off**2 - r_on**2)
    s = x * x * (3.0 - 2.0 * x)
    s = jnp.where(r <= r_on, 1.0, s)
    s = jnp.where(r >= r_off, 0.0, s)
    return s


def elec_energy(
    R,                 # (N,3) positions
    q,                 # (N,)  charges
    pair_i, pair_j,    # (P,)  int index arrays of included pairs (excl. already removed)
    ke=KE_KCAL_ANG,
    r_on=None,
    r_off=None,
    eps=1e-12,
):
    """Total explicit electrostatic energy (scalar).

    Passing an explicit pair list (rather than a full NxN mask) lets you encode
    exclusions/1-4 scaling once and reuse it; scaling can be folded into `q`
    products via an optional per-pair weight if you extend the signature.
    """
    rij = R[pair_i] - R[pair_j]                 # (P,3)
    r = jnp.sqrt(jnp.sum(rij * rij, axis=-1) + eps)
    qq = q[pair_i] * q[pair_j]
    e = ke * qq / r
    if r_off is not None:
        e = e * _switch(r, r_on if r_on is not None else 0.0, r_off)
    return jnp.sum(e)


# forces = -dE/dR, computed by autodiff -> conservative by construction.
_elec_grad = jax.grad(elec_energy, argnums=0)


@jax.jit
def elec_energy_and_forces(R, q, pair_i, pair_j, ke=KE_KCAL_ANG, r_on=None, r_off=None):
    e = elec_energy(R, q, pair_i, pair_j, ke, r_on, r_off)
    f = -_elec_grad(R, q, pair_i, pair_j, ke, r_on, r_off)
    return e, f


def build_pairs(n_atoms, exclusions=None):
    """All i<j pairs minus `exclusions` (a set/iterable of (i,j) tuples)."""
    import itertools
    excl = set()
    if exclusions:
        for a, b in exclusions:
            excl.add((min(a, b), max(a, b)))
    ii, jj = [], []
    for i, j in itertools.combinations(range(n_atoms), 2):
        if (i, j) in excl:
            continue
        ii.append(i); jj.append(j)
    return jnp.asarray(ii, dtype=jnp.int32), jnp.asarray(jj, dtype=jnp.int32)


if __name__ == "__main__":
    import numpy as np
    key = jax.random.PRNGKey(0)
    R = jax.random.normal(key, (12, 3)) * 3.0
    q = jnp.asarray(np.random.uniform(-0.8, 0.8, 12).astype(np.float32))
    q = q - jnp.mean(q)                       # net-neutral for sanity
    pi, pj = build_pairs(12, exclusions=[(0, 1), (2, 3)])
    e, f = elec_energy_and_forces(R, q, pi, pj)
    # finite-difference check on the conservative force
    h = 1e-4
    g = np.zeros_like(np.asarray(R))
    for a in range(12):
        for d in range(3):
            Rp = R.at[a, d].add(h); Rm = R.at[a, d].add(-h)
            g[a, d] = -(float(elec_energy(Rp, q, pi, pj)) -
                        float(elec_energy(Rm, q, pi, pj))) / (2 * h)
    print("E =", float(e))
    print("max |F_autodiff - F_fd| =", np.max(np.abs(np.asarray(f) - g)))
