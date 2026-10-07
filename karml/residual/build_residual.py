"""Build residual training targets and write an audit-friendly NPZ.

Output fields (per your spec):
  R, Z            geometry + species (passthrough)
  E_ref, F_ref    original reference (kept for audit)
  E_expl_elec     explicit electrostatic energy subtracted
  F_expl_elec     its analytic (conservative) forces
  E_residual      E_ref - E_expl_elec        <- ML energy target
  F_residual      F_ref - F_expl_elec        <- ML force target

At inference reconstruct:  E_total = E_ML + E_expl_elec ; F_total likewise.

Backend `elec_fn(R, q, pi, pj)` -> (E, F) is pluggable; pass the JAX one for
guaranteed autodiff consistency, or a Fortran one for speed. They must use the
SAME ke / cutoff / switch / exclusion convention as deployment.
"""
from __future__ import annotations

import numpy as np


def build_residual_npz(
    out_path,
    R,                 # (M, N, 3) frames
    Z,                 # (N,) species
    E_ref,             # (M,)
    F_ref,             # (M, N, 3)
    q,                 # (N,)  fixed MM-style charges (see note on response terms)
    pair_i, pair_j,    # (P,)  included pairs after exclusions
    elec_fn,           # callable(R_frame, q, pair_i, pair_j) -> (E, F)
    ke=332.0637128,
    r_on=None,
    r_off=None,
):
    R = np.asarray(R, dtype=np.float64)
    F_ref = np.asarray(F_ref, dtype=np.float64)
    E_ref = np.asarray(E_ref, dtype=np.float64)
    M = R.shape[0]

    E_elec = np.empty(M, dtype=np.float64)
    F_elec = np.empty_like(F_ref)
    for m in range(M):
        e, f = elec_fn(R[m], q, pair_i, pair_j)   # units/convention fixed by elec_fn
        E_elec[m] = float(e)
        F_elec[m] = np.asarray(f)

    E_res = E_ref - E_elec
    F_res = F_ref - F_elec                          # == F_ref + dE_elec/dR

    np.savez_compressed(
        out_path,
        R=R, Z=np.asarray(Z),
        E_ref=E_ref, F_ref=F_ref,
        E_expl_elec=E_elec, F_expl_elec=F_elec,
        E_residual=E_res, F_residual=F_res,
        q=np.asarray(q), pair_i=np.asarray(pair_i), pair_j=np.asarray(pair_j),
        ke=ke, r_on=(np.nan if r_on is None else r_on),
        r_off=(np.nan if r_off is None else r_off),
    )
    # cheap sanity: residual should be smaller than reference if elec dominates
    print(f"[residual] frames={M}  "
          f"RMS E_ref={np.sqrt(np.mean(E_ref**2)):.4f}  "
          f"RMS E_res={np.sqrt(np.mean(E_res**2)):.4f}  "
          f"RMS |F_ref|={np.sqrt(np.mean(F_ref**2)):.4f}  "
          f"RMS |F_res|={np.sqrt(np.mean(F_res**2)):.4f}")
    return out_path


if __name__ == "__main__":
    # end-to-end demo with the JAX backend
    import jax
    from mmml.residual.electrostatics_jax import (
        elec_energy_and_forces, build_pairs, KE_KCAL_ANG,
    )

    rng = np.random.default_rng(0)
    M, N = 8, 10
    R = rng.normal(size=(M, N, 3)) * 3.0
    Z = rng.integers(1, 9, size=N)
    q = rng.uniform(-0.5, 0.5, N); q -= q.mean()
    E_ref = rng.normal(size=M)
    F_ref = rng.normal(size=(M, N, 3))
    pi, pj = build_pairs(N, exclusions=[(0, 1)])

    def elec_fn(Rf, q, pi, pj):
        e, f = elec_energy_and_forces(
            jax.numpy.asarray(Rf), jax.numpy.asarray(q), pi, pj, KE_KCAL_ANG
        )
        return float(e), np.asarray(f)

    build_residual_npz(
        "/tmp/residual_demo.npz", R, Z, E_ref, F_ref, q, pi, pj, elec_fn,
    )
