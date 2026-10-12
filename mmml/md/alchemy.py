"""Alchemical decoupling of one molecule in the jax-md ML/MM hybrid (prototype; self-solvation free energies).

Hamiltonian, tagged molecule t, MMML_ALCH='{"molecule": t, "lambda_ml": .., "lambda_elec": .., "lambda_vdw": .., "alpha": ..}':

    H = H_hybrid(lambda_monomer[t] = lambda_ml) + (1 - lambda_ml) * U_sc(t; lambda_elec, lambda_vdw)

H_hybrid is linear in lambda_ml (ML dimer interactions, switched MM tail and the short-range wall of every pair that
contains t scale with lambda_ml; t's own monomer energy does not), so with W(t) = dH_hybrid/dlambda_ml the reduced
potential of any state is known from W and U_sc. Two stages:
    S1  lambda_ml 1 -> 0 at lambda_elec = lambda_vdw = 1          (hybrid ML/MM interaction of t -> plain MM; no overlap)
    S2  lambda_ml = 0, lambda_elec 1 -> 0, then lambda_vdw 1 -> 0  (soft-core MM decoupling)
dG_solv = -(dF_S1 + dF_S2); intramolecular terms are never scaled, so no gas-phase leg.

U_sc: CHARMM LJ (eps_ij = sqrt(eps_i eps_j), Rmin_ij = Rmin/2_i + Rmin/2_j) with Beutler soft-core
    U_LJ = lambda_vdw 4 eps (1/x^2 - 1/x),  x = alpha (1 - lambda_vdw) + r^6 / sigma^6,  sigma^6 = Rmin^6 / 2
plus Coulomb lambda_elec 332.0637 q_i q_j / r, each molecule pair multiplied by a cosine switch of the centroid distance
(r_on .. r_off, default 9 .. 12 Å; needs r_off < L/2). Atom vectors go through the minimum image of the molecule centroid.
Energies in eV.
"""
from __future__ import annotations

import json
import os

import jax
import jax.numpy as jnp
import numpy as np

KCAL2EV = 1.0 / 23.0605
COULOMB = 332.0637

# lambda states (lambda_ml, lambda_elec, lambda_vdw) evaluated for MBAR; kappa = 1 - lambda_ml
S1_ML = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
S2_ELEC = [1.0, 0.75, 0.5, 0.25, 0.0]
S2_VDW = [1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.0]
STATES = ([(l, 1.0, 1.0) for l in S1_ML] + [(0.0, e, 1.0) for e in S2_ELEC[1:]]
          + [(0.0, 0.0, v) for v in S2_VDW[1:]])          # 26 states, index 5 = (0, 1, 1) joins S1 and S2


def spec_from_env():
    s = os.environ.get("MMML_ALCH")
    if not s:
        return None
    d = json.loads(s) if s.strip().startswith("{") else json.load(open(s))
    d.setdefault("molecule", 0); d.setdefault("lambda_ml", 1.0); d.setdefault("lambda_elec", 1.0)
    d.setdefault("lambda_vdw", 1.0); d.setdefault("alpha", 0.5)
    return d


def lambda_monomer_for(n_monomers, spec, lam=None):
    a = np.ones(int(n_monomers)); a[int(spec["molecule"])] = spec["lambda_ml"] if lam is None else lam
    return a


def make_alch_mm(molpair_spec, box_L, molecule=0, alpha=0.5, r_on=9.0, r_off=12.0):
    """-> (energy(R, lam_e, lam_v) eV, energy_states(R) (n_states,) eV for STATES' (lam_e, lam_v) at kappa = 1)."""
    off = np.asarray(molpair_spec["monomer_offsets"]); n_at = int(off[-1])
    q = np.asarray(molpair_spec["q"], np.float64).reshape(-1)[:n_at]
    rmh = np.asarray(molpair_spec["rmin_half"], np.float64).reshape(-1)[:n_at]
    eps = np.abs(np.asarray(molpair_spec["eps"], np.float64).reshape(-1)[:n_at])
    mol = np.repeat(np.arange(len(off) - 1), np.diff(off))
    # The hybrid reads LJ from the CGenFF prm only: TIP3P types (OT, HT) are absent there and come out with eps = 0
    # unless MMML_MM_LJ_ALL_TYPES=1. The decoupling term needs real repulsion, so missing TIP3P LJ is filled with the
    # CHARMM values (toppar_water_ions: OT eps -0.1521, Rmin/2 1.7682; HT eps -0.046, Rmin/2 0.2245), identified by charge.
    q_all = np.asarray(molpair_spec["q"], np.float64).reshape(-1)[:n_at]
    eps = eps.copy(); rmh = rmh.copy()
    ot = (eps == 0) & np.isclose(q_all, -0.834); ht = (eps == 0) & np.isclose(q_all, 0.417)
    eps[ot], rmh[ot] = 0.1521, 1.7682
    eps[ht], rmh[ht] = 0.046, 0.2245
    if ot.any() or ht.any():
        print(f"[alchemy] soft-core LJ: filled TIP3P CHARMM LJ for {int(ot.sum())} OT + {int(ht.sum())} HT atoms "
              f"(eps = 0 in the hybrid's CGenFF-only table); {int((eps == 0).sum())} atoms still without LJ", flush=True)
    t_idx = np.arange(off[molecule], off[molecule + 1]); o_idx = np.setdiff1d(np.arange(n_at), t_idx)
    L = float(box_L)
    # pair parameters (tagged atoms x other atoms), constants
    eps_ij = jnp.asarray(np.sqrt(eps[t_idx][:, None] * eps[o_idx][None, :]))
    rmin = rmh[t_idx][:, None] + rmh[o_idx][None, :]
    s6 = jnp.asarray(np.maximum(rmin**6 / 2.0, 1e-6))
    qq = jnp.asarray(COULOMB * q[t_idx][:, None] * q[o_idx][None, :])
    o_mol = jnp.asarray(mol[o_idx]); n_mol = len(off) - 1
    counts = jnp.asarray(np.bincount(mol, minlength=n_mol).astype(np.float64))
    mol_j = jnp.asarray(mol)
    t_idx_j, o_idx_j = jnp.asarray(t_idx), jnp.asarray(o_idx)

    def mic(d):
        return d - L * jnp.round(d / L)

    def whole_and_centroids(R):
        # molecules made whole about their first atom; centroid = plain mean (the hybrid's switching centroid)
        first = R[jnp.asarray(off[:-1])][mol_j]
        Rw = first + mic(R - first)
        C = jax.ops.segment_sum(Rw, mol_j, num_segments=n_mol) / counts[:, None]
        return Rw, C

    def switch(Rc):
        x = jnp.clip((Rc - r_on) / (r_off - r_on), 0.0, 1.0)
        return 0.5 * (1.0 + jnp.cos(jnp.pi * x))

    def pair_terms(R):
        R = jnp.asarray(R)
        Rw, C = whole_and_centroids(R)
        dC = mic(C - C[molecule])                                          # (n_mol, 3) image of each molecule
        Sm = switch(jnp.sqrt(jnp.sum(dC * dC, -1) + 1e-12))                 # safe norm: dC[t] = 0
        S = Sm[o_mol][None, :]
        # atom vectors through the molecule's image (consistent for whole molecules, any extent)
        xo = Rw[o_idx_j] - C[o_mol] + dC[o_mol]
        xt = Rw[t_idx_j] - C[molecule]
        d = xo[None, :, :] - xt[:, None, :]
        r2 = jnp.sum(d * d, -1)
        r6 = r2**3
        r = jnp.sqrt(jnp.maximum(r2, 1e-6))
        return S, r6, r

    def lj(S, r6, lam_v):
        x = alpha * (1.0 - lam_v) + r6 / s6
        return jnp.sum(S * lam_v * 4.0 * eps_ij * (1.0 / x**2 - 1.0 / x))

    def el(S, r, lam_e):
        return lam_e * jnp.sum(S * qq / jnp.maximum(r, 0.3))

    def energy(R, lam_e, lam_v):
        S, r6, r = pair_terms(R)
        return (lj(S, r6, lam_v) + el(S, r, lam_e)) * KCAL2EV

    st = jnp.asarray([(e, v) for _, e, v in STATES])

    def energy_states(R):
        S, r6, r = pair_terms(R)
        return jax.vmap(lambda ev: (lj(S, r6, ev[1]) + el(S, r, ev[0])) * KCAL2EV)(st)

    return jax.jit(energy), jax.jit(energy_states)
