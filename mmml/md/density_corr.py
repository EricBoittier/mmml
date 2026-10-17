"""Short-range intermolecular density correction for the water/methanol ML/MM hybrid.

PET-OMol-L (and the students distilled from it) favour compression of the liquid relative to periodic GGA-D3
(RPBE-D3 / revPBE-D3 on 88 COM-scaled ~100-atom boxes, 0.88-1.12 rho_exp): +0.55 (methanol) to +0.90 GPa (water)
of pressure bias. The bias is fitted by a repulsive pair term over heavy atoms of different molecules

    U = scale * sum_{i<j, mol_i != mol_j, Z_i, Z_j > 1}  c_{t(i,j)} exp(-r_ij / lam) s(r_ij),

s a cosine switch from r_on to r_off, t the pair type (OO, OC, CC), c in kcal/mol (condensed_ref/scripts/
fit_density_correction*.py; CV residual 0.046 kcal/mol per molecule). ``scale`` (or per-type ``scale_types``) is
the knob tuned to the experimental density: at scale 1 the term moves the hybrid to the GGA-D3 compression energy,
and the GGAs are themselves 7-11 % too light.

Dense minimum-image sum over heavy atoms (O(n_heavy^2), ~3 M pairs at 38 A); orthorhombic box, r_off < L/2.
params json: {"lam_A": 0.6, "r_on_A": 5.0, "r_off_A": 6.0, "c_kcal_mol": {"OO": .., "OC": .., "CC": ..},
              "scale": 1.0, "scale_types": {"OO": 1.0, ...} (optional)}
"""
from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np

KCAL_TO_EV = 1.0 / 23.060548
_SYM = {6: "C", 8: "O"}


def _pair_key(a, b):
    return "".join(sorted((a, b), key=lambda s: "OC".index(s)))   # "OO", "OC", "CC"


def make_density_corr_fn(atomic_numbers, monomer_offsets, params_json, box_L, scale=None, dtype=jnp.float32):
    """energy(R, box=None, lam=None) in eV; ``box`` = current box (scalar, (3,) or (3, 3) diagonal) or the fixed one."""
    par = json.load(open(params_json)) if isinstance(params_json, str) else dict(params_json)
    lam, r_on, r_off = float(par["lam_A"]), float(par["r_on_A"]), float(par["r_off_A"])
    s_glob = float(par.get("scale", 1.0) if scale is None else scale)
    s_t = par.get("scale_types", {})
    Z = np.asarray(atomic_numbers).reshape(-1)
    offs = np.asarray(monomer_offsets).reshape(-1)
    mol = np.zeros(len(Z), np.int32)
    for m in range(len(offs) - 1):
        mol[offs[m]:offs[m + 1]] = m
    heavy = np.where(Z > 1)[0]
    sym = [_SYM[int(z)] for z in Z[heavy]]
    cmat = np.zeros((len(heavy), len(heavy)), np.float64)
    for key in ("OO", "OC", "CC"):
        c = float(par["c_kcal_mol"][key]) * KCAL_TO_EV * s_glob * float(s_t.get(key, 1.0))
        a = np.array([_pair_key(p, q) == key for p in sym for q in sym]).reshape(len(heavy), len(heavy))
        cmat[a] = c
    same = mol[heavy][:, None] == mol[heavy][None, :]
    iu = np.triu(np.ones_like(same), 1).astype(bool) & ~same
    cmat = np.where(iu, cmat, 0.0)
    idx = jnp.asarray(heavy, jnp.int32); C = jnp.asarray(cmat, dtype)
    L0 = jnp.asarray(np.full(3, float(box_L)), dtype)

    def _L(box):
        if box is None:
            return L0
        b = jnp.asarray(box, dtype)
        if b.ndim == 0:
            return jnp.full((3,), b)
        if b.ndim == 2:
            return jnp.diagonal(b)
        return b

    def energy(R, box=None, lam_alch=None):
        L = _L(box)
        X = jnp.asarray(R, dtype)[idx]
        d = X[:, None, :] - X[None, :, :]
        d = d - L * jnp.round(d / L)
        r2 = jnp.sum(d * d, -1)
        mask = C != 0
        r = jnp.sqrt(jnp.where(mask, r2, 1.0))
        t = jnp.clip((r - r_on) / (r_off - r_on), 0.0, 1.0)
        sw = 0.5 * (1.0 + jnp.cos(jnp.pi * t))
        return jnp.sum(jnp.where(mask, C * jnp.exp(-r / lam) * sw, 0.0))

    f = jax.jit(energy)
    f.n_heavy = int(len(heavy)); f.scale = s_glob
    return f
