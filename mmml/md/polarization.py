"""Many-body polarisation term for the water/methanol ML/MM hybrid (prototype).

Thole-damped induced point dipoles on fixed TIP3P / CGenFF charges with fitted per-type polarisabilities
(``params.json`` from training/watmeoh/prod/manybody). Only the many-body part is added, because the ML pair term
already contains every pair's polarisation inside the ML range:

    E_add = U_pol(system) - sum_{molecule pairs} s(r_com) U_pol(pair)

``s`` is the ML pair switch (1 below ``r_on``, cosine to 0 at ``r_off``). U_pol(system) is evaluated densely with the
minimum image and a smooth real-space taper (``r_taper_on`` .. ``r_taper_off``), dipoles by a fixed number of Jacobi
iterations (differentiable). Pair energies use a top-k selection of the closest molecule pairs (static shape).
Atom order per molecule as in the CHARMM PSF: TIP3 O,H,H; MEOH CB,OG,HG1,HB1,HB2,HB3.
Enable in jax-md with MMML_POL_PARAMS=/path/params.json (NVT/NVE/NpT; orthorhombic box). ``energy(R, box=None)``
takes the current box (scalar, (3,) or diagonal (3, 3)) so the barostat sees the polarisation strain derivative.
"""
from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np

KC = 332.0637 / 23.0605  # e^2/A -> eV (energies in eV for the hybrid)
NAMES = ["O_w", "H_w", "O_m", "H_o", "C_m", "H_m"]


def _molecules(Z):
    """Split the PSF atom sequence into molecules -> (types, charges, mol_id, atom table (n_mol, 6) padded -1)."""
    Z = list(map(int, Z)); i = 0; types, q, mol, table = [], [], [], []
    while i < len(Z):
        if Z[i] == 8 and Z[i + 1] == 1 and Z[i + 2] == 1:          # TIP3 O H H
            types += [0, 1, 1]; q += [-0.834, 0.417, 0.417]; idx = [i, i + 1, i + 2]; i += 3
        elif Z[i] == 6:                                              # MEOH CB OG HG1 HB1 HB2 HB3
            types += [4, 2, 3, 5, 5, 5]; q += [-0.04, -0.65, 0.42, 0.09, 0.09, 0.09]; idx = list(range(i, i + 6)); i += 6
        else:
            raise ValueError(f"unexpected atom sequence at {i}: Z={Z[i:i + 6]}")
        mol += [len(table)] * len(idx); table.append(idx + [-1] * (6 - len(idx)))
    return np.array(types), np.array(q), np.array(mol), np.array(table)


def _switch(r, r_on, r_off):
    x = jnp.clip((r - r_on) / (r_off - r_on), 0.0, 1.0)
    return 0.5 * (1 + jnp.cos(jnp.pi * x))


def _upol_dense(R, alpha, q, mol, a, L, taper, n_iter, tag=None, lam=1.0):
    d = R[:, None, :] - R[None, :, :]
    if L is not None:
        d = d - L * jnp.round(d / L)
    n = R.shape[0]
    r = jnp.sqrt(jnp.sum(d * d, -1) + jnp.eye(n))
    inter = (mol[:, None] != mol[None, :]).astype(R.dtype)
    if taper is not None:
        inter = inter * _switch(r, taper[0], taper[1])
    if tag is not None:   # alchemy: interactions between the tagged molecule and the rest scale with lam
        inter = inter * jnp.where(tag[:, None] != tag[None, :], lam, 1.0)
    u = r / (alpha[:, None] * alpha[None, :]) ** (1 / 6)
    e = jnp.exp(-a * u**3); l3 = 1 - e; l5 = 1 - (1 + a * u**3) * e
    E0 = jnp.einsum("ij,ijk->ik", inter * q[None, :] * l3 / r**3, d)
    w3 = inter * l3 / r**3; w5 = inter * 3 * l5 / r**5

    def Tmu(mu):  # sum_j T_ij mu_j without forming the 3n x 3n matrix
        dm = jnp.einsum("ijk,jk->ij", d, mu)
        return jnp.einsum("ij,ijk->ik", w5 * dm, d) - jnp.einsum("ij,jk->ik", w3, mu)

    # Jacobi iterations as a fori_loop over a rematerialised body: reverse mode then stores only the (n, 3) dipoles per
    # iteration instead of every n x n intermediate (several GB at ~2,600 atoms); same arithmetic.
    body = jax.checkpoint(lambda _, mu: alpha[:, None] * (E0 + Tmu(mu)))
    mu = jax.lax.fori_loop(0, n_iter, body, alpha[:, None] * E0)
    return -0.5 * KC * jnp.sum(mu * E0)


def _upol_sparse(R, alpha, q, mol, a, L, taper, n_iter, idx, tag=None, lam=1.0):
    """_upol_dense restricted to the neighbour pairs idx (n, K; padded with n) within taper[1]: same energy, since the
    taper sets every interaction beyond taper[1] to zero. Memory and work O(n K) instead of O(n^2)."""
    n = R.shape[0]
    valid = idx < n
    j = jnp.where(valid, idx, 0)
    d = R[:, None, :] - R[j]                                     # (n, K, 3), r_i - r_j as in the dense kernel
    d = d - L * jnp.round(d / L)
    r2 = jnp.sum(d * d, -1)
    inter = (valid & (mol[:, None] != mol[j])).astype(R.dtype)
    r = jnp.sqrt(jnp.where(inter > 0, r2, 1.0))
    inter = inter * _switch(r, taper[0], taper[1])
    if tag is not None:
        inter = inter * jnp.where(tag[:, None] != tag[j], lam, 1.0)
    u = r / (alpha[:, None] * alpha[j]) ** (1 / 6)
    e = jnp.exp(-a * u**3); l3 = 1 - e; l5 = 1 - (1 + a * u**3) * e
    E0 = jnp.einsum("ik,ikc->ic", inter * q[j] * l3 / r**3, d)
    w3 = inter * l3 / r**3; w5 = inter * 3 * l5 / r**5

    def Tmu(mu):
        mj = mu[j]                                                # (n, K, 3)
        dm = jnp.sum(d * mj, -1)
        return jnp.einsum("ik,ikc->ic", w5 * dm, d) - jnp.einsum("ik,ikc->ic", w3, mj)

    body = jax.checkpoint(lambda _, mu: alpha[:, None] * (E0 + Tmu(mu)))
    mu = jax.lax.fori_loop(0, n_iter, body, alpha[:, None] * E0)
    return -0.5 * KC * jnp.sum(mu * E0)


def make_pol_energy_fn(Z, box_L, params_json, r_on=4.5, r_off=6.0, taper=(9.0, 12.0), n_iter=12, pair_cap=None,
                       tagged=None, lam=1.0, neighbor_R=None, capacity_multiplier=1.3, system_term=True):
    """energy(R, box=None, lam=None) in eV. tagged: molecule index whose coupling to the rest (permanent field both ways and
    dipole-dipole) scales with lam (alchemical decoupling; lam=None -> the constructor's lam). tagged=None: no scaling.
    neighbor_R: positions to size an atom-pair neighbour list (cutoff taper[1]) for the system term -> O(n K) instead of
    O(n^2); fixed box only (NVT/NVE): the box argument is then ignored. The list is rebuilt on every call."""
    par = json.load(open(params_json))
    alpha_t = jnp.asarray([par["alpha_A3"][k] for k in NAMES]); a = float(par["thole_a"])
    types, q, mol, table = _molecules(Z)
    n_mol = len(table); n = len(types)
    alpha = alpha_t[types]; qj = jnp.asarray(q); molj = jnp.asarray(mol)
    # pair gathering: index n = dummy atom (q = 0, far away)
    tab = np.where(table < 0, n, table); tabj = jnp.asarray(tab)
    alpha_x = jnp.concatenate([alpha, jnp.ones(1)]); q_x = jnp.concatenate([qj, jnp.zeros(1)])
    tag = None if tagged is None else jnp.asarray(np.asarray(mol) == int(tagged))
    tag_x = None if tag is None else jnp.concatenate([tag, jnp.zeros(1, bool)])
    lam0 = float(lam)
    mass = np.where(np.array(Z) == 1, 1.008, np.where(np.array(Z) == 6, 12.011, 15.999))
    K = int(pair_cap or 10 * n_mol)
    iu = np.triu_indices(n_mol, 1)
    pbc = bool(box_L)
    L0 = jnp.asarray(float(box_L)) if pbc else None

    def _L(box):
        if not pbc:
            return None
        if box is None:
            return L0
        b = jnp.asarray(box)
        return jnp.diagonal(b) if b.ndim == 2 else b

    def com(R, L):
        Rx = jnp.concatenate([R, jnp.zeros((1, 3), R.dtype)]); mx = jnp.concatenate([jnp.asarray(mass), jnp.zeros(1)])
        X = Rx[tabj]; m = mx[tabj]
        # whole molecules: unwrap each atom relative to the molecule's first atom
        if L is not None:
            X = X[:, :1] + (X - X[:, :1]) - L * jnp.round((X - X[:, :1]) / L)
        return jnp.sum(X * m[..., None], 1) / jnp.sum(m, 1)[:, None], X

    nfn = nbrs0 = None
    if neighbor_R is not None and pbc:
        from jax_md import partition, space
        disp, _ = space.periodic(float(box_L))
        nfn = partition.neighbor_list(disp, float(box_L), float(taper[1]), dr_threshold=0.0,
                                      capacity_multiplier=float(capacity_multiplier), format=partition.Dense)
        R0 = jnp.asarray(neighbor_R, jnp.float32); R0 = R0 - float(box_L) * jnp.floor(R0 / float(box_L))
        nbrs0 = nfn.allocate(R0)

    def energy(R, box=None, lam=None):
        lam = lam0 if lam is None else lam
        R = jnp.asarray(R); L = _L(box)
        if L is not None:
            L = L.astype(R.dtype)
        if not system_term:     # pair-subtraction term only (the caller supplies the system term, e.g. pol_ewald)
            e_sys = jnp.zeros((), R.dtype)
        elif nfn is not None:
            Rw = R - L0.astype(R.dtype) * jnp.floor(R / L0.astype(R.dtype))
            nb = nfn.update(Rw, nbrs0)
            e_sys = _upol_sparse(R, alpha.astype(R.dtype), qj.astype(R.dtype), molj, a, L0.astype(R.dtype), taper, n_iter, nb.idx, tag, lam)
        else:
            e_sys = _upol_dense(R, alpha.astype(R.dtype), qj.astype(R.dtype), molj, a, L, taper if L is not None else None, n_iter, tag, lam)
        C, X = com(R, L)
        dc = C[iu[0]] - C[iu[1]]
        if L is not None:
            dc = dc - L * jnp.round(dc / L)
        rc = jnp.linalg.norm(dc, axis=-1)
        _, top = jax.lax.top_k(-rc, K)                       # K closest molecule pairs
        i, j = jnp.asarray(iu[0])[top], jnp.asarray(iu[1])[top]
        s = _switch(rc[top], r_on, r_off)
        Xi = X[i]
        Xj = X[j] + ((C[i] - dc[top]) - C[j])[:, None, :]    # place j at its minimum image relative to i
        Rp = jnp.concatenate([Xi, Xj], 1)                    # (K, 12, 3)
        ai = jnp.concatenate([alpha_x[tabj[i]], alpha_x[tabj[j]]], 1)
        qp = jnp.concatenate([q_x[tabj[i]], q_x[tabj[j]]], 1)
        mp = jnp.concatenate([jnp.zeros((K, 6)), jnp.ones((K, 6))], 1)
        # dummy (padding) atoms: move far away so they do not interact
        pad = jnp.concatenate([tabj[i], tabj[j]], 1) == n
        far = 1000.0 + 50.0 * jnp.arange(12)[None, :, None] * jnp.array([1.0, 0.7, 0.3])
        Rp = jnp.where(pad[..., None], far, Rp)
        mp = jnp.where(pad, 2.0 + jnp.arange(12)[None, :], mp)
        if tag is None:
            ep = jax.vmap(lambda r_, a_, q_, m_: _upol_dense(r_, a_, q_, m_, a, None, None, n_iter))(Rp, ai, qp, mp)
        else:
            tp = jnp.concatenate([tag_x[tabj[i]], tag_x[tabj[j]]], 1)
            ep = jax.vmap(lambda r_, a_, q_, m_, t_: _upol_dense(r_, a_, q_, m_, a, None, None, n_iter, t_, lam))(Rp, ai, qp, mp, tp)
        return e_sys - jnp.sum(s * ep)

    f = jax.jit(energy)
    if nfn is not None:   # host-side capacity check: overflow(R) -> True if the list would drop neighbours
        f_over = jax.jit(lambda R: nfn.update(jnp.asarray(R) - L0 * jnp.floor(jnp.asarray(R) / L0), nbrs0).did_buffer_overflow)
        try:
            f.overflow = f_over
        except AttributeError:
            pass
    return f
