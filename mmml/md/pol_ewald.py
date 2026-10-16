"""Induced dipoles with Ewald (P3M) electrostatics: the periodic system term of mmml.md.polarization without the 9-12 A
taper, at a fraction of the cost.

The induced dipoles solve mu = alpha E(mu), with the field E = -dU/dmu of the charge-dipole + dipole-dipole energy
(e^2/A units; Thole-damped, intermolecular only, all periodic images)::

    U(mu) = sum_{i != j} -q_j (mu_i . d_ij) t3_ij  +  1/2 sum_{i != j} [ (mu_i . mu_j) t3_ij - (mu_i . d_ij)(mu_j . d_ij) t5_ij ]
    t3 = l3(u) / r^3,  t5 = 3 l5(u) / r^5,  d_ij = r_i - r_j  (same kernel as polarization._upol_dense, without the taper)

split Ewald-style into
  * real space, pairs within r_c (neighbour list): screened B1/B2 kernels plus the short-range Thole correction
    (l3 - 1)/r^3, 3 (l5 - 1)/r^5, which vanishes beyond ~4 A;
  * reciprocal space: P3M mesh energy of the combined density q W + mu . grad W (dipoles assigned as the directional
    derivative of the charge assignment, so the same mesh, influence function and FFT serve both);
  * self: -2 beta^3 / (3 sqrt(pi)) sum mu^2 (the charge self term does not depend on mu);
  * exclusion: intramolecular pairs remove their smooth (bare - real-space) part.
Jacobi iterations exactly as in the dense kernel (n_iter, start alpha E0, checkpointed body); U_pol = -1/2 KC mu . E0.

    E = make_pol_ewald_energy_fn(Z, box_L, params_json, R_init)      # energy(R) in eV, fixed cubic box
"""
from __future__ import annotations

import json
import math

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import erfc

from mmml.md import polarization as pol
from mmml.md.p3m import P3M

KC = pol.KC
SQPI = math.sqrt(math.pi)


def _b12(r, beta):
    """Ewald-screened kernels B1 ~ 1/r^3, B2 ~ 3/r^5 (Smith 1982)."""
    g = jnp.exp(-(beta * r) ** 2)
    b0 = erfc(beta * r) / r
    b1 = (b0 + 2 * beta / SQPI * g) / r**2
    b2 = (3 * b1 + 4 * beta**3 / SQPI * g) / r**2
    return b1, b2


class PolEwald:
    def __init__(self, Z, box_L, params_json, R_init, r_cut=7.5, beta=0.42, mesh=None, order=5, n_iter=12,
                 capacity_multiplier=1.3, dtype=jnp.float32):
        par = json.load(open(params_json))
        alpha_t = np.asarray([par["alpha_A3"][k] for k in pol.NAMES]); self.a = float(par["thole_a"])
        types, q, mol, _ = pol._molecules(Z)
        self.alpha = jnp.asarray(alpha_t[types], dtype); self.q = jnp.asarray(q, dtype); self.mol = jnp.asarray(mol)
        self.L = float(box_L); self.beta = float(beta); self.rc = float(r_cut); self.n_iter = int(n_iter); self.dtype = dtype
        n = len(q)
        if mesh is None:
            mesh = int(next(m for m in (16, 20, 24, 30, 32, 36, 40, 48, 50, 60, 64, 72, 80, 96) if self.L / m <= 0.96))
        self.p3m = P3M(self.L, mesh, order, self.beta, dtype=dtype)
        # intramolecular pairs (i < j)
        ii, jj = [], []
        mol_np = np.asarray(mol)
        for m in np.unique(mol_np):
            at = np.where(mol_np == m)[0]
            for x in range(len(at)):
                for y in range(x + 1, len(at)):
                    ii.append(at[x]); jj.append(at[y])
        self.ei = jnp.asarray(np.array(ii, np.int32)); self.ej = jnp.asarray(np.array(jj, np.int32))
        from jax_md import partition, space
        disp, _ = space.periodic(self.L)
        self.nfn = partition.neighbor_list(disp, self.L, self.rc, dr_threshold=0.0,
                                           capacity_multiplier=float(capacity_multiplier), format=partition.Dense)
        R0 = jnp.asarray(R_init, jnp.float32); R0 = R0 - self.L * jnp.floor(R0 / self.L)
        self.nbrs0 = self.nfn.allocate(R0)
        self.ones = jnp.ones(n, dtype)

    # -------------------------------------------------------------------- pieces
    def _pairs(self, R):
        """Neighbour geometry (n, K): d, r, inter mask, screened + Thole kernels k1 ~ t3, k2 ~ t5."""
        n = R.shape[0]; L = jnp.asarray(self.L, R.dtype)
        idx = self.nfn.update(R - L * jnp.floor(R / L), self.nbrs0).idx
        valid = idx < n; j = jnp.where(valid, idx, 0)
        d = R[:, None, :] - R[j]; d = d - L * jnp.round(d / L)
        inter = valid & (self.mol[:, None] != self.mol[j])
        r = jnp.sqrt(jnp.where(inter, jnp.sum(d * d, -1), 1.0))
        b1, b2 = _b12(r, self.beta)
        u = r / (self.alpha[:, None] * self.alpha[j]) ** (1 / 6)
        e = jnp.exp(-self.a * u**3); l3 = 1 - e; l5 = 1 - (1 + self.a * u**3) * e
        m = inter.astype(R.dtype)
        k1 = m * (b1 + (l3 - 1) / r**3)
        k2 = m * (b2 + 3 * (l5 - 1) / r**5)
        return j, d, k1, k2

    def _excl(self, R):
        L = jnp.asarray(self.L, R.dtype)
        d = R[self.ei] - R[self.ej]; d = d - L * jnp.round(d / L)
        r = jnp.sqrt(jnp.sum(d * d, -1)); b1, b2 = _b12(r, self.beta)
        return d, 1 / r**3 - b1, 3 / r**5 - b2          # smooth parts carried by the mesh

    def u_mu(self, mu, R, geo, exc):
        """mu-dependent electrostatic energy (e^2/A)."""
        j, d, k1, k2 = geo
        q = self.q.astype(R.dtype)
        mud = jnp.sum(mu[:, None, :] * d, -1)                        # (n, K)  mu_i . d_ij
        mjd = jnp.sum(mu[j] * d, -1)                                 # mu_j . d_ij
        e_real = jnp.sum(-q[j] * mud * k1) + 0.5 * jnp.sum(jnp.sum(mu[:, None, :] * mu[j], -1) * k1 - mud * mjd * k2)
        # mesh: charges + dipoles (directional derivative of the unit-charge assignment along mu)
        Qq = self.p3m.mesh_charges(R, q)
        Qmu = jax.jvp(lambda x: self.p3m.mesh_charges(x, self.ones.astype(R.dtype)), (R,), (mu,))[1]
        rho = jnp.fft.rfftn(Qq + Qmu)
        e_rec = 1.0 / (2 * self.p3m.V) * jnp.sum(self.p3m.G * jnp.abs(rho) ** 2 * self.p3m.wz[None, None, :])
        e_self = -2 * self.beta**3 / (3 * SQPI) * jnp.sum(mu * mu)
        de, s1, s2 = exc; ei, ej = self.ei, self.ej
        sq = -q[ej] * jnp.sum(mu[ei] * de, -1) * s1 + q[ei] * jnp.sum(mu[ej] * de, -1) * s1
        sm = jnp.sum(mu[ei] * mu[ej], -1) * s1 - jnp.sum(mu[ei] * de, -1) * jnp.sum(mu[ej] * de, -1) * s2
        e_excl = -jnp.sum(sq + sm)
        return e_real + e_rec + e_self + e_excl

    def u_mu_only(self, mu, R, geo, exc, rho_q):
        """U(mu) - U(0): charge-dipole + dipole-dipole energy (e^2/A), charge mesh rho_q precomputed."""
        j, d, k1, k2 = geo
        q = self.q.astype(R.dtype)
        mud = jnp.sum(mu[:, None, :] * d, -1); mjd = jnp.sum(mu[j] * d, -1)
        e_real = jnp.sum(-q[j] * mud * k1) + 0.5 * jnp.sum(jnp.sum(mu[:, None, :] * mu[j], -1) * k1 - mud * mjd * k2)
        Qmu = jax.jvp(lambda x: self.p3m.mesh_charges(x, self.ones.astype(R.dtype)), (R,), (mu,))[1]
        rho_mu = jnp.fft.rfftn(Qmu)
        cross = 2 * jnp.real(jnp.conj(rho_q) * rho_mu) + jnp.abs(rho_mu) ** 2
        e_rec = 1.0 / (2 * self.p3m.V) * jnp.sum(self.p3m.G * cross * self.p3m.wz[None, None, :])
        e_self = -2 * self.beta**3 / (3 * SQPI) * jnp.sum(mu * mu)
        de, s1, s2 = exc; ei, ej = self.ei, self.ej
        sq = -q[ej] * jnp.sum(mu[ei] * de, -1) * s1 + q[ei] * jnp.sum(mu[ej] * de, -1) * s1
        sm = jnp.sum(mu[ei] * mu[ej], -1) * s1 - jnp.sum(mu[ei] * de, -1) * jnp.sum(mu[ej] * de, -1) * s2
        return e_real + e_rec + e_self - jnp.sum(sq + sm)

    def system_energy_scf(self, R, tol=1e-6, max_iter=60):
        """Converged induced dipoles (PCG on (1/alpha - T) mu = E0) and the variational energy
        U = sum mu^2 / (2 alpha) + U(mu) - U(0) = -1/2 mu . E0 at the solution. mu is held fixed (stop_gradient) for the
        forces: dU/dmu = 0 at the minimum, so dE/dR = dU/dR|_mu (Hellmann-Feynman); no backprop through the solver."""
        al = self.alpha.astype(R.dtype)[:, None]

        def U(mu, R_):
            geo = self._pairs(R_); exc = self._excl(R_)
            rho_q = jnp.fft.rfftn(self.p3m.mesh_charges(R_, self.q.astype(R_.dtype)))
            return jnp.sum(mu * mu / al) * 0.5 + self.u_mu_only(mu, R_, geo, exc, rho_q)

        Rs = jax.lax.stop_gradient(R)
        geo = self._pairs(Rs); exc = self._excl(Rs)
        rho_q = jnp.fft.rfftn(self.p3m.mesh_charges(Rs, self.q.astype(R.dtype)))
        gU = jax.grad(lambda m: self.u_mu_only(m, Rs, geo, exc, rho_q))
        z0 = jnp.zeros_like(R)
        E0 = -gU(z0)
        A = lambda m: m / al + gU(m) + E0                          # (1/alpha - T) mu
        x = al * E0; r = E0 - A(x); z = al * r; p = z; rz = jnp.sum(r * z); b2 = jnp.sum(E0 * E0)

        def cond(c):
            k, x, r, p, rz = c
            return (k < max_iter) & (jnp.sum(r * r) > (tol * tol) * b2)

        def step(c):
            k, x, r, p, rz = c
            Ap = A(p); a = rz / jnp.sum(p * Ap)
            x = x + a * p; r = r - a * Ap; z = al * r; rz2 = jnp.sum(r * z)
            return k + 1, x, r, z + (rz2 / rz) * p, rz2

        k, mu, r, _, _ = jax.lax.while_loop(cond, step, (0, x, r, p, rz))
        mu = jax.lax.stop_gradient(mu)
        return KC * U(mu, R)

    def system_energy(self, R):
        """Periodic induced-dipole energy (eV)."""
        geo = self._pairs(R); exc = self._excl(R)
        field = lambda mu: -jax.grad(self.u_mu)(mu, R, geo, exc)
        E0 = field(jnp.zeros_like(R))
        al = self.alpha.astype(R.dtype)[:, None]
        body = jax.checkpoint(lambda _, mu: al * field(mu))
        mu = jax.lax.fori_loop(0, self.n_iter, body, al * E0)
        return -0.5 * KC * jnp.sum(mu * E0)


def make_pol_ewald_energy_fn(Z, box_L, params_json, R_init, r_on=4.5, r_off=6.0, n_iter=12, pair_cap=None, scf=False,
                             scf_tol=1e-6, **kw):
    """energy(R, box=None, lam=None) in eV: Ewald system term minus the switched vacuum pair polarisation of close
    molecule pairs (the same pair term as polarization.make_pol_energy_fn). Fixed cubic box; no alchemical lambda."""
    pe = PolEwald(Z, box_L, params_json, R_init, n_iter=n_iter, **kw)
    pair_only = pol.make_pol_energy_fn(Z, box_L, params_json, r_on=r_on, r_off=r_off, n_iter=n_iter, pair_cap=pair_cap,
                                       system_term=False)

    def energy(R, box=None, lam=None):
        R = jnp.asarray(R)
        sys_e = pe.system_energy_scf(R, tol=scf_tol) if scf else pe.system_energy(R)
        return sys_e + pair_only(R)

    f = jax.jit(energy)
    try:
        f.overflow = jax.jit(lambda R: pe.nfn.update(jnp.asarray(R) - pe.L * jnp.floor(jnp.asarray(R) / pe.L), pe.nbrs0).did_buffer_overflow)
        f.pol_ewald = pe
    except AttributeError:
        pass
    return f
