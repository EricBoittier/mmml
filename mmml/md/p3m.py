"""Particle-particle particle-mesh (P3M) Ewald electrostatics for the hybrid MM tail, on device (JAX).

Ewald split of the Coulomb energy of point charges in a cubic periodic box (kcal/mol, K = 332.0637)::

    E = E_real + E_recip + E_self + E_excl
    E_real  = K sum_{i<j, inter} q_i q_j erfc(beta r_ij) / r_ij            (evaluated by the caller, e.g. mm_molpair)
    E_recip = K / (2 V) sum_{k != 0} G(k) |rho_M(k)|^2                       (this module, mesh)
    E_self  = -K beta / sqrt(pi) sum_i q_i^2
    E_excl  = -K sum_{i<j, same molecule} q_i q_j erf(beta r_ij) / r_ij      (intramolecular pairs are not interactions)

Mesh part (Hockney & Eastwood; Deserno & Holm, J. Chem. Phys. 109, 7678 (1998)): charges are assigned to an
n^3 mesh with cardinal B-splines of order P (Fourier transform U(k) = prod_d sinc(k_d h / 2)^P); rho_M(k) is the
discrete Fourier transform of the mesh charges. The influence function is the one that is optimal for the energy
(Stern & Calkins, J. Chem. Phys. 128, 214106 (2008)), since forces here come from automatic differentiation of the
energy::

    G(k) = sum_m U^2(k_m) phi(k_m) / [sum_m U^2(k_m)]^2,   k_m = k + 2 pi m / h,   phi(k) = 4 pi / k^2 exp(-k^2 / 4 beta^2)

with the alias sum over m in [-n_alias, n_alias]^3. G is precomputed once on the host for a fixed box (NVT/NVE).

    p3m = P3M(box_L=30.0, n_mesh=32, order=5, beta=0.40)
    e_kcal = p3m.recip(pos, q) + p3m.self_energy(q) + p3m.exclusion(pos, q, excl_i, excl_j)
``ewald_direct`` is a float64 numpy reference (explicit k sum) for validation.
"""
from __future__ import annotations

import math

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import erf

COULOMB_K = 3.32063711e2  # kcal/mol A / e^2, as mm_molpair / mm_energy_forces


def _sinc(x):
    return np.where(np.abs(x) < 1e-12, 1.0, np.sin(x) / np.where(np.abs(x) < 1e-12, 1.0, x))


def influence_function(box_L: float, n: int, order: int, beta: float, n_alias: int = 2) -> np.ndarray:
    """Energy-optimal G on the rfftn half grid (n, n, n//2+1), float64; G(0) = 0."""
    h = box_L / n
    kf = 2 * np.pi * np.fft.fftfreq(n, d=h)
    kr = 2 * np.pi * np.fft.rfftfreq(n, d=h)
    KX, KY, KZ = np.meshgrid(kf, kf, kr, indexing="ij")
    num = np.zeros_like(KX); den = np.zeros_like(KX)
    m = np.arange(-n_alias, n_alias + 1) * 2 * np.pi / h
    for mx in m:
        ax = KX + mx; ux = _sinc(ax * h / 2) ** (2 * order)
        for my in m:
            ay = KY + my; uy = _sinc(ay * h / 2) ** (2 * order)
            for mz in m:
                az = KZ + mz; uz = _sinc(az * h / 2) ** (2 * order)
                u2 = ux * uy * uz
                k2 = ax * ax + ay * ay + az * az
                phi = np.where(k2 > 0, 4 * np.pi / np.where(k2 > 0, k2, 1.0) * np.exp(-k2 / (4 * beta * beta)), 0.0)
                num += u2 * phi; den += u2
    G = np.where(den > 0, num / np.maximum(den, 1e-300) ** 2, 0.0)
    G[0, 0, 0] = 0.0
    return G


def _bspline_weights(w, order: int):
    """Cardinal B-spline weights for fractional offsets w in [0,1): (..., order); weight k goes to grid index floor(u) - (P-1) + k."""
    data = [1.0 - w, w] + [jnp.zeros_like(w)] * (order - 2)
    for j in range(3, order + 1):
        div = 1.0 / (j - 1)
        data[j - 1] = div * w * data[j - 2]
        for k in range(1, j - 1):
            data[j - k - 1] = div * ((w + k) * data[j - k - 2] + (j - k - w) * data[j - k - 1])
        data[0] = div * (1.0 - w) * data[0]
    return jnp.stack(data, axis=-1)


class P3M:
    def __init__(self, box_L: float, n_mesh: int = 32, order: int = 5, beta: float = 0.40, n_alias: int = 2, dtype=jnp.float32):
        self.L = float(box_L); self.n = int(n_mesh); self.P = int(order); self.beta = float(beta); self.dtype = dtype
        self.V = self.L ** 3
        self.G = jnp.asarray(influence_function(self.L, self.n, self.P, self.beta, n_alias), dtype)
        # rfftn half grid: interior kz planes appear twice in the full sum
        w = np.full(self.n // 2 + 1, 2.0); w[0] = 1.0
        if self.n % 2 == 0: w[-1] = 1.0
        self.wz = jnp.asarray(w, dtype)

    def mesh_charges(self, pos, q):
        n, P = self.n, self.P
        u = pos / (self.L / n)
        f = jnp.floor(u); w = u - f
        W = _bspline_weights(w, P)                                 # (N, 3, P)
        idx = (f.astype(jnp.int32)[:, :, None] + jnp.arange(P)[None, None, :] - (P - 1)) % n   # (N, 3, P): weight k -> floor(u) - (P-1) + k
        wx, wy, wz = W[:, 0], W[:, 1], W[:, 2]
        val = q[:, None, None, None] * wx[:, :, None, None] * wy[:, None, :, None] * wz[:, None, None, :]
        ix = jnp.broadcast_to(idx[:, 0][:, :, None, None], val.shape)
        iy = jnp.broadcast_to(idx[:, 1][:, None, :, None], val.shape)
        iz = jnp.broadcast_to(idx[:, 2][:, None, None, :], val.shape)
        return jnp.zeros((n, n, n), pos.dtype).at[ix.ravel(), iy.ravel(), iz.ravel()].add(val.ravel())

    def recip(self, pos, q):
        """Mesh (k != 0) energy, kcal/mol."""
        rho = jnp.fft.rfftn(self.mesh_charges(pos, q))
        s = jnp.abs(rho) ** 2
        return COULOMB_K / (2 * self.V) * jnp.sum(self.G * s * self.wz[None, None, :])

    def self_energy(self, q):
        return -COULOMB_K * self.beta / math.sqrt(math.pi) * jnp.sum(q * q)

    def exclusion(self, pos, q, ei, ej):
        """-K sum over intramolecular pairs (ei, ej) of q_i q_j erf(beta r) / r (molecules whole or MIC-safe)."""
        d = pos[ej] - pos[ei]; d = d - self.L * jnp.round(d / self.L)
        r = jnp.sqrt(jnp.sum(d * d, -1))
        return -COULOMB_K * jnp.sum(q[ei] * q[ej] * erf(self.beta * r) / r)


def real_kernel(r, beta):
    """erfc(beta r) / r for the caller's real-space pair sum."""
    return jax.scipy.special.erfc(beta * r) / r


def ewald_direct(pos, q, box_L, beta, mol, k_max=None, r_cut=None):
    """float64 numpy reference: full Ewald energy (kcal/mol) of the inter-molecular Coulomb interaction.
    Real space with minimum image (requires erfc(beta L/2) negligible), explicit k sum, self, intramolecular exclusion."""
    pos = np.asarray(pos, np.float64); q = np.asarray(q, np.float64); mol = np.asarray(mol); L = float(box_L)
    N = len(q); e_real = 0.0
    for i in range(N - 1):
        d = pos[i + 1:] - pos[i]; d -= L * np.round(d / L); r = np.linalg.norm(d, axis=1)
        inter = mol[i + 1:] != mol[i]
        from scipy.special import erfc as _erfc, erf as _erf
        e_real += np.sum(np.where(inter, q[i] * q[i + 1:] * _erfc(beta * r) / r, 0.0))
        e_real -= np.sum(np.where(~inter, q[i] * q[i + 1:] * _erf(beta * r) / r, 0.0))   # exclusion
    if k_max is None: k_max = 2 * beta * 4.5
    nm = int(math.ceil(k_max * L / (2 * np.pi)))
    g = np.arange(-nm, nm + 1)
    n = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    n = n[(n[:, 0] > 0) | ((n[:, 0] == 0) & (n[:, 1] > 0)) | ((n[:, 0] == 0) & (n[:, 1] == 0) & (n[:, 2] > 0))]
    k = 2 * np.pi / L * n; k2 = (k * k).sum(1); k = k[k2 <= k_max ** 2]; k2 = k2[k2 <= k_max ** 2]
    e_rec = 0.0
    for s in range(0, len(k), 4000):
        kk = k[s:s + 4000]; S = (q[:, None] * np.exp(1j * pos @ kk.T)).sum(0)
        e_rec += np.sum(4 * np.pi / k2[s:s + 4000] * np.exp(-k2[s:s + 4000] / (4 * beta ** 2)) * np.abs(S) ** 2)
    e_rec *= 2 / (2 * L ** 3)          # half k-space listed: x2
    e_self = -beta / np.sqrt(np.pi) * np.sum(q * q)
    return COULOMB_K * (e_real + e_rec + e_self), dict(real_minus_excl=COULOMB_K * e_real, recip=COULOMB_K * e_rec, self=COULOMB_K * e_self)
