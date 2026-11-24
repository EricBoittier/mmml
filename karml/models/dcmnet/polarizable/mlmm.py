"""ML/MM electrostatics with field-conditioned (polarisable) DCMnet charges.  Exploratory; electrostatics only.

System: M identical ML molecules (n atoms each, e.g. acetone) plus optional fixed external point charges (MM environment).
One-shot response (the same field the model was trained with, ``--source cgenff``):

    field at molecule I  <-  Gaussian-smeared (sigma) *fixed CGenFF atom charges* of every other molecule J != I and of the external charges
    q_I, p_I = merged_charges(model, params, Z, R_I - com_I, V_I, E_I)  (+ com_I)         # n * n_dcm sites, net charge 0
    E_es     = sum_{I<J} Coul(q_I, p_I; q_J, p_J) + sum_I Coul(q_I, p_I; q_ext, x_ext)       (unsmeared Coulomb, Hartree -> kcal/mol)

Forces are -dE_es/dR by autodiff through the site positions AND through the field (so partner and MM atoms feel the charge-response force).
Not included (by design): the learned polarisation / pair terms of the energy-head study (the labels have no dispersion; they must not replace LJ),
periodic images (cluster only), a smooth switch on the field sources (needed before PBC), self-consistency (a fixed-count unrolled
iteration is available with ``n_iter > 1`` but the checkpoint was trained one-shot, so n_iter=1 is the in-distribution choice).
Known caveat: with induced charges and no self (polarisation) energy, E_es over-counts induction (about 2x in linear response).
Float32: if JAX x64 is enabled wrap calls in ``jax.enable_x64(False)``.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from .base import BOHR_PER_ANG
from .field_model import FieldMessagePassingModel, field_inputs, merged_charges

HARTREE_KCAL = 627.5095
MASS = {1: 1.008, 6: 12.011, 8: 15.999, 17: 35.45}


def load_model(run_dir):
    """Model and params from a training run directory (model_config.json + best_params.msgpack)."""
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "model_config.json").read_text())
    model = FieldMessagePassingModel(**cfg["architecture"])
    params = jax.tree_util.tree_map(jnp.asarray, serialization.msgpack_restore((run_dir / "best_params.msgpack").read_bytes()))
    return model, params, cfg.get("field", {})


def coulomb(q1, p1, q2, p2, eps=1e-6):
    """Coulomb energy (Hartree) between two sets of point charges (e) at positions (A)."""
    r = jnp.sqrt(jnp.sum((p1[:, None] - p2[None]) ** 2, -1) + eps)
    return jnp.sum(q1[:, None] * q2[None] / (r * BOHR_PER_ANG))


class PolarMLMM:
    """Electrostatic energy / forces of M molecules (+ fixed external charges) with field-conditioned charges."""

    def __init__(self, model, params, Z, q_cgenff, sigma=1.0, n_iter=1):
        self.model, self.params = model, params
        self.Z = np.asarray(Z, np.int32)
        self.n = len(self.Z)
        self.qcg = jnp.asarray(q_cgenff, jnp.float32)
        self.w = jnp.asarray([MASS[int(z)] for z in self.Z], jnp.float32)
        self.sigma, self.n_iter = float(sigma), int(n_iter)

    # -- charges ---------------------------------------------------------------------------------------------------------
    def _mol_charges(self, params, Rm, sq, sx, polar):
        n = self.n
        com = (Rm * self.w[:, None]).sum(0) / self.w.sum()
        if polar:
            V, E = field_inputs(Rm, sq, sx, n, self.sigma)
        else:
            V = E = None
        q, p = merged_charges(self.model, params, self.Z, Rm - com, V, E)
        return q, p + com

    def charges(self, R, ext_q=None, ext_x=None, polar=True, params=None):
        """R (M,n,3) A -> site charges q (M, n*n_dcm) and positions p (M, n*n_dcm, 3)."""
        params = self.params if params is None else params
        M, n = R.shape[0], self.n
        if ext_q is None:
            ext_q, ext_x = jnp.zeros((0,), jnp.float32), jnp.zeros((0, 3), jnp.float32)
        allq = jnp.tile(self.qcg, M)
        allx = R.reshape(M * n, 3)
        own = jnp.repeat(jnp.arange(M), n)

        def one(I, Rm):
            sq = jnp.concatenate([jnp.where(own == I, 0.0, allq), ext_q])  # own atoms do not polarise themselves
            sx = jnp.concatenate([allx, ext_x])
            return self._mol_charges(params, Rm, sq, sx, polar)

        q, p = jax.vmap(one)(jnp.arange(M), R)
        for _ in range(self.n_iter - 1):  # optional fixed-count unrolled iteration on the model's own site charges (out of distribution for the one-shot checkpoint)
            raise NotImplementedError("n_iter > 1 not implemented: the checkpoint is trained one-shot with CGenFF sources")
        return q, p

    # -- energy ----------------------------------------------------------------------------------------------------------
    def energy(self, R, ext_q=None, ext_x=None, polar=True, params=None):
        """E_es in kcal/mol: inter-molecular site-site Coulomb plus site-external-charge Coulomb."""
        R = jnp.asarray(R, jnp.float32)
        q, p = self.charges(R, ext_q, ext_x, polar, params)
        M = R.shape[0]
        e = 0.0
        for I in range(M):
            for J in range(I + 1, M):
                e = e + coulomb(q[I], p[I], q[J], p[J])
            if ext_q is not None:
                e = e + coulomb(q[I], p[I], ext_q, ext_x)
        return e * HARTREE_KCAL

    def energy_and_forces(self, R, ext_q=None, ext_x=None, polar=True):
        """(E kcal/mol, F kcal/mol/A of the molecules, F_ext on the external charges)."""
        ext_q_ = None if ext_q is None else jnp.asarray(ext_q, jnp.float32)
        ext_x_ = None if ext_x is None else jnp.asarray(ext_x, jnp.float32)
        if ext_q is None:
            e, g = jax.value_and_grad(lambda r: self.energy(r, None, None, polar))(jnp.asarray(R, jnp.float32))
            return e, -g, None
        e, (g, gx) = jax.value_and_grad(lambda r, x: self.energy(r, ext_q_, x, polar), argnums=(0, 1))(jnp.asarray(R, jnp.float32), ext_x_)
        return e, -g, -gx
