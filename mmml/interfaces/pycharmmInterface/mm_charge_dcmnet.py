"""DCMnet distributed charge sites -> per-atom MM charges q_i(conformation), differentiable (acetone).

DCMnet (Boittier et al. 2024, ``MessagePassingModel``; model class vendored verbatim from the release
``dcmnet_model.py``) returns ``n_dcm`` charge sites per atom (charge q_s and an off-nuclear position p_s).
The molecule-pair MM tail (:mod:`mmml.interfaces.pycharmmInterface.mm_molpair`) is hard-wired to one charge on
each nucleus, so the sites are collapsed onto the nuclei by a *dipole-preserving projection*::

    qm_i = sum_{s in atom i} q_s                                   (monopole assignment)
    q    = qm + C^T (C C^T)^-1 (t - C qm),   C = [1; x_i^T] (4 x A),  t = [0; sum_s q_s p_s]

i.e. the smallest change of ``qm`` (in the 2-norm) that restores the net charge (0) and the molecular dipole of the
site distribution. It is linear in the site charges, exact in the monopole and dipole, loses the quadrupole and
higher moments of the site model (quantified in ``analysis/dcmnet_liquid_charges_2026-11-18``), is differentiable
(gradients flow through the network, the site positions and the projection geometry), and is translation and
rotation invariant. The model input is exactly as trained: mass-COM-centred whole molecule padded with Z = 0 dummy
atoms at the origin to 60 atoms (a compact 10-atom evaluation is NOT equivalent).

Everything is a pure function of the whole-molecule positions, so a caller that differentiates the energy
w.r.t. positions (forces) or w.r.t. an affine strain (atomic virial) gets the d(charge)/d(geometry) terms for free.
"""
from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np

NATOMS = 60
MASS = {1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999}
ARCH_KEYS = ("features", "max_degree", "num_iterations", "num_basis_functions", "cutoff", "max_atomic_number",
             "n_dcm", "include_pseudotensors", "dc_scale")
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "mm_charge_models"


def _imports():
    import e3x
    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    return e3x, nn, jax, jnp


def _build_model_class():
    e3x, nn, jax, jnp = _imports()

    class MessagePassingModel(nn.Module):
        features: int = 32
        max_degree: int = 2
        num_iterations: int = 3
        num_basis_functions: int = 8
        cutoff: float = 5.0
        max_atomic_number: int = 17
        n_dcm: int = 4
        include_pseudotensors: bool = False
        dc_scale: float = 0.175

        def mono(self, atomic_numbers, positions, dst_idx, src_idx, edge_w=None):
            positions_dst = e3x.ops.gather_dst(positions, dst_idx=dst_idx)
            positions_src = e3x.ops.gather_src(positions, src_idx=src_idx)
            displacements = positions_src - positions_dst
            basis = e3x.nn.basis(displacements, num=self.num_basis_functions, max_degree=self.max_degree,
                                 radial_fn=e3x.nn.reciprocal_bernstein,
                                 cutoff_fn=functools.partial(e3x.nn.smooth_cutoff, cutoff=self.cutoff))
            x = e3x.nn.Embed(num_embeddings=self.max_atomic_number + 1, features=self.features)(atomic_numbers)
            for i in range(self.num_iterations):
                if i == self.num_iterations - 1:
                    y = e3x.nn.MessagePass(max_degree=self.max_degree, include_pseudotensors=False)(
                        x, basis, edge_w, dst_idx=dst_idx, src_idx=src_idx)
                else:
                    y = e3x.nn.MessagePass(include_pseudotensors=self.include_pseudotensors)(
                        x, basis, edge_w, dst_idx=dst_idx, src_idx=src_idx)
                y = e3x.nn.add(x, y)
                y = e3x.nn.Dense(self.features)(y)
                y = e3x.nn.silu(y)
                y = e3x.nn.Dense(self.features)(y)
                x = e3x.nn.add(x, y)
            x = e3x.nn.TensorDense(features=self.n_dcm, max_degree=1, include_pseudotensors=False)(x)
            atomic_mono = e3x.nn.change_max_degree_or_type(x, max_degree=0, include_pseudotensors=False)
            element_bias = self.param("element_bias", lambda rng, shape: jnp.zeros(shape), (self.max_atomic_number + 1))
            atomic_mono = nn.Dense(self.n_dcm, use_bias=False)(atomic_mono)
            atomic_mono = atomic_mono.squeeze(axis=1)
            atomic_mono = atomic_mono.squeeze(axis=1)
            atomic_mono += element_bias[atomic_numbers][:, None]
            x = e3x.nn.hard_tanh(x) * self.dc_scale
            atomic_dipo = x[:, 0, 1:4, :]
            atomic_dipo += positions[:, :, None]
            return atomic_mono, atomic_dipo

        @nn.compact
        def __call__(self, atomic_numbers, positions, dst_idx, src_idx, edge_w=None):
            return self.mono(atomic_numbers, positions, dst_idx, src_idx, edge_w)

    return MessagePassingModel


def load_dcmnet(spec: str):
    """spec: a run directory (best_params.msgpack | params.msgpack + model_config.json) or a name under data/mm_charge_models."""
    from flax import serialization
    _, _, jax, jnp = _imports()
    p = Path(spec)
    if not p.exists():
        p = DATA_DIR / spec
    cfg = json.loads((p / "model_config.json").read_text())
    pf = p / "best_params.msgpack"
    if not pf.exists():
        pf = p / "params.msgpack"
    model = _build_model_class()(**{k: cfg["architecture"][k] for k in ARCH_KEYS})
    params = jax.tree_util.tree_map(jnp.asarray, serialization.msgpack_restore(pf.read_bytes()))
    return model, params


def project_sites_to_atoms(q_sites, p_sites, X, n_dcm):
    """Dipole-preserving projection of the sites of M molecules onto their A nuclei.
    q_sites (M, A*n), p_sites (M, A*n, 3), X (M, A, 3) -> q (M, A) with sum_i q_i = sum_s q_s and sum_i q_i x_i = sum_s q_s p_s."""
    _, _, jax, jnp = _imports()
    M, A, _ = X.shape
    qm = q_sites.reshape(M, A, n_dcm).sum(-1)
    C = jnp.concatenate([jnp.ones((M, 1, A), X.dtype), jnp.swapaxes(X, 1, 2)], axis=1)          # (M,4,A)
    t = jnp.concatenate([q_sites.sum(-1, keepdims=True), jnp.einsum("ms,msc->mc", q_sites, p_sites)], axis=1)   # (M,4)
    res = t - jnp.einsum("mka,ma->mk", C, qm)
    G = jnp.einsum("mka,mla->mkl", C, C)
    lam = jnp.linalg.solve(G, res[..., None])[..., 0]
    return qm + jnp.einsum("mka,mk->ma", C, lam)


def make_dcm_atom_charge_fn(spec: str, Z, *, dtype="float32", chunk: int | None = None, return_sites: bool = False,
                            padded: bool = False):
    """Return ``fn(X)``: whole molecules X (M, A, 3) [A] -> per-atom charges (M, A) [e] (net neutral, dipole of the sites).

    dtype: network precision ('float32' | 'float64'). chunk: molecules per network call (None = all at once;
    smaller chunks are rematerialised, bounded memory). return_sites: ``fn(X) -> (q_atoms, q_sites, p_sites)``.

    The network was trained on molecules padded to 60 atoms with Z = 0 dummies at the origin, and the dummies take
    part in the message passing (a compact 10-atom input is NOT equivalent). All 60 - A dummies are identical (same Z,
    same position, same neighbour multiset), so by symmetry they stay identical in every layer: the default
    (``padded=False``) evaluates ONE representative dummy and gives its outgoing edges the multiplicity as the
    ``weights`` of ``MessagePass`` (50 into a real atom, 49 into itself). This is exact (same function as the 60-atom
    graph, verified to 1e-12 in float64) with 111 instead of 3540 edges per molecule. ``padded=True`` = the literal
    60-atom graph (reference)."""
    e3x, _, jax, jnp = _imports()
    Z = np.asarray(Z, np.int32)
    A = len(Z)
    model, params = load_dcmnet(spec)
    nd = int(model.n_dcm)
    dt = jnp.float64 if dtype == "float64" else jnp.float32
    params = jax.tree_util.tree_map(
        lambda a: jnp.asarray(a, dt) if jnp.issubdtype(jnp.asarray(a).dtype, jnp.floating) else jnp.asarray(a), params)
    dst, src = e3x.ops.sparse_pairwise_indices(NATOMS)
    npad = NATOMS - A
    dstc, srcc = e3x.ops.sparse_pairwise_indices(A + 1)                       # A real atoms + 1 representative dummy (index A)
    dstc = jnp.concatenate([dstc, jnp.asarray([A])]); srcc = jnp.concatenate([srcc, jnp.asarray([A])])   # dummy <- other dummies
    wc = jnp.where(srcc == A, jnp.where(dstc == A, npad - 1.0, float(npad)), 1.0).astype(dt)[:, None, None, None]
    Zc = jnp.zeros((A + 1,), jnp.int32).at[:A].set(jnp.asarray(Z))
    mass = jnp.asarray([MASS[int(z)] for z in Z], jnp.float64)
    Zp = jnp.zeros((NATOMS,), jnp.int32).at[:A].set(jnp.asarray(Z))

    params0 = params

    def one(params, Rp):
        if padded:
            mono, dipo = model.apply(params, atomic_numbers=Zp, positions=Rp, dst_idx=dst, src_idx=src)
            nat = NATOMS
        else:
            mono, dipo = model.apply(params, atomic_numbers=Zc, positions=Rp[: A + 1], dst_idx=dstc, src_idx=srcc, edge_w=wc)
            nat = A + 1
        d = jnp.moveaxis(dipo, -1, -2).reshape(nat * nd, 3)
        m = mono.reshape(nat * nd)
        NDC = A * nd
        valid = (jnp.arange(nat * nd) < NDC).astype(m.dtype)
        m = (m * valid - (m * valid).sum() / NDC) * valid      # net charge 0 (training-time post-processing)
        return m[:NDC], d[:NDC]

    batched = jax.vmap(one, in_axes=(None, 0))

    def net(params, Rp):                       # (M, 60, 3) -> (M, A*nd), (M, A*nd, 3)
        M = Rp.shape[0]
        ck = M if chunk is None else min(int(chunk), M)
        nb = -(-M // ck); pad = nb * ck - M
        if pad:
            Rp = jnp.concatenate([Rp, jnp.broadcast_to(Rp[:1], (pad,) + Rp.shape[1:])], 0)
        fn = batched if nb == 1 else jax.checkpoint(batched)
        q, p = jax.lax.map(lambda r: fn(params, r), Rp.reshape((nb, ck) + Rp.shape[1:]))
        return q.reshape(nb * ck, -1)[:M], p.reshape(nb * ck, -1, 3)[:M]

    def fn(X, params=None):
        params = params0 if params is None else params
        out_dt = X.dtype
        c = jnp.sum(X * mass[None, :, None].astype(X.dtype), axis=1) / jnp.sum(mass).astype(X.dtype)
        Rc = (X - c[:, None, :]).astype(dt)
        M = X.shape[0]
        Rp = jnp.zeros((M, NATOMS if padded else A + 1, 3), dt).at[:, :A].set(Rc)
        if dt == jnp.float32:
            with jax.enable_x64(False):     # network constants stay f32 even if the host process runs x64
                q, p = net(params, Rp)
        else:
            q, p = net(params, Rp)
        q = q.astype(out_dt); p = p.astype(out_dt) + c[:, None, :]
        qa = project_sites_to_atoms(q, p, X, nd)
        return (qa, q, p) if return_sites else qa

    fn.n_dcm = nd
    fn.params0 = params0
    return fn
