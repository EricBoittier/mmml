"""Fused Pallas-Triton evaluation of PhysNet (e3x) on many small systems.

Drop-in for the hybrid ML/MM calculator's per-fragment PhysNet call (monomer and
dimer batches): energies and analytic forces for B systems of <= n atoms each.

    from mmml.models.physnetjax.physnetjax.models.physnet_fused import make_physnet_fused
    fn = make_physnet_fused(config, params, max_atoms=6)          # build once (static)
    E, F = fn(R, Z)        # R (B, n, 3) float32, Z (B, n) int32, Z == 0 -> padding atom
                           # E (B,) eV, F (B, n, 3) eV/A (zero on padding atoms)
    out = fn.apply_like(R_flat, Z_flat)  # MODEL.apply-like: {"energy": (B,1), "forces": (B*n,3)}

``config`` is the checkpoint ``config`` dict (PhysNet hyperparameters), ``params``
the flax params ({"params": {...}} or the bare tree). Supported architectures:
max_degree 0 (any num_iterations) and max_degree 1 with num_iterations 2 (the
current production student); charges=False; zbl on/off; any features that is a
power of two >= 16; num_basis_functions <= 16.

Design (see metatomic-runs/training/watmeoh/perf/pallas/README.md):
  * per message pass two Pallas kernels (forward, reverse) that compute pair
    geometry, Chebyshev RBF x smooth cutoff, filters and the neighbour sum for a
    tile of TB systems in registers -- no edge tensor ever reaches HBM;
  * the first pass folds the embedding into per-species filter weights
    (h_i = sum_s [sum_{j in s} phi_ij] @ (W * emb_s));
  * atom-level residual MLPs / readout / ZBL are plain JAX, differentiated by XLA;
    the message-pass kernels carry custom VJPs, so forces = -grad E.
  * precision: "fp32" (IEEE dots in-kernel, fp32 MLP; matches the float32
    reference to ~1e-6 eV / 2e-5 eV/A), "tf32x3" (3-pass TF32, fp32-accurate),
    "tf32", "bf16", "fp16" (low-precision dot operands, fp32 accumulation).

Pallas-Triton in jax 0.11 does not know the RTX 5090 / 3090 device kinds; they
are registered on import (sm_120 / sm_86).
"""
from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

__all__ = ["make_physnet_fused", "unpack_physnet_params"]

KPAD = 16
_ZBL_A = 0.8854 * 0.529177249
_ZBL_C = np.array([0.18175, 0.50986, 0.28022, 0.02817])
_ZBL_C = _ZBL_C / np.linalg.norm(_ZBL_C)
_ZBL_E = np.array([3.19980, 0.94229, 0.40290, 0.20162])


def _register_gpus():
    try:
        from jax._src.pallas.triton import gpu_info as _gi
    except Exception:  # pragma: no cover
        return
    for kind, arch, cc in [("NVIDIA GeForce RTX 5090", "12.0", 120),
                           ("NVIDIA GeForce RTX 5080", "12.0", 120),
                           ("NVIDIA GeForce RTX 3090", "8.6", 86),
                           ("NVIDIA GeForce RTX 3090 Ti", "8.6", 86),
                           ("NVIDIA GeForce RTX 3080", "8.6", 86)]:
        _gi.registry.setdefault(kind, lambda a=arch, c=cc: _gi.GpuInfo(gpu_version=None, arch_name=a,
                                                                       compute_capability=c))


_register_gpus()

_PREC = {
    "fp32": (jax.lax.Precision.HIGHEST, None),
    "tf32x3": (jax.lax.DotAlgorithmPreset.TF32_TF32_F32_X3, None),
    "tf32": (jax.lax.Precision.DEFAULT, None),
    "bf16": (None, jnp.bfloat16),
    "fp16": (None, jnp.float16),
}


# ----------------------------------------------------------------------------- params
def unpack_physnet_params(cfg, params):
    """Flax PhysNet params -> effective weights of the fused math (CG / tensor
    kernels folded into the filter matrices)."""
    p = params["params"] if "params" in params else params
    def _arr(o):  # JSON checkpoints hold nested lists: convert leaves, not list items
        return {k: _arr(v) for k, v in o.items()} if isinstance(o, dict) else jnp.asarray(np.asarray(o, np.float32))

    p = _arr(p)
    md = int(cfg.get("max_degree", 0))
    nit = int(cfg["num_iterations"])
    nres = abs(int(cfg.get("n_refinement_blocks", cfg.get("n_res", 3))))
    if bool(cfg.get("charges", False)):
        raise NotImplementedError("physnet_fused: charges=True models are not supported")
    if bool(cfg.get("efa", False)):
        raise NotImplementedError("physnet_fused: EFA models are not supported")
    if md > 1 or (md == 1 and nit != 2):
        raise NotImplementedError("physnet_fused: max_degree 1 is supported only with num_iterations=2")
    if nres < 1:
        raise NotImplementedError("physnet_fused: n_refinement_blocks >= 1 required")
    W = {"emb": p["Embed_0"]["embedding"][:, 0, 0, :], "mp": [], "ref": []}
    di = 0
    for it in range(nit):
        mp = p[f"MessagePass_{it}"]
        tk = mp["tensor"]["kernel"]
        e = {"Ws": mp["filter"]["0+"]["kernel"] * tk[0, 0, 0, 0, 0, 0][None], "kind": "s2s"}
        if md == 1 and it == 0:
            e["Wv"] = mp["filter"]["1-"]["kernel"] * tk[0, 1, 0, 0, 0, 1][None]
            e["kind"] = "s2v"
        elif md == 1 and it == 1:
            e["Wv"] = mp["filter"]["1-"]["kernel"] * tk[0, 1, 0, 1, 0, 0][None] / math.sqrt(3.0)
            e["kind"] = "v2s"
        W["mp"].append(e)
        blocks = []
        for _ in range(nres + 1):
            dd = p[f"Dense_{di}"]
            di += 1
            blocks.append({"W0": dd["0+"]["kernel"], "b0": dd["0+"]["bias"],
                           "W1": dd["1-"]["kernel"] if ("1-" in dd and e["kind"] == "s2v") else None})
        W["ref"].append(blocks)
    W["out"] = p[f"Dense_{di}"]["0+"]["kernel"][:, 0] * p[f"Dense_{di + 1}"]["kernel"][0, 0]
    W["ebias"] = p["energy_bias"] if (cfg.get("use_energy_bias", False) and "energy_bias" in p) \
        else jnp.zeros(W["emb"].shape[0])
    return W


# ----------------------------------------------------------------------------- kernels
def _geom(px, py, pz, zf, TB, n, rc, species, need_grad):
    E = TB * n * n
    am = (zf > 0).astype(jnp.float32)
    ii = jax.lax.broadcasted_iota(jnp.int32, (TB, n, n), 1)
    jj = jax.lax.broadcasted_iota(jnp.int32, (TB, n, n), 2)
    em3 = am[:, :, None] * am[:, None, :] * (ii != jj).astype(jnp.float32)
    emb = em3 > 0
    dx3 = jnp.where(emb, px[:, None, :] - px[:, :, None], 1.0)  # R_j - R_i
    dy3 = jnp.where(emb, py[:, None, :] - py[:, :, None], 1.0)
    dz3 = jnp.where(emb, pz[:, None, :] - pz[:, :, None], 1.0)
    r3 = jnp.sqrt(dx3 * dx3 + dy3 * dy3 + dz3 * dz3)
    r = r3.reshape(E, 1)
    em = em3.reshape(E, 1)
    inv_r = 1.0 / r
    g = {"u": [dx3.reshape(E, 1) * inv_r, dy3.reshape(E, 1) * inv_r, dz3.reshape(E, 1) * inv_r],
         "r": r, "em": em, "inv_r": inv_r}
    if species is not None:
        zj3 = jnp.broadcast_to(zf[:, None, :], (TB, n, n))
        g["oh"] = [(zj3 == float(s)).astype(jnp.float32).reshape(E, 1) for s in species]
    er = jnp.exp(-r)
    zz = jnp.clip(2.0 * er - 1.0, -1.0, 1.0)
    th = jnp.arccos(zz)
    kk = jax.lax.broadcasted_iota(jnp.int32, (1, KPAD), 1).astype(jnp.float32)
    kth = kk * th
    T = jnp.cos(kth)  # T_k(2 e^-r - 1) == e3x exponential_chebyshev
    xr = r * (1.0 / rc)
    inside = (xr < 1.0) & (em > 0)
    xs = jnp.where(inside, xr, 0.0)
    omx = 1.0 - xs * xs
    cut = jnp.where(inside, jnp.exp(1.0 - 1.0 / omx), 0.0)  # e3x smooth_cutoff
    g["phi"] = T * cut
    if need_grad:
        sin_th = jnp.sqrt(jnp.maximum(1.0 - zz * zz, 1e-12))
        dT = -kk * jnp.sin(kth) * (2.0 * er / sin_th)
        dcut = jnp.where(inside, cut * (-2.0 * xs / rc) / (omx * omx), 0.0)
        g["dphi"] = dT * cut + T * dcut
    return g


def _helpers(TB, n):
    E, A = TB * n * n, TB * n

    def bj(a, w):
        return jnp.broadcast_to(a.reshape(TB, 1, n, w), (TB, n, n, w)).reshape(E, w)

    def bi(a, w):
        return jnp.broadcast_to(a.reshape(TB, n, 1, w), (TB, n, n, w)).reshape(E, w)

    def sum_j(m, w):
        return m.reshape(A, n, w).sum(axis=1)

    def sum_i(m, w):
        return m.reshape(TB, n, n, w).sum(axis=1).reshape(A, w)

    return bj, bi, sum_j, sum_i


def _store_grad(g, rbar, ubar, refs, TB, n):
    u, inv_r, em = g["u"], g["inv_r"], g["em"]
    udot = ubar[0] * u[0] + ubar[1] * u[1] + ubar[2] * u[2] if ubar is not None else None
    for c, ref in enumerate(refs):
        gc = rbar * u[c]
        if ubar is not None:
            gc = gc + (ubar[c] - u[c] * udot) * inv_r
        g3 = (gc * em).reshape(TB, n, n)
        ref[...] = g3.sum(axis=1) - g3.sum(axis=2)  # dE/dR_i


class _MP:
    def __init__(self, W, cfg, n, TB, nw, TBb, nwb, prec, dot_dtype, species):
        self.F = int(W["emb"].shape[1])
        self.n, self.TB, self.nw, self.TBb, self.nwb = n, TB, nw, TBb, nwb
        self.rc = float(cfg["cutoff"])
        self.species = tuple(species)
        K = int(cfg["num_basis_functions"])
        if K > KPAD:
            raise NotImplementedError("physnet_fused: num_basis_functions > 16")
        wdt = jnp.float32 if dot_dtype is None else dot_dtype

        def padk(w):
            return jnp.zeros((KPAD, w.shape[1]), jnp.float32).at[:K].set(w)

        def dot(a, b):
            if dot_dtype is not None:
                return jnp.dot(a.astype(dot_dtype), b, preferred_element_type=jnp.float32)
            return jnp.dot(a, b, preferred_element_type=jnp.float32, precision=prec)

        self.dot = dot
        mp0 = W["mp"][0]
        self.vec0 = mp0["kind"] == "s2v"
        w0 = [padk(mp0["Ws"]) * W["emb"][s][None] for s in species]
        if self.vec0:
            w0 += [padk(mp0["Wv"]) * W["emb"][s][None] for s in species]
        self.w0 = [w.astype(wdt) for w in w0]
        self.w0T = [w.T.astype(wdt) for w in w0]
        self.later = []
        for mp in W["mp"][1:]:
            w = [padk(mp["Ws"])] + ([padk(mp["Wv"])] if mp["kind"] == "v2s" else [])
            self.later.append((mp["kind"] == "v2s", [x.astype(wdt) for x in w], [x.T.astype(wdt) for x in w]))
        self.mp0 = self._make_mp0()
        self.mpk = [self._make_mpk(v2s, w, wT) for v2s, w, wT in self.later]

    def _call(self, kern, tiles, feats, weights, outs, TB, nw, name):
        n, F = self.n, self.F
        spec_a = pl.BlockSpec((TB, n), lambda b: (b, 0))
        spec_f = pl.BlockSpec((TB, n, F), lambda b: (b, 0, 0))
        w_specs = [pl.BlockSpec(w.shape, lambda b: (0, 0)) for w in weights]
        B = tiles[0].shape[0]
        out_shape = [jax.ShapeDtypeStruct((B, n, F) if o == "f" else (B, n), jnp.float32) for o in outs]
        return pl.pallas_call(
            kern, out_shape=out_shape, grid=(B // TB,),
            in_specs=[spec_a] * len(tiles) + [spec_f] * len(feats) + w_specs,
            out_specs=[spec_f if o == "f" else spec_a for o in outs],
            compiler_params=plt.CompilerParams(num_warps=nw, num_stages=1), name=name,
        )(*tiles, *feats, *weights)

    def _make_mp0(self):
        F, n, rc, sp, dot, vec0 = self.F, self.n, self.rc, self.species, self.dot, self.vec0
        S = len(sp)
        nw0 = len(self.w0)

        def fwd_kernel(TB):
            bj, bi, sum_j, sum_i = _helpers(TB, n)

            def k(px, py, pz, zf, *rest):
                w, outs = rest[:nw0], rest[nw0:]
                g = _geom(px[...], py[...], pz[...], zf[...], TB, n, rc, sp, False)
                h, hv = None, [None] * 3
                for si in range(S):
                    ps = g["phi"] * g["oh"][si]
                    t = dot(sum_j(ps, KPAD), w[si][...])
                    h = t if h is None else h + t
                    if vec0:
                        for c in range(3):
                            t = dot(sum_j(ps * g["u"][c], KPAD), w[S + si][...])
                            hv[c] = t if hv[c] is None else hv[c] + t
                outs[0][...] = h.reshape(TB, n, F)
                if vec0:
                    for c in range(3):
                        outs[1 + c][...] = hv[c].reshape(TB, n, F)
            return k

        def bwd_kernel(TB):
            bj, bi, sum_j, sum_i = _helpers(TB, n)
            nf = 4 if vec0 else 1

            def k(px, py, pz, zf, *rest):
                A = TB * n
                feats, w, gout = rest[:nf], rest[nf:nf + nw0], rest[nf + nw0:]
                g = _geom(px[...], py[...], pz[...], zf[...], TB, n, rc, sp, True)
                phi, u = g["phi"], g["u"]
                hb = feats[0][...].reshape(A, F)
                hvb = [feats[1 + c][...].reshape(A, F) for c in range(3)] if vec0 else None
                phibar, ubar = None, ([None] * 3 if vec0 else None)
                for si in range(S):
                    acc = bi(dot(hb, w[si][...]), KPAD)
                    if vec0:
                        for c in range(3):
                            qb = bi(dot(hvb[c], w[S + si][...]), KPAD)
                            acc = acc + qb * u[c]
                            t = jnp.sum(qb * phi, axis=1, keepdims=True) * g["oh"][si]
                            ubar[c] = t if ubar[c] is None else ubar[c] + t
                    t = acc * g["oh"][si]
                    phibar = t if phibar is None else phibar + t
                rbar = jnp.sum(phibar * g["dphi"], axis=1, keepdims=True)
                _store_grad(g, rbar, ubar, gout, TB, n)
            return k

        kf, kb = fwd_kernel(self.TB), bwd_kernel(self.TBb)
        n_out = 4 if vec0 else 1

        @jax.custom_vjp
        def mp0(px, py, pz, zf):
            return tuple(self._call(kf, (px, py, pz, zf), (), self.w0, ["f"] * n_out, self.TB, self.nw, "pn_mp0_fwd"))

        def f(px, py, pz, zf):
            return mp0(px, py, pz, zf), (px, py, pz, zf)

        def b(res, cts):
            px, py, pz, zf = res
            gx, gy, gz = self._call(kb, res, tuple(cts), self.w0T, ["a"] * 3, self.TBb, self.nwb, "pn_mp0_bwd")
            return gx, gy, gz, jnp.zeros_like(zf)

        mp0.defvjp(f, b)
        return mp0

    def _make_mpk(self, v2s, w_l, wT_l):
        F, n, rc, dot = self.F, self.n, self.rc, self.dot
        nf = 4 if v2s else 1
        nwl = len(w_l)

        def fwd_kernel(TB):
            bj, bi, sum_j, sum_i = _helpers(TB, n)

            def k(px, py, pz, zf, *rest):
                A = TB * n
                feats, w, out = rest[:nf], rest[nf:nf + nwl], rest[nf + nwl]
                g = _geom(px[...], py[...], pz[...], zf[...], TB, n, rc, None, False)
                phi, u = g["phi"], g["u"]
                msg = dot(phi, w[0][...]) * bj(feats[0][...].reshape(A, F), F)
                if v2s:
                    wv = None
                    for c in range(3):
                        t = bj(feats[1 + c][...].reshape(A, F), F) * u[c]
                        wv = t if wv is None else wv + t
                    msg = msg + dot(phi, w[1][...]) * wv
                out[...] = sum_j(msg, F).reshape(TB, n, F)
            return k

        def bwd_kernel(TB):
            bj, bi, sum_j, sum_i = _helpers(TB, n)

            def k(px, py, pz, zf, *rest):
                A = TB * n
                feats = rest[:nf + 1]
                w, wT, outs = rest[nf + 1:nf + 1 + nwl], rest[nf + 1 + nwl:nf + 1 + 2 * nwl], rest[nf + 1 + 2 * nwl:]
                g = _geom(px[...], py[...], pz[...], zf[...], TB, n, rc, None, True)
                phi, u = g["phi"], g["u"]
                Hb = bi(feats[nf][...].reshape(A, F), F)
                a_e = bj(feats[0][...].reshape(A, F), F)
                phibar = dot(Hb * a_e, wT[0][...])
                outs[0][...] = sum_i(dot(phi, w[0][...]) * Hb, F).reshape(TB, n, F)
                ubar = None
                if v2s:
                    av = [bj(feats[1 + c][...].reshape(A, F), F) for c in range(3)]
                    wv = av[0] * u[0] + av[1] * u[1] + av[2] * u[2]
                    phibar = phibar + dot(Hb * wv, wT[1][...])
                    wbar = Hb * dot(phi, w[1][...])
                    ubar = [jnp.sum(wbar * av[c], axis=1, keepdims=True) for c in range(3)]
                    for c in range(3):
                        outs[1 + c][...] = sum_i(wbar * u[c], F).reshape(TB, n, F)
                rbar = jnp.sum(phibar * g["dphi"], axis=1, keepdims=True)
                _store_grad(g, rbar, ubar, outs[-3:], TB, n)
            return k

        kf, kb = fwd_kernel(self.TB), bwd_kernel(self.TBb)

        @jax.custom_vjp
        def mpk(px, py, pz, zf, feats):
            return self._call(kf, (px, py, pz, zf), tuple(feats), w_l, ["f"], self.TB, self.nw, "pn_mp_fwd")[0]

        def f(px, py, pz, zf, feats):
            return mpk(px, py, pz, zf, feats), (px, py, pz, zf, feats)

        def b(res, ct):
            px, py, pz, zf, feats = res
            outs = self._call(kb, (px, py, pz, zf), tuple(feats) + (ct,), w_l + wT_l,
                              ["f"] * nf + ["a"] * 3, self.TBb, self.nwb, "pn_mp_bwd")
            return outs[-3], outs[-2], outs[-1], jnp.zeros_like(zf), tuple(outs[:nf])

        mpk.defvjp(f, b)
        return mpk


# ----------------------------------------------------------------------------- model
def _refine(blocks, s, v, mm):
    ys = yv = None
    for blk in blocks[:-1]:
        g = jax.nn.sigmoid(s)
        ys = mm(g * s + s, blk["W0"]) + blk["b0"]
        if v is not None:
            yv = [mm(g * c + c, blk["W1"]) for c in v]
            v = [a + b for a, b in zip(v, yv)]
        s = s + ys
    last = blocks[-1]
    zs = mm(ys, last["W0"]) + last["b0"]
    G = jax.nn.sigmoid(zs)
    return G * zs, ([G * mm(c, last["W1"]) for c in yv] if v is not None else None)


def _zbl_dense(cfg, R, Z):
    n = Z.shape[1]
    am = (Z > 0).astype(R.dtype)
    em = am[:, :, None] * am[:, None, :] * (1.0 - jnp.eye(n, dtype=R.dtype))
    d = R[:, None, :, :] - R[:, :, None, :]
    d = jnp.where(em[..., None] > 0, d, 1.0)
    r = jnp.sqrt(jnp.sum(d * d, -1))
    Zf = jnp.maximum(Z.astype(R.dtype), 1.0)
    Zi, Zj = Zf[:, :, None], Zf[:, None, :]
    cuton, cutoff = float(cfg["zbl_cuton"]), float(cfg["zbl_cutoff"])
    a = _ZBL_A / (Zi ** 0.23 + Zj ** 0.23)
    x = r / a
    phi = sum(float(c) * jnp.exp(-float(e) * x) for c, e in zip(_ZBL_C, _ZBL_E))
    s = (cutoff - r) / (cutoff - cuton)
    sw = ((6.0 * s - 15.0) * s + 10.0) * s ** 3
    sw = jnp.where(r < cuton, 1.0, jnp.where(r >= cutoff, 0.0, sw))
    rep = 0.5 * 14.3996454784255 * Zi * Zj / r * phi * sw
    return jnp.sum(jnp.where(em > 0, rep, 0.0), (1, 2))


def _default_tile(n, F):
    # tuned on RTX 5090 / 4090 (water dimers n=8, monomers n=4, MeOH dimers n=16)
    if n <= 4:
        return 8, 4
    if n <= 8:
        return 2, 4
    return 1, 4


def make_physnet_fused(config, params, max_atoms, *, precision="fp32", mlp_precision=None,
                       species=(1, 6, 8), tile=None, tile_bwd=None, jit=True):
    """Build ``fn(R, Z) -> (E, F)`` for batches of systems with <= ``max_atoms`` atoms.

    precision: in-kernel dot precision, one of fp32 | tf32x3 | tf32 | bf16 | fp16.
    mlp_precision: atom-level MLP dots (default: fp32 if precision in fp32/tf32x3,
      else the same low-precision dtype with fp32 accumulation).
    species: atomic numbers that can occur. The first message pass is folded per
      species, so atoms of any other element would receive no first-pass messages --
      pass every element present in the system (default H, C, O).
    tile / tile_bwd: (systems per program, num_warps) for the forward / reverse kernels.
    Returns a callable ``fn(R, Z) -> (E, F)``; ``fn.apply_like(R_flat, Z_flat)`` returns
    ``{"energy": (B, 1), "forces": (B*n, 3)}`` like ``PhysNet.apply`` on the same batch.
    """
    W = unpack_physnet_params(config, params)
    F = int(W["emb"].shape[1])
    if F & (F - 1) or F < 16:
        raise NotImplementedError("physnet_fused: features must be a power of two >= 16")
    n = max(4, 1 << (int(max_atoms) - 1).bit_length())
    TB, nw = tile or _default_tile(n, F)
    TBb, nwb = tile_bwd or (TB, nw)
    prec, dot_dtype = _PREC[precision]
    mp = _MP(W, config, n, TB, nw, TBb, nwb, prec, dot_dtype, species)
    mlp_precision = mlp_precision or ("fp32" if dot_dtype is None else precision)
    _, mlp_dt = _PREC[mlp_precision]
    if mlp_dt is None:
        mm = functools.partial(jnp.matmul, precision=jax.lax.Precision.HIGHEST)
    else:
        def mm(a, b):
            return jnp.matmul(a.astype(mlp_dt), b.astype(mlp_dt), preferred_element_type=jnp.float32)
    use_zbl = bool(config.get("zbl", False))
    TBm = max(TB, TBb)
    Wout = W["out"]

    def energy(px, py, pz, Zp):
        zf = Zp.astype(jnp.float32)
        h = mp.mp0(px, py, pz, zf)
        x, xv = _refine(W["ref"][0], h[0], list(h[1:]) if len(h) > 1 else None, mm)
        for k, (v2s, _, _) in enumerate(mp.later):
            feats = (x,) + (tuple(xv) if v2s else ())
            x, xv = _refine(W["ref"][k + 1], mp.mpk[k](px, py, pz, zf, feats), None, mm)
        E = jnp.sum(jnp.matmul(x, Wout, precision=jax.lax.Precision.HIGHEST) * (Zp > 0), -1)
        if use_zbl:
            E = E + _zbl_dense(config, jnp.stack([px, py, pz], -1), Zp)
        return E

    def fn(R, Z):
        R = jnp.asarray(R, jnp.float32)
        Z = jnp.asarray(Z, jnp.int32)
        B0, n0 = Z.shape
        Bp = -(-B0 // TBm) * TBm
        Rp = jnp.pad(R, ((0, Bp - B0), (0, n - n0), (0, 0)))
        Zp = jnp.pad(Z, ((0, Bp - B0), (0, n - n0)))
        # keep padded atoms far away and distinct (they are masked anyway)
        Rp = jnp.where((Zp > 0)[..., None], Rp, 0.0)
        E, vjp = jax.vjp(lambda a, b, c: energy(a, b, c, Zp), Rp[..., 0], Rp[..., 1], Rp[..., 2])
        gx, gy, gz = vjp(jnp.ones_like(E))
        Fo = -jnp.stack([gx, gy, gz], -1)[:B0, :n0] * (Z > 0)[..., None]
        Eb = jnp.where(Z > 0, W["ebias"][Z], 0.0).sum(-1)
        return E[:B0] + Eb, Fo

    return _FusedFn(jax.jit(fn) if jit else fn, int(max_atoms),
                    {"n_pad": n, "tile": (TB, nw), "tile_bwd": (TBb, nwb), "precision": precision,
                     "mlp_precision": mlp_precision, "species": tuple(species)})


class _FusedFn:
    def __init__(self, fn, max_atoms, info):
        self._fn, self.max_atoms, self.info = fn, max_atoms, info

    def __call__(self, R, Z):
        return self._fn(R, Z)

    def apply_like(self, R_flat, Z_flat, n_atoms=None):
        """``PhysNet.apply``-shaped output for a flat (B * n_atoms) batch."""
        na = int(n_atoms or self.max_atoms)
        B = R_flat.shape[0] // na
        E, Fo = self._fn(R_flat.reshape(B, na, 3), Z_flat.reshape(B, na))
        return {"energy": E[:, None], "forces": Fo.reshape(B * na, 3)}
