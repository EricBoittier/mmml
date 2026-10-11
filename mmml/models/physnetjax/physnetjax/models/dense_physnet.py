"""Dense per-system PhysNet (e3x, max_degree <= 1, no charges) for small fragments.

Same math as :class:`PhysNet` on the all-pairs (i != j) edge list that
``prepare_batches_md`` builds, but on dense ``(B, n, n)`` tensors per system
(monomer / dimer), so XLA fuses the message passing into a few batched matmuls
and reductions instead of gather + segment_sum over a flat edge list. Forces by
autodiff. Weights come straight from the PhysNet params tree (checkpoint
compatible). Prototype: perf/pallas/scripts/pnet.py (watmeoh perf study).
"""
import math

import jax
import jax.numpy as jnp
import numpy as np


def dense_supported(model) -> bool:
    return (
        int(getattr(model, "max_degree", 9)) <= 1
        and not bool(getattr(model, "charges", False))
        and not bool(getattr(model, "efa", False))
        and int(getattr(model, "n_refinement_blocks", -1)) >= 1
        and type(model).__name__ == "PhysNet"
    )


def model_cfg(model) -> dict:
    keys = ("max_degree", "num_iterations", "n_refinement_blocks", "num_basis_functions",
            "cutoff", "zbl", "zbl_cuton", "zbl_cutoff")
    return {k: getattr(model, k) for k in keys}


def cheb_rbf(r, K):
    """T_k(2 exp(-r) - 1), k=0..K-1 by recurrence (== e3x exponential_chebyshev)."""
    x = 2.0 * jnp.exp(-r) - 1.0
    T = [jnp.ones_like(x), x]
    for _ in range(2, K):
        T.append(2.0 * x * T[-1] - T[-2])
    return jnp.stack(T[:K], -1)


def smooth_cutoff(r, rc):
    x = r / rc
    inside = x < 1
    xs = jnp.where(inside, x, 0.0)
    return jnp.where(inside, jnp.exp(1.0 - 1.0 / (1.0 - xs * xs)), 0.0)


def unpack(cfg, params):
    """Flatten the flax tree into the effective weights used by the fused math."""
    p = params["params"]
    md = int(cfg["max_degree"])
    nit = int(cfg["num_iterations"])
    nres = abs(int(cfg["n_refinement_blocks"]))
    W = {"emb": p["Embed_0"]["embedding"][:, 0, 0, :], "mp": [], "ref": []}
    dense_i = 0
    for it in range(nit):
        mp = p[f"MessagePass_{it}"]
        tk = mp["tensor"]["kernel"]  # (1, L1+1, 1, L2+1, 1, L3+1, F)
        last = it == nit - 1
        in_deg = 0 if it == 0 else (md if True else 0)
        e = {"Ws": mp["filter"]["0+"]["kernel"] * tk[0, 0, 0, 0, 0, 0][None]}
        if md >= 1:
            if not last:  # scalar x -> vector out: k[0,1,0,0,0,1]
                e["Wv"] = mp["filter"]["1-"]["kernel"] * tk[0, 1, 0, 0, 0, 1][None]
                e["kind"] = "s2v"
            elif in_deg >= 1 and it > 0:  # vector x . vector filter -> scalar: k[0,1,0,1,0,0]/sqrt3
                e["Wv"] = mp["filter"]["1-"]["kernel"] * tk[0, 1, 0, 1, 0, 0][None] / math.sqrt(3.0)
                e["kind"] = "v2s"
            else:
                e["kind"] = "s2s"
        else:
            e["kind"] = "s2s"
        W["mp"].append(e)
        blocks = []
        for _ in range(nres + 1):
            dd = p[f"Dense_{dense_i}"]
            dense_i += 1
            blocks.append({"W0": dd["0+"]["kernel"], "b0": dd["0+"]["bias"],
                           "W1": dd["1-"]["kernel"] if "1-" in dd else None})
        W["ref"].append(blocks)
    W["out1"] = p[f"Dense_{dense_i}"]["0+"]["kernel"][:, 0]
    W["out2"] = p[f"Dense_{dense_i + 1}"]["kernel"][0, 0]
    W["ebias"] = p["energy_bias"] if "energy_bias" in p else jnp.zeros(W["emb"].shape[0])
    return W


ZBL_A = 0.8854 * 0.529177249
ZBL_C = np.array([0.18175, 0.50986, 0.28022, 0.02817])
ZBL_C = ZBL_C / np.linalg.norm(ZBL_C)
ZBL_E = np.array([3.19980, 0.94229, 0.40290, 0.20162])


def zbl_pair(r, Zi, Zj, cuton, cutoff):
    a = ZBL_A / (Zi ** 0.23 + Zj ** 0.23 + 1e-30)
    x = r / a
    phi = sum(float(c) * jnp.exp(-float(e) * x) for c, e in zip(ZBL_C, ZBL_E))
    s = (cutoff - r) / (cutoff - cuton)
    sw = ((6.0 * s - 15.0) * s + 10.0) * s ** 3
    sw = jnp.where(r < cuton, 1.0, jnp.where(r >= cutoff, 0.0, sw))
    return 0.5 * 14.3996454784255 * Zi * Zj / r * phi * sw


def dense_energy(W, cfg, R, Z):
    """Per-system energies (B,) for R (B,n,3), Z (B,n) (Z=0 marks padding)."""
    B, n, _ = R.shape
    K = int(cfg["num_basis_functions"])
    rc = float(cfg["cutoff"])
    am = (Z > 0).astype(R.dtype)  # (B,n)
    eye = jnp.eye(n, dtype=R.dtype)
    em = am[:, :, None] * am[:, None, :] * (1.0 - eye)  # (B,n,n) dst i, src j
    d = R[:, None, :, :] - R[:, :, None, :]  # d[b,i,j] = R_j - R_i
    d = jnp.where(em[..., None] > 0, d, 1.0)  # padding/self edges: safe dummy vector, gated below
    r = jnp.sqrt(jnp.sum(d * d, -1))
    u = d / r[..., None]
    phi = cheb_rbf(r, K) * (smooth_cutoff(r, rc) * em)[..., None]  # (B,n,n,K)
    x0 = W["emb"][Z]  # (B,n,F)
    xv = None
    for mp, blocks in zip(W["mp"], W["ref"]):
        S = phi @ mp["Ws"]  # (B,n,n,F)
        if mp["kind"] == "s2v":
            V = phi @ mp["Wv"]
            h0 = jnp.einsum("bijf,bjf->bif", S, x0)
            h1 = jnp.einsum("bijc,bijf,bjf->bicf", u, V, x0)
        elif mp["kind"] == "v2s":
            V = phi @ mp["Wv"]
            h0 = jnp.einsum("bijf,bjf->bif", S, x0) + jnp.einsum("bijc,bijf,bjcf->bif", u, V, xv)
            h1 = None
        else:
            h0 = jnp.einsum("bijf,bjf->bif", S, x0)
            h1 = None
        # refinement (e3x semantics: silu gate on the scalar channel)
        x_s, x_v = h0, h1
        y_s = y_v = None
        for blk in blocks[:-1]:
            g = jax.nn.sigmoid(x_s)
            y_s = (g * x_s + x_s) @ blk["W0"] + blk["b0"]
            if x_v is not None:
                y_v = (g[:, :, None, :] * x_v + x_v) @ blk["W1"]
                x_v = x_v + y_v
            x_s = x_s + y_s
        blk = blocks[-1]
        if y_s is None:  # n_res == 0 (not used here)
            y_s, y_v = x_s, x_v
        z_s = y_s @ blk["W0"] + blk["b0"]
        z_v = y_v @ blk["W1"] if y_v is not None else None
        g = jax.nn.sigmoid(z_s)
        x0 = g * z_s
        xv = g[:, :, None, :] * z_v if z_v is not None else None
    Ei = (x0 @ W["out1"]) * W["out2"] + W["ebias"][Z]
    Ei = Ei * am
    E = jnp.sum(Ei, -1)
    if cfg.get("zbl", False):
        Zf = Z.astype(R.dtype)
        rep = zbl_pair(r, Zf[:, :, None], Zf[:, None, :], float(cfg["zbl_cuton"]), float(cfg["zbl_cutoff"]))
        E = E + jnp.sum(rep * em, (1, 2))
    return E


def make_dense_apply(model, params):
    """``f(R (B,n,3), Z (B,n)) -> (E (B,), F (B,n,3))``; Z == 0 marks padding."""
    cfg = model_cfg(model)
    W = unpack(cfg, params)

    def f(R, Z):
        Es, vjp = jax.vjp(lambda R_: dense_energy(W, cfg, R_, Z), R)
        (g,) = vjp(jnp.ones_like(Es))
        return Es, -g * (Z > 0)[..., None].astype(g.dtype)

    return f
