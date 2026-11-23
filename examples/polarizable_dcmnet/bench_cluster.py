"""Timing of the polarisable DCMnet on N-molecule clusters (all pairs, fixed CGenFF-charge field source).

Per evaluation: field at every molecule from the other molecules' CGenFF charges -> one charge pass per molecule (vmap,
exact merged padding graph) -> site-site Coulomb energy of the polarised charges -> gradient w.r.t. all positions
(autodiff through fields and charges).  Electrostatics layer only: not a full potential.

usage: python bench_cluster.py --params RUN_DIR --mono_data ft_dataset.npz [--ns 16,32,64]
(RUN_DIR holds model_config.json + best_params.msgpack; mono_data supplies Z, q_cgenff and a reference geometry R[0].)
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from karml.models.dcmnet.polarizable import BOHR_PER_ANG, FieldMessagePassingModel, field_inputs, merged_charges


def load_params(run_dir):
    cfg = json.load(open(Path(run_dir) / "model_config.json"))["architecture"]
    p = jax.tree_util.tree_map(
        jnp.asarray, serialization.msgpack_restore((Path(run_dir) / "best_params.msgpack").read_bytes())
    )
    return cfg, p


def make(model, N, Z, qcg, ref):
    n = len(Z)
    rng = np.random.default_rng(0)
    g = int(np.ceil(N ** (1 / 3)))
    sp = 5.2  # lattice spacing (A)
    cells = np.array([[i, j, k] for i in range(g) for j in range(g) for k in range(g)])[:N] * sp + rng.normal(
        0, 0.4, (N, 3)
    )
    Rs = []
    for c in cells:
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        Q = Q * np.sign(np.linalg.det(Q))
        Rs.append(ref @ Q.T + c)
    Rall = jnp.asarray(np.array(Rs), jnp.float32)
    owner = jnp.repeat(jnp.arange(N), n)
    nd = model.n_dcm

    def energy(params, Rall):
        allpos = Rall.reshape(-1, 3)
        allq = jnp.tile(qcg, N)

        def one(i, R):
            qsrc = jnp.where(owner == i, 0.0, allq)
            V, E = field_inputs(R, qsrc, allpos, n)
            c = R.mean(0)
            q, p = merged_charges(model, params, Z, R - c, V, E)
            return q, p + c

        q, p = jax.vmap(one)(jnp.arange(N), Rall)
        q = q.reshape(-1)
        p = p.reshape(-1, 3)
        own = jnp.repeat(jnp.arange(N), n * nd)
        r = jnp.linalg.norm(p[:, None] - p[None], axis=-1) * BOHR_PER_ANG + jnp.eye(q.shape[0]) * 1e3
        w = (own[:, None] != own[None]).astype(jnp.float32)
        return 0.5 * jnp.sum(w * q[:, None] * q[None] / r)

    return Rall, jax.jit(energy), jax.jit(jax.value_and_grad(energy, argnums=1))


def run(cfg, params, Z, qcg, ref, ns, reps=20):
    model = FieldMessagePassingModel(**cfg)
    res = []
    for N in ns:
        Rall, f, fg = make(model, N, Z, qcg, ref)
        for _ in range(2):
            jax.block_until_ready(f(params, Rall))
            jax.block_until_ready(fg(params, Rall))

        def t(fn):
            ts = []
            for _ in range(reps):
                t0 = time.perf_counter()
                jax.block_until_ready(fn(params, Rall))
                ts.append(time.perf_counter() - t0)
            return float(np.median(ts))

        tf, tg = t(f), t(fg)
        res.append(
            dict(
                N=N,
                atoms=N * len(Z),
                forward_ms=tf * 1e3,
                fwdgrad_ms=tg * 1e3,
                ns_day_05fs=86400.0 / tg * 0.5e-6,
                ns_day_1fs=86400.0 / tg * 1e-6,
            )
        )
        print("%5d %7d %12.2f %12.2f %14.2f" % (N, N * len(Z), tf * 1e3, tg * 1e3, res[-1]["ns_day_05fs"]), flush=True)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", required=True)
    ap.add_argument("--mono_data", required=True)
    ap.add_argument("--ns", default="16,32,64,128")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--json", default="", help="write results to this file")
    a = ap.parse_args(argv)
    cfg, p0 = load_params(a.params)
    M = np.load(a.mono_data)
    Z = np.asarray(M["Z"])
    dev = jax.devices()[0]
    print("device", dev, "features", cfg["features"], "l", cfg["max_degree"], flush=True)
    print("%5s %7s %12s %12s %14s" % ("N", "atoms", "forward ms", "fwd+grad ms", "ns/day@0.5fs"), flush=True)
    res = run(
        cfg,
        p0,
        Z,
        jnp.asarray(M["q_cgenff"], jnp.float32),
        np.asarray(M["R"][0], np.float32),
        [int(x) for x in a.ns.split(",")],
        a.reps,
    )
    if a.json:
        json.dump(dict(device=str(dev), results=res), open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
