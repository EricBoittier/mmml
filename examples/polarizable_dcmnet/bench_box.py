"""Timing of the polarisable DCMnet on periodic liquid boxes (exact merged padding graph, CGenFF-charge source).

Per evaluation (minimum image, K nearest molecules as neighbours): neighbour list from molecular centroids -> fields at
every molecule from the CGenFF charges of its K neighbours -> one charge pass per molecule (vmap) -> site-site Coulomb
energy of the polarised charges with the K neighbours -> gradient w.r.t. all positions.  Electrostatics only, truncated
at the K-neighbour shell (no Ewald, no induction/repulsion/dispersion/intramolecular terms): the cost of the
polarisable-charge layer, not a full potential.

Input ``--boxes boxes.npz`` has, for every system key (e.g. aco266, dcm308): ``<key>_R`` (N, n, 3) A coordinates of one
frame with whole molecules, ``<key>_L`` cubic box edge (A), ``<key>_Z`` (n,) atomic numbers, ``<key>_q`` (n,) CGenFF
charges.  Build it from one frame of a liquid trajectory (see docs/polarisable-dcmnet.md).

usage: python bench_box.py --params RUN_DIR --boxes boxes.npz [--systems aco266,dcm308 --Ks 24,48]
"""

import argparse
import json
import os
import time

os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")

import jax
import jax.numpy as jnp
import numpy as np
from bench_cluster import load_params

from karml.models.dcmnet.polarizable import BOHR_PER_ANG, FieldMessagePassingModel, field_inputs, merged_charges


def make(model, B, key, K):
    R0 = jnp.asarray(B[key + "_R"])
    L = float(B[key + "_L"])
    Z = np.asarray(B[key + "_Z"])
    qcg = jnp.asarray(B[key + "_q"])
    N, n = R0.shape[:2]

    def charges_all(params, R):
        c = R.mean(1)
        dc = c[None, :, :] - c[:, None, :]
        delta = jnp.round(dc / L) * L
        d2 = jnp.sum((dc - delta) ** 2, -1) + jnp.eye(N) * 1e9
        nbr = jax.lax.top_k(-d2, K)[1]
        dsel = delta[jnp.arange(N)[:, None], nbr]  # (N,K,3) minimum-image shifts
        nbrR = R[nbr] - dsel[:, :, None, :]
        psrc = nbrR.reshape(N, K * n, 3)
        qsrc = jnp.tile(qcg, (N, K))
        V, E = jax.vmap(lambda Ri, q, p: field_inputs(Ri, q, p, n))(R, qsrc, psrc)

        def one(Ri, ci, Vi, Ei):
            q, p = merged_charges(model, params, Z, Ri - ci, Vi, Ei)
            return q, p + ci

        q, p = jax.vmap(one)(R, c, V, E)
        return q, p, nbr, dsel

    def charge_pass(params, R):
        q, p, _, _ = charges_all(params, R)
        return jnp.sum(q * q) + jnp.sum(p)

    def energy(params, R):
        q, p, nbr, dsel = charges_all(params, R)
        pj = p[nbr] - dsel[:, :, None, :]
        qj = q[nbr]
        diff = p[:, None, :, None, :] - pj[:, :, None, :, :]
        r = jnp.sqrt(jnp.sum(diff**2, -1) + 1e-12) * BOHR_PER_ANG
        return 0.5 * jnp.sum(q[:, None, :, None] * qj[:, :, None, :] / r)

    return R0, N, n, jax.jit(charge_pass), jax.jit(energy), jax.jit(jax.value_and_grad(energy, argnums=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", required=True)
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--systems", default="aco266,aco379,dcm308,dcm439")
    ap.add_argument("--Ks", default="24,48")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--json", default="", help="write results to this file")
    a = ap.parse_args(argv)
    cfg, p0 = load_params(a.params)
    model = FieldMessagePassingModel(**cfg)
    B = np.load(a.boxes)
    dev = jax.devices()[0]
    print("device", dev, "features", cfg["features"], "l", cfg["max_degree"], flush=True)
    print(
        "%-7s %5s %6s %3s %13s %13s %13s %14s"
        % ("system", "N", "atoms", "K", "charge ms", "forward ms", "fwd+grad ms", "ns/day@0.5fs"),
        flush=True,
    )
    res = []
    for key in a.systems.split(","):
        for K in [int(x) for x in a.Ks.split(",")]:
            R0, N, n, cp, f, fg = make(model, B, key, K)
            for _ in range(2):
                for fn in (cp, f, fg):
                    jax.block_until_ready(fn(p0, R0))

            def t(fn, R0=R0):
                ts = []
                for _ in range(a.reps):
                    t0 = time.perf_counter()
                    jax.block_until_ready(fn(p0, R0))
                    ts.append(time.perf_counter() - t0)
                return float(np.median(ts))

            tc, tf, tg = t(cp), t(f), t(fg)
            res.append(
                dict(
                    system=key,
                    N=N,
                    atoms=N * n,
                    K=K,
                    charge_pass_ms=tc * 1e3,
                    forward_ms=tf * 1e3,
                    fwdgrad_ms=tg * 1e3,
                    ns_day_05fs=86400.0 / tg * 0.5e-6,
                )
            )
            print(
                "%-7s %5d %6d %3d %13.2f %13.2f %13.2f %14.2f"
                % (key, N, N * n, K, tc * 1e3, tf * 1e3, tg * 1e3, res[-1]["ns_day_05fs"]),
                flush=True,
            )
    if a.json:
        json.dump(dict(device=str(dev), results=res), open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
