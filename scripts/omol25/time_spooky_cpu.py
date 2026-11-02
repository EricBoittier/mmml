#!/usr/bin/env python3
"""CPU timing + XLA cost/memory analysis of one SpookyPhysNet E+F train step (value_and_grad
of E/F MSE, i.e. includes the force double-backward) at several (pad_atoms, batch) shapes.

Pairs are all-pairs within a structure (e3x.ops.sparse_pairwise_indices, as in the trainer),
so cost ~ B * n^2, independent of the 6 A cutoff.

  JAX_PLATFORMS=cpu PYTHONPATH=/mmhome/boittier/home/mmml-perf python -P time_spooky_cpu.py \
      --cache /tmp/omol_smoke/cache600 --out analysis/timing_cpu.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import orbax.checkpoint as ocp  # noqa: E402

from mmml.models.physnetjax.physnetjax.models.spooky_model import SpookyPhysNet  # noqa: E402
from mmml.models.physnetjax.physnetjax.training.spooky_training import build_spooky_batch_from_flat_data  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--features", type=int, default=64)
    p.add_argument("--max-degree", type=int, default=1)
    p.add_argument("--num-iterations", type=int, default=3)
    p.add_argument("--cutoff", type=float, default=6.0)
    p.add_argument("--shapes", default="24:16,48:8,80:4,120:2")
    p.add_argument("--reps", type=int, default=3)
    a = p.parse_args()
    data = {k: np.asarray(v) for k, v in ocp.PyTreeCheckpointer().restore(os.path.abspath(a.cache)).items()
            if not k.startswith("metadata_")}
    N = data["N"].reshape(-1)
    model = SpookyPhysNet(features=a.features, max_degree=a.max_degree, num_iterations=a.num_iterations,
                          cutoff=a.cutoff, max_padded_atoms=int(N.max()), use_energy_bias=True, zbl=True)
    rows = []
    params = None
    for spec in a.shapes.split(","):
        pad, B = map(int, spec.split(":"))
        idx = np.where(N <= pad)[0][:B]
        b = build_spooky_batch_from_flat_data(data, idx, pad_atoms=pad)
        if params is None:
            params = model.init(jax.random.PRNGKey(0), atomic_numbers=b["Z"], charges=b["Q_atoms"], spins=b["S_atoms"],
                                positions=b["R"], dst_idx=b["dst_idx"], src_idx=b["src_idx"],
                                batch_segments=b["batch_segments"], batch_size=B, batch_mask=b["batch_mask"],
                                atom_mask=b["atom_mask"])
            n_params = int(sum(x.size for x in jax.tree_util.tree_leaves(params)))

        def loss(prm, b=b, B=B):
            o = model.apply(prm, atomic_numbers=b["Z"], charges=b["Q_atoms"], spins=b["S_atoms"], positions=b["R"],
                            dst_idx=b["dst_idx"], src_idx=b["src_idx"], batch_segments=b["batch_segments"],
                            batch_size=B, batch_mask=b["batch_mask"], atom_mask=b["atom_mask"])
            el = jnp.mean((o["energy"].reshape(-1, 1) - b["E"]) ** 2)
            fl = jnp.sum(((o["forces"] - b["F"]) ** 2) * b["atom_mask"][:, None]) / jnp.sum(b["atom_mask"])
            return el + 52.91 * fl

        step = jax.jit(jax.value_and_grad(loss))
        comp = step.lower(params).compile()
        ca = comp.cost_analysis()
        ca = ca[0] if isinstance(ca, (list, tuple)) else ca
        try:
            ma = comp.memory_analysis()
            temp_mb = ma.temp_size_in_bytes / 1e6
        except Exception:
            temp_mb = None
        jax.block_until_ready(step(params))
        t0 = time.perf_counter()
        for _ in range(a.reps):
            jax.block_until_ready(step(params))
        dt = (time.perf_counter() - t0) / a.reps
        r = {"pad_atoms": pad, "batch": B, "pairs": int(b["dst_idx"].shape[0]), "cpu_s_per_step": dt,
             "cpu_ms_per_structure": 1e3 * dt / B, "gflop_per_step": (ca or {}).get("flops", float("nan")) / 1e9,
             "gflop_per_pair": (ca or {}).get("flops", float("nan")) / 1e9 / b["dst_idx"].shape[0],
             "xla_temp_MB": temp_mb, "temp_MB_per_kpair": (temp_mb / (b["dst_idx"].shape[0] / 1e3)) if temp_mb else None}
        print(r, flush=True)
        rows.append(r)
    out = {"model": {"features": a.features, "max_degree": a.max_degree, "num_iterations": a.num_iterations,
                     "cutoff": a.cutoff, "n_params": n_params}, "cpu": os.cpu_count(), "rows": rows}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
