#!/usr/bin/env python
"""Extract the exact held-out validation split used by a DES training run.

make_training splits with
    data_key = jax.random.split(jax.random.PRNGKey(seed), 2)[0]
    prepare_datasets(data_key, n_train, n_valid, [path], natoms=natoms)

so the split is reproducible from (seed, n_train, n_valid, path). Reproducing it
matters: des-full-fit and the des-lj-prod-* runs pass --valid-data == --data, so
their "validation" numbers are in-sample. Only des-hybrid-ws (100000/8500) has a
genuinely disjoint hold-out, and this recovers it as a standalone NPZ that
`mmml physnet-evaluate` can consume.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import jax
import numpy as np

from mmml.models.physnetjax.physnetjax.data.data import prepare_datasets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-train", type=int, default=100000)
    ap.add_argument("--n-valid", type=int, default=8500)
    ap.add_argument("--natoms", type=int, default=34)
    a = ap.parse_args()

    data_key, _ = jax.random.split(jax.random.PRNGKey(a.seed), 2)
    train, valid = prepare_datasets(
        data_key, a.n_train, a.n_valid, [str(a.npz)], natoms=a.natoms
    )

    out = {}
    for k, v in valid.items():
        arr = np.asarray(v)
        if arr.ndim >= 1:
            out[k] = arr
    # Carry the shared master tables through so the hybrid-MM evaluate path has
    # everything it needs; they are (n_types,) and not per-sample.
    src = np.load(a.npz, allow_pickle=True)
    for k in ("cgenff_master_sigmas", "cgenff_master_epsilons"):
        if k in src.files:
            out[k] = src[k]

    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output, **out)

    n = len(out.get("E", out.get("R", [])))
    print(f"hold-out: {n} frames -> {a.output}")
    print(f"  keys: {sorted(out)[:12]}")

    # Sanity: the hold-out must not overlap the training split.
    if "R" in train and "R" in out:
        tr = np.asarray(train["R"])
        va = out["R"]
        # cheap fingerprint comparison on a few frames
        trf = {float(np.sum(tr[i])) for i in range(min(2000, len(tr)))}
        overlap = sum(1 for i in range(min(2000, len(va)))
                      if float(np.sum(va[i])) in trf)
        print(f"  overlap check on 2000 frames: {overlap} collisions "
              f"({'clean' if overlap == 0 else 'INVESTIGATE'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
