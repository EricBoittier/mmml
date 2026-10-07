#!/usr/bin/env python
"""Per-atom outputs and model internals for a packed full-SPICE-α checkpoint.

Complements eval_full_packed.py (per-frame metrics). On a random subsample of
the test split, writes OUT_DIR/atoms_test.npz with per-atom Z, charge, atomic
dipole and predicted / reference forces, plus per-frame dipole split into the
charge part Σq(r-c) and the atomic-dipole part Σμ_i. For a few example frames
(one per subset) writes OUT_DIR/examples.npz with inputs (Z, R, field,
neighbour list), outputs and the per-layer features captured from the model.

Usage (GPU node):
  python scripts/spice_alpha/eval_detail_packed.py CKPT_DIR PARAMS_JSON OUT_DIR [--n-frames 30000]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ckpt_dir", type=Path)
    ap.add_argument("params", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--n-frames", type=int, default=30000)
    ap.add_argument("--pack-scale", type=float, default=0.5)
    return ap.parse_args()


def main():
    args = parse_args()
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", False)
    from karml.data.spice_alpha_ragged import SUBSET_IDS, load_ragged
    from karml.models.efield.packed import PackSpec, _forward, iter_packed_batches, packed_losses, prefetch
    from karml.models.efield.training import EFieldPhysNet, load_params

    cfg = json.loads((args.ckpt_dir / "config.json").read_text())
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    shards = sorted(Path(cfg["ragged_dir"]).glob("*.npz"))
    if cfg.get("subsets"):
        shards = [p for p in shards if p.stem in cfg["subsets"]]
    data = load_ragged(shards)
    split = dict(np.load(args.ckpt_dir / "split.npz"))
    sc = args.pack_scale
    spec = PackSpec(int(cfg["max_molecules"] * sc), int(cfg["max_atoms"] * sc), int(cfg["max_edges"] * sc),
                    cfg["cutoff"], coulomb_cutoff=cfg.get("coulomb_cutoff"),
                    max_coulomb_edges=int(cfg.get("max_coulomb_edges", 0) * sc))
    model = EFieldPhysNet(features=cfg["features"], max_degree=cfg["max_degree"],
                          num_iterations=cfg["num_iterations"], num_basis_functions=cfg["num_basis_functions"],
                          cutoff=cfg["cutoff"], max_atomic_number=55, include_pseudotensors=cfg.get("pseudotensors", True),
                          field_scale=0.001, zbl=False, electrostatics_damping_sigma=4.0, packed=True,
                          charge_activation=cfg.get("charge_activation", "silu"),
                          atomic_dipoles=cfg.get("atomic_dipoles", True),
                          parity_correct_field=cfg.get("parity_correct", False),
                          parity_correct_dipole=cfg.get("parity_correct", False),
                          strict_pseudotensors=cfg.get("strict_pseudotensors", False))
    params = {"params": load_params(args.params)["params"]}
    M = spec.max_molecules
    weights = {"energy": 0.0, "forces": 1.0, "dipole": 1.0, "charge": 0.0, "polar": 1.0}

    @jax.jit
    def evaluate(batch):
        _, (_, preds) = packed_losses(model.apply, params, batch, M, weights, 0.001, False)
        _, _, state = _forward(model.apply, params, batch, M)
        preds["charges"] = state["intermediates"]["atomic_charges"][-1]
        preds["atomic_dipoles"] = state["intermediates"]["atomic_dipoles"][-1]
        return preds

    off = data["offsets"]
    rng = np.random.default_rng(0)
    test = split["test"]
    idx = np.sort(rng.choice(test, size=min(args.n_frames, len(test)), replace=False))

    atoms = {k: [] for k in ("frame", "Z", "q", "mu", "F", "F_ref")}
    frames = {k: [] for k in ("frame", "subset", "n_atoms", "dipole", "dipole_ref", "dipole_q", "dipole_mu",
                              "polar", "polar_ref", "field")}
    k = 0
    for batch in prefetch(iter_packed_batches(data, idx, spec)):
        p = jax.device_get(evaluate({kk: jnp.asarray(v) for kk, v in batch.items()}))
        m = int(batch["mol_mask"].sum())
        Z, seg, R = batch["atomic_numbers"], batch["batch_segments"], batch["positions"]
        real = Z > 0
        fid = idx[k:k + m]
        atoms["frame"].append(fid[seg[real]])
        atoms["Z"].append(Z[real].astype(np.int8))
        atoms["q"].append(p["charges"][real])
        atoms["mu"].append(p["atomic_dipoles"][real])
        atoms["F"].append(p["forces"][real])
        atoms["F_ref"].append(batch["forces"][real])
        n = np.bincount(seg[real], minlength=M)[:m]
        com = np.stack([np.bincount(seg[real], weights=R[real, c], minlength=M)[:m] for c in range(3)], 1) / n[:, None]
        rc = R[real] - com[seg[real]]
        dq = np.stack([np.bincount(seg[real], weights=p["charges"][real] * rc[:, c], minlength=M)[:m]
                       for c in range(3)], 1)
        dmu = np.stack([np.bincount(seg[real], weights=p["atomic_dipoles"][real, c], minlength=M)[:m]
                        for c in range(3)], 1)
        frames["frame"].append(fid)
        frames["subset"].append(batch["subset"][:m])
        frames["n_atoms"].append(n)
        frames["dipole"].append(p["dipole"][:m])
        frames["dipole_ref"].append(batch["dipoles"][:m])
        frames["dipole_q"].append(dq)
        frames["dipole_mu"].append(dmu)
        frames["polar"].append(p["polar"][:m])
        frames["polar_ref"].append(batch["polar"][:m])
        frames["field"].append(batch["electric_field"][:m])
        k += m
    np.savez_compressed(out / "atoms_test.npz",
                        **{f"atom_{k}": np.concatenate(v) for k, v in atoms.items()},
                        **{f"frame_{k}": np.concatenate(v) for k, v in frames.items()})
    print(f"per-atom data for {k} test frames", flush=True)

    # Example frames: per subset, the test frame whose size is closest to the subset median.
    names = {v: s for s, v in SUBSET_IDS.items()}
    n_test = off[test + 1] - off[test]
    ex = {}
    for s in sorted(np.unique(data["subset"][test])):
        cand = test[data["subset"][test] == s]
        nc = n_test[data["subset"][test] == s]
        f = int(cand[np.argmin(np.abs(nc - np.median(nc)))])
        ex[names[int(s)]] = f
    store = {}
    for name, f in ex.items():
        single = PackSpec(2, int(off[f + 1] - off[f]) + 1, spec.max_edges, spec.cutoff,
                          spec.coulomb_cutoff, spec.max_coulomb_edges)
        batch = next(iter_packed_batches(data, np.array([f]), single))
        bj = {kk: jnp.asarray(v) for kk, v in batch.items()}
        _, (_, preds) = packed_losses(model.apply, params, bj, 2, weights, 0.001, False)
        (_, _), state = model.apply(
            params, atomic_numbers=bj["atomic_numbers"], positions=bj["positions"], Ef=bj["electric_field"],
            dst_idx_flat=bj["dst_idx_flat"], src_idx_flat=bj["src_idx_flat"], batch_segments=bj["batch_segments"],
            batch_size=2, coulomb_dst_idx_flat=bj.get("coulomb_dst_idx_flat"),
            coulomb_src_idx_flat=bj.get("coulomb_src_idx_flat"),
            capture_intermediates=True, mutable=["intermediates"])
        n = int(off[f + 1] - off[f])
        real_e = np.asarray(batch["dst_idx_flat"]) < n
        g = f"{name}/"
        store[g + "frame"] = np.array(f)
        store[g + "Z"] = batch["atomic_numbers"][:n]
        store[g + "R"] = batch["positions"][:n]
        store[g + "field"] = batch["electric_field"][0]
        store[g + "edges"] = np.stack([batch["dst_idx_flat"][real_e], batch["src_idx_flat"][real_e]], 1)
        store[g + "F_ref"] = batch["forces"][:n]
        store[g + "dipole_ref"] = batch["dipoles"][0]
        store[g + "polar_ref"] = batch["polar"][0]
        store[g + "F"] = np.asarray(preds["forces"])[:n]
        store[g + "dipole"] = np.asarray(preds["dipole"])[0]
        store[g + "polar"] = np.asarray(preds["polar"])[0]
        inter = jax.device_get(state["intermediates"])
        store[g + "q"] = np.asarray(inter["atomic_charges"][-1])[:n]
        store[g + "mu"] = np.asarray(inter["atomic_dipoles"][-1])[:n]
        # Captured submodule outputs with a per-atom leading axis (A, parity, (l+1)^2, features).
        for mod, v in inter.items():
            if not isinstance(v, dict) or "__call__" not in v:
                continue
            y = np.asarray(v["__call__"][0])
            if y.ndim == 4 and y.shape[0] == n + 1:
                store[g + f"feat/{mod}/l0"] = y[:n, 0, 0, :]
                if y.shape[2] >= 4:
                    store[g + f"feat/{mod}/l1norm"] = np.linalg.norm(y[:n, 0, 1:4, :], axis=1)
    np.savez_compressed(out / "examples.npz", **store)
    print("examples:", ex, flush=True)
    print("captured features:", sorted({k.split('/feat/')[1] for k in store if '/feat/' in k}), flush=True)


if __name__ == "__main__":
    main()
