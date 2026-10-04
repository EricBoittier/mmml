#!/usr/bin/env python
"""Evaluate a packed full-SPICE-α checkpoint on whole splits, per subset.

Rebuilds the model and pack budget from CKPT_DIR/config.json and the frame
split from CKPT_DIR/split.npz, then scores every frame of each requested split
(training validates on a 20k subsample only). Writes to OUT_DIR:
  metrics.json      RMSE / MAE per split and subset (dipole e·Å, polar e·Å²/V, forces eV/Å)
  preds_<split>.npz per-frame dipole / polar predictions and targets, subset, n_atoms
  charges.json      predicted atomic-charge statistics per element (test split)

Usage (GPU node):
  python scripts/spice_alpha/eval_full_packed.py CKPT_DIR PARAMS_JSON OUT_DIR [--splits test valid]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

BOHR3_PER_EA2V = 97.17


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ckpt_dir", type=Path)
    ap.add_argument("params", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--ragged-dir", type=Path, default=None, help="default: config.json ragged_dir")
    ap.add_argument("--splits", nargs="+", default=["test", "valid"])
    ap.add_argument("--max-frames", type=int, default=0, help="subsample each split (smoke test)")
    ap.add_argument("--pack-scale", type=float, default=1.0,
                    help="scale the training atom/edge/molecule budget (e.g. 0.5 for a 40 GB GPU)")
    return ap.parse_args()


def main():
    args = parse_args()
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", False)
    from mmml.data.spice_alpha_ragged import SUBSET_IDS, load_ragged
    from mmml.models.efield.packed import PackSpec, _forward, iter_packed_batches, packed_losses, prefetch
    from mmml.models.efield.training import EFieldPhysNet, load_params

    cfg = json.loads((args.ckpt_dir / "config.json").read_text())
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    shards = sorted(Path(args.ragged_dir or cfg["ragged_dir"]).glob("*.npz"))
    if cfg.get("subsets"):
        shards = [p for p in shards if p.stem in cfg["subsets"]]
    t0 = time.time()
    data = load_ragged(shards)
    split = dict(np.load(args.ckpt_dir / "split.npz"))
    print(f"loaded {len(data['N'])} frames ({time.time() - t0:.0f}s)", flush=True)

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
    # Energies are scored only for runs trained on them (the driver saves the per-element references).
    e_ref_path = args.ckpt_dir / "e_ref.npy"
    e_ref = np.load(e_ref_path) if e_ref_path.exists() and cfg.get("energy_weight", 0) > 0 else None
    weights = {"energy": 0.0, "forces": 1.0, "dipole": 1.0, "charge": 0.0, "polar": 1.0}

    @jax.jit
    def evaluate(batch):
        _, (_, preds) = packed_losses(model.apply, params, batch, M, weights, 0.001, False)
        _, _, state = _forward(model.apply, params, batch, M)
        preds["charges"] = state["intermediates"]["atomic_charges"][-1]
        return preds

    names = {v: k for k, v in SUBSET_IDS.items()}
    off = data["offsets"]
    metrics, charge_stats = {}, {}
    rng = np.random.default_rng(0)
    for split_name in args.splits:
        idx = np.sort(split[split_name])
        if args.max_frames and len(idx) > args.max_frames:
            idx = np.sort(rng.choice(idx, size=args.max_frames, replace=False))
        n = len(idx)
        rec = {k: np.zeros(s, np.float32) for k, s in
               (("dipole", (n, 3)), ("dipole_ref", (n, 3)), ("polar", (n, 3, 3)), ("polar_ref", (n, 3, 3)),
                ("q_total", (n,)), ("force_sq", (n,)), ("force_n", (n,)), ("energy", (n,)), ("energy_ref", (n,)))}
        q_by_z: dict[int, list[np.ndarray]] = {}
        k, t0 = 0, time.time()
        for batch in prefetch(iter_packed_batches(data, idx, spec, e_ref=e_ref)):
            p = jax.device_get(evaluate({kk: jnp.asarray(v) for kk, v in batch.items()}))
            m = int(batch["mol_mask"].sum())
            sl = slice(k, k + m)
            rec["dipole"][sl], rec["dipole_ref"][sl] = p["dipole"][:m], batch["dipoles"][:m]
            rec["polar"][sl], rec["polar_ref"][sl] = p["polar"][:m], batch["polar"][:m]
            seg, Z = batch["batch_segments"], batch["atomic_numbers"]
            real = Z > 0
            rec["q_total"][sl] = np.bincount(seg[real], weights=p["charges"][real], minlength=M)[:m]
            fsq = np.sum((p["forces"] - batch["forces"]) ** 2, axis=1)
            rec["force_sq"][sl] = np.bincount(seg[real], weights=fsq[real], minlength=M)[:m]
            rec["force_n"][sl] = 3 * np.bincount(seg[real], minlength=M)[:m]
            rec["energy"][sl], rec["energy_ref"][sl] = p["energy"][:m], batch["energies"][:m]
            if split_name == "test":
                for z in np.unique(Z[real]):
                    q_by_z.setdefault(int(z), []).append(p["charges"][Z == z])
            k += m
        assert k == n, (k, n)
        print(f"{split_name}: {n} frames in {time.time() - t0:.0f}s", flush=True)
        subset = data["subset"][idx]
        n_atoms = (off[idx + 1] - off[idx]).astype(np.int32)
        np.savez_compressed(out / f"preds_{split_name}.npz", frame=idx, subset=subset, n_atoms=n_atoms, **rec)

        derr = rec["dipole"] - rec["dipole_ref"]
        perr = (rec["polar"] - rec["polar_ref"]) / BOHR3_PER_EA2V
        diag = np.eye(3, dtype=bool)
        res = {}
        for key, m in [("all", np.ones(n, bool))] + [(names[s], subset == s) for s in np.unique(subset)]:
            pd, po = perr[m][:, diag], perr[m][:, ~diag]
            res[key] = {
                "n_frames": int(m.sum()),
                "dipole_rmse": float(np.sqrt(np.mean(derr[m] ** 2))),
                "dipole_mae": float(np.mean(np.abs(derr[m]))),
                "dipole_norm_rel": float(np.median(np.linalg.norm(derr[m], axis=1)
                                                   / np.maximum(np.linalg.norm(rec["dipole_ref"][m], axis=1), 1e-3))),
                "polar_diag_rmse": float(np.sqrt(np.mean(pd ** 2))),
                "polar_diag_mae": float(np.mean(np.abs(pd))),
                "polar_offdiag_rmse": float(np.sqrt(np.mean(po ** 2))),
                "polar_offdiag_mae": float(np.mean(np.abs(po))),
                "forces_rmse": float(np.sqrt(rec["force_sq"][m].sum() / rec["force_n"][m].sum())),
                "q_total_abs_max": float(np.max(np.abs(rec["q_total"][m]))),
                **({"energy_rmse": float(np.sqrt(np.mean((rec["energy"][m] - rec["energy_ref"][m]) ** 2))),
                    "energy_mae_per_atom": float(np.mean(np.abs(rec["energy"][m] - rec["energy_ref"][m]) / n_atoms[m]))}
                   if e_ref is not None else {}),
            }
        metrics[split_name] = res
        if q_by_z:
            for z, chunks in sorted(q_by_z.items()):
                q = np.concatenate(chunks)
                charge_stats[z] = {"n": int(q.size), "min": float(q.min()), "p01": float(np.percentile(q, 1)),
                                   "mean": float(q.mean()), "p99": float(np.percentile(q, 99)), "max": float(q.max())}

    meta = {"ckpt_dir": str(args.ckpt_dir), "params": str(args.params), "units": {
        "dipole": "e·Å", "polar": "e·Å²/V (= Bohr³ / 97.17)", "forces": "eV/Å"}}
    (out / "metrics.json").write_text(json.dumps({"meta": meta, **metrics}, indent=2))
    if charge_stats:
        (out / "charges.json").write_text(json.dumps(charge_stats, indent=2))
    for split_name, res in metrics.items():
        print(f"== {split_name}")
        for key, r in sorted(res.items()):
            print(f"  {key:20s} n={r['n_frames']:6d} dipole {r['dipole_rmse']:.4f} polar diag "
                  f"{r['polar_diag_rmse']:.4f} offdiag {r['polar_offdiag_rmse']:.4f} forces {r['forces_rmse']:.3f}")
    for z, s in sorted(charge_stats.items()):
        print(f"  Z={z:2d} n={s['n']:8d} q min {s['min']:+.3f} p01 {s['p01']:+.3f} mean {s['mean']:+.3f} "
              f"p99 {s['p99']:+.3f} max {s['max']:+.3f}")


if __name__ == "__main__":
    main()
