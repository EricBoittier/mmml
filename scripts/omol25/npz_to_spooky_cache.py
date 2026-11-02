#!/usr/bin/env python3
"""Merge omol_to_npz shards (optionally a random subset) into the Orbax flat cache that
scripts/train_so3lr_spooky_extxyz.py reads with ``--mode train --cache-path <dir>``.

Keys written (exactly what cache_extxyz_to_orbax writes): R Z F mol_offsets E N Q S D
+ metadata_n_structures / metadata_n_atoms_total / metadata_max_atoms, and
metadata.json sidecar (dipole_key + provenance: subset names/counts, source shards).
Optionally also a single merged .npz (--npz-out) for other mmml tools.

  python -I npz_to_spooky_cache.py --shards npz/train_4M --n-structures 200000 --seed 0 \
      --out caches/omol_pilot200k
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter

import numpy as np

FLAT = ("R", "Z", "F", "E", "N", "Q", "S", "D")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shards", required=True, nargs="+", help="shard dirs or npz files")
    p.add_argument("--out", required=True, help="Orbax cache directory to create")
    p.add_argument("--n-structures", type=int, default=None, help="uniform random subset size (default: all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--npz-out", default=None, help="also write a merged npz here")
    p.add_argument("--linear-ref-from", default=None,
                   help="apply (not fit) the linear_ref stored in another cache's metadata.json (e.g. val with train ref)")
    p.add_argument("--linear-ref", choices=("none", "fit"), default="none",
                   help="fit: additionally subtract a least-squares per-element offset E ~ sum_Z n_Z b_Z + b_Q*Q "
                        "(fit on this subset; b written to metadata.json). Cancels in ML/MM dimer-minus-monomer "
                        "terms and leaves forces unchanged; makes initial E targets ~zero-mean.")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    files = []
    for s in args.shards:
        files += sorted(glob.glob(os.path.join(s, "*.npz"))) if os.path.isdir(s) else [s]
    sizes = []
    for f in files:
        with np.load(f, allow_pickle=False) as d:
            sizes.append(int(d["N"].shape[0]))
    n_all = int(sum(sizes))
    rng = np.random.default_rng(args.seed)
    if args.n_structures is not None and args.n_structures < n_all:
        pick = np.sort(rng.choice(n_all, size=args.n_structures, replace=False))
    else:
        pick = np.arange(n_all)
    starts = np.concatenate([[0], np.cumsum(sizes)])

    parts = {k: [] for k in FLAT}
    subset_counts = Counter()
    for fi, f in enumerate(files):
        sel = pick[(pick >= starts[fi]) & (pick < starts[fi + 1])] - starts[fi]
        if sel.size == 0:
            continue
        with np.load(f, allow_pickle=False) as d:
            off = d["mol_offsets"]
            atom_idx = np.concatenate([np.arange(off[i], off[i + 1]) for i in sel])
            for k in ("R", "Z", "F"):
                parts[k].append(d[k][atom_idx])
            for k in ("E", "N", "Q", "S", "D"):
                parts[k].append(d[k][sel])
            names = d["subset_names"]
            subset_counts.update(names[d["subset"][sel]].tolist())
    data = {k: np.concatenate(v) for k, v in parts.items()}
    data["R"] = data["R"].astype(np.float64); data["F"] = data["F"].astype(np.float64)
    data["Z"] = data["Z"].astype(np.int32); data["N"] = data["N"].astype(np.int32).reshape(-1, 1)
    for k in ("E", "Q", "S"):
        data[k] = data[k].astype(np.float64).reshape(-1, 1)
    data["D"] = data["D"].astype(np.float64).reshape(-1, 3)
    data["mol_offsets"] = np.concatenate([[0], np.cumsum(data["N"].reshape(-1))]).astype(np.int64)
    linref = None
    if args.linear_ref_from:
        with open(args.linear_ref_from) as fh:
            linref = dict(json.load(fh)["linear_ref"], applied_from=os.path.abspath(args.linear_ref_from))
        tab = np.zeros(119)
        tab[np.asarray(linref["Z"])] = linref["b_eV"]
        mol_of_atom = np.repeat(np.arange(data["N"].shape[0]), data["N"].reshape(-1))
        shift = np.bincount(mol_of_atom, weights=tab[data["Z"]], minlength=data["N"].shape[0])
        shift += linref["b_Q_eV_per_e"] * data["Q"].reshape(-1)
        data["E"] = (data["E"].reshape(-1) - shift).reshape(-1, 1)
        print(f"applied linear ref from {args.linear_ref_from}; E std now {data['E'].std():.3f} eV", flush=True)
    elif args.linear_ref == "fit":
        mol_of_atom = np.repeat(np.arange(data["N"].shape[0]), data["N"].reshape(-1))
        counts = np.zeros((data["N"].shape[0], 119))
        np.add.at(counts, (mol_of_atom, data["Z"]), 1.0)
        used = np.where(counts.sum(0) > 0)[0]
        y = data["E"].reshape(-1)
        # columns: element counts + total charge Q (linear in Q, so it also cancels in
        # E_AB - E_A - E_B when Q_AB = Q_A + Q_B; no Q^2 / spin terms on purpose)
        X = np.concatenate([counts[:, used], data["Q"].reshape(-1, 1)], axis=1)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        b, b_q = coef[:-1], float(coef[-1])
        resid = y - X @ coef
        linref = {"Z": used.tolist(), "b_eV": b.tolist(), "b_Q_eV_per_e": b_q, "E_std_before": float(y.std()),
                  "E_std_after": float(resid.std()), "E_mae_after_per_atom": float(np.mean(np.abs(resid) / data["N"].reshape(-1)))}
        data["E"] = resid.reshape(-1, 1)
        print(f"linear ref: E std {linref['E_std_before']:.2f} -> {linref['E_std_after']:.3f} eV", flush=True)
    n_mol, n_at = data["N"].shape[0], int(data["mol_offsets"][-1])
    data["metadata_n_structures"] = np.asarray(n_mol, dtype=np.int64)
    data["metadata_n_atoms_total"] = np.asarray(n_at, dtype=np.int64)
    data["metadata_max_atoms"] = np.asarray(int(data["N"].max()), dtype=np.int32)

    if args.npz_out:
        np.savez(args.npz_out, **{k: v for k, v in data.items() if not k.startswith("metadata_")})

    import orbax.checkpoint as ocp
    from flax.training import orbax_utils

    out = os.path.abspath(args.out)
    if os.path.exists(out) and not args.force:
        raise FileExistsError(f"{out} exists (use --force)")
    ocp.PyTreeCheckpointer().save(out, data, save_args=orbax_utils.save_args_from_target(data), force=args.force)
    meta = {"dipole_key": "", "source_shards": files, "n_structures": n_mol, "n_atoms_total": n_at,
            "max_atoms": int(data["N"].max()), "seed": args.seed, "subset_counts": dict(subset_counts.most_common()),
            "energy": "E_DFT - sum E_iso(Z,q=0) [eV] (see shard filter_report.json)",
            "dipole_is_zero": bool(np.all(data["D"] == 0)), "linear_ref": linref,
            "est_host_ram_GB": round(sum(v.nbytes for v in data.values()) / 1e9, 2)}
    with open(os.path.join(out, "metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(json.dumps({k: meta[k] for k in ("n_structures", "n_atoms_total", "max_atoms", "est_host_ram_GB")}), "->", out)


if __name__ == "__main__":
    main()
