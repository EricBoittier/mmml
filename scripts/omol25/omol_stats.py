#!/usr/bin/env python3
"""Dataset statistics for OMol25 inputs -> analysis/stats_<split>.json.

Element counts (atoms and structures containing), natoms distribution, charge,
multiplicity and (charge, mult) joint counts, data_id subsets, and survival under
the default omol_to_npz filter (per subset). Read-only; parallel over .aselmdb files.

  python -I omol_stats.py --input raw/val --split val --workers 16
  python -I omol_stats.py --input raw/colabfit_sample/co/co_0.parquet --split colabfit_val_co0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import omol_io  # noqa: E402
from omol_to_npz import SYMBOLS, add_filter_args, make_filter  # noqa: E402


def stats_one(job):
    args, inp, fmt, fslice = job
    reasons = make_filter(args)
    el_atoms = np.zeros(119, np.int64)
    el_structs = np.zeros(119, np.int64)
    natoms, charge, mult, qs, sub = Counter(), Counter(), Counter(), Counter(), Counter()
    sub_kept, fail_first, fail_any = Counter(), Counter(), Counter()
    sub_natoms = {}
    e_per_atom = []
    fmax = []
    kw = {"max_rows": args.max_rows}
    if fslice is not None:
        kw["file_slice"] = fslice
    has_dipole = 0
    for rec in omol_io.iter_records(inp, fmt, **kw):
        Z = rec["Z"]
        bc = np.bincount(Z, minlength=119)
        el_atoms += bc
        el_structs += bc > 0
        n = len(Z)
        natoms[n] += 1
        charge[rec["charge"]] += 1
        mult[rec["spin"]] += 1
        qs[f"{rec['charge']},{rec['spin']}"] += 1
        sub[rec["data_id"]] += 1
        sub_natoms.setdefault(rec["data_id"], []).append(n)
        has_dipole += rec["dipole"] is not None
        if len(fmax) < 200_000:
            fmax.append(float(np.max(np.linalg.norm(rec["F"], axis=1))))
            e_per_atom.append(rec["E"] / n)
        r = reasons(rec)
        if r:
            fail_first[r[0]] += 1
            for x in r:
                fail_any[x] += 1
        else:
            sub_kept[rec["data_id"]] += 1
    return dict(el_atoms=el_atoms, el_structs=el_structs, natoms=natoms, charge=charge, mult=mult, qs=qs,
                sub=sub, sub_kept=sub_kept, fail_first=fail_first, fail_any=fail_any,
                sub_natoms={k: np.asarray(v, np.int32) for k, v in sub_natoms.items()},
                fmax=np.asarray(fmax), e_per_atom=np.asarray(e_per_atom), has_dipole=has_dipole)


def pct(a, qs=(0, 1, 5, 25, 50, 75, 95, 99, 100)):
    a = np.asarray(a)
    return {f"p{q}": float(np.percentile(a, q)) for q in qs} | {"mean": float(a.mean())} if a.size else {}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--format", default="auto", choices=("auto", "aselmdb", "parquet"))
    p.add_argument("--split", required=True)
    p.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis"))
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--max-rows", type=int, default=None, help="cap per task")
    add_filter_args(p)
    args = p.parse_args()
    t0 = time.time()
    fmt = args.format if args.format != "auto" else ("parquet" if args.input.endswith(".parquet") else "aselmdb")
    if fmt == "aselmdb":
        nf = len(omol_io.list_aselmdb_files(args.input))
        jobs = [(args, args.input, fmt, slice(i, i + 1)) for i in range(nf)]
    else:
        jobs = [(args, args.input, fmt, None)]
    if args.workers > 1:
        with Pool(min(args.workers, len(jobs))) as pool:
            res = pool.map(stats_one, jobs, chunksize=1)
    else:
        res = [stats_one(j) for j in jobs]

    el_atoms = sum(r["el_atoms"] for r in res)
    el_structs = sum(r["el_structs"] for r in res)
    C = {k: sum((r[k] for r in res), Counter()) for k in
         ("natoms", "charge", "mult", "qs", "sub", "sub_kept", "fail_first", "fail_any")}
    sub_natoms = {}
    for r in res:
        for k, v in r["sub_natoms"].items():
            sub_natoms.setdefault(k, []).append(v)
    n = sum(C["sub"].values())
    nat = np.repeat(np.array(list(C["natoms"].keys())), np.array(list(C["natoms"].values())))
    n_kept = sum(C["sub_kept"].values())
    out = {
        "split": args.split, "input": os.path.abspath(args.input), "format": fmt, "n_structures": n,
        "n_atoms_total": int(el_atoms.sum()),
        "structures_with_dipole": int(sum(r["has_dipole"] for r in res)),
        "natoms": pct(nat) | {"frac_le_120": float((nat <= 120).mean()), "frac_le_350": float((nat <= 350).mean()),
                               "hist_bins_10": {f"{b}-{b + 9}": int(((nat >= b) & (nat < b + 10)).sum())
                                                for b in range(0, int(nat.max()) + 1, 10)}},
        "elements": {SYMBOLS[z]: {"atoms": int(el_atoms[z]), "structures": int(el_structs[z]),
                                  "frac_structures": float(el_structs[z] / n)}
                     for z in np.argsort(-el_structs) if el_structs[z] > 0},
        "charge": {str(k): v for k, v in sorted(C["charge"].items())},
        "multiplicity": {str(k): v for k, v in sorted(C["mult"].items())},
        "charge_multiplicity_top30": dict(C["qs"].most_common(30)),
        "subsets": {k: {"n": v, "frac": v / n, "natoms": pct(np.concatenate(sub_natoms[k]), (50, 95, 100)),
                        "kept_default_filter": C["sub_kept"].get(k, 0)} for k, v in C["sub"].most_common()},
        "default_filter": {"elements": args.elements, "max_atoms": args.max_atoms,
                           "max_abs_charge": args.max_abs_charge, "max_multiplicity": args.max_multiplicity,
                           "min_atoms": args.min_atoms, "n_kept": n_kept, "survival": n_kept / max(n, 1),
                           "fail_first_reason": dict(C["fail_first"].most_common()),
                           "fail_any_reason": dict(C["fail_any"].most_common())},
        "max_atomic_force_eV_per_A_sample": pct(np.concatenate([r["fmax"] for r in res])),
        "E_dft_per_atom_eV_sample": pct(np.concatenate([r["e_per_atom"] for r in res])),
        "seconds": round(time.time() - t0, 1),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"stats_{args.split}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"{n:,} structures; default-filter survival {100 * out['default_filter']['survival']:.1f}% -> {path}")


if __name__ == "__main__":
    main()
