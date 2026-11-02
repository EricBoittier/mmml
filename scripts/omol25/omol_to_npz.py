#!/usr/bin/env python3
"""OMol25 (ASE-LMDB or ColabFit parquet) -> filtered, sharded flat npz for SpookyPhysNet.

Output shard format == the flat dict consumed by
scripts/train_so3lr_spooky_extxyz.py (restore_cached_data -> stack_device_batches ->
spooky_training.build_spooky_batch_from_flat_data):

    R (n_atoms_total,3) f8 [A]      Z (n_atoms_total,) i4      F (n_atoms_total,3) f8 [eV/A]
    mol_offsets (n_mol+1,) i8       N (n_mol,1) i4
    E (n_mol,1) f8 [eV]  = E_DFT - sum_i E_iso(Z_i, q=0)   (omol_elem_refs, neutral atoms)
    Q (n_mol,1) f8  total charge    S (n_mol,1) f8  spin multiplicity 2S+1 (OMol25 'spin')
    D (n_mol,3) f8  dipole [e*A] if present in the source, else zeros (required key; only
                     used with --predict-charges, so leave that off unless D is real)
  extra provenance (dropped by npz_to_spooky_cache.py): E_dft, subset (i2), subset_names, keys.

dtypes are deliberate: build_spooky_batch_from_flat_data calls np.asarray(data["R"], float64)
on the *whole* array every batch, so float32 storage would trigger a full copy per step.

Units: OMol25 energies are eV (wB97M-V/def2-TZVPD total energies), forces eV/A.

Usage (one process per input file group; shards are independent):
  python -I omol_to_npz.py --input raw/train_4M --out npz/train_4M --workers 16
  python -I omol_to_npz.py --input raw/colabfit_sample/co/co_0.parquet --out npz/cf_sample --max-rows 20000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import omol_io  # noqa: E402

SYMBOLS = ("X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
           "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr "
           "Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn").split()
SYM2Z = {s: i for i, s in enumerate(SYMBOLS)}
DEFAULT_ELEMENTS = "H,C,N,O,F,P,S,Cl,Br,I,Li,Na,K,Mg,Ca,Zn"


def add_filter_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("filter")
    g.add_argument("--elements", default=DEFAULT_ELEMENTS, help="allowed elements (comma list); 'all' disables")
    g.add_argument("--min-atoms", type=int, default=2, help="drop isolated atoms (no edges)")
    g.add_argument("--max-atoms", type=int, default=120)
    g.add_argument("--max-abs-charge", type=int, default=2)
    g.add_argument("--max-multiplicity", type=int, default=3)
    g.add_argument("--max-force", type=float, default=None, help="drop if any |F_i| > this [eV/A] (off by default)")
    g.add_argument("--subsets", default=None, help="keep only these data_id values (comma list)")
    g.add_argument("--exclude-subsets", default=None, help="drop these data_id values (comma list)")


def make_filter(args):
    allowed = None
    if args.elements.lower() != "all":
        allowed = np.zeros(119, dtype=bool)
        for s in args.elements.split(","):
            allowed[SYM2Z[s.strip()]] = True
    keep_sub = set(args.subsets.split(",")) if args.subsets else None
    drop_sub = set(args.exclude_subsets.split(",")) if args.exclude_subsets else set()

    def reasons(rec) -> list[str]:
        r = []
        n = len(rec["Z"])
        if allowed is not None and not allowed[rec["Z"]].all():
            r.append("elements")
        if n < args.min_atoms:
            r.append("min_atoms")
        if n > args.max_atoms:
            r.append("max_atoms")
        if abs(rec["charge"]) > args.max_abs_charge:
            r.append("charge")
        if rec["spin"] > args.max_multiplicity:
            r.append("multiplicity")
        if args.max_force is not None and float(np.max(np.linalg.norm(rec["F"], axis=1))) > args.max_force:
            r.append("max_force")
        if (keep_sub is not None and rec["data_id"] not in keep_sub) or rec["data_id"] in drop_sub:
            r.append("subset")
        return r

    return reasons


class ShardWriter:
    def __init__(self, out_dir: str, prefix: str, shard_size: int, ref_table: np.ndarray | None):
        self.out_dir, self.prefix, self.shard_size, self.ref = out_dir, prefix, shard_size, ref_table
        self.i_shard = 0
        self.files: list[str] = []
        self._reset()

    def _reset(self):
        self.buf = defaultdict(list)
        self.n = 0

    def add(self, rec):
        Z = rec["Z"]
        e_ref = float(np.sum(self.ref[Z])) if self.ref is not None else 0.0
        if not np.isfinite(e_ref):
            raise ValueError(f"{rec['key']}: missing isolated-atom reference for Z in {sorted(set(Z.tolist()))}")
        b = self.buf
        b["R"].append(rec["R"]); b["Z"].append(Z); b["F"].append(rec["F"])
        b["N"].append(len(Z)); b["E"].append(rec["E"] - e_ref); b["E_dft"].append(rec["E"])
        b["Q"].append(rec["charge"]); b["S"].append(rec["spin"])
        b["D"].append(rec["dipole"] if rec["dipole"] is not None else np.zeros(3))
        b["subset"].append(rec["data_id"]); b["keys"].append(rec["key"])
        self.n += 1
        if self.n >= self.shard_size:
            self.flush()

    def flush(self):
        if self.n == 0:
            return
        b = self.buf
        N = np.asarray(b["N"], dtype=np.int32)
        names = sorted(set(b["subset"]))
        name_idx = {s: i for i, s in enumerate(names)}
        out = {
            "R": np.concatenate(b["R"]).astype(np.float64),
            "Z": np.concatenate(b["Z"]).astype(np.int32),
            "F": np.concatenate(b["F"]).astype(np.float64),
            "mol_offsets": np.concatenate([[0], np.cumsum(N)]).astype(np.int64),
            "N": N.reshape(-1, 1),
            "E": np.asarray(b["E"], dtype=np.float64).reshape(-1, 1),
            "E_dft": np.asarray(b["E_dft"], dtype=np.float64).reshape(-1, 1),
            "Q": np.asarray(b["Q"], dtype=np.float64).reshape(-1, 1),
            "S": np.asarray(b["S"], dtype=np.float64).reshape(-1, 1),
            "D": np.asarray(b["D"], dtype=np.float64).reshape(-1, 3),
            "subset": np.asarray([name_idx[s] for s in b["subset"]], dtype=np.int16),
            "subset_names": np.asarray(names),
            "keys": np.asarray(b["keys"]),
        }
        path = os.path.join(self.out_dir, f"{self.prefix}_{self.i_shard:04d}.npz")
        np.savez(path, **out)
        self.files.append(path)
        self.i_shard += 1
        self._reset()


def convert_one(job):
    """Convert one input (file or file-slice) into shards; return stats."""
    args, inp, prefix, file_slice = job
    reasons = make_filter(args)
    ref = None
    if args.ref_mode == "neutral":
        ref = omol_io.neutral_ref_table(omol_io.load_iso_atom_refs(args.refs))
    w = ShardWriter(args.out, prefix, args.shard_size, ref)
    tot, kept = Counter(), Counter()
    fail_first, fail_any = Counter(), Counter()
    kw = {"max_rows": args.max_rows}
    if file_slice is not None:
        kw["file_slice"] = file_slice
    for rec in omol_io.iter_records(inp, args.format, **kw):
        tot[rec["data_id"]] += 1
        r = reasons(rec)
        if r:
            fail_first[r[0]] += 1
            for x in r:
                fail_any[x] += 1
            continue
        kept[rec["data_id"]] += 1
        w.add(rec)
    w.flush()
    return {"total": dict(tot), "kept": dict(kept), "fail_first": dict(fail_first),
            "fail_any": dict(fail_any), "shards": w.files}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="dir of *.aselmdb, a .aselmdb file, or ColabFit .parquet")
    p.add_argument("--format", default="auto", choices=("auto", "aselmdb", "parquet"))
    p.add_argument("--out", required=True)
    p.add_argument("--refs", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                  "raw/hf_refs/references/iso_atom_elem_refs.yaml"))
    p.add_argument("--ref-mode", default="neutral", choices=("neutral", "none"),
                   help="neutral: E - sum E_iso(Z, q=0) [eV]; none: raw DFT total energy")
    p.add_argument("--shard-size", type=int, default=100_000, help="structures per npz shard")
    p.add_argument("--workers", type=int, default=1, help="processes (aselmdb: one task per .aselmdb file)")
    p.add_argument("--max-rows", type=int, default=None, help="cap per task (smoke tests)")
    add_filter_args(p)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    fmt = args.format
    if fmt == "auto":
        fmt = "parquet" if args.input.endswith(".parquet") else "aselmdb"
    if fmt == "aselmdb":
        files = omol_io.list_aselmdb_files(args.input)
        jobs = [(args, args.input, f"shard_f{i:04d}", slice(i, i + 1)) for i in range(len(files))]
    else:
        jobs = [(args, args.input, "shard_p0", None)]
    args.format = fmt

    with Pool(min(args.workers, len(jobs))) as pool:
        results = pool.map(convert_one, jobs, chunksize=1) if args.workers > 1 else [convert_one(j) for j in jobs]

    agg = {k: Counter() for k in ("total", "kept", "fail_first", "fail_any")}
    shards = []
    for r in results:
        for k in agg:
            agg[k].update(r[k])
        shards += r["shards"]
    n_tot, n_keep = sum(agg["total"].values()), sum(agg["kept"].values())
    report = {
        "input": os.path.abspath(args.input), "format": fmt, "ref_mode": args.ref_mode,
        "energy_units": "eV", "force_units": "eV/A",
        "filter": {k: getattr(args, k) for k in ("elements", "min_atoms", "max_atoms", "max_abs_charge",
                                                  "max_multiplicity", "max_force", "subsets", "exclude_subsets")},
        "n_total": n_tot, "n_kept": n_keep, "survival": n_keep / max(n_tot, 1),
        "fail_first_reason": dict(agg["fail_first"].most_common()),
        "fail_any_reason": dict(agg["fail_any"].most_common()),
        "per_subset": {s: {"total": agg["total"][s], "kept": agg["kept"].get(s, 0),
                           "survival": agg["kept"].get(s, 0) / agg["total"][s]}
                       for s, _ in agg["total"].most_common()},
        "shards": sorted(shards), "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out, "filter_report.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"kept {n_keep:,}/{n_tot:,} ({100 * report['survival']:.1f}%) in {len(shards)} shards, "
          f"{report['seconds']} s -> {args.out}/filter_report.json")


if __name__ == "__main__":
    main()
