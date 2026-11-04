"""One pass per OMol25 LMDB file: chemical-space map (as map_chemspace) + save training candidates.
Saved (select_<file>.npz): every structure that (a) contains a target fragment (water-only clusters, CH4O, C2H6O, C8H18O,
CnH2n+2O n>=3, CnH2n+2 n>=6), (b) is a single molecule with <=12 atoms (any Q/S), (c) is a diatomic, or (d) a 10 % random
sample of non-covalent multi-molecule structures (n_frag>=2, closest heavy contact>=2 A, <=120 atoms).
Arrays: Z int8, R/F float32, E float64 (raw E_DFT, eV), Q, S, mol_id int16, offsets, reason bitmask, data_id."""
import os, re, sys, time, zlib, json
import numpy as np, lmdb
try:
    import orjson as _js
except ImportError:
    _js = json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fragment_census import frag_one, formula
sys.path.insert(0, "/mmhome/boittier/home/metatomic-runs/training/omol_universal/scripts")
from omol_io import _np_from_json
TGT = {"CH4O", "C2H6O", "C8H18O"}; ALK = re.compile(r"^C(\d+)H(\d+)(O?)$")
def target(fr):
    s = set(fr)
    if s == {"H2O"} and len(fr) >= 2: return True
    if s & TGT: return True
    for f in s:
        m = ALK.match(f)
        if m and int(m[2]) == 2 * int(m[1]) + 2 and ((m[3] and int(m[1]) >= 3) or (not m[3] and int(m[1]) >= 6)): return True
    return False
def map_select_file(f, out_map, out_sel, frac=0.10, seed=0):
    rng = np.random.default_rng(seed + abs(hash(os.path.basename(f))) % 2**31)
    env = lmdb.open(f, subdir=False, readonly=True, lock=False, readahead=True, meminit=False)
    M = {k: [] for k in ("n_atoms", "Q", "S", "n_frag", "max_frag", "dmin_inter")}; comp, did, elem = [], [], []
    S = {k: [] for k in ("Z", "R", "F", "mol_id")}; SS = {k: [] for k in ("E", "Q", "S", "N", "reason")}; sdid = []
    t0 = time.time()
    with env.begin() as txn:
        for k, v in txn.cursor():
            if not k.decode().isdigit(): continue
            d = _js.loads(zlib.decompress(v)); info = {**(d.get("data") or {}), **(d.get("key_value_pairs") or {})}
            Z = _np_from_json(d["numbers"]).astype(np.int32); R = _np_from_json(d["positions"]).astype(np.float64).reshape(-1, 3)
            nc, lab, dmin = frag_one(Z, R); q = int(round(float(info.get("charge", 0)))); s = int(round(float(info.get("spin", 1))))
            fr = [formula(Z[lab == j]) for j in range(nc)] if nc <= 12 else None
            M["n_atoms"].append(len(Z)); M["Q"].append(q); M["S"].append(s); M["n_frag"].append(nc); M["max_frag"].append(int(np.bincount(lab).max())); M["dmin_inter"].append(dmin)
            comp.append("|".join(sorted(fr)) if fr is not None else f"{nc} frags|" + formula(Z)); did.append(str(info.get("data_id", ""))); elem.append(np.bincount(Z, minlength=57)[:57])
            reason = 0
            if fr is not None and target(fr): reason |= 1
            if nc == 1 and len(Z) <= 12: reason |= 2
            if len(Z) == 2: reason |= 4
            if nc >= 2 and dmin >= 2.0 and len(Z) <= 120 and rng.random() < frac: reason |= 8
            if reason:
                S["Z"].append(Z.astype(np.int8)); S["R"].append(R.astype(np.float32)); S["F"].append(_np_from_json(d["forces"]).astype(np.float32).reshape(-1, 3))
                S["mol_id"].append(lab.astype(np.int16)); SS["E"].append(float(d["energy"])); SS["Q"].append(q); SS["S"].append(s); SS["N"].append(len(Z)); SS["reason"].append(reason); sdid.append(did[-1])
    env.close()
    np.savez(out_map, **{k: np.asarray(v) for k, v in M.items()}, comp=np.asarray(comp), data_id=np.asarray(did), elem=np.asarray(elem, np.int16))
    if SS["N"]:
        np.savez(out_sel, **{k: np.concatenate(v) for k, v in S.items()}, E=np.array(SS["E"]), Q=np.array(SS["Q"], np.int8), S=np.array(SS["S"], np.int8),
                 offsets=np.concatenate([[0], np.cumsum(SS["N"])]).astype(np.int64), reason=np.array(SS["reason"], np.int8), data_id=np.array(sdid))
    return len(comp), len(SS["N"]), time.time() - t0
