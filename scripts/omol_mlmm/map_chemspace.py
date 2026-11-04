"""Chemical-space map of an OMol25 ASE-LMDB split (read-only; zlib-JSON records, no pickle).
One output npz per LMDB file (parallel over files / array tasks): per structure n_atoms, element counts (Z<=56 table),
Q, multiplicity, data_id, n_frag, largest fragment, closest inter-fragment heavy-atom contact, sorted fragment formulas.
    python map_chemspace.py <lmdb_dir> <out_dir> <task_index> <n_tasks> [max_per_file]
"""
import json, os, sys, time, zlib
from collections import Counter
import numpy as np, lmdb
try:
    import orjson as _js
except ImportError:
    _js = json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fragment_census import frag_one, formula
sys.path.insert(0, "/mmhome/boittier/home/metatomic-runs/training/omol_universal/scripts")
from omol_io import list_aselmdb_files, _np_from_json

def map_file(f, out, max_n=None):
    env = lmdb.open(f, subdir=False, readonly=True, lock=False, readahead=True, meminit=False)   # sequential scan: readahead (cold NFS 4.8 -> ? ms)
    rows = {k: [] for k in ("n_atoms", "Q", "S", "n_frag", "max_frag", "dmin_inter")}; comp, dataid, elem = [], [], []
    t0 = time.time()
    with env.begin() as txn:
        cur = txn.cursor()
        for k, v in cur:
            ks = k.decode()
            if not ks.isdigit():
                continue
            d = _js.loads(zlib.decompress(v)); info = {**(d.get("data") or {}), **(d.get("key_value_pairs") or {})}
            Z = _np_from_json(d["numbers"]).astype(np.int32); R = _np_from_json(d["positions"]).astype(np.float64).reshape(-1, 3)
            nc, lab, dmin = frag_one(Z, R); sizes = np.bincount(lab)
            rows["n_atoms"].append(len(Z)); rows["Q"].append(int(round(float(info.get("charge", 0)))))
            rows["S"].append(int(round(float(info.get("spin", 1))))); rows["n_frag"].append(nc)
            rows["max_frag"].append(int(sizes.max())); rows["dmin_inter"].append(dmin)
            comp.append("|".join(sorted(formula(Z[lab == j]) for j in range(nc))) if nc <= 12 else f"{nc} frags|" + formula(Z))
            dataid.append(str(info.get("data_id", ""))); elem.append(np.bincount(Z, minlength=57)[:57])
            if max_n and len(comp) >= max_n:
                break
    env.close()
    np.savez(out, **{k: np.asarray(v) for k, v in rows.items()}, comp=np.asarray(comp), data_id=np.asarray(dataid),
             elem=np.asarray(elem, np.int16))
    return len(comp), time.time() - t0

if __name__ == "__main__":
    src, outd, ti, nt = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); mx = int(sys.argv[5]) if len(sys.argv) > 5 else None
    os.makedirs(outd, exist_ok=True)
    files = list_aselmdb_files(src)[ti::nt]
    for f in files:
        o = os.path.join(outd, os.path.basename(f).replace(".aselmdb", ".npz"))
        if os.path.exists(o) and not mx:
            continue
        n, dt = map_file(f, o, mx); print(f"{os.path.basename(f)} {n} structures {dt:.1f}s ({1e3 * dt / max(n, 1):.2f} ms/structure)", flush=True)
