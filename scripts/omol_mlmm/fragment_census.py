"""Fragment census of the filtered OMol25 npz shards: connected components by covalent radii (ASE natural_cutoffs x1.2).
Per structure: n_frag, fragment sizes, per-fragment formula; per subset: counts of multi-fragment structures,
size / composition stats, nearest inter-fragment heavy-atom distance. Output: mlmm_ft/fragment_census.{json,npz}.
    python fragment_census.py <npz_dir> <out_prefix> [n_workers]
"""
import glob, json, os, sys
from collections import Counter
from multiprocessing import Pool
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from ase.data import covalent_radii, chemical_symbols

R = covalent_radii.copy()

def frag_one(Z, X):
    n = len(Z); t = cKDTree(X); pairs = t.query_pairs(2.6, output_type="ndarray")
    if len(pairs):
        d = np.linalg.norm(X[pairs[:, 0]] - X[pairs[:, 1]], axis=1)
        ok = d < 1.2 * (R[Z[pairs[:, 0]]] + R[Z[pairs[:, 1]]]); p = pairs[ok]
    else:
        p = np.zeros((0, 2), int)
    g = coo_matrix((np.ones(len(p)), (p[:, 0], p[:, 1])), shape=(n, n))
    nc, lab = connected_components(g, directed=False)
    dmin = np.nan
    if nc > 1:
        heavy = Z > 1
        best = np.inf
        for a in range(nc):
            ia = np.where((lab == a) & heavy)[0] if (heavy & (lab == a)).any() else np.where(lab == a)[0]
            ib = np.where((lab != a) & heavy)[0] if (heavy & (lab != a)).any() else np.where(lab != a)[0]
            if len(ia) and len(ib):
                best = min(best, cKDTree(X[ib]).query(X[ia])[0].min())
        dmin = best
    return nc, lab, dmin

def formula(z):
    c = Counter(chemical_symbols[i] for i in z); return "".join(f"{k}{c[k] if c[k] > 1 else ''}" for k in sorted(c, key=lambda s: (s != "C", s != "H", s)))

def shard(f):
    d = np.load(f, allow_pickle=False); off = d["mol_offsets"]; names = d["subset_names"]; sub = names[d["subset"]]
    out = []
    for i in range(len(off) - 1):
        Z = d["Z"][off[i]:off[i + 1]]; X = d["R"][off[i]:off[i + 1]]
        nc, lab, dmin = frag_one(Z, X)
        sizes = np.bincount(lab); fr = [formula(Z[lab == k]) for k in range(nc)] if nc > 1 else []
        out.append((os.path.basename(f), i, str(sub[i]), int(len(Z)), int(nc), int(sizes.max()), float(dmin), int(d["Q"][i]) if d["Q"].ndim == 1 else int(d["Q"][i].ravel()[0]), "|".join(sorted(fr)) if nc <= 12 else f"{nc} frags"))
    return out

if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(sys.argv[1], "*.npz"))); outp = sys.argv[2]; nw = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    with Pool(nw) as p:
        rows = [r for part in p.imap(shard, files) for r in part]
    sub = np.array([r[2] for r in rows]); nfr = np.array([r[4] for r in rows]); dmin = np.array([r[6] for r in rows])
    rep = {}
    for s in sorted(set(sub)):
        m = sub == s; mm = m & (nfr > 1)
        comps = Counter(r[8] for r, k in zip(rows, mm) if k)
        rep[s] = {"n": int(m.sum()), "multi_frag": int(mm.sum()), "frac_multi": float(mm.sum() / m.sum()),
                  "n_frag_hist": {int(k): int(v) for k, v in sorted(Counter(nfr[m]).items())[:20]},
                  "dmin_inter_pctl": np.nanpercentile(dmin[mm], [5, 50, 95]).tolist() if mm.any() else None,
                  "top_compositions": comps.most_common(25)}
    rep["_all"] = {"n": len(rows), "multi_frag": int((nfr > 1).sum())}
    json.dump(rep, open(outp + ".json", "w"), indent=1)
    np.savez(outp + ".npz", shard=np.array([r[0] for r in rows]), row=np.array([r[1] for r in rows]), subset=sub,
             n_atoms=np.array([r[3] for r in rows]), n_frag=nfr, max_frag=np.array([r[5] for r in rows]), dmin_inter=dmin,
             Q=np.array([r[7] for r in rows]), comp=np.array([r[8] for r in rows]))
    print(json.dumps({k: {kk: v[kk] for kk in ("n", "multi_frag")} for k, v in rep.items()}, indent=0))
