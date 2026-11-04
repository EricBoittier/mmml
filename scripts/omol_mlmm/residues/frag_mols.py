"""Frequent small molecules (covalent fragments <= 32 atoms) in OMol25 4M: count occurrences per molecular graph, typer level
per atom, charge/spin when the fragment is the whole structure, one example geometry per graph. Graph key = formula + sorted
r2 signatures (cheap; RDKit canonical skeleton SMILES computed afterwards on unique keys only).
    python frag_mols.py <shard_dir> <out.npz/json prefix> <fraction> <workers>"""
import glob, os, sys, time, json, collections, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "envtyper"))
import cgenff_envtyper as E
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
TY = None
EL = {1: "H", 3: "Li", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 11: "Na", 12: "Mg", 14: "Si", 15: "P", 16: "S", 17: "Cl", 19: "K", 20: "Ca", 30: "Zn", 35: "Br", 53: "I"}
def init():
    global TY; TY = E.CGenFFTyper()
def formula(z):
    c = collections.Counter(EL.get(int(x), str(x)) for x in z); keys = (["C", "H"] if "C" in c else []) + sorted(k for k in c if k not in ("C", "H") or "C" not in c)
    return "".join(f"{k}{c[k] if c[k] > 1 else ''}" for k in keys)
def shard(args):
    f, frac = args; rng = np.random.default_rng(abs(hash(os.path.basename(f))) % 2**32)
    with np.load(f, allow_pickle=False) as d:
        Z, R, off, Q, S = d["Z"], d["R"], d["mol_offsets"], d["Q"].reshape(-1), d["S"].reshape(-1); sub = d["subset_names"][d["subset"]]
    M = {}
    for i in np.nonzero(rng.random(len(off) - 1) < frac)[0]:
        z = Z[off[i]:off[i + 1]].astype(int); r = R[off[i]:off[i + 1]].astype(float); adj = E.graph_from_xyz(z, r); n = len(z)
        rows = [a for a in range(n) for _ in adj[a]]; cols = [b for a in range(n) for b in adj[a]]
        nc, lab = connected_components(csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)), directed=False)
        for c in range(nc):
            idx = np.nonzero(lab == c)[0]
            if len(idx) > 32: continue
            if nc > 1:   # must be non-covalently separated (>= 2 A) from the rest, else it is a reactive/bonded contact
                rest = np.nonzero(lab != c)[0]
                if np.min(np.linalg.norm(r[idx][:, None] - r[rest][None], axis=-1)) < 2.0: continue
            zi = z[idx]; m = {a: k for k, a in enumerate(idx)}; ad = [[m[b] for b in adj[a]] for a in idx]
            sig = E.signatures(zi, ad); key = formula(zi) + "|" + ";".join(sorted(sig[2]))
            e = M.get(key)
            if e is None:
                a = TY.assign(sig)
                e = M[key] = dict(n=0, subsets=collections.Counter(), Q=collections.Counter(), S=collections.Counter(), natoms=len(idx),
                                  levels=[x[1] for x in a], types=[x[0] or "" for x in a], ex=dict(Z=zi.tolist(), R=np.round(r[idx], 4).tolist()), ex_single=None)
            e["n"] += 1; e["subsets"][str(sub[i])] += 1
            if nc == 1:
                e["Q"][int(Q[i])] += 1; e["S"][int(S[i])] += 1
                if e["ex_single"] is None: e["ex_single"] = dict(Z=zi.tolist(), R=np.round(r[idx], 4).tolist(), Q=int(Q[i]), S=int(S[i]))
    return M
if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(sys.argv[1], "*.npz"))); frac = float(sys.argv[3]); W = int(sys.argv[4]); t0 = time.time(); ALL = {}
    with Pool(W, initializer=init) as p:
        for k, M in enumerate(p.imap_unordered(shard, [(f, frac) for f in files])):
            for key, e in M.items():
                a = ALL.get(key)
                if a is None: ALL[key] = e; continue
                a["n"] += e["n"]; a["subsets"].update(e["subsets"]); a["Q"].update(e["Q"]); a["S"].update(e["S"])
                if a["ex_single"] is None: a["ex_single"] = e["ex_single"]
            if k % 20 == 0: print(f"shard {k}/{len(files)} {time.time()-t0:.0f}s {len(ALL)} graphs", flush=True)
    L = sorted(ALL.items(), key=lambda kv: -kv[1]["n"])
    out = [dict(key=k, **{**e, "subsets": dict(e["subsets"].most_common(4)), "Q": dict(e["Q"]), "S": dict(e["S"])}) for k, e in L if e["n"] >= 3]
    json.dump(dict(fraction=frac, seconds=time.time() - t0, n_graphs=len(ALL), n_occ=sum(e["n"] for e in ALL.values()), mols=out), open(sys.argv[2] + ".json", "w"))
    print(f"{len(ALL)} graphs, {len(out)} with >=3 occurrences, {time.time()-t0:.0f}s")
