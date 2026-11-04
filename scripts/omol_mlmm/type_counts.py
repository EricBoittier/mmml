"""CGenFF type coverage of an OMol25 npz-shard set (or a random fraction of it): per type atom counts, structures
containing the type, per OMol25 subset; typing levels; untyped elements. Parallel over shards; arrays read once.
    python type_counts.py <shard_dir> <out.json> [fraction=1.0] [workers=16]"""
import glob, json, os, sys, time, collections, numpy as np
from multiprocessing import Pool
import cgenff_envtyper as E
TY = None
def init():
    global TY; TY = E.CGenFFTyper(extra=os.environ.get("EXTRA_RTF"))
def shard(args):
    f, frac = args; rng = np.random.default_rng(abs(hash(os.path.basename(f))) % 2**32)
    with np.load(f, allow_pickle=False) as d:
        Z, R, off = d["Z"], d["R"], d["mol_offsets"]; sub = d["subset_names"][d["subset"]]
    n = len(off) - 1; pick = np.nonzero(rng.random(n) < frac)[0]
    tc = collections.Counter(); sc = collections.Counter(); lev = collections.Counter(); un = collections.Counter(); bysub = collections.defaultdict(collections.Counter)
    for i in pick:
        z = Z[off[i]:off[i + 1]].astype(int); r = R[off[i]:off[i + 1]]
        a = TY.assign(E.signatures(z, E.graph_from_xyz(z, r))); ts = set()
        for (t, l, p), zz in zip(a, z):
            lev[l if l < 10 else 10] += 1
            if t is None: un[int(zz)] += 1; t = f"untyped_Z{int(zz)}"
            tc[t] += 1; ts.add(t)
        for t in ts: sc[t] += 1; bysub[str(sub[i])][t] += 1
    return len(pick), tc, sc, lev, un, bysub
if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(sys.argv[1], "*.npz"))); frac = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0; W = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    t0 = time.time(); N = 0; TC = collections.Counter(); SC = collections.Counter(); LV = collections.Counter(); UN = collections.Counter(); BS = collections.defaultdict(collections.Counter)
    with Pool(W, initializer=init) as p:
        for k, (n, tc, sc, lev, un, bs) in enumerate(p.imap_unordered(shard, [(f, frac) for f in files])):
            N += n; TC.update(tc); SC.update(sc); LV.update(lev); UN.update(un)
            for s, c in bs.items(): BS[s].update(c)
            if k % 10 == 0: print(f"{k + 1}/{len(files)} {N} structures {time.time() - t0:.0f}s", flush=True)
    ty = E.CGenFFTyper(); cg_types = sorted(t for t, z in ty.mass.items() if t in {x for r in E.parse_rtfs()[1] if r["file"] == "top_all36_cgenff.rtf" for x in r["types"].values()})
    json.dump(dict(structures=N, fraction=frac, seconds=time.time() - t0, atom_counts=dict(TC), structure_counts=dict(SC), levels={str(k): v for k, v in LV.items()},
                   untyped_by_Z={str(k): v for k, v in UN.items()}, by_subset={s: dict(c) for s, c in BS.items()}, cgenff_rtf_types=cg_types), open(sys.argv[2], "w"))
    print(f"done {N} structures in {time.time() - t0:.0f}s")
