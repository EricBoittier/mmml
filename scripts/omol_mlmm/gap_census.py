"""Where the CGenFF type set does not describe OMol25 chemistry: per r1 environment (element, degree, nH, ring + neighbour
invariants) count atoms that are (a) untyped, (b) typed only at r0 (element-level), (c) typed from an ambiguous table entry
(purity < 0.8). Keep example (shard,row,atom) and subset/charge context. 10 % sample of the filtered 4M shards.
    python gap_census.py <shard_dir> <out.json> [fraction] [workers]"""
import glob, json, os, sys, time, collections, numpy as np
from multiprocessing import Pool
import cgenff_envtyper as E
TY = None
def init():
    global TY; TY = E.CGenFFTyper(extra=os.environ.get("EXTRA_RTF"))
def shard(args):
    f, frac = args; rng = np.random.default_rng(abs(hash(os.path.basename(f))) % 2**32)
    with np.load(f, allow_pickle=False) as d:
        Z, R, off, Q, S = d["Z"], d["R"], d["mol_offsets"], d["Q"].reshape(-1), d["S"].reshape(-1); sub = d["subset_names"][d["subset"]]
    cnt = {k: collections.Counter() for k in ("untyped", "r0", "ambig")}; ex = collections.defaultdict(list); subc = collections.defaultdict(collections.Counter)
    amb_types = collections.defaultdict(collections.Counter); LV = collections.Counter(); natoms = 0; nstruct = 0
    for i in np.nonzero(rng.random(len(off) - 1) < frac)[0]:
        z = Z[off[i]:off[i + 1]].astype(int); r = R[off[i]:off[i + 1]]; adj = E.graph_from_xyz(z, r); sig = E.signatures(z, adj)
        a = TY.assign(sig); LV.update(x[1] for x in a); natoms += len(z); nstruct += 1
        for k, (t, lev, pur) in enumerate(a):
            cls = "untyped" if t is None else "r0" if lev in (0, 6, 10, 16, 20, 26) else "ambig" if pur < 0.8 else None
            if cls is None: continue
            key = sig[1][k]; cnt[cls][key] += 1; subc[key][str(sub[i])] += 1
            if cls == "ambig": amb_types[key][t] += 1
            if len(ex[key]) < 3: ex[key].append([os.path.basename(f), int(i), int(k), int(Q[i]), int(S[i])])
    return nstruct, natoms, cnt, ex, subc, amb_types, LV
if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(sys.argv[1], "*.npz"))); frac = float(sys.argv[3]); W = int(sys.argv[4])
    t0 = time.time(); N = A = 0; C = {k: collections.Counter() for k in ("untyped", "r0", "ambig")}; EX = {}; SUB = collections.defaultdict(collections.Counter); AMB = collections.defaultdict(collections.Counter); LVT = collections.Counter()
    with Pool(W, initializer=init) as p:
        for n, a, cnt, ex, subc, amb, lv in p.imap_unordered(shard, [(f, frac) for f in files]):
            N += n; A += a; LVT.update(lv)
            for k in C: C[k].update(cnt[k])
            for key, v in ex.items(): EX.setdefault(key, v)
            for key, v in subc.items(): SUB[key].update(v)
            for key, v in amb.items(): AMB[key].update(v)
    out = {"levels": {str(k): v for k, v in sorted(LVT.items())}, "structures": N, "atoms": A, "fraction": frac, "seconds": time.time() - t0}
    for k in C:
        out[k] = [dict(env=key, atoms=c, subsets=dict(SUB[key].most_common(3)), examples=EX.get(key, []), table_types=dict(AMB[key].most_common(4)) if k == "ambig" else None)
                  for key, c in C[k].most_common(300)]
        out[f"{k}_total"] = sum(C[k].values()); out[f"{k}_n_env"] = len(C[k])
    json.dump(out, open(sys.argv[2], "w"), indent=1)
    print(f"{N} structures {A} atoms in {out['seconds']:.0f}s; untyped {out['untyped_total']} ({out['untyped_n_env']} envs), r0 {out['r0_total']} ({out['r0_n_env']}), ambiguous {out['ambig_total']} ({out['ambig_n_env']})")
