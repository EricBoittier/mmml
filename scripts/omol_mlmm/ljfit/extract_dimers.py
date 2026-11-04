"""Non-covalent dimers for fitting LJ of provisional XN* types. From OMol25 4M multi-molecule structures: molecules that
contain an XN* atom (typer v3 + auto library) and whose graph is a known residue (charge known), paired with their
nearest partner molecule (also a known residue), closest contact < 4.5 A, both <= 24 atoms. Keeps up to K per
(XN type, partner) and N per XN type.
    python extract_dimers.py <shard_dir> <out.json> <fraction> <workers>"""
import glob, json, os, sys, collections, numpy as np
from multiprocessing import Pool
sys.path.insert(0, "../../envtyper"); sys.path.insert(0, ".."); import cgenff_envtyper as E
from frag_mols import formula
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
AUTO = os.path.abspath("../omol_auto.rtf")
def residue_keys():
    """graph key -> (resi, net charge, source) for CHARMM primary residues and the OMol25 auto residues."""
    K = {}; mass, res = E.parse_rtfs()
    for r in res:
        if not r["atoms"] or r["file"] not in E.PRIMARY: continue
        Z, adj, st = E.residue_graph(r, mass)
        if any(st) or 0 in Z: continue
        K.setdefault(formula(Z) + "|" + ";".join(sorted(E.signatures(Z, adj)[2])), (r["name"], int(round(sum(r["charges"].values()))), "charmm"))
    for o in json.load(open("../omol_auto.json"))["molecules"]:
        if o["existing"]: continue
        Z = np.array(o["Z"]); adj = E.graph_from_xyz(Z, np.array(o["R"]))
        K.setdefault(formula(Z) + "|" + ";".join(sorted(E.signatures(Z, adj)[2])), (o["resi"], int(o["Q"]), "auto"))
    return K
TY = KEYS = None
def init():
    global TY, KEYS; TY = E.CGenFFTyper(extra=AUTO); KEYS = residue_keys()
def shard(args):
    f, frac = args; rng = np.random.default_rng(abs(hash(os.path.basename(f))) % 2**32); out = []
    with np.load(f, allow_pickle=False) as d: Z, R, off, sub = d["Z"], d["R"], d["mol_offsets"], d["subset_names"][d["subset"]]
    for i in np.nonzero(rng.random(len(off) - 1) < frac)[0]:
        z = Z[off[i]:off[i + 1]].astype(int); r = R[off[i]:off[i + 1]].astype(float)
        if len(z) > 150: continue
        adj = E.graph_from_xyz(z, r); n = len(z)
        rr = [a for a in range(n) for _ in adj[a]]; cc = [b for a in range(n) for b in adj[a]]
        nc, lab = connected_components(csr_matrix((np.ones(len(rr)), (rr, cc)), shape=(n, n)), directed=False)
        if nc < 2: continue
        sig = E.signatures(z, adj); ty = [t or "" for t, _, _ in TY.assign(sig)]
        xn = {lab[k] for k, t in enumerate(ty) if t.startswith("XN")}
        if not xn: continue
        frag = {}
        for c in range(nc):
            idx = np.nonzero(lab == c)[0]; key = formula(z[idx]) + "|" + ";".join(sorted(sig[2][k] for k in idx))   # r2 within fragment == fragment r2 (no inter-fragment bonds)
            frag[c] = (idx, KEYS.get(key))
        for c in xn:
            ia, ka = frag[c]
            if ka is None or len(ia) > 24: continue
            best = None
            for c2, (ib, kb) in frag.items():
                if c2 == c or kb is None or len(ib) > 24: continue
                dmin = np.min(np.linalg.norm(r[ia][:, None] - r[ib][None], axis=-1))
                if dmin < 4.5 and (best is None or dmin < best[0]): best = (dmin, c2)
            if best is None or best[0] < 1.5: continue
            ib, kb = frag[best[1]]
            out.append(dict(xn=sorted({ty[k] for k in ia if ty[k].startswith("XN")}), A=ka[0], QA=ka[1], B=kb[0], QB=kb[1], dmin=float(best[0]),
                            ZA=z[ia].tolist(), RA=np.round(r[ia], 4).tolist(), ZB=z[ib].tolist(), RB=np.round(r[ib], 4).tolist(),
                            tA=[ty[k] for k in ia], tB=[ty[k] for k in ib], subset=str(sub[i])))
    return out
if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(sys.argv[1], "*.npz"))); frac = float(sys.argv[3]); W = int(sys.argv[4]); allD = []
    with Pool(W, initializer=init) as p:
        for k, o in enumerate(p.imap_unordered(shard, [(f, frac) for f in files])):
            allD += o
            if k % 10 == 0: print(k, len(allD), flush=True)
    rng = np.random.default_rng(0); rng.shuffle(allD); per = collections.Counter(); pair = collections.Counter(); keep = []
    for d in allD:
        t = d["xn"][0]; pp = (t, d["B"])
        if per[t] >= 12 or pair[pp] >= 3: continue
        per[t] += 1; pair[pp] += 1; keep.append(d)
    json.dump(dict(n_found=len(allD), found_by_type=dict(collections.Counter(d["xn"][0] for d in allD)), dimers=keep), open(sys.argv[2], "w"))
    print("found", len(allD), dict(collections.Counter(d["xn"][0] for d in allD)), "kept", len(keep), dict(per))
