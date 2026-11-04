"""ML/MM fine-tune cache v2 from the streamed full-OMol25 selection (select_full/*_sel.npz), 4 Nov 2026.
Energy: E_DFT - sum E_iso(Z, q=0) - (sum_Z b_Z + b_Q Q)  (same convention/linear ref as the 300k cache; train with
--energy-add-charge-ref 5.661417917773716).  Types: CGenFF envtyper v3 + OMol25 auto-residue library; LJ from CHARMM
parameter files; provisional XN* types and untyped atoms -> element-level LJ (until the dimer fit).
Mix (priority target > non-covalent > single): non-covalent clusters 50 %, target chemistry 25 % (water-only clusters
<= half of it), single molecules <= 12 atoms 25 %.
    python build_full_cache.py --n 1000000 --out <cache> [--workers 16]"""
import argparse, glob, json, os, sys, time, numpy as np
from multiprocessing import Pool
U = "/mmhome/boittier/home/metatomic-runs/training/omol_universal"
sys.path.insert(0, f"{U}/mlmm_ft/envtyper"); sys.path.insert(0, f"{U}/mlmm_ft/scripts"); sys.path.insert(0, f"{U}/scripts")
import cgenff_envtyper as E, ljtab, build_mlmm_cache as B, omol_io
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
ALLOWED = {1, 3, 6, 7, 8, 9, 11, 12, 15, 16, 17, 19, 20, 30, 35, 53}
EXTRA = f"{U}/mlmm_ft/residues/omol_auto.rtf"
TY = None
def init():
    global TY; TY = E.CGenFFTyper(extra=EXTRA)
def scan(f):
    """eligible rows + class per file (cheap: no typing)."""
    with np.load(f, allow_pickle=False) as d:
        off = d["offsets"]; Z = d["Z"]; Q = d["Q"].reshape(-1); S = d["S"].reshape(-1); rs = d["reason"].reshape(-1)
    n = np.diff(off); ok = (n >= 2) & (n <= 120) & (np.abs(Q) <= 2) & (S <= 3)
    bad = np.zeros(len(n), bool); zmask = ~np.isin(Z, list(ALLOWED)); idx = np.repeat(np.arange(len(n)), n); bad[np.unique(idx[zmask])] = True
    ok &= ~bad
    hmo = np.isin(Z, (1, 8)); allw = np.ones(len(n), bool); allw[np.unique(idx[~hmo])] = False
    cls = np.where(rs & 1, np.where(allw, 0, 1), np.where(rs & 8, 2, np.where(rs & 6, 3, -1)))   # 0 water cluster, 1 other target, 2 noncov, 3 single/diatomic
    rows = np.nonzero(ok & (cls >= 0))[0]
    return f, rows, cls[rows]
def build(job):
    f, rows = job
    with np.load(f, allow_pickle=False) as d:
        off = d["offsets"]; Za, Ra, Fa = d["Z"], d["R"], d["F"]; Ea, Qa, Sa = d["E"].reshape(-1), d["Q"].reshape(-1), d["S"].reshape(-1)
    out = dict(R=[], Z=[], F=[], E=[], N=[], Q=[], S=[], mol_id=[], tname=[], lev=[])
    for r in rows:
        z = Za[off[r]:off[r + 1]].astype(int); R = Ra[off[r]:off[r + 1]].astype(np.float64); adj = E.graph_from_xyz(z, R); n = len(z)
        rr = [a for a in range(n) for _ in adj[a]]; cc = [b for a in range(n) for b in adj[a]]
        _, lab = connected_components(csr_matrix((np.ones(len(rr)), (rr, cc)), shape=(n, n)), directed=False)
        a = TY.assign(E.signatures(z, adj))
        out["R"].append(R); out["Z"].append(z.astype(np.int32)); out["F"].append(Fa[off[r]:off[r + 1]].astype(np.float64))
        out["E"].append(float(Ea[r])); out["N"].append(n); out["Q"].append(float(Qa[r])); out["S"].append(float(Sa[r]))
        out["mol_id"].append(lab.astype(np.int32)); out["tname"].append([t or "" for t, _, _ in a]); out["lev"].append(np.array([l for _, l, _ in a], np.int8))
    voc = sorted({t for tl in out["tname"] for t in tl}); vi = {t: i for i, t in enumerate(voc)}   # compact: int16 indices + local vocab
    out["tname"] = [np.array([vi[t] for t in tl], np.int16) for tl in out["tname"]]; out["voc"] = voc
    for k in ("R", "F"): out[k] = [x.astype(np.float32) for x in out[k]]
    return out
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=1_000_000); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--linear-ref-from", default=f"{U}/caches/omol_train4M_pilot200k/metadata.json")
    a = ap.parse_args(); t0 = time.time()
    files = sorted(glob.glob(f"{U}/mlmm_ft/select_full/*_sel.npz"))
    with Pool(a.workers) as p: S = p.map(scan, files)
    cls = np.concatenate([c for _, _, c in S]); fid = np.concatenate([np.full(len(r), i) for i, (_, r, _) in enumerate(S)]); row = np.concatenate([r for _, r, _ in S])
    avail = {k: int((cls == k).sum()) for k in range(4)}; print(f"{len(files)} files; eligible by class (water-cl, target, noncov, single): {avail}  {time.time()-t0:.0f}s", flush=True)
    quota = {2: a.n // 2, 1: a.n // 4 - min(a.n // 8, avail[0]), 0: min(a.n // 8, avail[0]), 3: a.n // 4}
    short = sum(max(0, quota[k] - avail[k]) for k in quota); quota = {k: min(quota[k], avail[k]) for k in quota}
    for k in (2, 3, 1, 0):          # hand any shortfall to classes with spare structures
        extra = min(short, avail[k] - quota[k]); quota[k] += extra; short -= extra
    rng = np.random.default_rng(a.seed); pick = np.concatenate([rng.choice(np.nonzero(cls == k)[0], quota[k], replace=False) for k in range(4)])
    print("quota", quota, flush=True)
    jobs = []
    for i in range(len(files)):
        rr = np.sort(row[pick[fid[pick] == i]])
        for c in np.array_split(rr, max(1, len(rr) // 5000)): jobs.append((files[i], c))
    parts = dict(R=[], Z=[], F=[], E=[], N=[], Q=[], S=[], mol_id=[], tname=[], lev=[]); GV = {}
    with Pool(a.workers, initializer=init) as p:
        for k, res in enumerate(p.imap_unordered(build, jobs)):
            voc = res.pop("voc"); gmap = np.array([GV.setdefault(t, len(GV)) for t in voc], np.int32)
            res["tname"] = [gmap[x] for x in res["tname"]]
            for key in parts: parts[key].extend(res[key])
            if k % 20 == 0: print(f"chunk {k+1}/{len(jobs)}: {len(parts['N'])} structures {time.time()-t0:.0f}s", flush=True)
    Z = np.concatenate(parts["Z"]); gvoc = np.array(sorted(GV, key=GV.get)); nm = np.concatenate(parts["tname"]); lev = np.concatenate(parts["lev"])
    lj = ljtab.lj_table()
    okv = np.array([t in lj for t in gvoc]); ok = okv[np.concatenate(parts["tname"])]; used = np.zeros(len(gvoc), bool); used[np.unique(np.concatenate(parts['tname']))] = True; final = sorted(gvoc[okv & used].tolist()); fall = sorted(set(Z[~ok].tolist()))
    tlist = final + [f"X_{z}" for z in fall]; pos = {t: i for i, t in enumerate(tlist)}
    tidx = np.empty(len(nm), np.int32); vpos = np.array([pos.get(t, -1) for t in gvoc]); tidx[ok] = vpos[np.concatenate(parts["tname"])][ok]
    zpos = np.full(119, -1); zpos[fall] = [pos[f"X_{z}"] for z in fall]; tidx[~ok] = zpos[Z[~ok]]
    def el_lj(z): i = 0 if z == 1 else B.ZMAP.get(int(z), 17); return B.PT[i][2], B.PT[i][1]
    eps = np.array([lj[t][0] if t in lj else el_lj(int(t[2:]))[0] for t in tlist]); rmin = np.array([lj[t][1] if t in lj else el_lj(int(t[2:]))[1] for t in tlist])
    iso = omol_io.neutral_ref_table(omol_io.load_iso_atom_refs(f"{U}/raw/hf_refs/references/iso_atom_elem_refs.yaml"))
    lr = json.load(open(a.linear_ref_from))["linear_ref"]; tab = np.zeros(119); tab[np.asarray(lr["Z"])] = lr["b_eV"]; bq = lr["b_Q_eV_per_e"]
    offs = np.concatenate([[0], np.cumsum(parts["N"])]).astype(np.int64); Qv = np.array(parts["Q"])
    Ezs = np.add.reduceat(iso[Z] + tab[Z], offs[:-1]); Ev = np.array(parts["E"]) - Ezs - bq * Qv
    data = {"R": np.concatenate(parts["R"]).astype(np.float64), "Z": Z.astype(np.int32), "F": np.concatenate(parts["F"]).astype(np.float64), "E": Ev.reshape(-1, 1),
            "N": np.array(parts["N"], np.int32).reshape(-1, 1), "Q": Qv.reshape(-1, 1), "S": np.array(parts["S"]).reshape(-1, 1),
            "D": np.zeros((len(Qv), 3)), "mol_id": np.concatenate(parts["mol_id"]), "cgenff_type_idx": tidx,
            "cgenff_master_sigmas": 2 * rmin / 2 ** (1 / 6), "cgenff_master_epsilons": eps, "mol_offsets": offs}
    n, na = len(Qv), int(offs[-1])
    data["metadata_n_structures"] = np.asarray(n, np.int64); data["metadata_n_atoms_total"] = np.asarray(na, np.int64); data["metadata_max_atoms"] = np.asarray(int(max(parts["N"])), np.int32)
    import orbax.checkpoint as ocp
    from flax.training import orbax_utils
    out = os.path.abspath(a.out); ocp.PyTreeCheckpointer().save(out, data, save_args=orbax_utils.save_args_from_target(data), force=True)
    cnt = np.bincount(tidx, minlength=len(tlist))
    meta = dict(n_structures=n, n_atoms_total=na, max_atoms=int(max(parts["N"])), E_std=float(Ev.std()), E_mean=float(Ev.mean()),
                selection=dict(source="select_full (streamed full OMol25 train)", files=len(files), quota={str(k): int(v) for k, v in quota.items()}, classes="0 water clusters, 1 other target chemistry, 2 non-covalent clusters, 3 single molecules <=12 atoms", seed=a.seed),
                linear_ref=dict(lr, applied_from=a.linear_ref_from), energy="E_DFT - sum E_iso(Z,q=0) - (sum_Z b_Z + b_Q Q) [eV]",
                types_source="CGenFF envtyper v3 + OMol25 auto-residue library (mlmm_ft/residues/omol_auto.rtf); LORO 95.1 %",
                n_types=len(tlist), type_names=tlist, type_atoms=cnt.tolist(), fallback_atoms=int((~ok).sum()), fallback_frac=float((~ok).mean()),
                level_counts={str(k): int(v) for k, v in zip(*np.unique(lev, return_counts=True))}, dipole_key="", dipole_is_zero=True)
    json.dump(meta, open(os.path.join(out, "metadata.json"), "w"), indent=1)
    print(f"{n} structures, {na} atoms, {len(tlist)} types ({len(fall)} element fallbacks, {100*(~ok).mean():.2f} % atoms), E std {Ev.std():.3f} eV, {time.time()-t0:.0f}s -> {out}")
if __name__ == "__main__":
    main()
