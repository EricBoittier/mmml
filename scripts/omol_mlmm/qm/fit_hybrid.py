"""Hybrid charges = scaled-ESP fit restrained to environment-transferred CGenFF charges (weights by match level), total
charge + symmetry constrained.  Hyper-parameters (ESP scale lam, restraint weights per level group) chosen by 2-fold CV on
CGenFF residues present in OMol25, with each validation residue left out of the transfer tables (LORO).
    python fit_hybrid.py <esp_dir>  -> hybrid_charges.json, hybrid_cv.json"""
import json, glob, os, sys, itertools, numpy as np
sys.path.insert(0, "../../envtyper"); import cgenff_envtyper as E, envcharge as EC
D = sys.argv[1]; J = {j["name"]: j for j in json.load(open("jobs.json"))}
mass, res = E.parse_rtfs(); prim = [r for r in res if r["file"] in E.PRIMARY]
C = EC.EnvCharger(prim, mass); T = E.EnvTyper(prim, mass); byname = {k[1]: k for k in C.res}
GROUP = {2: 0, 1: 1, 3: 1, 4: 2, 5: 2, 0: 2, 6: 2, -1: 3}         # r2 | r1,H-parent | back-off/r0 | none
def prep(name):
    f = os.path.join(D, name + "_esp.npz")
    if not os.path.exists(f): return None
    e = np.load(f, allow_pickle=False); Z = e["Z"]; R = e["R_bohr"] * 0.529177210903; j = J[name]
    adj = E.graph_from_xyz(Z, R); sig = E.signatures(Z, adj); key = byname.get(name[4:]) if name.startswith("VAL_") else None
    if key: C._upd(*C.res[key][:3], -1); T.remove(key)
    ty = [t or "" for t, _, _ in T.assign(sig)]

    qe, lev, _ = C.predict(sig, ty, j["Q"])
    if key: C._upd(*C.res[key][:3], +1); T.add(key)
    A = 1 / np.linalg.norm(e["pts_bohr"][:, None] - e["R_bohr"][None], axis=-1)
    groups = sorted(set(sig[2])); M = np.zeros((len(Z), len(groups)))
    for i, g in enumerate(sig[2]): M[i, groups.index(g)] = 1
    AM = A @ M; Pn = len(e["V"])
    return dict(AtA=AM.T @ AM / Pn, AtV=AM.T @ e["V"] / Pn, qe=qe, g=np.array([GROUP[int(l)] for l in lev]), M=M, Q=j["Q"], Z=Z, qc=np.array(j["q_cgenff"]) if "q_cgenff" in j else None)
def solve(p, lam, w):
    M = p["M"]; W = np.array([w[g] for g in p["g"]])
    H = p["AtA"] + M.T @ (W[:, None] * M); b = lam * p["AtV"] + M.T @ (W * p["qe"]); c = M.sum(0)
    K = np.block([[H, c[:, None]], [c[None], np.zeros((1, 1))]]); x = np.linalg.solve(K, np.concatenate([b, [p["Q"]]]))
    return M @ x[:-1]
if __name__ == "__main__":
    names = sorted(n for n in J if os.path.exists(os.path.join(D, n + "_esp.npz")))
    P = {n: prep(n) for n in names}; P = {n: p for n, p in P.items() if p is not None}
    val = [n for n in P if n.startswith("VAL_")]; rng = np.random.default_rng(0); rng.shuffle(val); folds = [val[0::2], val[1::2]]
    mae = lambda ns, f: float(np.mean(np.concatenate([np.abs(f(P[n]) - P[n]["qc"]) for n in ns])))
    base = dict(env_transfer_loro=mae(val, lambda p: p["qe"]))
    for lam in (1.0, 0.74): base[f"esp_only_lam{lam}"] = mae(val, lambda p, l=lam: solve(p, l, [1e-6] * 4))
    grid = list(itertools.product((0.6, 0.7, 0.8, 0.9, 1.0), (1e-3, 1e-2, 1e-1, 1, 10), (1e-4, 1e-3, 1e-2, 1e-1, 1), (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)))
    def best(ns):
        sc = [(mae(ns, lambda p, g=g: solve(p, g[0], [g[1] * 10, g[1], g[2], g[3]])), g) for g in grid]; return min(sc)
    cv = []
    for k in (0, 1):
        s, g = best(folds[k]); cv.append(dict(train_mae=s, hp=g, test_mae=mae(folds[1 - k], lambda p: solve(p, g[0], [g[1] * 10, g[1], g[2], g[3]]))))
    s, g = best(val); wv = [g[1] * 10, g[1], g[2], g[3]]
    # per match-quality breakdown: atoms whose environment fell back (group>=2) -- where new molecules need help
    lo = lambda f: float(np.mean(np.concatenate([np.abs(f(P[n]) - P[n]["qc"])[P[n]["g"] >= 2] for n in val if (P[n]["g"] >= 2).any()])))
    out = dict(n_val=len(val), baselines=base, cv=cv, final_hp=dict(lam=g[0], w=wv), final_train_mae=s,
               low_level_atoms=dict(env=lo(lambda p: p["qe"]), hybrid=lo(lambda p: solve(p, g[0], wv)), esp=lo(lambda p: solve(p, 0.74, [1e-6] * 4))))
    json.dump(out, open("hybrid_cv.json", "w"), indent=1); print(json.dumps(out, indent=1))
    Q = {n: dict(q=solve(p, g[0], wv).round(4).tolist(), q_env=p["qe"].round(4).tolist(), groups=p["g"].tolist()) for n, p in P.items() if not n.startswith("VAL_")}
    json.dump(Q, open("hybrid_charges.json", "w")); print(len(Q), "residues charged")
