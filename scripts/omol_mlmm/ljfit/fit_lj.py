"""Fit LJ (eps, Rmin/2) of provisional XN* types to wB97M-V/def2-TZVPD dimer interaction energies.
E_MM = Coulomb(residue charges, 332.0637/r) + LJ (CHARMM: eps_ij = sqrt(eps_i eps_j), Rmin_ij = Rmin/2_i + Rmin/2_j;
E = eps [(Rmin/r)^12 - 2 (Rmin/r)^6]); known types from CHARMM par files, XN* free.  Weighted least squares
(w = 1 / (1 + max(E_DFT, 0)/10)), joint over all types; bounds eps [0.002, 3], Rmin/2 [0.6, 3.2].
    python fit_lj.py out  -> lj_fit.json, ../omol_auto_lj.prm"""
import json, glob, sys, os, numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, "../../envtyper"); sys.path.insert(0, "../../scripts"); import cgenff_envtyper as E, ljtab, build_mlmm_cache as B
LJ = ljtab.lj_table(); D = json.load(open("dimers.json"))["dimers"]
mass, res = E.parse_rtfs(); R = {r["name"]: r for r in res if r["file"] in E.PRIMARY}
A = {o["resi"]: o for o in json.load(open("../omol_auto.json"))["molecules"] if not o["existing"]}
def sig_charges(name):
    """r2 signature -> mean charge for residue `name` (CHARMM or auto)."""
    if name in A:
        o = A[name]; Z = np.array(o["Z"]); s = E.signatures(Z, E.graph_from_xyz(Z, np.array(o["R"])))[2]; q = o["charges"]
    else:
        r = R[name]; Z, adj, st = E.residue_graph(r, mass); s = E.signatures(Z, adj, st)[2]; q = [r["charges"][a] for a in r["atoms"]]
    m = {}
    for k, x in zip(s, q): m.setdefault(k, []).append(x)
    return {k: float(np.mean(v)) for k, v in m.items()}
def charges(name, Z, Rx):
    m = sig_charges(name); s = E.signatures(np.array(Z), E.graph_from_xyz(np.array(Z), np.array(Rx)))[2]; return np.array([m[k] for k in s])
def el_lj(z): i = 0 if z == 1 else B.ZMAP.get(int(z), 17); return B.PT[i][2], B.PT[i][1]
VALID = {"XNBR1", "XNBR2", "XNC1", "XNC2", "XNC3", "XNCL1", "XNCL2", "XNF1", "XNF2", "XNH1", "XNI1", "XNI2", "XNO2", "XNP1", "XNP2", "XNS1", "XNS2"}
# not MM-valid (gas-phase dianions / reactive intermediates): XNC4 CS, XNC5 C2(2-), XNN1 NH(2-), XNO1 O(2-), XNP3-8, XNS3 S(2-)
pts = []; xn = set()
for f in sorted(glob.glob(os.path.join(sys.argv[1], "d*.json"))):
    r = json.load(open(f))
    if "scan" not in r: continue
    d = D[r["index"]]
    try: qa, qb = charges(d["A"], d["ZA"], d["RA"]), charges(d["B"], d["ZB"], d["RB"])
    except KeyError: continue
    if any(z in (3, 11, 12, 19, 20, 30) for z in d["ZA"] + d["ZB"]): continue      # metal partners: coordination, not LJ
    for kk in ("A", "B"):                                                                 # typer v3.1 polar-H rule
        Zk, Rk = np.array(d["Z" + kk]), np.array(d["R" + kk]); tk = list(d["t" + kk])
        for i, z in enumerate(Zk):
            if z == 1 and tk[i][:3] in ("HGA", "HGR", "HCA"):
                dd = np.linalg.norm(Rk - Rk[i], axis=1); dd[i] = 9; j = int(dd.argmin())
                if Zk[j] in E.POLAR_H and dd[j] < 1.7: tk[i] = E.POLAR_H[int(Zk[j])]
        d["t" + kk] = tk
    allxn = {t for t in d["tA"] + d["tB"] if t.startswith("XN")}
    if not allxn <= VALID: continue
    if abs(d["QA"]) > 0 and abs(d["QB"]) > 0: continue          # ion-ion pairs: induction/CT dominated, not an LJ target
    for s in r["scan"]:
        RB = np.array(s["RB"]); rij = np.linalg.norm(np.array(d["RA"])[:, None] - RB[None], axis=-1)
        pts.append(dict(r=rij, qq=np.outer(qa, qb), tA=d["tA"], tB=d["tB"], ZA=d["ZA"], ZB=d["ZB"], E=s["E_int_kcal"], xn=d["xn"], scale=s["scale"]))
        xn |= {t for t in d["tA"] + d["tB"] if t.startswith("XN")}
xn = sorted(xn); ix = {t: i for i, t in enumerate(xn)}
zof = {}
for p in pts:
    for t, z in list(zip(p["tA"], p["ZA"])) + list(zip(p["tB"], p["ZB"])): zof.setdefault(t, z)
def params(t, x):
    if t in ix: return x[2 * ix[t]], x[2 * ix[t] + 1]
    if t in LJ: return LJ[t]
    return el_lj(zof.get(t, 6))
def emm(p, x):
    eA = np.array([params(t, x) for t in p["tA"]]); eB = np.array([params(t, x) for t in p["tB"]])
    eps = np.sqrt(np.outer(eA[:, 0], eB[:, 0])); rm = eA[:, 1][:, None] + eB[:, 1][None]; sr6 = (rm / p["r"]) ** 6
    return float((332.0637 * p["qq"] / p["r"]).sum() + (eps * (sr6 * sr6 - 2 * sr6)).sum())
x0 = np.array([v for t in xn for v in el_lj(zof[t])]); Ev = np.array([p["E"] for p in pts]); w = 1 / (1 + np.maximum(Ev, 0) / 10)
res_f = lambda x: w * (np.array([emm(p, x) for p in pts]) - Ev)
lb = np.array([v for t in xn for v in ((0.005, 0.2) if zof[t] == 1 else (0.005, 0.9))]); ub = np.array([v for t in xn for v in ((1.0, 1.5) if zof[t] == 1 else (1.0, 2.8))])
sol = least_squares(res_f, np.clip(x0, lb, ub), bounds=(lb, ub), loss="soft_l1", f_scale=2.0)
E0 = np.array([emm(p, x0) for p in pts]); E1 = np.array([emm(p, sol.x) for p in pts])
per = {}
for t in xn:
    m = np.array([t in p["xn"] for p in pts]); near = m & (Ev < 10)
    per[t] = dict(element=int(zof[t]), eps=float(sol.x[2 * ix[t]]), rmin_half=float(sol.x[2 * ix[t] + 1]), eps0=float(x0[2 * ix[t]]), rmin_half0=float(x0[2 * ix[t] + 1]),
                  n_points=int(m.sum()), rmse_before=float(np.sqrt(np.mean((E0[near] - Ev[near]) ** 2))) if near.any() else None,
                  rmse_after=float(np.sqrt(np.mean((E1[near] - Ev[near]) ** 2))) if near.any() else None)
near = Ev < 10
out = dict(n_points=len(pts), n_dimers=len(pts) // 4, types=per, rmse_all_before=float(np.sqrt(np.mean((E0[near] - Ev[near]) ** 2))), rmse_all_after=float(np.sqrt(np.mean((E1[near] - Ev[near]) ** 2))),
           note="RMSE over scan points with E_DFT < 10 kcal/mol; electrostatics from residue charges; LJ of CGenFF types fixed")
json.dump(out, open("lj_fit.json", "w"), indent=1)
L = ["* LJ for OMol25 provisional types (fitted to wB97M-V/def2-TZVPD dimer scans; mlmm_ft/residues/ljfit/fit_lj.py)", "*", "", "NONBONDED nbxmod  5 atom cdiel fshift vatom vdistance vfswitch -", "cutnb 14.0 ctofnb 12.0 ctonnb 10.0 eps 1.0 e14fac 1.0 wmin 1.5", ""]
for t, v in per.items(): L.append(f"{t:8s} 0.0 {-v['eps']:9.4f} {v['rmin_half']:8.4f}   ! n={v['n_points']} rmse {v['rmse_before']:.2f}->{v['rmse_after']:.2f} kcal/mol" if v["rmse_before"] is not None else f"{t:8s} 0.0 {-v['eps']:9.4f} {v['rmin_half']:8.4f}")
open("../omol_auto_lj.prm", "w").write("\n".join(L) + "\nEND\n")
print(f"{len(pts)} points, {len(xn)} types; RMSE (E<10) {out['rmse_all_before']:.2f} -> {out['rmse_all_after']:.2f} kcal/mol")
for t, v in per.items(): print(t, v)

# ---- assignment to EXISTING CHARMM/CGenFF types (no new types): per XN type try every existing type of the same element
# (LJ from the CHARMM par files), others XN at their fitted values; pick the lowest weighted loss on that type's points.
cand = {}
for t in LJ:
    z = mass.get(t)
    if z: cand.setdefault(z, []).append(t)
xa = sol.x.copy(); assign = {}
def loss(x, m): r = w[m] * (np.array([emm(p, x) for p, k in zip(pts, m) if k]) - Ev[m]); return float(np.sum(np.where(np.abs(r) < 2, r * r / 2, 2 * np.abs(r) - 2)))
for t in xn:
    m = np.array([t in p["tA"] + p["tB"] for p in pts]); best = None
    for c in cand.get(per[t]["element"], []):
        x = xa.copy(); x[2 * ix[t]], x[2 * ix[t] + 1] = LJ[c]; l = loss(x, m)
        if best is None or l < best[0]: best = (l, c)
    assign[t] = best[1] if best else None
xs = sol.x.copy()
for t, c in assign.items():
    if c: xs[2 * ix[t]], xs[2 * ix[t] + 1] = LJ[c]
E2 = np.array([emm(p, xs) for p in pts])
for t in xn:
    m = np.array([t in p["xn"] for p in pts]) & (Ev < 10)
    per[t].update(assigned=assign[t], assigned_eps=LJ[assign[t]][0] if assign[t] else None, assigned_rmin_half=LJ[assign[t]][1] if assign[t] else None,
                  rmse_assigned=float(np.sqrt(np.mean((E2[m] - Ev[m]) ** 2))) if m.any() else None, n_candidates=len(cand.get(per[t]["element"], [])))
out.update(types=per, rmse_all_assigned=float(np.sqrt(np.mean((E2[near] - Ev[near]) ** 2))), assignment={t: assign[t] for t in xn})
json.dump(out, open("lj_fit.json", "w"), indent=1)
print(f"assigned existing types: RMSE (E<10) {out['rmse_all_assigned']:.2f} kcal/mol (element default {out['rmse_all_before']:.2f}, free fit {out['rmse_all_after']:.2f})")
for t in xn: print(f"{t:7s} -> {assign[t]:8s} rmse {per[t]['rmse_before']} -> free {per[t]['rmse_after']} / assigned {per[t]['rmse_assigned']}")
