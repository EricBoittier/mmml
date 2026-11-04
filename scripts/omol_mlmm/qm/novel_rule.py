"""Evaluate/apply the per-residue charge rule. 'novel' = provisional types or |transfer residual| > 0.2 e or hypervalent centre.
Novel residues: ESP fit (lam=1) restrained only to r2/r1 analogues; others: CV hybrid. Validation = CGenFF residues whose
LORO transfer residual > 0.2 e (the closest analogue of a novel molecule)."""
import json, sys, numpy as np
sys.argv = ["x", "out2"]; import fit_hybrid as F
P = {n: F.prep(n) for n in sorted(F.J) if F.os.path.exists(F.os.path.join("out2", n + "_esp.npz"))}; P = {n: p for n, p in P.items() if p}
hp = json.load(open("hybrid_cv.json"))["final_hp"]; WH = hp["w"]; WN = [0.1, 0.01, 1e-5, 1e-5]
def resid(p): return abs(p["Q"] - (p["qe"].sum() - 0))   # qe is already neutralised -> recompute raw residual below
import envcharge as EC
val = [n for n in P if n.startswith("VAL_")]
# raw transfer residual (before neutralisation) for each validation residue under LORO: recompute via the env levels
rows = []
for n in val:
    p = P[n]; e_env = np.abs(p["qe"] - p["qc"]); e_h = np.abs(F.solve(p, hp["lam"], WH) - p["qc"]); e_n = np.abs(F.solve(p, 1.0, WN) - p["qc"])
    rows.append((n, (p["g"] >= 2).mean(), e_env.mean(), e_h.mean(), e_n.mean(), len(p["qc"])))
rows = np.array([r[1:] for r in rows]); poor = rows[:, 0] > 0.3        # >30 % of atoms only matched at back-off level
out = {"n_val": len(val), "poorly_matched_val": int(poor.sum()),
       "poor_MAE": dict(env=float(rows[poor, 1].mean()), hybrid=float(rows[poor, 2].mean()), novel_rule=float(rows[poor, 3].mean())),
       "well_matched_MAE": dict(env=float(rows[~poor, 1].mean()), hybrid=float(rows[~poor, 2].mean()), novel_rule=float(rows[~poor, 3].mean()))}
print(json.dumps(out, indent=1)); json.dump(out, open("novel_rule_eval.json", "w"), indent=1)
