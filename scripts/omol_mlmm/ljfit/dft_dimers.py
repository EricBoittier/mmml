"""wB97M-V/def2-TZVPD (density fitting, gpu4pyscf) interaction energies for the LJ-fit dimers: monomers once, dimer at
COM-separation scale s in (0.95, 1.0, 1.15, 1.35) (B translated along the A->B COM vector). No counterpoise (OMol25 style).
    python dft_dimers.py dimers.json <task> <ntasks> <outdir>"""
import json, os, sys, time, numpy as np
import pyscf
from gpu4pyscf.dft import rks
SYM = {1: "H", 3: "Li", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 11: "Na", 12: "Mg", 15: "P", 16: "S", 17: "Cl", 19: "K", 20: "Ca", 30: "Zn", 35: "Br", 53: "I"}
HARTREE_KCAL = 627.509474
SCALES = (0.95, 1.0, 1.15, 1.35)
def energy(Z, R, q):
    mol = pyscf.M(atom=[(SYM[int(z)], r) for z, r in zip(Z, R)], basis="def2-tzvpd", ecp={"I": "def2-tzvpd"} if 53 in Z else None, charge=int(q), spin=0, unit="Angstrom", verbose=0)
    mf = rks.RKS(mol, xc="wb97m-v").density_fit(); mf.grids.atom_grid = (99, 590); mf.nlcgrids.atom_grid = (50, 194); mf.conv_tol = 1e-9; mf.max_cycle = 100
    e = mf.kernel()
    if not mf.converged: raise RuntimeError("SCF not converged")
    return float(e)
if __name__ == "__main__":
    D = json.load(open(sys.argv[1]))["dimers"]; t, nt, out = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]; os.makedirs(out, exist_ok=True)
    for k in range(t, len(D), nt):
        f = os.path.join(out, f"d{k:04d}.json")
        if os.path.exists(f): continue
        d = D[k]; t0 = time.time()
        try:
            ZA, RA, ZB, RB = np.array(d["ZA"]), np.array(d["RA"]), np.array(d["ZB"]), np.array(d["RB"])
            EA = energy(ZA, RA, d["QA"]); EB = energy(ZB, RB, d["QB"])
            ca, cb = RA.mean(0), RB.mean(0); res = []
            for s in SCALES:
                RBs = RB + (s - 1) * (cb - ca); Eab = energy(np.concatenate([ZA, ZB]), np.concatenate([RA, RBs]), d["QA"] + d["QB"])
                res.append(dict(scale=s, RB=RBs.round(4).tolist(), E_int_kcal=(Eab - EA - EB) * HARTREE_KCAL))
            r = dict(index=k, **{x: d[x] for x in ("xn", "A", "B", "QA", "QB", "dmin")}, scan=res, seconds=time.time() - t0)
        except Exception as e:
            r = dict(index=k, error=f"{type(e).__name__}: {e}", seconds=time.time() - t0)
        json.dump(r, open(f, "w")); print(k, d["A"], d["B"], f"{r['seconds']:.0f}s", [round(x["E_int_kcal"], 2) for x in r.get("scan", [])] or r.get("error"), flush=True)
