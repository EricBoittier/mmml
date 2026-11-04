"""RESP charges (HF/6-31G*, anions 6-31+G*, I: def2-SVP+ECP) on Merz-Kollman shells (1.4/1.6/1.8/2.0 x vdW, ~3 pts/A^2),
hyperbolic restraint a=0.0005 (heavy atoms), b=0.1, total-charge constraint, equivalence of topologically identical atoms
(same r2 signature).  Geometry: the OMol25 example geometry of the residue.
    python resp.py <jobs.json> <task_id> <n_tasks> <outdir>"""
import json, sys, os, time, numpy as np
from pyscf import gto, scf, lib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "envtyper")); import cgenff_envtyper as E
VDW = {1: 1.2, 3: 1.82, 5: 1.92, 6: 1.7, 7: 1.55, 8: 1.52, 9: 1.47, 11: 2.27, 14: 2.1, 15: 1.8, 16: 1.8, 17: 1.75, 35: 1.85, 53: 1.98}
SYM = {1: "H", 3: "Li", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 11: "Na", 14: "Si", 15: "P", 16: "S", 17: "Cl", 35: "Br", 53: "I"}
def shell_points(Z, R, density=3.0):
    pts = []
    for f in (1.4, 1.6, 1.8, 2.0):
        for i, z in enumerate(Z):
            r = VDW.get(int(z), 1.8) * f; n = max(20, int(4 * np.pi * r * r * density))
            k = np.arange(n) + 0.5; phi = np.arccos(1 - 2 * k / n); th = np.pi * (1 + 5 ** 0.5) * k
            p = R[i] + r * np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
            d = np.linalg.norm(p[:, None] - R[None], axis=-1); rad = np.array([VDW.get(int(x), 1.8) for x in Z]) * f
            pts.append(p[(d >= rad[None] - 1e-6).all(1)])
    return np.concatenate(pts)
def esp(mol, dm, pts_bohr):
    v = np.zeros(len(pts_bohr))
    for i0 in range(0, len(pts_bohr), 400):
        g = pts_bohr[i0:i0 + 400]; v[i0:i0 + 400] = -np.einsum("gij,ij->g", mol.intor("int1e_grids", grids=g), dm)
    rn = mol.atom_coords(); zc = mol.atom_charges()
    v += (zc[None] / np.linalg.norm(pts_bohr[:, None] - rn[None], axis=-1)).sum(1)
    return v
def resp_fit(R_bohr, pts_bohr, V, Q, Z, eq, a=0.0005, b=0.1):
    n = len(R_bohr); A0 = 1 / np.linalg.norm(pts_bohr[:, None] - R_bohr[None], axis=-1)   # (P, n)
    # equivalence: parametrise q = M p, groups of identical atoms share p
    groups = sorted(set(eq)); M = np.zeros((n, len(groups)));
    for i, g in enumerate(eq): M[i, groups.index(g)] = 1
    A = A0 @ M; heavy = (np.asarray(Z) > 1).astype(float)
    p = np.linalg.lstsq(A, V, rcond=None)[0]
    for _ in range(50):
        q = M @ p; rw = heavy * a / np.sqrt(q * q + b * b)            # hyperbolic restraint, linearised
        H = A.T @ A + M.T @ np.diag(rw) @ M; c = M.sum(0)
        K = np.block([[H, c[:, None]], [c[None], np.zeros((1, 1))]]); rhs = np.concatenate([A.T @ V, [Q]])
        pn = np.linalg.solve(K, rhs)[:-1]
        if np.abs(pn - p).max() < 1e-6: p = pn; break
        p = pn
    q = M @ p; rrms = np.sqrt(np.mean((A0 @ q - V) ** 2) / np.mean(V ** 2))
    return q, rrms
def run(job):
    Z = np.array(job["Z"]); R = np.array(job["R"]); Q = int(job["Q"])
    heavy = 53 in Z or 35 in Z
    if any(int(z) not in SYM for z in Z): raise ValueError("element not supported")
    basis = "def2-svp" if heavy else ("6-31+g*" if Q < 0 else "6-31g*")
    mol = gto.M(atom=[(SYM[int(z)], r) for z, r in zip(Z, R)], basis=basis, charge=Q, spin=0, unit="Angstrom", verbose=0,
                ecp="def2-svp" if heavy else None)
    mf = scf.RHF(mol); mf.conv_tol = 1e-8; mf.max_cycle = 200; mf.kernel()
    if not mf.converged:
        mf = scf.newton(mf); mf.kernel()
    pts = shell_points(Z, R); pb = pts / lib.param.BOHR
    V = esp(mol, mf.make_rdm1(), pb)
    adj = E.graph_from_xyz(Z, R); sig = E.signatures(Z, adj); eq = sig[2]
    q, rrms = resp_fit(mol.atom_coords(), pb, V, Q, Z, eq)
    np.savez(os.path.join(OUT, job["name"] + "_esp.npz"), pts_bohr=pb, V=V, R_bohr=mol.atom_coords(), Z=Z)
    dip = np.linalg.norm(((q[:, None]) * (R - R.mean(0))).sum(0)) / 0.20819434   # Debye
    return dict(name=job["name"], q=q.round(4).tolist(), rrms=float(rrms), converged=bool(mf.converged), basis=basis, npts=len(pts), E_hf=float(mf.e_tot), dipole_D=float(dip))
if __name__ == "__main__":
    jobs = json.load(open(sys.argv[1])); t, nt, out = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]; os.makedirs(out, exist_ok=True); OUT = out
    lib.num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    for j in jobs[t::nt]:
        f = os.path.join(out, j["name"] + ".json")
        if os.path.exists(f): continue
        t0 = time.time()
        try: r = run(j)
        except Exception as e: r = dict(name=j["name"], error=f"{type(e).__name__}: {e}")
        r["seconds"] = time.time() - t0; json.dump(r, open(f, "w")); print(j["name"], len(j["Z"]), f"{r['seconds']:.1f}s", r.get("rrms", r.get("error")), flush=True)
