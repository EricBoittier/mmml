#!/usr/bin/env python
"""POV-Ray renders of the three embedding schemes used for the solvated peptides, composed into one figure:
(a) gas-phase ML peptide, (b) mechanical embedding (ML peptide inside CHARMM36/TIP3P water), (c) ML first solvation
shell (peptide + waters within --shell Å of a peptide N/O as one ML region, the rest MM). Same molecule (trialanine
zwitterion, β start), same orientation and framing; MM water drawn as thin grey sticks, ML atoms as CPK balls and sticks,
hydrogen bonds of the ML shell dashed.

  python embedding_figs.py OUT_DIR [--conf beta] [--tag _20261014] [--rcut 7.0]
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path.home() / "mmml_dev"))
from render_pov import ELEM, bonds, orient, render, v   # noqa: E402
from ase.data import covalent_radii                      # noqa: E402
from mmml.utils.plotting.styles import apply_plot_style  # noqa: E402

T = Path.home() / "mmml_new/ckpts/tripeptides"
MMCOL = (0.62, 0.66, 0.70)


def sph(p, r, col, t=0.0):
    rgb = f"<{col[0]:.3f},{col[1]:.3f},{col[2]:.3f}" + (f",{t:.2f}>" if t else ">")
    return f"sphere {{ {v(p)}, {r:.3f} pigment {{ {'rgbt' if t else 'rgb'} {rgb} }} finish {{ phong 0.6 phong_size 60 ambient 0.22 diffuse 0.75 }} }}\n"


def cyl(a, b, r, col, t=0.0):
    rgb = f"<{col[0]:.3f},{col[1]:.3f},{col[2]:.3f}" + (f",{t:.2f}>" if t else ">")
    return f"cylinder {{ {v(a)}, {v(b)}, {r:.3f} pigment {{ {'rgbt' if t else 'rgb'} {rgb} }} finish {{ ambient 0.25 }} }}\n"


def dashed(a, b, col, r=0.04, n=7):
    out = []
    for k in range(n):
        if k % 2 == 0:
            out.append(cyl(a + (b - a) * k / n, a + (b - a) * (k + 1) / n, r, col))
    return "".join(out)


def ml_atoms(Z, R, scale=0.30):
    s = [sph(r, scale * (0.55 + covalent_radii[z]), ELEM.get(int(z), (0.7, 0.7, 0.7))) for z, r in zip(Z, R)]
    s += [cyl(R[i], R[j], 0.10, (0.62, 0.62, 0.60)) for i, j in bonds(Z, R)]
    return "".join(s)


def mm_waters(W, t=0.45):
    """W: (n,3,3) O,H,H positions; thin grey sticks."""
    s = []
    for w in W:
        s.append(sph(w[0], 0.16, MMCOL, t)); s.append(sph(w[1], 0.10, MMCOL, t)); s.append(sph(w[2], 0.10, MMCOL, t))
        s.append(cyl(w[0], w[1], 0.06, MMCOL, t)); s.append(cyl(w[0], w[2], 0.06, MMCOL, t))
    return "".join(s)


def camera(box, w, h):
    lo, hi = -box, box
    aspect = w / h
    width = max(hi[0] - lo[0], (hi[1] - lo[1]) * aspect)
    return f"""#version 3.7;
global_settings {{ assumed_gamma 1.0 }}
camera {{ perspective location <0,0,-400> right x*{aspect:.4f} up y angle {2 * np.degrees(np.arctan(width / 800)):.4f} look_at <0,0,0> }}
light_source {{ <-20,30,-60> color rgb 1.0 }}
light_source {{ <30,-10,-40> color rgb 0.45 shadowless }}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path); ap.add_argument("--conf", default="beta"); ap.add_argument("--tag", default="_20261014")
    ap.add_argument("--rcut", type=float, default=7.0); ap.add_argument("--shell-tag", default="_shell_20261014")
    args = ap.parse_args()
    out = args.out; out.mkdir(parents=True, exist_ok=True)
    mech = np.load(T / "mlmm" / f"AAA_zw_{args.conf}_20261012.npz")
    shell = np.load(T / "mlmm_shell" / f"AAA_zw_{args.conf}{args.shell_tag}.npz")
    gas = np.load(T / "trialanine_zw" / "trialanine_zw_20261012.npz")
    Z = mech["Z"].astype(int); npep = int(mech["n_pep"]); nw = int(mech["n_water"])
    # orientation from the mechanically embedded peptide; same rotation for every panel (the shell geometry is re-aligned by Kabsch)
    Rm = mech["R_all"]; Ro, Vt, c = orient(Rm[:npep])
    Wm = ((Rm[npep:npep + 3 * nw] - c) @ Vt.T).reshape(nw, 3, 3)
    def kabsch_to(P, Q):          # rotation/translation taking P (peptide) onto Q
        pc, qc = P.mean(0), Q.mean(0); H = (P - pc).T @ (Q - qc); U, _, VT = np.linalg.svd(H); d = np.sign(np.linalg.det(VT.T @ U.T))
        Rot = VT.T @ np.diag([1, 1, d]) @ U.T; return lambda X: (X - pc) @ Rot.T + qc
    Rs = shell["R_all"]; ns = int(shell["n_shell"]); f = kabsch_to(Rs[:npep], Ro)
    Rs_pep = f(Rs[:npep]); Ws = f(Rs[npep:npep + 3 * nw]).reshape(nw, 3, 3)
    Zg = gas["Z"].astype(int); g = kabsch_to(gas[f"{args.conf}_R"], Ro); Rg = g(gas[f"{args.conf}_R"])
    box = np.array([args.rcut + 3.5, args.rcut + 1.5, 20.0])
    def near(W, Rp):
        d = np.linalg.norm(W[:, 0][:, None] - Rp[None], axis=-1).min(1); return d < args.rcut
    w, h = 1200, 900
    scenes = {}
    scenes["gas"] = camera(box, w, h) + ml_atoms(Zg, Rg)
    scenes["mech"] = camera(box, w, h) + ml_atoms(Z, Ro) + mm_waters(Wm[near(Wm, Ro)])
    shellW = Ws[:ns]; restW = Ws[ns:]
    Zreg = np.concatenate([Z, np.tile([8, 1, 1], ns)]); Rreg = np.vstack([Rs_pep, shellW.reshape(-1, 3)])
    hb = ""
    polar = Rs_pep[np.isin(Z, (7, 8))]
    for wat in shellW:
        d = np.linalg.norm(polar - wat[0], axis=1); k = int(np.argmin(d))
        if d[k] < 3.4: hb += dashed(wat[0], polar[k], (0.04, 0.43, 0.40))
    scenes["shell"] = camera(box, w, h) + ml_atoms(Zreg, Rreg) + hb + mm_waters(restW[near(restW, Ro)])
    pngs = {k: render(out, f"pov_embed_{k}_{args.conf}{args.tag}", s, w=w, h=h) for k, s in scenes.items()}
    # compose
    apply_plot_style("icml")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35))
    titles = [("a", "gas phase\nML peptide alone"), ("b", f"mechanical embedding\nML peptide in {nw} MM (TIP3P) waters"), ("c", f"ML first solvation shell\npeptide + {ns} waters ML, rest MM")]
    for ax, (k, (lab, t)) in zip(axes, zip(("gas", "mech", "shell"), titles)):
        ax.imshow(mpimg.imread(pngs[k])); ax.set_axis_off()
        ax.text(0.01, 0.98, lab, transform=ax.transAxes, fontweight="bold", fontsize=9, va="top")
        ax.set_title(t, fontsize=7, linespacing=1.15)
    fig.subplots_adjust(wspace=0.02, left=0.01, right=0.99, top=0.84, bottom=0.02)
    p = out / f"fig_embedding_{args.conf}{args.tag}.png"; fig.savefig(p, dpi=300); fig.savefig(p.with_suffix(".pdf")); plt.close(fig)
    json.dump(dict(conf=args.conf, n_water=nw, n_shell=ns, rcut_drawn=args.rcut, n_mm_drawn_mech=int(near(Wm, Ro).sum()), n_mm_drawn_shell=int(near(restW, Ro).sum())), open(out / f"embedding_figs{args.tag}.json", "w"), indent=1)
    print("wrote", p, {k: str(x.name) for k, x in pngs.items()})


if __name__ == "__main__":
    main()
