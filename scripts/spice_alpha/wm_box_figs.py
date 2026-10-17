#!/usr/bin/env python
"""POV-Ray renders of the water–methanol boxes of the handoff at one scale (30, 38, 50, 64 Å, x = 0.5), heavy atoms only:
water O blue, methanol C grey and O orange-red, cell edges drawn; composed into fig_wm_boxes<tag>.png/pdf.
  python wm_box_figs.py OUT_DIR [--tag _20261014]"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.image as mpimg
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path.home() / "mmml_dev"))
from render_pov import render, v   # noqa: E402
from mmml.utils.plotting.styles import apply_plot_style  # noqa: E402

B = Path.home() / "handoff/handoff_large_20261014"
BOXES = [("30 Å", B / "boxes30/x0.500_f1.000_r1"), ("38 Å", B / "boxes_large/x0.500_L3.8_r0"), ("50 Å", B / "boxes_large/x0.500_L5_r0"), ("64 Å", B / "boxes_large/rep2_x050_L64")]
COL = {"WO": (0.20, 0.45, 0.85), "MC": (0.42, 0.42, 0.44), "MO": (0.88, 0.40, 0.18)}


def read_crd(p):
    names, res, R = [], [], []
    for line in open(p):
        f = line.split()
        if len(f) >= 7 and f[0].isdigit() and f[1].isdigit():
            names.append(f[3]); res.append(f[2]); R.append([float(f[4]), float(f[5]), float(f[6])])
    return np.array(names), np.array(res), np.array(R)


def scene(names, res, R, L, w, h):
    R = R - L * np.floor(R / L)                      # wrap atoms into the cell
    # camera: oblique view of the cube from a corner, same distance scaling for every box (scale by 1/L of the largest)
    c = np.full(3, L / 2)
    look = c
    dirn = np.array([1.0, 0.65, -1.4]); dirn /= np.linalg.norm(dirn)
    cam = c + dirn * 2.6 * 64.0                        # fixed distance: boxes appear at their true relative size
    s = [f"""#version 3.7;
global_settings {{ assumed_gamma 1.0 }}
camera {{ perspective location {v(cam)} right x*{w/h:.4f} up y angle 36 look_at {v(look)} }}
light_source {{ {v(cam + np.array([-60, 120, -40]))} color rgb 1.0 }}
light_source {{ {v(cam + np.array([80, 20, 60]))} color rgb 0.4 shadowless }}
"""]
    def sph(p, r, col): return f"sphere {{ {v(p)}, {r} pigment {{ rgb <{col[0]},{col[1]},{col[2]}> }} finish {{ phong 0.4 ambient 0.3 diffuse 0.7 }} }}\n"
    for n, rs, p in zip(names, res, R):
        if rs == "TIP3" and n == "OH2": s.append(sph(p, 0.95, COL["WO"]))
        elif rs == "MEOH" and n == "CB": s.append(sph(p, 1.0, COL["MC"]))
        elif rs == "MEOH" and n == "OG": s.append(sph(p, 0.9, COL["MO"]))
    E = np.array([[0, 0, 0], [L, 0, 0], [L, L, 0], [0, L, 0], [0, 0, L], [L, 0, L], [L, L, L], [0, L, L]], float)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for i, j in edges:
        s.append(f"cylinder {{ {v(E[i])}, {v(E[j])}, 0.25 pigment {{ rgb <0.15,0.15,0.15> }} finish {{ ambient 0.6 }} no_shadow }}\n")
    return "".join(s)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out", type=Path); ap.add_argument("--tag", default="_20261014")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    w, h = 1100, 1000; pngs, info = [], []
    for lab, d in BOXES:
        names, res, R = read_crd(d / "model.crd"); b = json.load(open(d / "box.json")); L = float(b["box_A"])
        png = render(a.out, f"pov_wmbox_{lab.split()[0]}{a.tag}", scene(names, res, R, L, w, h), w=w, h=h)
        pngs.append(png); info.append((lab, b)); print("rendered", png.name, len(R), "atoms")
    apply_plot_style("icml")
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.3))
    extra = {"30 Å": "7.3 ns/day · 5.0 GB (5090)\n4.0 ns/day · 2.9 GB (A100)", "38 Å": "3.6 ns/day · 10 GB (5090)\n≈18–20 GB with dipoles", "50 Å": "1.5 ns/day · 23 GB (5090)\nrunning on A100 40 GB", "64 Å": "≈0.7 ns/day · ≈48 GB\nH200 only"}
    for ax, png, (lab, b), k in zip(axes, pngs, info, "abcd"):
        ax.imshow(mpimg.imread(png)); ax.set_axis_off()
        ax.text(0.01, 0.99, k, transform=ax.transAxes, fontweight="bold", fontsize=9, va="top")
        ax.set_title(f"{lab}\n{b['n_water'] + b['n_meoh']:,} molecules · {b['n_atoms']:,} atoms", fontsize=6.8, linespacing=1.15)
        ax.text(0.5, -0.02, extra[lab], transform=ax.transAxes, fontsize=5.8, ha="center", va="top", color="#444444", linespacing=1.2)
    fig.subplots_adjust(wspace=0.02, left=0.01, right=0.99, top=0.82, bottom=0.14)
    p = a.out / f"fig_wm_boxes{a.tag}.png"; fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.04); fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04); plt.close(fig); print("wrote", p)


if __name__ == "__main__":
    main()
