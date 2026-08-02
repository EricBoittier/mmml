#!/usr/bin/env python3
"""POV-Ray teaching style for distributed charge models and equivalent moments."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import molecule
from PIL import Image, ImageDraw, ImageFont

from scripts.render_povray_multipoles import _arrow
from scripts.render_povray_style_catalog import scene, vec


def main() -> int:
    out = Path("docs/images/povray-overlays/distributed_charge_model.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    water = molecule("H2O"); water.center(vacuum=0)
    R = water.positions-water.positions.mean(0)
    # Two views of the same molecule: explicit DCM sites and their equivalent
    # low-order molecular moments. Values are deliberately schematic.
    left, right = R+[-2.7, 0, 0], R+[2.7, 0, 0]
    atoms = Atoms(numbers=np.tile(water.numbers, 2), positions=np.vstack([left, right]))
    base = scene(atoms, "glossy", "", vectors=False, geometry=False, font="",
                 width=1400, height=820, frame_scale=1.45)
    sites, charges, parents = [], [], []
    offsets = [np.array([.20, 0, .20]), np.array([-.20, 0, .20])]
    for atom_i, (p, z) in enumerate(zip(left, water.numbers)):
        q_each = -.40 if z == 8 else .20
        local = offsets if z == 8 else [np.array([.10, .12, 0]), np.array([-.10, -.12, 0])]
        for off in local:
            sites.append(p+off); charges.append(q_each); parents.append(p)
    sites, charges = np.asarray(sites), np.asarray(charges)
    overlay = []
    for site, charge, parent in zip(sites, charges, parents):
        color = (.16,.38,.92) if charge > 0 else (.88,.10,.18)
        overlay.append(f'cylinder {{ {vec(parent)}, {vec(site)}, 0.018 pigment {{ color rgbt <.25,.28,.34,.45> }} no_shadow }}')
        overlay.append(f'sphere {{ {vec(site)}, {0.12+0.16*abs(charge):.3f} pigment {{ color rgb {vec(color)} }} finish {{ emission .12 phong .7 }} no_shadow }}')
    origin = right.mean(0)
    rel = sites-left.mean(0); mu = np.sum(charges[:,None]*rel, axis=0)
    mu_u = mu/max(np.linalg.norm(mu),1e-12)
    overlay += _arrow(origin, origin+1.45*mu_u, (.95,.62,.06), radius=.055)
    Q = sum(q*(3*np.outer(r,r)-np.dot(r,r)*np.eye(3)) for q,r in zip(charges,rel))
    vals, axes = np.linalg.eigh(Q)
    for val, axis in zip(vals, axes.T):
        color=(.48,.20,.78) if val>=0 else (.05,.62,.62)
        overlay += _arrow(origin+.18*axis, origin+.68*axis, color, radius=.017)
        overlay += _arrow(origin-.18*axis, origin-.68*axis, color, radius=.017)
    stem=out.with_suffix("")
    stem.with_suffix(".pov").write_text(base+"\n".join(overlay)+"\n")
    stem.with_suffix(".ini").write_text(f'Input_File_Name="{stem.name}.pov"\nOutput_File_Name="{stem.name}.png"\nWidth=1400\nHeight=820\nAntialias=On\nOutput_Alpha=On\nDisplay=Off\n')
    subprocess.run([shutil.which("povray") or "povray",stem.with_suffix(".ini").name],cwd=stem.parent,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
    image=Image.open(out).convert("RGBA"); alpha=image.getchannel("A"); baseline=alpha.getextrema()[0]
    if 0<baseline<255: image.putalpha(alpha.point(lambda x:max(0,round((x-baseline)*255/(255-baseline)))))
    draw=ImageDraw.Draw(image,"RGBA"); fp="/System/Library/Fonts/Supplemental/Verdana.ttf"
    title=ImageFont.truetype(fp,34); body=ImageFont.truetype(fp,21)
    draw.rounded_rectangle((28,24,1370,118),radius=14,fill=(250,250,252,225))
    draw.text((48,34),"Distributed charge model and equivalent molecular multipoles",font=title,fill=(18,22,30,255))
    draw.text((95,78),"explicit off-atom sites (2 per parent)",font=body,fill=(18,22,30,255))
    draw.text((760,78),"same charge distribution summarized by μ and Q",font=body,fill=(18,22,30,255))
    draw.text((54,740),"schematic values · blue +q · red −q · gold μ · violet/teal Q axes",font=body,fill=(18,22,30,255))
    image.save(out); print(out)

    # Companion ESP slice from the exact same schematic distributed sites.
    import matplotlib.pyplot as plt
    from mmml.utils.plotting.styles import apply_plot_style, default_cmap
    apply_plot_style("icml")
    # Water lies in the yz plane in ASE's standard orientation.
    uu = np.linspace(-2.2, 2.2, 280); vv = np.linspace(-2.0, 2.4, 280)
    U, V = np.meshgrid(uu, vv)
    esp = np.zeros_like(U)
    rel_sites = sites-left.mean(0)
    for site, charge in zip(rel_sites, charges):
        distance = np.sqrt((U-site[1])**2+(V-site[2])**2+.16**2)
        esp += charge/distance
    limit = float(np.percentile(np.abs(esp), 96))
    fig, ax = plt.subplots(figsize=(7.5, 6.2), constrained_layout=True)
    levels = np.linspace(-limit, limit, 25)
    cf = ax.contourf(U, V, np.clip(esp, -limit, limit), levels=levels,
                     cmap=default_cmap("diverging"), extend="both")
    ax.contour(U, V, esp, levels=[0], colors="0.15", linewidths=1.1)
    parent = R-R.mean(0)
    colors = [(.86,.16,.13) if z==8 else (.86,.87,.90) for z in water.numbers]
    ax.scatter(parent[:,1], parent[:,2], s=180, c=colors, edgecolor="0.25", zorder=5)
    site_colors = [(.16,.38,.92) if q>0 else (.88,.10,.18) for q in charges]
    ax.scatter(rel_sites[:,1], rel_sites[:,2], s=35, c=site_colors,
               edgecolor="white", linewidth=.6, zorder=6)
    ax.set_aspect("equal"); ax.set(xlabel="molecular-plane coordinate (Å)",
        ylabel="molecular-plane coordinate (Å)",
        title="ESP generated by the distributed charge sites")
    fig.colorbar(cf, ax=ax, label="electrostatic potential (arb. units)")
    esp_out = out.with_name("distributed_charge_esp.png")
    fig.savefig(esp_out, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig); print(esp_out)
    return 0


if __name__ == "__main__": raise SystemExit(main())
