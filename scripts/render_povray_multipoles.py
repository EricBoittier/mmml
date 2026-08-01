#!/usr/bin/env python3
"""Render a DCM electrostatics glyph: charges, dipole, and point-charge quadrupole."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from PIL import Image, ImageDraw, ImageFont

from scripts.render_povray_style_catalog import scene, vec


def _arrow(start, end, color, radius=.045) -> list[str]:
    d = end-start; length = float(np.linalg.norm(d))
    if length < 1e-10:
        return []
    u = d/length; neck = end-u*min(.22, .28*length)
    pigment = f"pigment {{ color rgb {vec(color)} }} finish {{ emission 0.16 phong 0.5 }}"
    return [f"cylinder {{ {vec(start)}, {vec(neck)}, {radius} {pigment} }}",
            f"cone {{ {vec(neck)}, {radius*2.8}, {vec(end)}, 0 {pigment} }}"]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path("examples/mp2_nms15_train.npz"))
    p.add_argument("--output", type=Path,
                   default=Path("docs/images/povray-overlays/dcm_multipoles.png"))
    p.add_argument("--width", type=int, default=1200)
    p.add_argument("--height", type=int, default=900)
    a = p.parse_args(argv)
    d = np.load(a.input, allow_pickle=True)
    names = np.asarray(d["res_name"]).astype(str)
    idx = int(np.flatnonzero(names == "DCM")[0]); n = int(d["N"][idx])
    Z = np.asarray(d["Z"][idx, :n]); R = np.asarray(d["R"][idx, :n])
    q = np.asarray(d["cgenff_charge"][idx, :n]); dipole = np.asarray(d["D"][idx])
    atoms = Atoms(numbers=Z, positions=R)
    base = scene(atoms, "glossy", "", vectors=False, geometry=False,
                 font="", width=a.width, height=a.height, frame_scale=2.05)
    xyz = R-R.mean(0)
    overlay = []
    # Signed partial-charge halos: red is negative, blue is positive.
    for pos, charge, atomic_number in zip(xyz, q, Z):
        color = (0.16, .38, .92) if charge > 0 else (.88, .12, .20)
        radius = max(.24, .52*covalent_radii[atomic_number]) + .07 + .55*abs(float(charge))
        overlay.append(f"sphere {{ {vec(pos)}, {radius:.4f} pigment {{ color rgbt <{color[0]},{color[1]},{color[2]},0.78> }} finish {{ emission 0.05 phong 0.35 }} no_shadow }}")
    # Dipole: stored quantum label, direction and magnitude in e Å.
    mu = dipole/max(float(np.linalg.norm(dipole)), 1e-12)
    overlay += _arrow(np.zeros(3), mu*2.25, (.95, .62, .06), radius=.065)
    # Traceless quadrupole derived from the displayed point charges. This is a
    # visualization proxy, not an ab-initio quadrupole label.
    Q = sum(charge*(3*np.outer(r, r)-np.dot(r, r)*np.eye(3)) for charge, r in zip(q, xyz))
    values, axes = np.linalg.eigh(Q)
    for value, axis in zip(values, axes.T):
        length = .52 + .30*abs(value)/max(float(np.max(np.abs(values))), 1e-12)
        color = (.48, .20, .78) if value >= 0 else (.05, .62, .62)
        overlay += _arrow(axis*.20, axis*length, color, radius=.018)
        overlay += _arrow(-axis*.20, -axis*length, color, radius=.018)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    stem = a.output.with_suffix("")
    stem.with_suffix(".pov").write_text(base+"\n".join(overlay)+"\n")
    stem.with_suffix(".ini").write_text(
        f'Input_File_Name="{stem.name}.pov"\nOutput_File_Name="{stem.name}.png"\nWidth={a.width}\nHeight={a.height}\nAntialias=On\nOutput_Alpha=On\nDisplay=Off\n')
    subprocess.run([shutil.which("povray") or "povray", stem.with_suffix(".ini").name],
                   cwd=stem.parent, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.STDOUT)
    image = Image.open(a.output).convert("RGBA")
    alpha = image.getchannel("A"); baseline = alpha.getextrema()[0]
    if 0 < baseline < 255:
        image.putalpha(alpha.point(lambda x: max(0, round((x-baseline)*255/(255-baseline)))))
    draw = ImageDraw.Draw(image, "RGBA")
    font_path = "/System/Library/Fonts/Supplemental/Verdana.ttf"
    title_font = ImageFont.truetype(font_path, 34); body_font = ImageFont.truetype(font_path, 23)
    draw.rounded_rectangle((24, 22, 520, 194), radius=14, fill=(250,250,252,225))
    draw.text((42, 34), "DCM electrostatic multipoles", font=title_font, fill=(18,22,30,255))
    draw.text((42, 84), "blue/red halos   +q / −q", font=body_font, fill=(18,22,30,255))
    draw.text((42, 116), f"gold dipole   |μ| = {np.linalg.norm(dipole):.3f} e Å", font=body_font, fill=(18,22,30,255))
    draw.text((42, 148), "violet/teal   ± point-charge Q axes", font=body_font, fill=(18,22,30,255))
    image.save(a.output)
    print(f"Rendered {a.output}; source frame {idx}; charges sum {q.sum():.6f} e")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
