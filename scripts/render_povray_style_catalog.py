#!/usr/bin/env python3
"""Render a reproducible POV-Ray material/annotation catalog from XYZ artifacts."""

from __future__ import annotations

import argparse
import html
import shutil
import subprocess
from pathlib import Path

import numpy as np
from ase.data import covalent_radii
from ase.io import read

COLORS = {"H": (0.86, 0.87, 0.90), "C": (0.22, 0.24, 0.28),
          "N": (0.18, 0.32, 0.88), "O": (0.86, 0.16, 0.13),
          "Cl": (0.20, 0.72, 0.28), "S": (0.92, 0.75, 0.12)}
STYLES = {
    "glossy": "finish { phong 0.85 phong_size 90 reflection 0.06 }",
    "chalky": "finish { diffuse 0.88 roughness 0.12 } normal { bumps 0.16 scale 0.08 }",
    "milky": "finish { diffuse 0.55 phong 0.25 reflection 0.03 }",
    "metallic": "finish { metallic reflection 0.28 phong 0.7 phong_size 70 }",
}


def bonds(atoms) -> list[tuple[int, int]]:
    xyz, nums = atoms.positions, atoms.numbers
    return [(i, j) for i in range(len(atoms)) for j in range(i + 1, len(atoms))
            if np.linalg.norm(xyz[i] - xyz[j]) < 1.18 * (covalent_radii[nums[i]] + covalent_radii[nums[j]])]


def vec(v) -> str:
    return "<" + ",".join(f"{x:.6f}" for x in v) + ">"


def scene(atoms, style: str, title: str, *, vectors: bool, geometry: bool,
          font: str, width: int, height: int, frame_scale: float = 1.72) -> str:
    xyz = atoms.positions - atoms.positions.mean(0)
    span = max(float(np.ptp(xyz, axis=0).max()), 3.0)
    camera = np.array([span * 1.15, span * .80, -span * 2.7])
    finish = STYLES[style]
    lines = [
        '#version 3.7;', 'global_settings { assumed_gamma 1.0 }',
        'background { color rgbt <1,1,1,1> }',
        f'camera {{ orthographic location {vec(camera)} look_at <0,0,0> right x*{frame_scale*span} up y*{frame_scale*span*height/width} }}',
        f'light_source {{ {vec((-span, span*2, -span*2))} color rgb <1,1,1> area_light <2,0,0>,<0,2,0>,5,5 adaptive 1 jitter }}',
        f'light_source {{ {vec((span*2, -span, -span))} color rgb <0.35,0.40,0.52> }}',
        # Mostly transparent shadow catcher: reusable alpha, but the soft
        # contact shadows still ground the molecule on a page or slide.
        'plane { y, -2.15 pigment { color rgbt <0.91,0.92,0.94,0.82> } finish { diffuse 0.75 } }',
    ]
    for i, j in bonds(atoms):
        a, b = xyz[i], xyz[j]
        lines.append(f'cylinder {{ {vec(a)}, {vec(b)}, 0.105 pigment {{ color rgb <0.55,0.57,0.61> }} finish {{ diffuse 0.7 phong 0.25 }} }}')
    for sym, p, z in zip(atoms.get_chemical_symbols(), xyz, atoms.numbers):
        color = COLORS.get(sym, (0.55, 0.55, 0.58))
        pigment = (f"color rgbt <{color[0]},{color[1]},{color[2]},0.28>"
                   if style == "milky" else f"color rgb {vec(color)}")
        radius = max(0.24, covalent_radii[z] * .52)
        lines.append(f'sphere {{ {vec(p)}, {radius:.4f} texture {{ pigment {{ {pigment} }} {finish} }} }}')
    if vectors:
        try:
            forces = atoms.get_forces()
        except (RuntimeError, ValueError):
            forces = None
    else:
        forces = None
    if forces is not None:
        scale = 1.35 / max(float(np.linalg.norm(forces, axis=1).max()), 1e-12)
        for p, force in zip(xyz, forces):
            d = force * scale; n = np.linalg.norm(d)
            if n < .025: continue
            u = d / n; start = p + .42 * u; q = start + d; neck = q - .22 * u
            lines += [
                f'cylinder {{ {vec(start)}, {vec(neck)}, 0.045 pigment {{ color rgb <0.78,0.04,0.12> }} finish {{ emission 0.18 }} }}',
                f'cone {{ {vec(neck)}, 0.12, {vec(q)}, 0 pigment {{ color rgb <0.78,0.04,0.12> }} finish {{ emission 0.18 }} }}',
            ]
    if geometry:
        # Reaction coordinate guide between nucleophile N and electrophile C.
        syms = atoms.get_chemical_symbols()
        if "N" in syms and "C" in syms:
            i, j = syms.index("N"), syms.index("C"); a, b = xyz[i].copy(), xyz[j].copy()
            a[2] -= .35; b[2] -= .35
            lines.append(f'cylinder {{ {vec(a)}, {vec(b)}, 0.04 pigment {{ color rgb <0.48,0.08,0.72> }} finish {{ emission 0.25 }} }}')
            lines.append(f'sphere {{ {vec((a+b)/2)}, 0.075 pigment {{ color rgb <0.48,0.08,0.72> }} }}')
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--font", default="/System/Library/Fonts/Supplemental/Verdana.ttf")
    a = p.parse_args(argv)
    frames = read(a.input, index=":")
    picks = [(0, "reactant"), (len(frames)//2, "transition"), (len(frames)-1, "product")]
    variants = [("glossy", False, False), ("chalky", False, False),
                ("milky", False, False), ("metallic", False, False),
                ("glossy", True, False), ("chalky", True, True)]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    povray = shutil.which("povray")
    cards = []
    for frame_idx, frame_name in picks:
        for material, vectors, geometry in variants:
            suffix = material + ("_forces" if vectors else "") + ("_geometry" if geometry else "")
            stem = a.out_dir / f"{frame_name}_{suffix}"
            label = f"{frame_name.title()} | {material}" + (" | forces" if vectors else "") + (" | N--C guide" if geometry else "")
            stem.with_suffix(".pov").write_text(scene(frames[frame_idx], material, label,
                vectors=vectors, geometry=geometry, font=a.font, width=a.width, height=a.height))
            stem.with_suffix(".ini").write_text(
                f'Input_File_Name="{stem.name}.pov"\nOutput_File_Name="{stem.name}.png"\nWidth={a.width}\nHeight={a.height}\nAntialias=On\nAntialias_Threshold=0.15\nOutput_Alpha=On\nDisplay=Off\n')
            if povray:
                subprocess.run([povray, stem.with_suffix(".ini").name], cwd=a.out_dir,
                               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                # Labels belong to the image plane, not molecular world space:
                # this keeps them fixed and unclipped for every camera framing.
                from PIL import Image, ImageDraw, ImageFont
                image = Image.open(stem.with_suffix(".png")).convert("RGBA")
                alpha = image.getchannel("A")
                baseline = alpha.getextrema()[0]
                if 0 < baseline < 255:
                    alpha = alpha.point(lambda value: max(
                        0, round((value - baseline) * 255 / (255 - baseline))))
                    image.putalpha(alpha)
                draw = ImageDraw.Draw(image, "RGBA")
                label_font = ImageFont.truetype(a.font, max(18, a.width // 34))
                box = draw.textbbox((0, 0), label, font=label_font)
                draw.rounded_rectangle((18, 16, box[2] + 42, box[3] + 34),
                                       radius=10, fill=(248, 249, 252, 218))
                draw.text((30, 22), label, font=label_font, fill=(20, 25, 35, 255))
                image.save(stem.with_suffix(".png"))
            cards.append((stem.with_suffix('.png').name, label, stem.with_suffix('.pov').name))
    body = "\n".join(f'<figure><a href="{html.escape(pov)}"><img src="{html.escape(png)}"></a><figcaption>{html.escape(label)}</figcaption></figure>' for png,label,pov in cards)
    (a.out_dir / "index.html").write_text(f'''<!doctype html><meta charset="utf-8"><title>POV-Ray style catalog</title><style>body{{font:16px system-ui;background:#f3f4f7;margin:2rem}}h1{{margin-bottom:.2rem}}main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:1rem}}figure{{margin:0;background:white;padding:.7rem;border-radius:10px;box-shadow:0 2px 12px #0002}}img{{width:100%;height:auto}}figcaption{{padding:.5rem .2rem}}</style><h1>POV-Ray molecular style catalog</h1><p>Source: {html.escape(str(a.input))}. Click an image for its reproducible POV source.</p><main>{body}</main>''')
    print(f"Rendered {len(cards)} catalog entries -> {a.out_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
