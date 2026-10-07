#!/usr/bin/env python3
"""POV-Ray figures for ``docs/md-cg-capabilities-checklist.md``.

Renders real bundled geometries (trialanine snapshot, acetone unit cell) and
small schematic scenes (cubic cell, minimum-image wrap, rigid-body move).
No CHARMM build and no dynamics.

    uv run python scripts/render_md_cg_capability_figures.py

POV-Ray is resolved from ``PATH`` or ``~/.local/share/karml-povray``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.data import covalent_radii
from ase.io import read, write
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "images" / "md-cg"

ELEM = {
    "H": (0.82, 0.84, 0.88),
    "C": (0.28, 0.30, 0.34),
    "N": (0.18, 0.36, 0.86),
    "O": (0.86, 0.18, 0.16),
}
RAD = {"H": 0.28, "C": 0.52, "N": 0.50, "O": 0.48}

ROLE = {
    "core": (0.05, 0.55, 0.40),
    "near": (0.16, 0.38, 0.88),
    "far": (0.90, 0.52, 0.12),
}
NEAR_A = 8.0
CELL_SIDE_A = 24.0


def _povray() -> tuple[Path, Path]:
    """Resolve a POV-Ray whose orthographic camera parses.

    The conda 3.7.0.10 build under ``karml-povray`` rejects a normal
    orthographic camera ("viewing angle has to be smaller than 180
    degrees"). Prefer the 3.7.0.8 tree that accepts ASE's camera block.
    """
    candidates = []
    env = shutil.os.environ.get("POVRAY")
    if env:
        candidates.append(Path(env))
    candidates.append(Path.home() / ".local" / "share" / "povray-working" / "bin" / "povray")
    found = shutil.which("povray")
    if found:
        candidates.append(Path(found))
    candidates.append(Path.home() / ".local" / "share" / "karml-povray" / "bin" / "povray")
    binary = next((path for path in candidates if path.is_file()), None)
    if binary is None:
        raise SystemExit("POV-Ray not found (set POVRAY, or install povray on PATH)")
    include = binary.resolve().parents[1] / "share" / "povray-3.7" / "include"
    if not (include / "colors.inc").is_file():
        raise SystemExit(f"POV-Ray includes missing under {include}")
    return binary, include


def _font(size: int) -> ImageFont.FreeTypeFont:
    listed = subprocess.check_output(
        ["fc-match", "-f", "%{file}", "Noto Sans:style=Regular"], text=True
    ).strip()
    path = Path(listed)
    if not path.is_file():
        raise SystemExit(f"Noto Sans font not found ({listed})")
    return ImageFont.truetype(str(path), size=size)


def _caption(path: Path, title: str, subtitle: str) -> None:
    image = Image.open(path).convert("RGB")
    bar_h = 78
    canvas = Image.new("RGB", (image.width, image.height + bar_h), (255, 255, 255))
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.line([(0, image.height), (image.width, image.height)], fill=(220, 224, 230), width=1)
    draw.text((18, image.height + 10), title, font=_font(22), fill=(22, 26, 32))
    draw.text((18, image.height + 42), subtitle, font=_font(16), fill=(70, 78, 90))
    canvas.save(path)


def _legend(path: Path, entries: list[tuple[str, tuple[float, float, float]]]) -> None:
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    font = _font(18)
    pad, row_h, swatch = 14, 28, 14
    width = 280
    height = pad * 2 + row_h * len(entries)
    x0, y0 = 16, 16
    draw.rounded_rectangle(
        (x0, y0, x0 + width, y0 + height), radius=10, fill=(255, 255, 255, 230)
    )
    for i, (label, rgb) in enumerate(entries):
        y = y0 + pad + i * row_h
        color = tuple(int(255 * c) for c in rgb)
        draw.ellipse((x0 + 12, y, x0 + 12 + swatch, y + swatch), fill=color)
        draw.text((x0 + 36, y - 2), label, font=font, fill=(22, 26, 32, 255))
    image.save(path)


def _render_pov(pov_text: str, png: Path, *, width: int, height: int) -> None:
    binary, include = _povray()
    png.parent.mkdir(parents=True, exist_ok=True)
    pov = png.with_suffix(".pov")
    ini = png.with_suffix(".ini")
    pov.write_text(pov_text)
    ini.write_text(
        "\n".join(
            [
                f'Input_File_Name="{pov.name}"',
                f'Output_File_Name="{png.name}"',
                f"Width={width}",
                f"Height={height}",
                "Antialias=On",
                "Antialias_Threshold=0.05",
                "Display=Off",
                "Quality=9",
                "",
            ]
        )
    )
    proc = subprocess.run(
        [str(binary), f"+L{include}", ini.name],
        cwd=png.parent,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not png.is_file():
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-12:]
        raise SystemExit("POV-Ray failed:\n" + "\n".join(tail))
    pov.unlink(missing_ok=True)
    ini.unlink(missing_ok=True)


def _vec(v) -> str:
    return "<" + ",".join(f"{float(x):.5f}" for x in v) + ">"


def _header(camera, look, *, span: float, width: int, height: int) -> list[str]:
    aspect = height / width
    return [
        "#version 3.7;",
        "global_settings { assumed_gamma 1.0 }",
        "background { color rgb <1,1,1> }",
        (
            f"camera {{ orthographic location {_vec(camera)} look_at {_vec(look)} "
            f"right x*{span:.4f} up y*{span * aspect:.4f} }}"
        ),
        f"light_source {{ {_vec(camera + np.array([-span, span, -span]))} color rgb <1,1,1> "
        "area_light <1.2,0,0>, <0,1.2,0>, 5, 5 adaptive 1 circular }",
        f"light_source {{ {_vec(np.array([span, -0.4 * span, span]))} color rgb <0.35,0.40,0.55> }}",
    ]


def _sphere(p, radius, rgb, *, transmit: float = 0.0, finish: str = "phong 0.6 phong_size 40") -> str:
    pigment = (
        f"color rgbt <{rgb[0]:.3f},{rgb[1]:.3f},{rgb[2]:.3f},{transmit:.3f}>"
        if transmit else f"color rgb {_vec(rgb)}"
    )
    return (
        f"sphere {{ {_vec(p)}, {radius:.4f} texture {{ pigment {{ {pigment} }} "
        f"finish {{ {finish} }} }} }}"
    )


def _cyl(a, b, radius, rgb, *, transmit: float = 0.0) -> str:
    pigment = (
        f"color rgbt <{rgb[0]:.3f},{rgb[1]:.3f},{rgb[2]:.3f},{transmit:.3f}>"
        if transmit else f"color rgb {_vec(rgb)}"
    )
    return (
        f"cylinder {{ {_vec(a)}, {_vec(b)}, {radius:.4f} pigment {{ {pigment} }} "
        "finish { phong 0.3 } }"
    )


def _arrow(start, end, rgb, radius=0.06) -> list[str]:
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 1e-8:
        return []
    unit = direction / length
    head = min(0.28, 0.28 * length)
    neck = end - unit * head
    return [
        _cyl(start, neck, radius, rgb),
        f"cone {{ {_vec(neck)}, {radius * 2.4:.4f}, {_vec(end)}, 0 pigment {{ color rgb {_vec(rgb)} }} }}",
    ]


def _cell_edges(origin, axes, radius, rgb) -> list[str]:
    """Twelve edges of the parallelepiped spanned by ``axes`` from ``origin``."""
    origin = np.asarray(origin, dtype=float)
    axes = [np.asarray(v, dtype=float) for v in axes]
    lines = []
    for j in (0, 1):
        for k in (0, 1):
            start = origin + j * axes[1] + k * axes[2]
            lines.append(_cyl(start, start + axes[0], radius, rgb))
    for i in (0, 1):
        for k in (0, 1):
            start = origin + i * axes[0] + k * axes[2]
            lines.append(_cyl(start, start + axes[1], radius, rgb))
    for i in (0, 1):
        for j in (0, 1):
            start = origin + i * axes[0] + j * axes[1]
            lines.append(_cyl(start, start + axes[2], radius, rgb))
    return lines


def _intramolecular_bonds(numbers, positions, groups: list[np.ndarray]) -> list[tuple[int, int]]:
    """Covalent pairs as ``(i, j)``.

    ASE's POV writer treats a third tuple entry as a fractional cell offset
    (``offset @ cell``), not as a colour. Passing an RGB triple there draws
    the stick across the unit cell.
    """
    bonds = []
    for idx in groups:
        idx = np.asarray(idx, dtype=int)
        for a_i, i in enumerate(idx):
            for j in idx[a_i + 1 :]:
                dist = float(np.linalg.norm(positions[i] - positions[j]))
                limit = 1.15 * (covalent_radii[numbers[i]] + covalent_radii[numbers[j]])
                if dist < min(limit, 1.7):
                    bonds.append((int(i), int(j)))
    return bonds


def _groups_by_residue(atoms) -> list[np.ndarray]:
    res = np.asarray(atoms.arrays["residuenumbers"])
    return [np.flatnonzero(res == r) for r in sorted(set(res.tolist()))]


def _render_atoms(
    atoms: Atoms,
    png: Path,
    *,
    colors: np.ndarray,
    radii: np.ndarray,
    bonds: list,
    rotation: str,
    width: int = 1200,
    show_cell: bool = True,
) -> None:
    pov = png.with_suffix(".pov")
    png.parent.mkdir(parents=True, exist_ok=True)
    has_cell = bool(atoms.cell is not None and atoms.cell.rank == 3 and show_cell)
    write(
        str(pov),
        atoms,
        format="pov",
        radii=radii,
        colors=np.asarray(colors, dtype=float),
        rotation=rotation,
        show_unit_cell=2 if has_cell else 0,
        povray_settings=dict(
            canvas_width=width,
            background="White",
            transparent=False,
            display=False,
            camera_type="orthographic",
            celllinewidth=0.055 if has_cell else 0.0,
            bondlinewidth=0.12,
            bondatoms=bonds,
        ),
    )
    ini = png.with_suffix(".ini")
    ini.write_text(ini.read_text().replace("Pause_When_Done=True", "Pause_When_Done=False"))
    binary, include = _povray()
    ini = png.with_suffix(".ini")
    proc = subprocess.run(
        [str(binary), f"+L{include}", ini.name],
        cwd=png.parent,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not png.is_file():
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-12:]
        raise SystemExit(f"POV-Ray failed for {png.name}:\n" + "\n".join(tail))
    pov.unlink(missing_ok=True)
    ini.unlink(missing_ok=True)


def _element_style(atoms: Atoms) -> tuple[np.ndarray, np.ndarray]:
    symbols = atoms.get_chemical_symbols()
    colors = np.array([ELEM.get(s, (0.6, 0.6, 0.6)) for s in symbols])
    radii = np.array([RAD.get(s, 0.5) for s in symbols])
    return colors, radii


def _charmm_symbols(atoms: Atoms) -> None:
    """CHARMM PDBs leave the element column blank; recover it from atom names."""
    names = [str(n).strip() for n in atoms.arrays["atomtypes"]]
    symbols = []
    for name in names:
        if name[:2].upper() in {"CL", "BR"}:
            symbols.append(name[:2].title() if name[:2].upper() == "CL" else "Br")
            continue
        symbols.append({"C": "C", "H": "H", "N": "N", "O": "O", "S": "S"}.get(name[0].upper(), "C"))
    atoms.set_chemical_symbols(symbols)


def figure_peptide_and_mixed() -> dict[str, int]:
    full = read(REPO / "examples" / "atoms.pdb")
    n_core = 42
    pos = np.asarray(full.positions, dtype=float)
    numbers = np.asarray(full.numbers)
    symbols = full.get_chemical_symbols()

    peptide = Atoms(symbols=symbols[:n_core], positions=pos[:n_core])
    peptide.center()
    p_colors, p_radii = _element_style(peptide)
    p_bonds = _intramolecular_bonds(peptide.numbers, peptide.positions, [np.arange(n_core)])
    peptide_png = OUT / "peptide.png"
    _render_atoms(
        peptide, peptide_png, colors=p_colors, radii=p_radii, bonds=p_bonds,
        rotation="18x,22y,0z", width=1100, show_cell=False,
    )
    _caption(
        peptide_png,
        "Trialanine core (42 atoms, examples/atoms.pdb)",
        "ML intramolecular energy is evaluated on this fragment alone.",
    )

    # Recenter the snapshot in a cube and keep molecules that sit fully inside it.
    side = CELL_SIDE_A
    origin_shift = 0.5 * side - pos[:n_core].mean(axis=0)
    shifted = pos + origin_shift
    keep: list[int] = list(range(n_core))
    role = np.zeros(len(full), dtype=int)
    n_near = n_far = 0
    for oxygen in range(n_core, len(full), 3):
        group = shifted[oxygen : oxygen + 3]
        if np.any(group < 0.4) or np.any(group > side - 0.4):
            continue
        dist = float(np.linalg.norm(shifted[oxygen] - shifted[:n_core].mean(axis=0)))
        kind = 1 if dist <= NEAR_A else 2
        if kind == 1:
            n_near += 1
        else:
            n_far += 1
        role[oxygen : oxygen + 3] = kind
        keep.extend([oxygen, oxygen + 1, oxygen + 2])
    cropped = Atoms(
        numbers=numbers[keep],
        positions=shifted[keep],
        cell=[side, side, side],
        pbc=True,
    )
    role_keep = role[keep]
    palette = np.array([ROLE["core"], ROLE["near"], ROLE["far"]])
    colors = palette[role_keep]
    radii = np.array([RAD.get(symbols[i], 0.5) for i in keep])
    groups = [np.flatnonzero(np.arange(len(keep)) < n_core)]
    # water triples follow the peptide block, in keep-order
    cursor = n_core
    while cursor < len(keep):
        groups.append(np.arange(cursor, cursor + 3))
        cursor += 3
    bonds = _intramolecular_bonds(cropped.numbers, cropped.positions, groups)
    mixed = OUT / "mixed-cell.png"
    _render_atoms(
        cropped, mixed, colors=colors, radii=radii, bonds=bonds,
        rotation="-62x,18y,0z", width=1280, show_cell=True,
    )
    _legend(
        mixed,
        [
            ("ML core  ·  ml_intra", ROLE["core"]),
            (f"ML shell, under {NEAR_A:.0f} A  ·  ml_pep_water", ROLE["near"]),
            ("MM bulk  ·  mm_nonbonded + vdw_core", ROLE["far"]),
        ],
    )
    _caption(
        mixed,
        f"Mixed system inside a {side:.0f} Å cubic cell",
        f"Crop of examples/atoms.pdb: peptide + {n_near} near waters + {n_far} far waters. Wireframe is the periodic cell.",
    )

    # Full snapshot, role-colored, no invented cell.
    all_role = np.zeros(len(full), dtype=int)
    com = pos[:n_core].mean(axis=0)
    n_near_all = n_far_all = 0
    for oxygen in range(n_core, len(full), 3):
        dist = float(np.linalg.norm(pos[oxygen] - com))
        kind = 1 if dist <= NEAR_A else 2
        all_role[oxygen : oxygen + 3] = kind
        if kind == 1:
            n_near_all += 1
        else:
            n_far_all += 1
    overview_atoms = full.copy()
    overview_atoms.center()
    groups = [np.arange(n_core)]
    for oxygen in range(n_core, len(full), 3):
        groups.append(np.arange(oxygen, oxygen + 3))
    overview = OUT / "mixed-overview.png"
    _render_atoms(
        overview_atoms,
        overview,
        colors=palette[all_role],
        radii=np.array([RAD.get(s, 0.5) for s in symbols]),
        bonds=_intramolecular_bonds(overview_atoms.numbers, overview_atoms.positions, groups),
        rotation="-58x,24y,0z",
        width=1280,
        show_cell=False,
    )
    _legend(
        overview,
        [
            ("ML core", ROLE["core"]),
            (f"ML shell, under {NEAR_A:.0f} A  ({n_near_all} waters)", ROLE["near"]),
            (f"MM bulk  ({n_far_all} waters)", ROLE["far"]),
        ],
    )
    _caption(
        overview,
        "Same coloring on the full 200-water snapshot",
        "The ML shell is a thin layer around the peptide. Bulk water stays on the classical term.",
    )
    return {"near_in_cell": n_near, "far_in_cell": n_far, "near_all": n_near_all, "far_all": n_far_all}


def figure_acetone_cell() -> None:
    crystal = read(REPO / "karml/data/structures/acetone_pbca_150k_cod7110464.cif")
    crystal.wrap()
    colors, radii = _element_style(crystal)
    # Bond only covalent contacts. The CIF cell is orthorhombic Pbca.
    pos = crystal.positions
    numbers = crystal.numbers
    bonds = []
    for i in range(len(crystal)):
        for j in range(i + 1, len(crystal)):
            dist = float(np.linalg.norm(pos[i] - pos[j]))
            limit = 1.15 * (covalent_radii[numbers[i]] + covalent_radii[numbers[j]])
            if dist < min(limit, 1.7):
                bonds.append((i, j))
    png = OUT / "acetone-cell.png"
    _render_atoms(
        crystal, png, colors=colors, radii=radii, bonds=bonds,
        rotation="-72x,8y,12z", width=1280, show_cell=True,
    )
    lengths = crystal.cell.lengths()
    _caption(
        png,
        "Acetone crystal unit cell (Pbca, 150 K, COD 7110464)",
        f"Orthorhombic cell  {lengths[0]:.2f} × {lengths[1]:.2f} × {lengths[2]:.2f} Å.  The wireframe is one lattice repeat.",
    )


def _water() -> Atoms:
    water = molecule("H2O")
    water.center(vacuum=0.0)
    return water


def figure_schematic_cell() -> None:
    """Cubic cell, a few waters, and one periodic image outside the +a face."""
    side = 10.0
    water = _water()
    rng = np.random.default_rng(7)
    pieces = []
    # 2×2×2 grid, inset from the faces so every molecule is wholly inside.
    grid = [2.6, 7.4]
    for x in grid:
        for y in grid:
            for z in grid:
                mol = water.copy()
                mol.rotate(float(rng.uniform(0, 180)), "y")
                mol.rotate(float(rng.uniform(0, 180)), "x")
                mol.translate([x, y, z])
                pieces.append(mol)
    inside = pieces[0]
    for mol in pieces[1:]:
        inside += mol
    # Ghost of the molecule nearest the +a face, translated by one cell vector.
    donor_src = max(pieces, key=lambda mol: float(mol.positions[:, 0].mean()))
    donor = donor_src.copy()
    donor.translate([side, 0.0, 0.0])

    # Bias the frame toward +a so the periodic image sits fully in view.
    span = 36.0
    look = np.array([8.5, side / 2, side / 2])
    camera = look + np.array([16.0, 11.0, -18.0])
    lines = _header(camera, look, span=span, width=1280, height=900)
    lines += _cell_edges((0, 0, 0), [(side, 0, 0), (0, side, 0), (0, 0, side)], 0.045, (0.25, 0.28, 0.32))
    lines += _arrow((0, 0, 0), (3.2, 0, 0), (0.75, 0.16, 0.18), 0.07)
    lines += _arrow((0, 0, 0), (0, 3.2, 0), (0.12, 0.55, 0.28), 0.07)
    lines += _arrow((0, 0, 0), (0, 0, 3.2), (0.16, 0.36, 0.82), 0.07)
    for symbol, xyz in zip(inside.get_chemical_symbols(), inside.positions):
        lines.append(_sphere(xyz, RAD[symbol], ELEM[symbol]))
    for i, j in _intramolecular_bonds(
        inside.numbers, inside.positions, [np.arange(3) + 3 * k for k in range(len(inside) // 3)]
    ):
        lines.append(_cyl(inside.positions[i], inside.positions[j], 0.07, (0.45, 0.47, 0.50)))
    for symbol, xyz in zip(donor.get_chemical_symbols(), donor.positions):
        lines.append(_sphere(xyz, RAD[symbol], ELEM[symbol], transmit=0.55))
    # Image vector from a real molecule to its ghost.
    src = donor_src.positions.mean(axis=0)
    dst = donor.positions.mean(axis=0)
    lines += _arrow(src, dst, (0.45, 0.48, 0.55), 0.05)
    png = OUT / "unit-cell.png"
    _render_pov("\n".join(lines) + "\n", png, width=1280, height=900)
    _caption(
        png,
        "One cubic cell and its periodic image",
        "Black edges are the cell. Red, green, and blue arrows are a, b, and c. The faded water is the +a image.",
    )


def figure_mic() -> None:
    """Direct vector vs minimum-image vector across one face."""
    side = 8.0
    # Partner sits just inside the left face; the other site sits just inside the right face.
    left = np.array([0.7, 4.0, 4.0])
    right = np.array([7.2, 4.3, 3.7])
    image = right - np.array([side, 0.0, 0.0])  # periodic copy on the −a side
    span = 20.0
    look = np.array([3.0, 4.0, 4.0])
    camera = look + np.array([12.0, 8.0, -14.0])
    lines = _header(camera, look, span=span, width=1280, height=860)
    lines += _cell_edges((0, 0, 0), [(side, 0, 0), (0, side, 0), (0, 0, side)], 0.04, (0.25, 0.28, 0.32))
    lines.append(_sphere(left, 0.38, (0.16, 0.38, 0.88)))
    lines.append(_sphere(right, 0.38, (0.86, 0.18, 0.16)))
    lines.append(_sphere(image, 0.38, (0.86, 0.18, 0.16), transmit=0.45))
    # Long in-box vector.
    lines += _arrow(left + np.array([0.15, 0.15, 0]), right + np.array([-0.15, 0.05, 0]), (0.75, 0.16, 0.18), 0.055)
    # Short minimum-image vector, to the periodic copy.
    lines += _arrow(left + np.array([-0.05, -0.2, 0]), image + np.array([0.45, -0.05, 0]), (0.12, 0.55, 0.28), 0.055)
    png = OUT / "mic-wrap.png"
    _render_pov("\n".join(lines) + "\n", png, width=1280, height=860)
    _caption(
        png,
        "Minimum image: the short periodic vector",
        "Red arrow stays inside the cell (long). Green arrow uses the faded image across the face (short). Pair lists use the green one.",
    )


def figure_rigid() -> None:
    """A rigid monomer: translation of the centre of mass plus a rotation about it."""
    water = _water()
    water.translate([0.0, 0.0, 0.0])
    moved = water.copy()
    com = water.positions.mean(axis=0)
    moved.rotate(38.0, "z", center=com)
    moved.translate([3.4, 0.6, 0.0])
    span = 8.2
    look = np.array([1.8, 0.3, 0.0])
    camera = look + np.array([0.4, 6.5, -9.0])
    lines = _header(camera, look, span=span, width=1280, height=860)
    # Ghost of the starting pose.
    for symbol, xyz in zip(water.get_chemical_symbols(), water.positions):
        lines.append(_sphere(xyz, RAD[symbol], (0.65, 0.68, 0.72), transmit=0.35))
    for i, j in ((0, 1), (0, 2)):
        lines.append(_cyl(water.positions[i], water.positions[j], 0.06, (0.55, 0.57, 0.60), transmit=0.35))
    for symbol, xyz in zip(moved.get_chemical_symbols(), moved.positions):
        lines.append(_sphere(xyz, RAD[symbol], ELEM[symbol]))
    for i, j in ((0, 1), (0, 2)):
        lines.append(_cyl(moved.positions[i], moved.positions[j], 0.07, (0.45, 0.47, 0.50)))
    new_com = moved.positions.mean(axis=0)
    lines.append(_sphere(com, 0.12, (0.15, 0.15, 0.18)))
    lines.append(_sphere(new_com, 0.12, (0.15, 0.15, 0.18)))
    lines += _arrow(com, new_com, (0.75, 0.16, 0.18), 0.055)
    # Rotation arc around the new COM, in the xy plane.
    arc = []
    for deg in range(-10, 50, 8):
        theta = np.deg2rad(deg)
        arc.append(new_com + 1.15 * np.array([np.cos(theta), np.sin(theta), 0.0]))
    for a, b in zip(arc, arc[1:]):
        lines.append(_cyl(a, b, 0.035, (0.16, 0.36, 0.82)))
    lines += _arrow(arc[-2], arc[-1] + 0.15 * (arc[-1] - arc[-2]), (0.16, 0.36, 0.82), 0.04)
    png = OUT / "rigid-move.png"
    _render_pov("\n".join(lines) + "\n", png, width=1280, height=860)
    _caption(
        png,
        "One rigid-body trial move",
        "Grey is the start. The monomer translates (red, centre of mass) and rotates (blue) as one piece. Bonds inside it do not change.",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    counts = figure_peptide_and_mixed()
    figure_acetone_cell()
    figure_schematic_cell()
    figure_mic()
    figure_rigid()
    print("wrote", OUT)
    print("counts", counts)


if __name__ == "__main__":
    main()
