#!/usr/bin/env python3
"""Render a certified liquid box with POV-Ray.

Used to eyeball the boxes that back the dH_vap validation
(``scripts/build_des_validation_boxes.sh``) before committing GPU hours to MD on
them. A box that packed badly -- a void, an interpenetrating pair, a molecule
outside the cell -- is obvious in a picture and nearly invisible in a density
number, because Packmol reports the density it was *asked* for.

Camera is always orthographic (house style for liquid boxes). The periodic cell
is drawn when present on the structure, or when ``--box-side`` / a sibling
``box.json`` supplies it — CHARMM stage PDBs often omit ``CRYST1``.

POV-Ray must be on PATH (``brew install povray`` / ``apt install povray``).
Without it this still writes the ``.pov``/``.ini`` pair, which renders anywhere.

Example::

    python scripts/render_liquid_box_povray.py boxes/tip3/model.pdb \\
        -o docs/images/des-so3lr-dimers/box_tip3.png --label "TIP3  298 K"
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from ase.io import read, write

# Element colours tuned for a light background; ASE's jmol defaults render
# hydrogen pure white, which disappears entirely.
COLORS = {
    "H": (0.85, 0.85, 0.88),
    "C": (0.35, 0.35, 0.38),
    "N": (0.20, 0.35, 0.85),
    "O": (0.85, 0.20, 0.18),
    "S": (0.90, 0.78, 0.20),
    "Cl": (0.30, 0.75, 0.30),
    "Ar": (0.45, 0.75, 0.80),
    "He": (0.70, 0.85, 0.90),
    "Ne": (0.55, 0.80, 0.85),
    "Kr": (0.40, 0.70, 0.78),
    "Xe": (0.35, 0.60, 0.72),
}
RADII = {"H": 0.30, "C": 0.60, "N": 0.58, "O": 0.55, "S": 0.85, "Cl": 0.85}


def _atoms_have_cell(atoms) -> bool:
    cell = getattr(atoms, "cell", None)
    return bool(cell is not None and getattr(cell, "rank", 0) == 3)


def resolve_box_side_A(
    structure: Path,
    *,
    box_side: float | None = None,
    box_json: Path | None = None,
) -> float | None:
    """Cubic box side (Å) from CLI, ``box.json``, or neither."""
    if box_side is not None:
        side = float(box_side)
        if side <= 0:
            raise ValueError(f"--box-side must be positive, got {side}")
        return side
    candidates: list[Path] = []
    if box_json is not None:
        candidates.append(Path(box_json))
    parent = Path(structure).resolve().parent
    candidates.extend([parent / "box.json", parent.parent / "box.json"])
    for path in candidates:
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        side = data.get("box_side_A")
        if side is None:
            continue
        side_f = float(side)
        if side_f <= 0:
            raise ValueError(f"{path}: box_side_A must be positive, got {side_f}")
        return side_f
    return None


def attach_cell_if_needed(atoms, side_A: float | None):
    """Set a cubic cell when atoms lack one and ``side_A`` is known.

    Packmol / CHARMM stage PDBs are usually centered near the origin with no
    ``CRYST1``. ASE draws the unit cell from the coordinate origin along the
    cell vectors, so attaching ``[L,L,L]`` without shifting leaves the wireframe
    sitting in the positive octant while the liquid sits around zero. Shift the
    atom centroid to ``L/2`` so the cube encloses the liquid.
    """
    if _atoms_have_cell(atoms) or side_A is None:
        return atoms, False
    atoms = atoms.copy()
    pos = np.asarray(atoms.get_positions(), dtype=np.float64)
    atoms.set_positions(pos - pos.mean(axis=0) + 0.5 * float(side_A))
    atoms.set_cell([side_A, side_A, side_A])
    atoms.set_pbc(True)
    return atoms, True


def write_liquid_box_pov(
    atoms,
    pov_path: Path,
    *,
    width: int = 1200,
    rotation: str = "-70x, 10y, 0z",
    colors=None,
    radii=None,
) -> bool:
    """Write ASE POV-Ray inputs. Returns whether the unit cell is drawn."""
    has_cell = _atoms_have_cell(atoms)
    # `write` emits <stem>.pov and <stem>.ini; the renderer consumes the .ini.
    write(
        str(pov_path),
        atoms,
        format="pov",
        radii=radii,
        colors=colors,
        rotation=rotation,
        # 2 = draw cell and fit bbox to atoms+cell; 0 = omit when no lattice.
        show_unit_cell=2 if has_cell else 0,
        povray_settings=dict(
            # ASE raises "Can't set *both* width and height!" -- it derives the
            # aspect ratio from the projected geometry and refuses to be
            # over-constrained, so only the width is passed.
            canvas_width=width,
            background="White",
            transparent=False,
            display=False,
            # Orthographic is house style for liquid/crystal boxes (no foreshortening).
            camera_type="orthographic",
            # Cell edges make packing faults and stray molecules obvious.
            celllinewidth=0.05 if has_cell else 0.0,
            bondlinewidth=0.0,
        ),
    )
    return has_cell


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("structure", type=Path, help="PDB/CRD/XYZ of the box")
    ap.add_argument("-o", "--output", type=Path, required=True, help="PNG to write")
    ap.add_argument("--width", type=int, default=1200)
    ap.add_argument("--height", type=int, default=1200)
    ap.add_argument("--rotation", default="-70x, 10y, 0z",
                    help="ASE rotation string for the camera")
    ap.add_argument("--label", default=None, help="unused by POV-Ray; kept for provenance")
    ap.add_argument("--elements-from", type=Path, default=None,
                    help="PDB with a real element column to take symbols from, "
                         "matched by atom order (for CHARMM PDBs, which omit it)")
    ap.add_argument("--box-side", type=float, default=None,
                    help="cubic box side in Å when the structure has no CRYST1/cell")
    ap.add_argument("--box-json", type=Path, default=None,
                    help="box.json with box_side_A (default: next to the structure)")
    ap.add_argument("--keep-pov", action="store_true", help="do not delete the .pov/.ini")
    a = ap.parse_args(argv)

    atoms = read(a.structure)
    n = len(atoms)
    syms = atoms.get_chemical_symbols()

    # CHARMM writes PDBs with the element column (77-78) blank -- the trailing
    # field is the segid. ASE then falls back to the atom NAME, so CHARMM's `OG`
    # hydroxyl oxygen is read as Og (oganesson) and `HG1` as Hg (mercury). The
    # geometry is unaffected but every element-derived property is wrong, and it
    # fails silently: the render just comes out grey. Elements are therefore
    # taken from a file that does carry the column, matched by atom order.
    if a.elements_from is not None:
        ref = read(a.elements_from)
        if len(ref) != n:
            print(f"  ERROR: --elements-from has {len(ref)} atoms, structure has {n}")
            return 1
        syms = ref.get_chemical_symbols()
        atoms.set_chemical_symbols(syms)
        print(f"  elements taken from {a.elements_from.name}")

    IMPLAUSIBLE = {"Og", "Hg", "Cn", "Ts", "Nh", "Mc", "Lv", "Fl"}
    bad = IMPLAUSIBLE & set(syms)
    if bad:
        print(f"  ERROR: implausible elements {sorted(bad)} -- the element column is "
              f"probably blank and the atom names were parsed instead.\n"
              f"         Pass --elements-from <a PDB with an element column>.")
        return 1

    try:
        side = resolve_box_side_A(
            a.structure, box_side=a.box_side, box_json=a.box_json
        )
    except (ValueError, json.JSONDecodeError, OSError) as exc:
        print(f"  ERROR: could not resolve box side: {exc}")
        return 1
    atoms, cell_attached = attach_cell_if_needed(atoms, side)

    uniq = sorted(set(syms))
    print(f"{a.structure.name}: {n} atoms, elements {uniq}")
    if _atoms_have_cell(atoms):
        L = atoms.cell.lengths()
        src = "from structure" if not cell_attached else (
            f"from --box-side/{a.box_json.name if a.box_json else 'box.json'}"
        )
        print(f"  cell: {L[0]:.2f} x {L[1]:.2f} x {L[2]:.2f} A ({src})")
    else:
        print("  cell: none (not drawn)")

    colors = np.array([COLORS.get(s, (0.6, 0.6, 0.6)) for s in syms])
    radii = np.array([RADII.get(s, 0.6) for s in syms])

    a.output.parent.mkdir(parents=True, exist_ok=True)
    stem = a.output.with_suffix("")
    pov_path = stem.with_suffix(".pov")

    write_liquid_box_pov(
        atoms,
        pov_path,
        width=a.width,
        rotation=a.rotation,
        colors=colors,
        radii=radii,
    )
    print(f"  wrote {pov_path.name} + {stem.with_suffix('.ini').name}")

    exe = shutil.which("povray")
    if exe is None:
        print("  POV-Ray not on PATH -- .pov/.ini written; render them elsewhere")
        return 0

    ini = stem.with_suffix(".ini")
    # POV-Ray runs with cwd set to the .ini's directory (it resolves the .pov it
    # references relatively), so it must be handed the bare filename. Passing the
    # path as given fails from that cwd, and POV-Ray reports it as
    # "Failed to parse command-line option" rather than as a missing file.
    proc = subprocess.run(
        [exe, ini.name], capture_output=True, text=True, cwd=str(stem.parent)
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-6:]
        print(f"  POV-Ray failed (rc={proc.returncode}):")
        for line in tail:
            print(f"    {line}")
        return 1

    if not a.output.exists():
        print(f"  POV-Ray reported success but {a.output} is missing")
        return 1
    print(f"  rendered {a.output} ({a.output.stat().st_size // 1024} KB)")

    if not a.keep_pov:
        for p in (pov_path, ini):
            p.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
