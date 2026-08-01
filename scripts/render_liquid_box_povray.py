#!/usr/bin/env python3
"""Render a certified liquid box with POV-Ray.

Used to eyeball the boxes that back the dH_vap validation
(``scripts/build_des_validation_boxes.sh``) before committing GPU hours to MD on
them. A box that packed badly -- a void, an interpenetrating pair, a molecule
outside the cell -- is obvious in a picture and nearly invisible in a density
number, because Packmol reports the density it was *asked* for.

POV-Ray must be on PATH (``brew install povray`` / ``apt install povray``).
Without it this still writes the ``.pov``/``.ini`` pair, which renders anywhere.

Example::

    python scripts/render_liquid_box_povray.py boxes/tip3/model.pdb \\
        -o docs/images/des-so3lr-dimers/box_tip3.png --label "TIP3  298 K"
"""

from __future__ import annotations

import argparse
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

    uniq = sorted(set(syms))
    print(f"{a.structure.name}: {n} atoms, elements {uniq}")
    if atoms.cell is not None and atoms.cell.rank == 3:
        L = atoms.cell.lengths()
        print(f"  cell: {L[0]:.2f} x {L[1]:.2f} x {L[2]:.2f} A")

    colors = np.array([COLORS.get(s, (0.6, 0.6, 0.6)) for s in syms])
    radii = np.array([RADII.get(s, 0.6) for s in syms])

    a.output.parent.mkdir(parents=True, exist_ok=True)
    stem = a.output.with_suffix("")
    pov_path = stem.with_suffix(".pov")

    # `write` emits <stem>.pov and <stem>.ini; the renderer consumes the .ini.
    write(
        str(pov_path),
        atoms,
        format="pov",
        radii=radii,
        colors=colors,
        rotation=a.rotation,
        povray_settings=dict(
            # ASE raises "Can't set *both* width and height!" -- it derives the
            # aspect ratio from the projected geometry and refuses to be
            # over-constrained, so only the width is passed.
            canvas_width=a.width,
            background="White",
            transparent=False,
            display=False,
            camera_type="perspective",
            # Cell edges make packing faults and stray molecules obvious.
            celllinewidth=0.05,
            bondlinewidth=0.0,
        ),
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
