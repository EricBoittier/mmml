#!/usr/bin/env python3
"""Export one NH3 + CH3Cl geometry as a strictly formatted CGenFF PDB.

``examples/m/07_export_solute_pdb.py`` writes a whitespace-delimited layout so
that the five-character residue name ``CH3CL`` fits. CHARMM tolerates it, but
the coordinate columns end up shifted by one, so strict readers reject the file
-- ``mmml make-box`` fails in ``ase.io.read`` with "Invalid or missing
coordinate(s)". Here the chloromethane residue is named ``MECL`` (four
characters, see ``top_mecl.rtf``) and every field sits in its standard PDB
column, so ASE, Packmol and CHARMM all read the same file.

Atom order here is NOT the canonical ML order. CHARMM builds a PSF by reading
the sequence from the PDB, which requires each residue's atoms to be contiguous;
the canonical order (Cl, N, C, H(N)x3, H(C)x3) interleaves AMM1 and MECL, and
Packmol then renumbers the split residue (AMM1 came out as resid 9999). So the
file is written residue-grouped -- AMM1 then MECL, each in RTF atom order -- and
the permutation back to canonical order is written alongside it as JSON.

Consequence for the reaction coordinate: in this ordering the CV atoms are
C1 = 4, CL1 = 5, N1 = 0, so xi = r(C-Cl) - r(C-N) is ``--cv-difference 4,5,4,0``
here, versus ``2,0,2,1`` for the canonical window seeds.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parent.parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

# Canonical-order helpers live in the seeding script (module name starts with a
# digit, so it cannot be imported with a plain ``import`` statement).
_mod = __import__("01_seed_windows")

# Residue-grouped PDB order: (atom name, residue, resid, element, canonical index).
# Canonical order is Cl=0, N=1, C=2, H(N)=3,4,5, H(C)=6,7,8.
_LAYOUT = [
    ("N1", "AMM1", 1, "N", 1),
    ("H11", "AMM1", 1, "H", 3),
    ("H12", "AMM1", 1, "H", 4),
    ("H13", "AMM1", 1, "H", 5),
    ("C1", "MECL", 2, "C", 2),
    ("CL1", "MECL", 2, "CL", 0),
    ("H11", "MECL", 2, "H", 6),
    ("H12", "MECL", 2, "H", 7),
    ("H13", "MECL", 2, "H", 8),
]
# Index of each CV atom in PDB/PSF order.
_PDB_IDX = {name: i for i, (name, *_rest) in enumerate(
    [(l[0] + l[1], *l[1:]) for l in _LAYOUT])}
CV_DIFFERENCE_PDB_ORDER = "4,5,4,0"  # xi = r(C1-CL1) - r(C1-N1)


def pdb_atom_line(serial, name, resname, resid, xyz, element):
    """One strictly column-aligned PDB ATOM record.

    Columns (1-based): 1-6 record, 7-11 serial, 13-16 name, 18-21 resName,
    22 chainID, 23-26 resSeq, 31-38 x, 39-46 y, 47-54 z, 55-60 occupancy,
    61-66 tempFactor, 77-78 element.
    """
    x, y, z = (float(v) for v in xyz)
    # Atom names of fewer than four characters start in column 14 by convention.
    name_field = f" {name:<3s}" if len(name) < 4 else name[:4]
    return (
        f"ATOM  {serial:5d} {name_field}"
        f" {resname:<4s}"
        f"{'A':1s}{resid:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}"
        f"{1.0:6.2f}{0.0:6.2f}"
        f"{'':10s}{element:>2s}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scan", type=Path, default=REPO_ROOT / "examples/m/scan_nh3_ch3cl.npz"
    )
    parser.add_argument(
        "--xi",
        type=float,
        default=0.0,
        help="Pick the scan frame nearest this xi = r(C-Cl) - r(C-N) (A). "
        "Default 0.0 builds the box around a transition-state-like solute so "
        "the solvent shell equilibrates around the right charge distribution.",
    )
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args()

    artifacts = Path(
        os.environ.get("MENSH_ARTIFACTS", REPO_ROOT / "artifacts/menshutkin")
    )
    out = args.output or artifacts / "solute_amm1_mecl.pdb"

    z_all, r_all, xi = _mod.load_scan(args.scan)
    idx = int(np.argmin(np.abs(xi - args.xi)))
    z, r = z_all[idx], r_all[idx]

    # Centre so Packmol places the solute at the box centre without a jump.
    from ase.data import atomic_masses

    m = atomic_masses[z.astype(int)]
    r = r - (m[:, None] * r).sum(axis=0) / m.sum()

    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "REMARK  NH3 + CH3Cl solute for the Menshutkin campaign",
        f"REMARK  scan frame {idx}, xi = {xi[idx]:+.3f} A "
        f"(r(C-Cl) = {np.linalg.norm(r[0] - r[2]):.3f}, "
        f"r(C-N) = {np.linalg.norm(r[2] - r[1]):.3f})",
    ]
    prev_resid = None
    serial = 0
    for name, resname, resid, element, canon in _LAYOUT:
        if prev_resid is not None and resid != prev_resid:
            lines.append("TER")
        serial += 1
        lines.append(pdb_atom_line(serial, name, resname, resid, r[canon], element))
        prev_resid = resid
    lines += ["TER", "END"]
    out.write_text("\n".join(lines) + "\n")

    # Permutation so downstream code can move between PDB/PSF order and the
    # canonical ML order without re-deriving it from element symbols.
    import json

    pdb_to_canonical = [canon for *_x, canon in _LAYOUT]
    canonical_to_pdb = [0] * len(pdb_to_canonical)
    for pdb_i, canon_i in enumerate(pdb_to_canonical):
        canonical_to_pdb[canon_i] = pdb_i
    (out.with_suffix(".json")).write_text(json.dumps({
        "pdb_order": [f"{l[1]}:{l[0]}" for l in _LAYOUT],
        "pdb_to_canonical": pdb_to_canonical,
        "canonical_to_pdb": canonical_to_pdb,
        "cv_difference_pdb_order": CV_DIFFERENCE_PDB_ORDER,
        "cv_difference_canonical_order": "2,0,2,1",
        "xi_A": float(xi[idx]),
        "scan_frame": int(idx),
    }, indent=2) + "\n")

    print(f"scan frame {idx}: xi = {xi[idx]:+.3f} A (target {args.xi:+.3f})")
    print(f"Wrote {out}")

    # Fail here rather than three steps later inside make-box.
    from ase.io import read

    atoms = read(str(out))
    got = atoms.get_chemical_symbols()
    want = ["N", "H", "H", "H", "C", "Cl", "H", "H", "H"]
    if got != want:
        print(f"FAIL: ASE read back {got}, expected {want}", file=sys.stderr)
        return 1
    if not np.allclose(atoms.get_positions(), r[pdb_to_canonical], atol=1e-3):
        print("FAIL: ASE round-trip changed the coordinates", file=sys.stderr)
        return 1
    print(f"Wrote {out.with_suffix('.json')} (atom-order mapping)")
    print(f"PASS: ASE round-trip OK ({len(atoms)} atoms, {got})")
    print(f"      CV in PDB/PSF order: --cv-difference {CV_DIFFERENCE_PDB_ORDER}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
