#!/usr/bin/env python3
"""Create a frequency-scaled, collision-free molecule cloud for DES residues."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor

from mmml.data.cgenff_dataset import load_reference


def read_counts(path: Path, top: int) -> list[tuple[str, int]]:
    rows = []
    pattern = re.compile(r"^\| `([^`]+)` \| ([\d,]+) \|")
    for line in path.read_text().splitlines():
        match = pattern.match(line)
        if match:
            rows.append((match.group(1), int(match.group(2).replace(",", ""))))
    return rows[:top]


def _worst(row, side):
    if not row:
        return float("inf")
    total = sum(row)
    return max(side * side * max(row) / total**2, total**2 / (side * side * min(row)))


def treemap(values, x=0.0, y=0.0, width=1.0, height=1.0):
    values = list(values)
    rects = []
    while values:
        side = min(width, height)
        row = [values.pop(0)]
        while values and _worst(row + [values[0]], side) <= _worst(row, side):
            row.append(values.pop(0))
        total = sum(row)
        if width >= height:
            row_width = total / height
            cy = y
            for value in row:
                rh = value / row_width
                rects.append((x, cy, row_width, rh))
                cy += rh
            x += row_width
            width -= row_width
        else:
            row_height = total / width
            cx = x
            for value in row:
                rw = value / row_height
                rects.append((cx, y, rw, row_height))
                cx += rw
            y += row_height
            height -= row_height
    return rects


def residue_mol(residue: dict):
    mol = Chem.RWMol()
    atom_index = {}
    for name, z in zip(residue["atoms"], residue["z_elements"], strict=True):
        atom_index[name] = mol.AddAtom(Chem.Atom(int(z)))
    for a, b in residue["bonds"]:
        # Some CHARMM water topologies encode H1-H2 as a rigid-geometry
        # constraint. It is not a chemical bond and makes water draw as a
        # misleading triangle.
        both_hydrogen = (
            a in atom_index and b in atom_index
            and mol.GetAtomWithIdx(atom_index[a]).GetAtomicNum() == 1
            and mol.GetAtomWithIdx(atom_index[b]).GetAtomicNum() == 1
        )
        if (a in atom_index and b in atom_index and not both_hydrogen
                and mol.GetBondBetweenAtoms(atom_index[a], atom_index[b]) is None):
            mol.AddBond(atom_index[a], atom_index[b], Chem.BondType.SINGLE)
    result = mol.GetMol()
    rdDepictor.Compute2DCoords(result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coverage", type=Path,
                   default=Path("docs/images/des-so3lr-dimers/resi_coverage.md"))
    p.add_argument("-o", "--output", type=Path,
                   default=Path("docs/images/des-so3lr-dimers/molecule_cloud.png"))
    p.add_argument("--top", type=int, default=40)
    args = p.parse_args(argv)

    counts = read_counts(args.coverage, args.top)
    # Square-root area scaling preserves frequency rank while keeping the tail
    # legible; exact sampled counts remain printed in every tile.
    weights = np.sqrt([count for _, count in counts])
    weights = 0.925 * weights / weights.sum()
    rects = treemap(weights, y=0.0, height=0.925)
    ref = load_reference()
    colors = plt.get_cmap("YlGnBu")(np.linspace(0.22, 0.72, len(counts)))

    fig, ax = plt.subplots(figsize=(16, 10), facecolor="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for rank, ((name, count), (x, y, w, h)) in enumerate(zip(counts, rects, strict=True)):
        pad = 0.0025
        ax.add_patch(plt.Rectangle((x + pad, y + pad), max(0, w - 2 * pad), max(0, h - 2 * pad),
                                   facecolor=colors[rank], alpha=0.20,
                                   edgecolor=colors[rank], linewidth=1.0))
        residue = ref.residues.get(name)
        if residue and w > 0.045 and h > 0.055:
            mol = residue_mol(residue)
            size = int(np.clip(800 * min(w, h), 70, 190))
            image = Draw.MolToImage(mol, size=(size, size), kekulize=False)
            image = Image.fromarray(np.asarray(image))
            zoom = min(w * fig.get_figwidth(), h * fig.get_figheight()) * 0.30
            box = AnnotationBbox(OffsetImage(image, zoom=zoom),
                                 (x + w / 2, y + h * 0.58), frameon=False)
            ax.add_artist(box)
        fontsize = float(np.clip(7 + 24 * np.sqrt(w * h), 7, 18))
        ax.text(x + w / 2, y + max(0.012, h * 0.12), f"{name}  {count:,}",
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold" if rank < 8 else "normal", color="#14202b")

    ax.text(0.005, 0.992, "DES residue molecule cloud", ha="left", va="top",
            fontsize=20, fontweight="bold", color="#14202b")
    ax.text(0.005, 0.957, f"Top {len(counts)} typeable residues · tile area ∝ √ sampled frequency",
            ha="left", va="top", fontsize=10, color="#4c5964")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
