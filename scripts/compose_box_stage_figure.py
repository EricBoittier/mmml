#!/usr/bin/env python
"""Compose the per-stage POV-Ray renders into one labelled figure.

The panels are deliberately shown together: individually they are
indistinguishable even though the CHARMM MM pretreat does real work (1.39 A RMSD
on TIP3, 0.56 A on MEOH). What the pictures establish is the absence of packing
faults -- voids, interpenetrating pairs, molecules outside the cell -- which a
density number cannot show, because Packmol reports the density it was asked for.

The stage-to-stage RMSDs are *computed here* from the same PDBs that were
rendered, rather than typed into the panel captions, so the number under a panel
cannot drift away from the picture above it. They are minimum-image; see _rmsd.

Usage::

    python scripts/compose_box_stage_figure.py OUT.png RENDER_DIR \\
        --species tip3 --structures artifacts/.../boxes/tip3 \\
        --title "TIP3 liquid box ..." [--note "..."]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from mmml.utils.plotting.styles import apply_plot_style

# (render suffix, panel title, structure filename for the RMSD chain)
STAGES = [
    ("1_packmol", "1  Packmol placement", "init-packmol-sphere.pdb"),
    ("2_pretreat", "2  CHARMM MM pretreat", "01_mm.pdb"),
    ("3_prepladder", "3  prep ladder", "latest.pdb"),
    ("4_final", "4  final model.pdb", "model.pdb"),
]


def _coords(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    from ase.io import read

    return read(str(path)).get_positions()


def _rmsd(cur: np.ndarray, prev: np.ndarray, box: float | None) -> tuple[float, int]:
    """RMSD between two stages, and how many atoms wrapped.

    These PDBs carry no CRYST1 record, so a plain ``cur - prev`` counts a molecule
    that crossed a periodic face as having moved a full box length. In the TIP3
    box that is 24 atoms out of 2,196 and it inflates the RMSD from 0.07 A to
    1.43 A -- an artefact that reads as a real structural change. With the box
    side known we apply the minimum image convention; without it we cannot, and
    the caller says so rather than printing the contaminated number.
    """
    d = cur - prev
    if box:
        d = d - box * np.round(d / box)
    r = np.linalg.norm(d, axis=1)
    n_wrapped = int((np.linalg.norm(cur - prev, axis=1) > box / 2).sum()) if box else 0
    return float(np.sqrt((r**2).mean())), n_wrapped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("output", type=Path)
    ap.add_argument("render_dir", type=Path)
    ap.add_argument("--species", required=True, help="render filename prefix, e.g. tip3")
    ap.add_argument("--structures", type=Path, default=None,
                    help="directory holding the stage PDBs, for the RMSD chain")
    ap.add_argument("--title", required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("--box-side", type=float, default=None,
                    help="cubic box side in A for the minimum-image RMSD; "
                         "read from box.json when present")
    a = ap.parse_args()

    box = a.box_side
    if box is None and a.structures is not None:
        bj = a.structures / "box.json"
        if bj.exists():
            import json

            box = json.loads(bj.read_text()).get("box_side_A")
    if box:
        print(f"  minimum-image RMSD, box side {box} A")
    else:
        print("  WARNING: no box side -- RMSDs are wrap-contaminated")

    # RMSD of each stage against the previous one. Atom counts must match; if a
    # stage is missing or reordered we say so rather than printing a wrong number.
    subs: list[str] = []
    prev: np.ndarray | None = None
    n_atoms = 0
    total_wrapped = 0
    for _, _, struct in STAGES:
        cur = _coords(a.structures / struct) if a.structures else None
        if cur is None:
            subs.append("")
        elif prev is None:
            subs.append("reference")
            n_atoms = len(cur)
        elif cur.shape != prev.shape:
            subs.append(f"{len(cur)} atoms — not comparable")
        else:
            r, nw = _rmsd(cur, prev, box)
            total_wrapped += nw
            subs.append(f"RMSD {r:.4f} Å" + (f"  ({nw} wrapped)" if nw else ""))
        if cur is not None:
            prev = cur

    apply_plot_style("icml")
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.8))
    for ax, (suffix, title, _), sub in zip(axes, STAGES, subs):
        p = a.render_dir / f"box_{a.species}_{suffix}.png"
        if not p.exists():
            ax.text(0.5, 0.5, f"missing\n{p.name}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="0.4")
        else:
            # Trim the generous white margin POV-Ray leaves around the cluster.
            img = mpimg.imread(p)
            h, w = img.shape[:2]
            ax.imshow(img[int(0.10 * h):int(0.95 * h), int(0.08 * w):int(0.95 * w)])
        ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
        ax.set_xlabel(sub, fontsize=9.5, color="0.30")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    fig.suptitle(a.title, fontsize=13, fontweight="bold")
    note = a.note or (
        f"Minimum-image RMSD over all {n_atoms:,} atoms against the previous stage"
        + (f" (box {box:g} Å; {total_wrapped} atoms cross a periodic face and would "
           "otherwise register as moving a full box length)" if box else "")
        + ". The panels are visually identical because preparation moves the atoms "
        "very little."
    )
    fig.text(0.5, 0.012, note, ha="center", fontsize=9.5, color="0.30")
    fig.tight_layout(rect=(0, 0.085, 1, 0.94))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=150)
    print(f"wrote {a.output}")
    for (_, t, _), s in zip(STAGES, subs):
        print(f"  {t:26s} {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
