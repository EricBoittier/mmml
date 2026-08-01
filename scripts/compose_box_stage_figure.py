#!/usr/bin/env python
"""Compose the per-stage POV-Ray renders into one labelled figure.

The panels are deliberately shown together: individually they are
indistinguishable, and that is the result -- box preparation moves atoms by
0.37 A RMSD at the MM pretreat step and essentially nothing afterwards.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from mmml.utils.plotting.styles import apply_plot_style

OUT = Path(sys.argv[1])
SRC = Path(sys.argv[2])

PANELS = [
    ("box_tip3_1_packmol.png", "1  Packmol placement", "reference"),
    ("box_tip3_2_pretreat.png", "2  CHARMM MM pretreat", "RMSD 0.3661 Å"),
    ("box_tip3_3_prepladder.png", "3  prep ladder", "RMSD 0.0000 Å"),
    ("box_tip3_4_final.png", "4  final model.pdb", "RMSD 0.0220 Å"),
]

apply_plot_style("icml")
fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.8))
for ax, (fname, title, sub) in zip(axes, PANELS):
    p = SRC / fname
    if not p.exists():
        ax.text(0.5, 0.5, f"missing\n{fname}", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="0.4")
    else:
        # Trim the generous white margin POV-Ray leaves around the cluster.
        img = mpimg.imread(p)
        h, w = img.shape[:2]
        ax.imshow(img[int(0.10 * h):int(0.95 * h), int(0.08 * w):int(0.95 * w)])
    ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
    ax.set_xlabel(sub, fontsize=9.5, color="0.30")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

fig.suptitle(
    "TIP3 liquid box through preparation — 732 molecules, 28.0 Å cube, "
    "0.9975 g/cm³ (target 0.9970)",
    fontsize=13, fontweight="bold",
)
fig.text(0.5, 0.012,
         "RMSD is over all 2,196 atoms against the previous stage. The panels are "
         "visually identical because preparation moves atoms by <0.4 Å in total; "
         "the prep ladder moves them not at all.",
         ha="center", fontsize=9.5, color="0.30")
fig.tight_layout(rect=(0, 0.085, 1, 0.94))
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150)
print(f"wrote {OUT}")
