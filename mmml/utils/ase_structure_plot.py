"""Orthographic ASE structure figures with covalent bonds (MkDocs / workflow plots)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ase import Atoms
    from matplotlib.axes import Axes

# Orthographic paper-space scale (ASE Angstroms/cm). Tuned per view so structures fill the frame.
SCALE_MONOMER = 42.0
SCALE_BOX = 13.5
SCALE_CRYSTAL = 24.0
SCALE_TRIALANINE_BOX = 11.5
SCALE_TRIALANINE_PEPTIDE = 38.0
SCALE_PEPTIDE_ML = 38.0

DOCS_STRUCTURE_STYLE = {
    "figure_facecolor": "#f8fafc",
    "axes_facecolor": "#f8fafc",
    "bond_color": "#64748b",
    "bond_width": 1.35,
    "atom_edge": "#1e293b",
    "atom_edge_width": 0.65,
    "title_color": "#0f172a",
    "unit_cell_alpha": 0.55,
}


def use_matplotlib_agg() -> None:
    import matplotlib

    matplotlib.use("Agg")


def bond_segments_2d(atoms: Atoms, writer) -> "np.ndarray":
    """Covalent bond segments in image-plane coordinates (with MIC for PBC)."""
    import numpy as np
    from ase.geometry import find_mic
    from ase.neighborlist import natural_cutoffs, neighbor_list

    if len(atoms) == 0:
        return np.empty((0, 2, 2))

    cutoffs = natural_cutoffs(atoms, mult=1.08)
    i, j = neighbor_list("ij", atoms, cutoffs, self_interaction=False)
    if len(i) == 0:
        return np.empty((0, 2, 2))

    pos = atoms.get_positions()
    pos_i = pos[i]
    pos_j = pos[j]
    if atoms.pbc.any():
        vecs = pos_j - pos_i
        vecs, _ = find_mic(vecs, atoms.cell, atoms.pbc)
        pos_j = pos_i + vecs

    im_i = writer.to_image_plane_positions(pos_i)[:, :2]
    im_j = writer.to_image_plane_positions(pos_j)[:, :2]
    return np.stack([im_i, im_j], axis=1)


def draw_orthographic_structure(
    atoms: Atoms,
    ax: Axes,
    *,
    rotation: str,
    scale: float,
    show_unit_cell: int,
) -> None:
    """Orthographic ASE view: bonds under atoms, equal aspect, styled patches."""
    from ase.io.utils import make_patch_list
    from ase.visualize.plot import Matplotlib
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle, PathPatch

    style = DOCS_STRUCTURE_STYLE
    writer = Matplotlib(
        atoms,
        ax,
        rotation=rotation,
        radii=0.88,
        scale=scale,
        show_unit_cell=show_unit_cell,
        auto_bbox_size=1.1,
    )

    segments = bond_segments_2d(atoms, writer)
    if len(segments):
        ax.add_collection(
            LineCollection(
                segments,
                colors=style["bond_color"],
                linewidths=style["bond_width"],
                capstyle="round",
                zorder=1,
            )
        )

    for patch in make_patch_list(writer):
        patch.set_zorder(3)
        if isinstance(patch, Circle):
            patch.set_edgecolor(style["atom_edge"])
            patch.set_linewidth(style["atom_edge_width"])
            patch.set_alpha(0.97)
        elif isinstance(patch, PathPatch):
            patch.set_edgecolor("#3b82f6")
            patch.set_facecolor("none")
            patch.set_linewidth(1.0)
            patch.set_linestyle((0, (4, 3)))
            patch.set_alpha(style["unit_cell_alpha"])
        ax.add_patch(patch)

    ax.set_xlim(0, writer.w)
    ax.set_ylim(0, writer.h)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()


def save_structure_figure(
    atoms: Atoms,
    path: Path | str,
    *,
    title: str,
    rotation: str = "25x,15y,0z",
    scale: float = SCALE_MONOMER,
) -> Path:
    """Write a PNG with orthographic ASE projection and covalent bonds."""
    import matplotlib.pyplot as plt

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    style = DOCS_STRUCTURE_STYLE
    pbc = bool(getattr(atoms, "pbc", None) is not None and any(atoms.pbc))
    show_cell = 2 if pbc else 0

    fig, ax = plt.subplots(
        figsize=(6.5, 5.0),
        dpi=150,
        facecolor=style["figure_facecolor"],
    )
    ax.set_facecolor(style["axes_facecolor"])
    draw_orthographic_structure(
        atoms,
        ax,
        rotation=rotation,
        scale=scale,
        show_unit_cell=show_cell,
    )
    ax.set_title(
        title,
        fontsize=11.5,
        fontweight="500",
        color=style["title_color"],
        pad=10,
    )
    fig.tight_layout()
    fig.savefig(
        out,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )
    plt.close(fig)
    return out


__all__ = [
    "DOCS_STRUCTURE_STYLE",
    "SCALE_BOX",
    "SCALE_CRYSTAL",
    "SCALE_MONOMER",
    "SCALE_PEPTIDE_ML",
    "SCALE_TRIALANINE_BOX",
    "SCALE_TRIALANINE_PEPTIDE",
    "bond_segments_2d",
    "draw_orthographic_structure",
    "save_structure_figure",
    "use_matplotlib_agg",
]
