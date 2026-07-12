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
# PBC pedagogy (four waters): lower scale + larger radii so O/H stay visible in docs.
SCALE_PBC_WATER = 16.0
SCALE_PBC_WATER_SUPER = 13.0
PBC_ATOM_RADII = 1.28
PBC_ROTATION = "20x,12y,0z"

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
    radii: float = 0.88,
    charmm_image_tags: "np.ndarray | None" = None,
    atom_colors: "np.ndarray | None" = None,
    writer: "Matplotlib | None" = None,
) -> "Matplotlib":
    """Orthographic ASE view: bonds under atoms, equal aspect, styled patches.

    When ``charmm_image_tags`` is set (0 = primary, 1 = IMAGE translation), image
    sites are drawn in orange at lower opacity; primaries use Jmol element colors.

    ``atom_colors`` is an (N, 3) array of per-atom RGB triples that overrides
    the default Jmol element colors — e.g. to color atoms by *role* (ML core /
    ML shell / MM region) rather than by element, for architecture diagrams.

    Pass an existing ``writer`` (e.g. after drawing cell outlines) to reuse the
    same projection.
    """
    import numpy as np
    from ase.io.utils import make_patch_list
    from ase.visualize.plot import Matplotlib
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle, PathPatch

    style = DOCS_STRUCTURE_STYLE
    if writer is None:
        writer = Matplotlib(
            atoms,
            ax,
            rotation=rotation,
            radii=radii,
            colors=atom_colors,
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

    tags = None
    if charmm_image_tags is not None:
        tags = np.asarray(charmm_image_tags, dtype=np.int8).reshape(-1)
        if int(tags.shape[0]) != len(atoms):
            raise ValueError(
                f"charmm_image_tags length {tags.shape[0]} != n_atoms {len(atoms)}"
            )

    for idx, patch in enumerate(make_patch_list(writer)):
        is_image = tags is not None and int(tags[idx]) == 1
        patch.set_zorder(2 if is_image else 3)
        if isinstance(patch, Circle):
            patch.set_edgecolor(style["atom_edge"])
            patch.set_linewidth(style["atom_edge_width"])
            if is_image:
                patch.set_facecolor("#fb923c")
                patch.set_alpha(0.45)
            else:
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
    return writer


def save_structure_figure(
    atoms: Atoms,
    path: Path | str,
    *,
    title: str,
    rotation: str = "25x,15y,0z",
    scale: float = SCALE_MONOMER,
    atom_colors: "np.ndarray | None" = None,
    legend_entries: "list[tuple[str, str]] | None" = None,
) -> Path:
    """Write a PNG with orthographic ASE projection and covalent bonds.

    ``atom_colors`` / ``legend_entries`` support role-colored (rather than
    element-colored) diagrams: pass an (N, 3) RGB array plus a
    ``[(label, hex_color), ...]`` legend to color-code e.g. an ML-scored core
    vs. an ML-shell vs. an MM region.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

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
        atom_colors=atom_colors,
    )
    ax.set_title(
        title,
        fontsize=11.5,
        fontweight="500",
        color=style["title_color"],
        pad=10,
    )
    if legend_entries:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=9,
                markerfacecolor=color,
                markeredgecolor=style["atom_edge"],
                markeredgewidth=0.65,
                label=label,
            )
            for label, color in legend_entries
        ]
        ax.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.06),
            ncol=len(handles),
            fontsize=9,
            frameon=False,
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
    "PBC_ATOM_RADII",
    "PBC_ROTATION",
    "SCALE_BOX",
    "SCALE_CRYSTAL",
    "SCALE_MONOMER",
    "SCALE_PBC_WATER",
    "SCALE_PBC_WATER_SUPER",
    "SCALE_PEPTIDE_ML",
    "SCALE_TRIALANINE_BOX",
    "SCALE_TRIALANINE_PEPTIDE",
    "bond_segments_2d",
    "draw_orthographic_structure",
    "save_structure_figure",
    "use_matplotlib_agg",
]
