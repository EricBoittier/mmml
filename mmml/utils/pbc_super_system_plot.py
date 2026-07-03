"""ASE/matplotlib figures for CHARMM IMAGE super-system vs MIC / LR / domain docs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from mmml.utils.ase_structure_plot import DOCS_STRUCTURE_STYLE, use_matplotlib_agg

# Orthographic scale for the small 4×TIP3 teaching cell (Å per cm).
SCALE_PBC_DEMO = 28.0


def four_waters_cubic_cell(*, side_A: float = 14.0) -> "Atoms":
    """Four TIP3-like waters in a cubic PBC cell (schematic teaching geometry)."""
    from ase import Atoms
    from ase.build import molecule

    tip3 = molecule("H2O")
    tip3.center(vacuum=2.0)
    sites = np.array(
        [
            [0.35, 0.35, 0.35],
            [0.65, 0.35, 0.65],
            [0.35, 0.65, 0.65],
            [0.65, 0.65, 0.35],
        ],
        dtype=np.float64,
    ) * float(side_A)
    blocks: list[Atoms] = []
    for site in sites:
        w = tip3.copy()
        w.translate(site - w.get_center_of_mass())
        blocks.append(w)
    atoms = sum(blocks[1:], blocks[0])
    atoms.set_cell(np.diag([side_A, side_A, side_A]))
    atoms.set_pbc(True)
    return atoms


def _image_shifts_3d(radius: int = 1) -> list[np.ndarray]:
    shifts: list[np.ndarray] = []
    for ix in range(-radius, radius + 1):
        for iy in range(-radius, radius + 1):
            for iz in range(-radius, radius + 1):
                if ix == iy == iz == 0:
                    continue
                shifts.append(np.array([ix, iy, iz], dtype=np.float64))
    return shifts


def charmm_super_system_atoms(primary: "Atoms", *, shell: int = 1) -> tuple["Atoms", np.ndarray]:
    """Primary unit cell plus explicit translated copies (CHARMM IMAGE-style super system)."""
    from ase import Atoms

    cell = np.asarray(primary.cell.array, dtype=np.float64)
    symbols = list(primary.get_chemical_symbols())
    positions = [np.asarray(primary.get_positions(), dtype=np.float64)]
    tags: list[int] = [0] * len(primary)

    for shift in _image_shifts_3d(shell):
        disp = shift @ cell
        positions.append(positions[0] + disp)
        tags.extend([1] * len(primary))

    all_pos = np.vstack(positions)
    all_sym = symbols * (1 + len(_image_shifts_3d(shell)))
    out = Atoms(symbols=all_sym, positions=all_pos, cell=primary.cell, pbc=primary.pbc)
    out.set_array("charmm_image", np.asarray(tags, dtype=np.int8))
    return out, np.asarray(tags, dtype=np.int8)


def _draw_cell_outline(ax, writer, cell: np.ndarray, origin: np.ndarray, *, color: str, alpha: float, lw: float) -> None:
    """Draw one parallelepiped cell outline in the ASE image plane."""
    corners = np.array(
        [
            origin,
            origin + cell[0],
            origin + cell[0] + cell[1],
            origin + cell[1],
            origin + cell[2],
            origin + cell[0] + cell[2],
            origin + cell[0] + cell[1] + cell[2],
            origin + cell[1] + cell[2],
        ],
        dtype=np.float64,
    )
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    xy = writer.to_image_plane_positions(corners)[:, :2]
    for i, j in edges:
        ax.plot(
            [xy[i, 0], xy[j, 0]],
            [xy[i, 1], xy[j, 1]],
            color=color,
            alpha=alpha,
            linewidth=lw,
            linestyle=(0, (5, 4)),
            zorder=0,
        )


def _draw_atoms_colored(
    ax,
    writer,
    positions: np.ndarray,
    symbols: Sequence[str],
    mask: np.ndarray,
    *,
    face_primary: str = "#3b82f6",
    face_image: str = "#fb923c",
    radius: float = 0.42,
) -> None:
    from matplotlib.patches import Circle

    xy = writer.to_image_plane_positions(positions)[:, :2]
    radii = np.array([radius if s == "O" else radius * 0.55 for s in symbols], dtype=np.float64)
    for idx, (x, y) in enumerate(xy):
        color = face_primary if mask[idx] == 0 else face_image
        r = float(radii[idx])
        ax.add_patch(
            Circle(
                (x, y),
                r,
                facecolor=color,
                edgecolor="#1e293b",
                linewidth=0.55,
                alpha=0.92 if mask[idx] == 0 else 0.45,
                zorder=4 if mask[idx] == 0 else 2,
            )
        )


def plot_primary_cell(path: Path | str, *, side_A: float = 14.0) -> Path:
    """Unit cell only: four waters and the simulation box."""
    import matplotlib.pyplot as plt
    from ase.visualize.plot import Matplotlib

    use_matplotlib_agg()
    primary = four_waters_cubic_cell(side_A=side_A)
    style = DOCS_STRUCTURE_STYLE
    fig, ax = plt.subplots(figsize=(6.5, 5.2), dpi=150, facecolor=style["figure_facecolor"])
    ax.set_facecolor(style["axes_facecolor"])
    writer = Matplotlib(
        primary,
        ax,
        rotation="20x,12y,0z",
        radii=0.9,
        scale=SCALE_PBC_DEMO,
        show_unit_cell=2,
        auto_bbox_size=1.15,
    )
    _draw_cell_outline(
        ax,
        writer,
        primary.cell.array,
        np.zeros(3),
        color="#3b82f6",
        alpha=0.85,
        lw=1.4,
    )
    mask = np.zeros(len(primary), dtype=np.int8)
    _draw_atoms_colored(ax, writer, primary.get_positions(), primary.get_chemical_symbols(), mask)
    ax.set_xlim(0, writer.w)
    ax.set_ylim(0, writer.h)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(
        "Primary unit cell (N atoms in PSF)",
        fontsize=11.5,
        fontweight="500",
        color=style["title_color"],
        pad=10,
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def plot_charmm_super_system(path: Path | str, *, side_A: float = 14.0, shell: int = 1) -> Path:
    """CHARMM IMAGE-style super system: primary + translated image sites."""
    import matplotlib.pyplot as plt
    from ase.visualize.plot import Matplotlib

    use_matplotlib_agg()
    primary = four_waters_cubic_cell(side_A=side_A)
    super_atoms, tags = charmm_super_system_atoms(primary, shell=shell)
    style = DOCS_STRUCTURE_STYLE
    fig, ax = plt.subplots(figsize=(7.0, 5.6), dpi=150, facecolor=style["figure_facecolor"])
    ax.set_facecolor(style["axes_facecolor"])
    writer = Matplotlib(
        super_atoms,
        ax,
        rotation="20x,12y,0z",
        radii=0.9,
        scale=SCALE_PBC_DEMO * 0.82,
        show_unit_cell=0,
        auto_bbox_size=1.12,
    )
    cell = primary.cell.array
    for shift in [(0, 0, 0)] + [tuple(s) for s in _image_shifts_3d(shell)]:
        origin = np.asarray(shift, dtype=np.float64) @ cell
        is_home = shift == (0, 0, 0)
        _draw_cell_outline(
            ax,
            writer,
            cell,
            origin,
            color="#3b82f6" if is_home else "#94a3b8",
            alpha=0.9 if is_home else 0.35,
            lw=1.5 if is_home else 0.9,
        )
    _draw_atoms_colored(
        ax,
        writer,
        super_atoms.get_positions(),
        super_atoms.get_chemical_symbols(),
        tags,
    )
    ax.set_xlim(0, writer.w)
    ax.set_ylim(0, writer.h)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(
        "CHARMM super system: primary (blue) + image translations (orange)",
        fontsize=11.0,
        fontweight="500",
        color=style["title_color"],
        pad=10,
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def plot_mic_convention(path: Path | str, *, side_A: float = 14.0) -> Path:
    """MIC: store N atoms; draw shortest periodic vector to a partner."""
    import matplotlib.pyplot as plt
    from ase.geometry import find_mic
    from ase.visualize.plot import Matplotlib

    use_matplotlib_agg()
    primary = four_waters_cubic_cell(side_A=side_A)
    style = DOCS_STRUCTURE_STYLE
    fig, ax = plt.subplots(figsize=(6.5, 5.2), dpi=150, facecolor=style["figure_facecolor"])
    ax.set_facecolor(style["axes_facecolor"])
    writer = Matplotlib(
        primary,
        ax,
        rotation="20x,12y,0z",
        radii=0.9,
        scale=SCALE_PBC_DEMO,
        show_unit_cell=0,
        auto_bbox_size=1.15,
    )
    _draw_cell_outline(ax, writer, primary.cell.array, np.zeros(3), color="#3b82f6", alpha=0.85, lw=1.4)
    mask = np.zeros(len(primary), dtype=np.int8)
    _draw_atoms_colored(ax, writer, primary.get_positions(), primary.get_chemical_symbols(), mask)

    # Highlight O–O pair across the box boundary (waters 0 and 3 are diagonally placed).
    pos = primary.get_positions()
    i, j = 0, 3
    vec = pos[j] - pos[i]
    mic_vec, _ = find_mic(vec.reshape(1, 3), primary.cell, primary.pbc)[0]
    partner = pos[i] + mic_vec[0]
    im_i = writer.to_image_plane_positions(pos[[i]])[0, :2]
    im_j = writer.to_image_plane_positions(partner.reshape(1, 3))[0, :2]
    ax.annotate(
        "",
        xy=im_j,
        xytext=im_i,
        arrowprops=dict(arrowstyle="<->", color="#dc2626", lw=2.0, shrinkA=6, shrinkB=6),
        zorder=6,
    )
    ax.text(
        (im_i[0] + im_j[0]) / 2,
        (im_i[1] + im_j[1]) / 2 + 0.35,
        "MIC vector",
        ha="center",
        fontsize=9,
        color="#dc2626",
        fontweight="600",
        zorder=7,
    )
    ax.set_xlim(0, writer.w)
    ax.set_ylim(0, writer.h)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(
        "Minimum-image (MIC): N stored atoms, periodic shift in pair loop",
        fontsize=10.5,
        fontweight="500",
        color=style["title_color"],
        pad=10,
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def plot_spatial_decomposition(path: Path | str) -> Path:
    """Schematic 2×2 domain decomposition with halo (DOMDEC / ML spatial MPI)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle

    use_matplotlib_agg()
    style = DOCS_STRUCTURE_STYLE
    fig, ax = plt.subplots(figsize=(6.8, 5.4), dpi=150, facecolor=style["figure_facecolor"])
    ax.set_facecolor(style["axes_facecolor"])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.axis("off")

    colors = ["#dbeafe", "#dcfce7", "#fef9c3", "#fce7f3"]
    ranks = ["rank 0", "rank 1", "rank 2", "rank 3"]
    for k, (xc, yc) in enumerate([(0.6, 4.2), (5.2, 4.2), (0.6, 0.6), (5.2, 0.6)]):
        ax.add_patch(
            Rectangle((xc, yc), 3.6, 3.0, facecolor=colors[k], edgecolor="#334155", linewidth=1.2, zorder=1)
        )
        ax.add_patch(
            FancyBboxPatch(
                (xc + 0.15, yc + 0.15),
                3.3,
                2.7,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor="none",
                edgecolor="#ef4444",
                linewidth=1.0,
                linestyle=(0, (4, 3)),
                zorder=2,
            )
        )
        ax.text(xc + 1.8, yc + 1.35, ranks[k], ha="center", va="center", fontsize=10, color="#0f172a")

    ax.add_patch(
        Rectangle(
            (0.45, 0.45),
            9.1,
            7.1,
            fill=False,
            edgecolor="#3b82f6",
            linewidth=1.6,
            linestyle=(0, (6, 4)),
            zorder=0,
        )
    )
    ax.text(5.0, 7.75, "simulation box (PBC)", ha="center", fontsize=10.5, color="#1d4ed8", fontweight="600")
    ax.text(
        5.0,
        -0.35,
        "Red inner frame: halo / ghost monomers for ML–ML pairs at domain boundaries",
        ha="center",
        fontsize=9.2,
        color="#64748b",
    )
    ax.set_title(
        "Spatial decomposition (DOMDEC / ML spatial MPI)",
        fontsize=11.5,
        fontweight="500",
        color=style["title_color"],
        pad=12,
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def plot_lr_methods_schematic(path: Path | str) -> Path:
    """Near-field pairs vs far-field Ewald/PME/FMM-style k-space."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

    use_matplotlib_agg()
    style = DOCS_STRUCTURE_STYLE
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), dpi=150, facecolor=style["figure_facecolor"])
    titles = ["MIC / pair list", "PME / Ewald grid", "FMM tree (concept)"]
    for ax, title in zip(axes, titles, strict=True):
        ax.set_facecolor(style["axes_facecolor"])
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.add_patch(Rectangle((0.5, 0.5), 9, 9, fill=False, edgecolor="#3b82f6", linewidth=1.2, linestyle=(0, (5, 4))))
        ax.text(5, 9.35, title, ha="center", fontsize=10.5, fontweight="600", color=style["title_color"])

    # MIC panel
    ax0 = axes[0]
    ax0.scatter([3, 7, 4, 6], [3, 7, 7, 3], s=55, c="#3b82f6", zorder=3)
    ax0.add_patch(Circle((5, 5), 3.2, fill=False, edgecolor="#dc2626", linewidth=1.5, linestyle=(0, (4, 3))))
    ax0.text(5, 1.2, "cutnb sphere\n(all pairs inside)", ha="center", fontsize=8.5, color="#64748b")

    # PME panel
    ax1 = axes[1]
    for x in np.linspace(1.2, 8.8, 8):
        ax1.axvline(x, color="#cbd5e1", lw=0.6, zorder=0)
    for y in np.linspace(1.2, 8.8, 8):
        ax1.axhline(y, color="#cbd5e1", lw=0.6, zorder=0)
    ax1.scatter([3, 7, 4, 6], [3, 7, 7, 3], s=45, c="#3b82f6", zorder=3)
    ax1.text(5, 1.2, "short-range pairs +\nk-space reciprocal sum", ha="center", fontsize=8.5, color="#64748b")

    # FMM panel
    ax2 = axes[2]
    levels = [8.5, 6.5, 4.5, 2.8]
    cols = ["#e2e8f0", "#cbd5e1", "#94a3b8", "#64748b"]
    for size, col in zip(levels, cols, strict=True):
        ax2.add_patch(
            FancyBboxPatch(
                (5 - size / 2, 5 - size / 2),
                size,
                size,
                boxstyle="square,pad=0",
                facecolor=col,
                edgecolor="#334155",
                linewidth=0.8,
                zorder=1,
            )
        )
    ax2.scatter([5], [5], s=40, c="#3b82f6", zorder=4)
    ax2.text(5, 1.2, "hierarchical\nmultipole far field", ha="center", fontsize=8.5, color="#64748b")

    fig.suptitle(
        "Long-range electrostatics: truncated pairs vs grid vs multipole",
        fontsize=11.5,
        fontweight="500",
        color=style["title_color"],
        y=1.02,
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def generate_pbc_doc_figures(out_dir: Path | str) -> dict[str, Path]:
    """Write all PBC pedagogy figures under ``out_dir``."""
    root = Path(out_dir)
    return {
        "primary_cell": plot_primary_cell(root / "primary_cell.png"),
        "charmm_super_system": plot_charmm_super_system(root / "charmm_super_system.png"),
        "mic_convention": plot_mic_convention(root / "mic_convention.png"),
        "spatial_decomposition": plot_spatial_decomposition(root / "spatial_decomposition.png"),
        "lr_methods": plot_lr_methods_schematic(root / "lr_methods_schematic.png"),
    }


__all__ = [
    "charmm_super_system_atoms",
    "four_waters_cubic_cell",
    "generate_pbc_doc_figures",
    "plot_charmm_super_system",
    "plot_lr_methods_schematic",
    "plot_mic_convention",
    "plot_primary_cell",
    "plot_spatial_decomposition",
]
