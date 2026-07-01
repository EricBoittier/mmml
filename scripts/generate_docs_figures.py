#!/usr/bin/env python3
"""Generate static figures for MkDocs (ASE ``plot_atoms`` + matplotlib).

Run from repo root::

    uv run python scripts/generate_docs_figures.py

Writes PNGs under ``docs/images/`` for use in Markdown (MkDocs does not execute
inline Python). Structures use **orthographic** ASE projection (fixed scale per
view), Jmol colors, covalent bonds, and a light styled background.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

REPO = Path(__file__).resolve().parents[1]
IMG = REPO / "docs" / "images"
STRUCT = IMG / "structures"
PLOTS = IMG / "plots"

# Orthographic paper-space scale (ASE Angstroms/cm). Tuned per view so structures fill the frame.
_SCALE_MONOMER = 42.0
_SCALE_BOX = 13.5
_SCALE_CRYSTAL = 24.0

_STYLE = {
    "figure_facecolor": "#f8fafc",
    "axes_facecolor": "#f8fafc",
    "bond_color": "#64748b",
    "bond_width": 1.35,
    "atom_edge": "#1e293b",
    "atom_edge_width": 0.65,
    "title_color": "#0f172a",
    "unit_cell_alpha": 0.55,
}

if TYPE_CHECKING:
    from ase import Atoms
    from matplotlib.axes import Axes


def _use_agg() -> None:
    import matplotlib

    matplotlib.use("Agg")


def _bond_segments_2d(atoms: Atoms, writer) -> np.ndarray:
    """Covalent bond segments in image-plane coordinates (with MIC for PBC)."""
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


def _draw_orthographic_structure(
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

    writer = Matplotlib(
        atoms,
        ax,
        rotation=rotation,
        radii=0.88,
        scale=scale,
        show_unit_cell=show_unit_cell,
        auto_bbox_size=1.1,
    )

    segments = _bond_segments_2d(atoms, writer)
    if len(segments):
        ax.add_collection(
            LineCollection(
                segments,
                colors=_STYLE["bond_color"],
                linewidths=_STYLE["bond_width"],
                capstyle="round",
                zorder=1,
            )
        )

    for patch in make_patch_list(writer):
        patch.set_zorder(3)
        if isinstance(patch, Circle):
            patch.set_edgecolor(_STYLE["atom_edge"])
            patch.set_linewidth(_STYLE["atom_edge_width"])
            patch.set_alpha(0.97)
        elif isinstance(patch, PathPatch):
            # unit cell edges
            patch.set_edgecolor("#3b82f6")
            patch.set_facecolor("none")
            patch.set_linewidth(1.0)
            patch.set_linestyle((0, (4, 3)))
            patch.set_alpha(_STYLE["unit_cell_alpha"])
        ax.add_patch(patch)

    ax.set_xlim(0, writer.w)
    ax.set_ylim(0, writer.h)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()


def _save_structure_figure(
    atoms: Atoms,
    path: Path,
    *,
    title: str,
    rotation: str,
    scale: float,
) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    pbc = bool(getattr(atoms, "pbc", None) is not None and any(atoms.pbc))
    show_cell = 2 if pbc else 0

    fig, ax = plt.subplots(
        figsize=(6.5, 5.0),
        dpi=150,
        facecolor=_STYLE["figure_facecolor"],
    )
    ax.set_facecolor(_STYLE["axes_facecolor"])
    _draw_orthographic_structure(
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
        color=_STYLE["title_color"],
        pad=10,
    )
    fig.tight_layout()
    fig.savefig(
        path,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )
    plt.close(fig)


def figure_make_res(out: Path) -> None:
    import ase.io

    from mmml.paths import default_aco_template_pdb

    atoms = ase.io.read(default_aco_template_pdb())
    _save_structure_figure(
        atoms,
        out,
        title="make-res: acetone monomer (ACO, CGENFF)",
        rotation="25x,15y,0z",
        scale=_SCALE_MONOMER,
    )


def figure_make_box(out: Path) -> None:
    """Periodic box of acetone monomers (illustrates make-box / Packmol output)."""
    from ase import Atoms
    from ase.build import molecule

    side = 22.0
    monomer = molecule("CH3COCH3")
    monomer.center(vacuum=3.0)

    offsets = [
        (4.0, 4.0, 4.0),
        (12.0, 4.0, 6.0),
        (6.0, 11.0, 5.0),
        (14.0, 12.0, 8.0),
        (5.0, 15.0, 14.0),
        (13.0, 6.0, 15.0),
        (8.0, 9.0, 12.0),
        (16.0, 14.0, 4.0),
    ]
    symbols: list[str] = []
    positions: list[np.ndarray] = []
    for dx, dy, dz in offsets:
        for sym, pos in zip(monomer.get_chemical_symbols(), monomer.get_positions()):
            symbols.append(sym)
            positions.append(pos + np.array([dx, dy, dz]))

    box = Atoms(symbols=symbols, positions=positions, cell=[side, side, side], pbc=True)
    _save_structure_figure(
        box,
        out,
        title=f"make-box: 8× acetone in {side:.0f} Å cube (illustrative)",
        rotation="55x,25y,0z",
        scale=_SCALE_BOX,
    )


def _fallback_crystal_atoms() -> Atoms:
    """Orthorhombic benzene dimer when PyXtal is not installed."""
    from ase import Atoms
    from ase.build import molecule

    benzene = molecule("C6H6")
    benzene.translate([2.5, 2.5, 2.0])
    benzene2 = molecule("C6H6")
    benzene2.translate([8.0, 7.0, 9.0])
    symbols = benzene.get_chemical_symbols() + benzene2.get_chemical_symbols()
    positions = list(benzene.get_positions()) + list(benzene2.get_positions())
    return Atoms(
        symbols=symbols,
        positions=positions,
        cell=[14.0, 12.0, 16.0],
        pbc=True,
    )


def figure_build_crystal(out: Path) -> bool:
    """PyXtal DCM cell scaled to solid density when available; else ASE fallback."""
    try:
        from mmml.interfaces.pyxtal_placement import (
            MolecularCrystalBuildRequest,
            build_molecular_crystal_random,
            have_pyxtal,
            scale_atoms_cell_to_density,
        )
        from mmml.paths import default_dcm_molecule_xyz

        if have_pyxtal():
            dcm_xyz = str(default_dcm_molecule_xyz())
            result = build_molecular_crystal_random(
                MolecularCrystalBuildRequest(
                    molecules=[dcm_xyz],
                    stoichiometry=[4],
                    space_group=14,
                    dimension=3,
                    factor=1.0,
                    seed=42,
                    max_attempts=40,
                )
            )
            atoms = result.atoms
            scale_atoms_cell_to_density(atoms, 1.36)
            title = "build-crystal: DCM (Z=4, SG 14) at ρ=1.36 g/cm³"
            rotation = "15x,70y,0z"
        else:
            raise ImportError("pyxtal not installed")
    except Exception as exc:
        print(f"build-crystal figure: using ASE fallback ({exc})", file=sys.stderr)
        atoms = _fallback_crystal_atoms()
        title = "build-crystal: benzene dimer in periodic cell (illustrative)"
        rotation = "30x,55y,0z"

    _save_structure_figure(
        atoms,
        out,
        title=title,
        rotation=rotation,
        scale=_SCALE_CRYSTAL,
    )
    return True


def _style_matplotlib_rc() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "axes.edgecolor": "#cbd5e1",
            "axes.labelcolor": "#334155",
            "xtick.color": "#64748b",
            "ytick.color": "#64748b",
            "grid.color": "#e2e8f0",
        }
    )


def figure_liquid_box_schematic(out: Path) -> None:
    """Schematic density-prep ladder (matplotlib only)."""
    import matplotlib.pyplot as plt

    _style_matplotlib_rc()
    stages = ["Packmol", "MC staged", "MC target", "Lattice", "Certified"]
    density = [0.55, 0.78, 0.95, 0.99, 1.00]

    fig, ax = plt.subplots(figsize=(6.5, 3.6), dpi=150, facecolor=_STYLE["figure_facecolor"])
    ax.set_facecolor(_STYLE["axes_facecolor"])
    ax.plot(
        stages,
        density,
        "o-",
        color="#2563eb",
        linewidth=2.2,
        markersize=8,
        markerfacecolor="white",
        markeredgewidth=2,
        label="ρ / ρ_target",
    )
    ax.axhline(1.0, color="#94a3b8", linestyle="--", linewidth=1.2, label="target density")
    ax.set_ylim(0.4, 1.05)
    ax.set_ylabel("Relative density")
    ax.set_title("liquid-box: density prep ladder (schematic)", fontweight="500", pad=8)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.35, linestyle="-")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.autofmt_xdate(rotation=18, ha="right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def figure_compose_workflow(out: Path) -> None:
    """Bar chart: typical atom counts for structure builders."""
    import matplotlib.pyplot as plt

    _style_matplotlib_rc()
    labels = ["make-res\n(1 monomer)", "make-box\n(8× ACO)", "build-crystal\n(supercell)"]
    counts = [10, 80, 48]
    colors = ["#059669", "#2563eb", "#7c3aed"]

    fig, ax = plt.subplots(figsize=(6.2, 3.5), dpi=150, facecolor=_STYLE["figure_facecolor"])
    ax.set_facecolor(_STYLE["axes_facecolor"])
    bars = ax.bar(
        labels,
        counts,
        color=colors,
        edgecolor="white",
        linewidth=1.2,
        width=0.62,
    )
    ax.set_ylabel("Atoms (examples)")
    ax.set_title("Structure builders — example system sizes", fontweight="500", pad=8)
    for bar, n in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            str(n),
            ha="center",
            fontsize=9,
            color=_STYLE["title_color"],
        )
    ax.set_ylim(0, max(counts) * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def generate(*, check: bool = False) -> int:
    _use_agg()
    _style_matplotlib_rc()
    targets: dict[Path, str] = {
        STRUCT / "make-res-aco.png": "make_res",
        STRUCT / "make-box-acetone.png": "make_box",
        STRUCT / "build-crystal.png": "build_crystal",
        PLOTS / "liquid-box-density-ladder.png": "liquid_box",
        PLOTS / "structure-builder-sizes.png": "workflow",
    }

    builders = {
        "make_res": lambda p: figure_make_res(p),
        "make_box": lambda p: figure_make_box(p),
        "liquid_box": lambda p: figure_liquid_box_schematic(p),
        "workflow": lambda p: figure_compose_workflow(p),
    }

    changed = 0
    for path, key in targets.items():
        if check:
            if not path.is_file():
                print(f"missing: {path.relative_to(REPO)}", file=sys.stderr)
                changed += 1
            continue
        before = path.read_bytes() if path.is_file() else None
        if key == "build_crystal":
            figure_build_crystal(path)
        else:
            builders[key](path)
        after = path.read_bytes()
        if before != after:
            changed += 1

    if check:
        return 1 if changed else 0
    print(f"generate_docs_figures: wrote {len(targets)} images ({changed} updated)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Exit 1 if images missing")
    args = parser.parse_args()
    return generate(check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
