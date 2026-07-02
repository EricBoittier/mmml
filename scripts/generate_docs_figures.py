#!/usr/bin/env python3
"""Generate static figures for MkDocs (ASE ``plot_atoms`` + matplotlib).

Run from repo root::

    uv run python scripts/generate_docs_figures.py

Structure coordinates come from bundled assets under ``mmml/data/`` (CHARMM /
Packmol). Refresh those first when coordinates change::

    uv run python scripts/export_docs_structure_assets.py

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

from mmml.utils.ase_structure_plot import (
    DOCS_STRUCTURE_STYLE as _STYLE,
    SCALE_BOX as _SCALE_BOX,
    SCALE_CRYSTAL as _SCALE_CRYSTAL,
    SCALE_MONOMER as _SCALE_MONOMER,
    SCALE_TRIALANINE_BOX as _SCALE_TRIALANINE_BOX,
    SCALE_TRIALANINE_PEPTIDE as _SCALE_TRIALANINE_PEPTIDE,
    save_structure_figure as _save_structure_figure,
    use_matplotlib_agg as _use_agg,
)

if TYPE_CHECKING:
    from ase import Atoms


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
    """Periodic box of acetone monomers from Packmol (``make-box`` workflow)."""
    import ase.io

    from mmml.paths import default_make_box_aco_pdb

    pdb = default_make_box_aco_pdb()
    if not pdb.is_file():
        raise FileNotFoundError(
            f"Missing {pdb}. Run: "
            "./scripts/mmml-charmm-mpirun.sh python scripts/export_docs_structure_assets.py"
        )
    box = ase.io.read(pdb)
    side = float(box.cell.lengths()[0]) if box.cell is not None else 22.0
    n_monomers = sum(1 for s in box.get_chemical_symbols() if s == "O")
    _save_structure_figure(
        box,
        out,
        title=f"make-box: {n_monomers}× acetone in {side:.0f} Å cube (Packmol)",
        rotation="55x,25y,0z",
        scale=_SCALE_BOX,
    )


def _fallback_crystal_atoms() -> Atoms:
    """Experimental benzene P2₁/c cell (COD 4501704) when DCM CIF / PyXtal unavailable."""
    from ase.io import read

    from mmml.paths import default_benzene_crystal_cif

    cif = default_benzene_crystal_cif()
    if not cif.is_file():
        raise FileNotFoundError(f"Missing bundled benzene CIF: {cif}")
    return read(str(cif))


def figure_build_crystal(out: Path) -> bool:
    """Experimental DCM Pbcn cell (COD 2100015) when ASE can read it; else PyXtal/ASE fallback."""
    try:
        from ase.io import read

        from mmml.paths import default_dcm_crystal_cif

        cif = default_dcm_crystal_cif()
        if cif.is_file():
            atoms = read(str(cif))
            title = "build-crystal: DCM Pbcn (COD 2100015, ρ≈1.97 g/cm³)"
            rotation = "15x,70y,0z"
        else:
            raise FileNotFoundError(cif)
    except Exception as exc:
        print(f"build-crystal figure: using PyXtal/ASE fallback ({exc})", file=sys.stderr)
        try:
            from mmml.interfaces.pyxtal_placement import (
                MolecularCrystalBuildRequest,
                build_molecular_crystal_random,
                have_pyxtal,
            )
            from mmml.paths import default_dcm_molecule_xyz

            if have_pyxtal():
                result = build_molecular_crystal_random(
                    MolecularCrystalBuildRequest(
                        molecules=[str(default_dcm_molecule_xyz())],
                        stoichiometry=[4],
                        space_group=60,
                        dimension=3,
                        factor=1.0,
                        seed=42,
                        max_attempts=40,
                    )
                )
                atoms = result.atoms
                title = "build-crystal: DCM (Z=4, Pbcn) PyXtal placement"
                rotation = "15x,70y,0z"
            else:
                raise ImportError("pyxtal not installed")
        except Exception as exc2:
            print(f"build-crystal figure: benzene CIF fallback ({exc2})", file=sys.stderr)
            atoms = _fallback_crystal_atoms()
            title = "build-crystal: benzene P2₁/c (COD 4501704)"
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


def _trialanine_docs_atoms() -> Atoms:
    from mmml.interfaces.pycharmmInterface.trialanine_water_box import (
        load_trialanine_water_atoms_for_docs,
    )

    return load_trialanine_water_atoms_for_docs()


def figure_trialanine_water_box(out: Path) -> None:
    atoms = _trialanine_docs_atoms()
    _save_structure_figure(
        atoms,
        out,
        title="trialanine-water-box: CGENFF TRIA + 10× TIP3 (28 Å cube)",
        rotation="55x,25y,0z",
        scale=_SCALE_TRIALANINE_BOX,
    )


def figure_trialanine_peptide_zoom(out: Path) -> None:
    from mmml.interfaces.pycharmmInterface.trialanine_water_box import (
        peptide_atoms_from_trialanine_box,
    )

    full = _trialanine_docs_atoms()
    peptide = peptide_atoms_from_trialanine_box(full)
    _save_structure_figure(
        peptide,
        out,
        title="TRIA peptide (RESI TRIA, 42 atoms; CHARMM CGENFF build)",
        rotation="25x,15y,0z",
        scale=_SCALE_TRIALANINE_PEPTIDE,
    )


def figure_trialanine_build_pipeline(out: Path) -> None:
    """Schematic build steps for ``build_trialanine_water_box_in_charmm``."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    _style_matplotlib_rc()
    stages = [
        "CGENFF\n+ TRIA RTF",
        "sequence\n+ IC",
        "center\npeptide",
        "grid\nTIP3",
        "PBC\nNBOND",
    ]
    fig, ax = plt.subplots(figsize=(7.0, 2.4), dpi=150, facecolor=_STYLE["figure_facecolor"])
    ax.set_facecolor(_STYLE["axes_facecolor"])
    ax.set_xlim(0, len(stages))
    ax.set_ylim(0, 1)
    ax.axis("off")

    for i, label in enumerate(stages):
        x = i + 0.5
        box = FancyBboxPatch(
            (x - 0.38, 0.28),
            0.76,
            0.44,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor="#e0f2fe",
            edgecolor="#2563eb",
            linewidth=1.4,
        )
        ax.add_patch(box)
        ax.text(x, 0.5, label, ha="center", va="center", fontsize=9, color=_STYLE["title_color"])
        if i < len(stages) - 1:
            ax.annotate(
                "",
                xy=(x + 0.42, 0.5),
                xytext=(x + 0.58, 0.5),
                arrowprops=dict(arrowstyle="->", color="#64748b", lw=1.5),
            )

    ax.set_title(
        "trialanine_water_box: CGENFF-only build (no Packmol)",
        fontweight="500",
        fontsize=11,
        color=_STYLE["title_color"],
        pad=12,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _write_bundled_trialanine_reference_extxyz() -> None:
    """No-op: trialanine reference is exported by ``export_docs_structure_assets.py``."""


def figure_compose_workflow(out: Path) -> None:
    """Bar chart: typical atom counts for structure builders."""
    import matplotlib.pyplot as plt

    _style_matplotlib_rc()
    labels = [
        "make-res\n(1 monomer)",
        "make-box\n(8× ACO)",
        "trialanine\n(42+30 H₂O)",
        "build-crystal\n(supercell)",
    ]
    counts = [10, 80, 72, 48]
    colors = ["#059669", "#2563eb", "#0d9488", "#7c3aed"]

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
        STRUCT / "trialanine-water-box.png": "trialanine_box",
        STRUCT / "trialanine-peptide-zoom.png": "trialanine_peptide",
        PLOTS / "liquid-box-density-ladder.png": "liquid_box",
        PLOTS / "structure-builder-sizes.png": "workflow",
        PLOTS / "trialanine-build-pipeline.png": "trialanine_pipeline",
    }

    builders = {
        "make_res": lambda p: figure_make_res(p),
        "make_box": lambda p: figure_make_box(p),
        "liquid_box": lambda p: figure_liquid_box_schematic(p),
        "workflow": lambda p: figure_compose_workflow(p),
        "trialanine_box": lambda p: figure_trialanine_water_box(p),
        "trialanine_peptide": lambda p: figure_trialanine_peptide_zoom(p),
        "trialanine_pipeline": lambda p: figure_trialanine_build_pipeline(p),
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

    if not check:
        _write_bundled_trialanine_reference_extxyz()

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
