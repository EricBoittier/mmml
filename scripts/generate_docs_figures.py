#!/usr/bin/env python3
"""Generate static figures for MkDocs (ASE ``plot_atoms`` + matplotlib).

Run from repo root::

    uv run python scripts/generate_docs_figures.py

Writes PNGs under ``docs/images/`` for use in Markdown (MkDocs does not execute
inline Python). The generating code is quoted in ``docs/cli/structure-building.md``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
IMG = REPO / "docs" / "images"
STRUCT = IMG / "structures"
PLOTS = IMG / "plots"


def _use_agg() -> None:
    import matplotlib

    matplotlib.use("Agg")


def _save_plot_atoms(atoms, path: Path, *, title: str, rotation: str = "10x,10y,90z") -> None:
    import matplotlib.pyplot as plt
    from ase.visualize.plot import plot_atoms

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=120)
    plot_atoms(
        atoms,
        ax,
        radii=0.35,
        rotation=rotation,
        show_unit_cell=bool(getattr(atoms, "pbc", None) is not None and any(atoms.pbc)),
    )
    ax.set_title(title, fontsize=11)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_make_res(out: Path) -> None:
    import ase.io

    from mmml.paths import default_aco_template_pdb

    pdb = default_aco_template_pdb()
    atoms = ase.io.read(pdb)
    _save_plot_atoms(
        atoms,
        out,
        title="make-res: acetone monomer (ACO, CGENFF)",
        rotation="20x,30y,0z",
    )


def figure_make_box(out: Path) -> None:
    """Periodic box of acetone monomers (illustrates make-box / Packmol output)."""
    import numpy as np
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
    _save_plot_atoms(
        box,
        out,
        title=f"make-box: 8× acetone in {side:.0f} Å cube (illustrative)",
        rotation="65x,35y,0z",
    )


def _fallback_crystal_atoms():
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
    """PyXtal benzene cell when available; else ASE illustrative periodic cell."""
    try:
        from mmml.interfaces.pyxtal_placement import (
            MolecularCrystalBuildRequest,
            build_molecular_crystal_random,
            have_pyxtal,
        )

        if have_pyxtal():
            atoms, _meta = build_molecular_crystal_random(
                MolecularCrystalBuildRequest(
                    molecules=["c1ccccc1"],
                    stoichiometry=[2],
                    space_group=14,
                    dimension=3,
                    volume_factor=1.1,
                    seed=7,
                    max_attempts=40,
                )
            )
            title = "build-crystal: benzene (Z=2, space group 14)"
            rotation = "10x,80y,0z"
        else:
            raise ImportError("pyxtal not installed")
    except Exception as exc:
        print(f"build-crystal figure: using ASE fallback ({exc})", file=sys.stderr)
        atoms = _fallback_crystal_atoms()
        title = "build-crystal: benzene dimer in periodic cell (illustrative)"
        rotation = "25x,60y,0z"

    _save_plot_atoms(atoms, out, title=title, rotation=rotation)
    return True


def figure_liquid_box_schematic(out: Path) -> None:
    """Schematic density-prep ladder (matplotlib only)."""
    import matplotlib.pyplot as plt

    stages = ["Packmol", "MC staged", "MC target", "Lattice", "Certified"]
    density = [0.55, 0.78, 0.95, 0.99, 1.00]
    target = 1.0

    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=120)
    ax.plot(stages, density, "o-", color="#2563eb", linewidth=2, markersize=7, label="ρ / ρ_target")
    ax.axhline(target, color="#64748b", linestyle="--", linewidth=1, label="target density")
    ax.set_ylim(0.4, 1.05)
    ax.set_ylabel("Relative density")
    ax.set_title("liquid-box: density prep ladder (schematic)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate(rotation=20, ha="right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_compose_workflow(out: Path) -> None:
    """Bar chart: typical atom counts for structure builders."""
    import matplotlib.pyplot as plt

    labels = ["make-res\n(1 monomer)", "make-box\n(8× ACO)", "build-crystal\n(supercell)"]
    counts = [10, 80, 48]
    colors = ["#059669", "#2563eb", "#7c3aed"]

    fig, ax = plt.subplots(figsize=(6.0, 3.4), dpi=120)
    bars = ax.bar(labels, counts, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_ylabel("Atoms (examples)")
    ax.set_title("Structure builders — example system sizes")
    for bar, n in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, str(n), ha="center", fontsize=9)
    ax.set_ylim(0, max(counts) * 1.15)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate(*, check: bool = False) -> int:
    _use_agg()
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
