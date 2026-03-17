#!/usr/bin/env python3
"""
Pure PyCHARMM runner: heating and equilibration only (no MM/ML).

Runs CHARMM setup, minimization, heating, and equilibration. Does not run
ASE MD, JAX-MD, or any ML calculator. Use this for classical CHARMM-only
simulations or to prepare structures before running mmml run (MM/ML).

Usage:
    python -m mmml.cli.run.run_pycharmm --pdbfile pdb/init-packmol.pdb --cell 40
    mmml run-pycharmm --pdbfile pdb/init-packmol.pdb --cell 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mmml.interfaces.pycharmmInterface.import_pycharmm import coor
from mmml.interfaces.pycharmmInterface.setupBox import setup_box_generic
from mmml.cli.run.pycharmm_runner import (
    run_equilibration,
    run_heat,
    run_pycharmm_setup_and_minimize,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Pure PyCHARMM: heating and equilibration only (no MM/ML)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--pdbfile",
        type=Path,
        required=True,
        help="Path to the PDB file (requires correct atom names and types for CHARMM).",
    )
    parser.add_argument(
        "--cell",
        type=float,
        default=1000.0,
        help="Cubic cell side length in Å for periodic boundary conditions (default: 1000).",
    )
    parser.add_argument(
        "--skip-setup-energy-show",
        action="store_true",
        help="Skip energy.show() in setup_box for faster startup.",
    )
    parser.add_argument(
        "--pycharmm-minimize/--no-pycharmm-minimize",
        dest="pycharmm_minimize",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Run PyCHARMM nbonds/minimize before heat (default: True).",
    )
    parser.add_argument(
        "--pycharmm-minimize-steps",
        type=int,
        default=1000,
        metavar="N",
        help="ABNR minimization steps (default: 1000).",
    )
    parser.add_argument(
        "--view-braille",
        action="store_true",
        help="Display braille molecular viewer at each phase.",
    )
    return parser.parse_args()


def _make_braille_show_frame(live, args):
    """Return show_frame callback for braille viewer, or None if disabled."""
    if not getattr(args, "view_braille", False):
        return None
    from mmml.utils.visualize.braille_molecule import render_atoms_braille
    from rich.console import Group
    from rich.panel import Panel
    from rich.text import Text

    w = h = 100

    def show_frame(atoms, step, phase="md"):
        s = render_atoms_braille(atoms, width=w, height=h)
        live.update(Panel(Text.from_ansi(s), title=f"Step {step} ({phase})", border_style="cyan"))

    return show_frame


def run_pycharmm(args: argparse.Namespace):
    """Run pure PyCHARMM heating and equilibration."""
    pdbfilename = str(args.pdbfile)
    skip_energy_show = getattr(args, "skip_setup_energy_show", False)

    # Setup box and load PDB
    atoms = setup_box_generic(
        pdbfilename,
        side_length=args.cell,
        skip_energy_show=skip_energy_show,
    )

    if getattr(args, "view_braille", False):
        from rich.console import Console
        from rich.live import Live
        from rich.console import Group
        live_cm = Live(Group(), refresh_per_second=8, console=Console())
    else:
        from contextlib import nullcontext
        live_cm = nullcontext(None)

    with live_cm as live:
        show_frame = _make_braille_show_frame(live, args) if live is not None else None

        atoms = run_pycharmm_setup_and_minimize(atoms, args, show_frame=show_frame)
        atoms = run_heat(atoms, args, show_frame=show_frame)
        atoms = run_equilibration(atoms, args, show_frame=show_frame)

    # Sync final positions from CHARMM
    atoms.set_positions(coor.get_positions())
    return atoms


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    if not args.pdbfile.exists():
        print(f"Error: PDB file not found: {args.pdbfile}", file=sys.stderr)
        return 1
    run_pycharmm(args)
    print("PyCHARMM heating and equilibration complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
