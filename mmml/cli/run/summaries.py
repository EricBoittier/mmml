"""
Rich-formatted summaries for MD simulation system, forces, positions, charges, and masses.

Replaces raw print of arrays/energies with informative, readable summaries.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


def _ensure_np(arr: Any) -> np.ndarray:
    """Convert JAX/other arrays to numpy for stats."""
    if hasattr(arr, "__array__"):
        return np.asarray(arr)
    return np.array(arr)


def print_system_summary(
    atoms: Any,
    n_monomers: int,
    atoms_per_monomer_list: list[int],
    cell: Optional[Any] = None,
    cutoff_params: Optional[Any] = None,
    calculator_info: Optional[str] = None,
    console: Optional[Console] = None,
) -> None:
    """Print a Rich summary of the simulation system."""
    c = console or Console()
    table = Table(title="[bold cyan]System Summary[/bold cyan]", show_header=True)
    table.add_column("Property", style="bright_cyan", no_wrap=True)
    table.add_column("Value", style="white")

    natoms = len(atoms)
    formula = atoms.get_chemical_formula(mode="hill") if hasattr(atoms, "get_chemical_formula") else "N/A"
    table.add_row("Atoms", str(natoms))
    table.add_row("Formula", formula)
    table.add_row("Monomers", str(n_monomers))
    table.add_row("Atoms per monomer", str(atoms_per_monomer_list))

    if cell is not None:
        cell_arr = np.asarray(cell)
        if cell_arr.size >= 3:
            if cell_arr.ndim == 1:
                a, b, c = cell_arr.flat[:3]
            else:
                a = np.linalg.norm(cell_arr[0])
                b = np.linalg.norm(cell_arr[1])
                c = np.linalg.norm(cell_arr[2])
            table.add_row("Cell (Å)", f"a={a:.2f}, b={b:.2f}, c={c:.2f}")
        table.add_row("PBC", str(getattr(atoms, "pbc", False)))
    else:
        table.add_row("Cell", "None (non-periodic)")
        table.add_row("PBC", "False")

    if cutoff_params is not None:
        table.add_row("ML cutoff (Å)", str(getattr(cutoff_params, "ml_cutoff", "N/A")))
        table.add_row("MM switch-on (Å)", str(getattr(cutoff_params, "mm_switch_on", "N/A")))
        table.add_row("MM cutoff (Å)", str(getattr(cutoff_params, "mm_cutoff", "N/A")))

    if calculator_info:
        table.add_row("Calculator", calculator_info)

    c.print(Panel(table, title="[bold]System[/bold]", border_style="cyan"))


def print_forces_summary(
    forces: Any,
    energy_eV: Optional[float] = None,
    unit: str = "eV/Å",
    console: Optional[Console] = None,
) -> None:
    """Print a Rich summary of forces (and optionally energy) instead of raw arrays."""
    c = console or Console()
    F = _ensure_np(forces)
    if F.size == 0:
        c.print(Panel("[yellow]No forces (empty array)[/yellow]", title="[bold]Forces[/bold]", border_style="yellow"))
        return

    table = Table(title="[bold green]Forces Summary[/bold green]", show_header=True)
    table.add_column("Statistic", style="bright_green", no_wrap=True)
    table.add_column("Value", style="white")

    if energy_eV is not None:
        table.add_row("Energy (eV)", f"{energy_eV:.6f}")

    mag = np.linalg.norm(F, axis=-1)
    table.add_row("Shape", str(F.shape))
    table.add_row("Min component", f"{float(np.min(F)):.4f} {unit}")
    table.add_row("Max component", f"{float(np.max(F)):.4f} {unit}")
    table.add_row("Mean |F|", f"{float(np.mean(mag)):.4f} {unit}")
    table.add_row("Max |F|", f"{float(np.max(mag)):.4f} {unit}")
    table.add_row("Std |F|", f"{float(np.std(mag)):.4f} {unit}")

    c.print(Panel(table, title="[bold]Forces[/bold]", border_style="green"))


def print_positions_summary(
    positions: Any,
    atoms: Optional[Any] = None,
    title: str = "Positions",
    console: Optional[Console] = None,
) -> None:
    """Print a Rich summary of positions instead of raw arrays."""
    c = console or Console()
    R = _ensure_np(positions)
    if R.ndim == 3:
        # (n_frames, n_atoms, 3)
        n_frames, n_atoms, _ = R.shape
        R_flat = R.reshape(-1, 3)
    else:
        n_frames = 1
        n_atoms = R.shape[0] if R.ndim >= 2 else 0
        R_flat = R.reshape(-1, 3)

    if R_flat.size == 0:
        c.print(Panel("[yellow]No positions (empty array)[/yellow]", title=f"[bold]{title}[/bold]", border_style="yellow"))
        return

    table = Table(title=f"[bold blue]{title}[/bold blue]", show_header=True)
    table.add_column("Statistic", style="bright_blue", no_wrap=True)
    table.add_column("Value", style="white")

    table.add_row("Shape", str(R.shape))
    if n_frames > 1:
        table.add_row("Frames", str(n_frames))
    table.add_row("Atoms", str(n_atoms))

    mins = np.min(R_flat, axis=0)
    maxs = np.max(R_flat, axis=0)
    table.add_row("Bounds x (Å)", f"[{mins[0]:.2f}, {maxs[0]:.2f}]")
    table.add_row("Bounds y (Å)", f"[{mins[1]:.2f}, {maxs[1]:.2f}]")
    table.add_row("Bounds z (Å)", f"[{mins[2]:.2f}, {maxs[2]:.2f}]")

    com = np.mean(R_flat, axis=0)
    table.add_row("COM (Å)", f"({com[0]:.2f}, {com[1]:.2f}, {com[2]:.2f})")

    c.print(Panel(table, title=f"[bold]{title}[/bold]", border_style="blue"))


def print_charges_summary(
    charges: Any,
    console: Optional[Console] = None,
) -> None:
    """Print a Rich summary of atomic charges instead of raw arrays."""
    c = console or Console()
    q = _ensure_np(charges).flatten()
    if q.size == 0:
        c.print(Panel("[yellow]No charges (empty array)[/yellow]", title="[bold]Charges[/bold]", border_style="yellow"))
        return

    table = Table(title="[bold magenta]Charges Summary[/bold magenta]", show_header=True)
    table.add_column("Statistic", style="bright_magenta", no_wrap=True)
    table.add_column("Value", style="white")

    table.add_row("Count", str(len(q)))
    table.add_row("Sum (total charge)", f"{float(np.sum(q)):.6f} e")
    table.add_row("Min", f"{float(np.min(q)):.4f} e")
    table.add_row("Max", f"{float(np.max(q)):.4f} e")
    table.add_row("Mean", f"{float(np.mean(q)):.4f} e")
    table.add_row("Std", f"{float(np.std(q)):.4f} e")

    c.print(Panel(table, title="[bold]Charges[/bold]", border_style="magenta"))


def print_masses_summary(
    masses: Any,
    console: Optional[Console] = None,
) -> None:
    """Print a Rich summary of atomic masses instead of raw arrays."""
    c = console or Console()
    m = _ensure_np(masses).flatten()
    if m.size == 0:
        c.print(Panel("[yellow]No masses (empty array)[/yellow]", title="[bold]Masses[/bold]", border_style="yellow"))
        return

    table = Table(title="[bold yellow]Masses Summary[/bold yellow]", show_header=True)
    table.add_column("Statistic", style="bright_yellow", no_wrap=True)
    table.add_column("Value", style="white")

    table.add_row("Count", str(len(m)))
    table.add_row("Total mass (amu)", f"{float(np.sum(m)):.4f}")
    table.add_row("Min (amu)", f"{float(np.min(m)):.4f}")
    table.add_row("Max (amu)", f"{float(np.max(m)):.4f}")
    table.add_row("Mean (amu)", f"{float(np.mean(m)):.4f}")

    c.print(Panel(table, title="[bold]Masses[/bold]", border_style="yellow"))
