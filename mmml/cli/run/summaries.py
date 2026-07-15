"""
Rich-formatted summaries for MD simulation system, forces, positions, charges, and masses.

Replaces raw print of arrays/energies with informative, readable summaries.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


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
                a_len, b_len, c_len = cell_arr.flat[:3]
            else:
                a_len = np.linalg.norm(cell_arr[0])
                b_len = np.linalg.norm(cell_arr[1])
                c_len = np.linalg.norm(cell_arr[2])
            table.add_row("Cell (Å)", f"a={a_len:.2f}, b={b_len:.2f}, c={c_len:.2f}")
        table.add_row("PBC", str(getattr(atoms, "pbc", False)))
    else:
        table.add_row("Cell", "None (non-periodic)")
        table.add_row("PBC", "False")

    if cutoff_params is not None:
        table.add_row(
            "ML switch width (Å)",
            str(getattr(cutoff_params, "ml_switch_width", getattr(cutoff_params, "ml_cutoff", "N/A"))),
        )
        table.add_row("MM switch-on (Å)", str(getattr(cutoff_params, "mm_switch_on", "N/A")))
        table.add_row(
            "MM switch width (Å)",
            str(getattr(cutoff_params, "mm_switch_width", getattr(cutoff_params, "mm_cutoff", "N/A"))),
        )

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


def print_flat_bottom_summary(
    result: Any,
    *,
    flat_bottom_radius: float | None,
    flat_bottom_k: float = 1.0,
    flat_bottom_mode: str = "system",
    label: str = "",
    console: Optional[Console] = None,
) -> None:
    """Print hybrid energy, flat-bottom term, and COM diagnostics."""
    if flat_bottom_radius is None or float(flat_bottom_radius) <= 0.0:
        return
    c = console or Console()
    mode = str(flat_bottom_mode).lower().strip()
    hybrid = float(np.asarray(result.hybrid_energy).reshape(()))
    flat_e = float(np.asarray(result.flat_bottom_E).reshape(()))
    total = float(np.asarray(result.energy).reshape(()))
    com = np.asarray(result.com, dtype=float).reshape(3)
    com_dist = float(np.asarray(result.com_dist).reshape(()))
    r_fb = float(flat_bottom_radius) if flat_bottom_radius is not None else 0.0
    excess = max(0.0, com_dist - r_fb) if r_fb > 0 else 0.0
    active = r_fb > 0 and excess > 1e-8
    dist_label = "max |COM_m| (Å)" if mode == "monomer" else "|COM| (Å)"

    table = Table(title="[bold magenta]Flat-bottom / COM[/bold magenta]", show_header=True)
    table.add_column("Quantity", style="bright_magenta", no_wrap=True)
    table.add_column("Value", style="white")
    if label:
        table.add_row("Stage", label)
    table.add_row("flat_bottom_mode", mode)
    table.add_row("E_hybrid (ML+MM)", f"{hybrid:.6f} eV")
    table.add_row("E_flat_bottom", f"{flat_e:.6f} eV")
    table.add_row("E_total", f"{total:.6f} eV")
    if mode == "system":
        table.add_row("COM (Å)", f"({com[0]:.4f}, {com[1]:.4f}, {com[2]:.4f})")
    table.add_row(dist_label, f"{com_dist:.4f}")
    table.add_row("R_flat_bottom (Å)", f"{r_fb:.4f}" if r_fb > 0 else "off")
    table.add_row("k_flat_bottom (eV/Å²)", f"{float(flat_bottom_k):.4f}" if r_fb > 0 else "—")
    excess_label = "max |COM_m| - R" if mode == "monomer" else "|COM| - R"
    table.add_row(f"{excess_label} excess (Å)", f"{excess:.4f}" if r_fb > 0 else "—")
    table.add_row("Restraint active", "yes" if active else "no")
    c.print(Panel(table, title="[bold]Flat-bottom[/bold]", border_style="magenta"))


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


# ─────────────────────────────────────────────────────────────────────────────
# Calculator / cutoff summary
# ─────────────────────────────────────────────────────────────────────────────

_BAR_WIDTH = 38  # characters for ASCII energy-range ruler


def _ascii_ruler(total_len: float, regions: list[tuple[float, float, str, str]], width: int = _BAR_WIDTH) -> Text:
    """
    Build a Rich Text object representing a horizontal ruler.

    regions: list of (start_Å, end_Å, rich_color, label).  Segments are
    rendered left→right; gaps between regions are filled with dim dashes.
    """
    if total_len <= 0:
        return Text("(no ruler)")

    chars = ["-"] * width
    colors: list[Optional[str]] = [None] * width

    for start, end, color, _label in regions:
        lo = max(0, int(round(start / total_len * width)))
        hi = min(width, int(round(end / total_len * width)))
        for i in range(lo, hi):
            chars[i] = "█"
            colors[i] = color

    t = Text()
    i = 0
    while i < width:
        c_color = colors[i]
        j = i + 1
        while j < width and colors[j] == c_color:
            j += 1
        seg = "".join(chars[i:j])
        if c_color:
            t.append(seg, style=c_color)
        else:
            t.append(seg, style="dim white")
        i = j
    return t


def _scale_bar(fraction: float, width: int = _BAR_WIDTH, color: str = "cyan") -> Text:
    """Simple filled progress bar for a [0, 1] fraction."""
    filled = max(0, min(width, int(round(fraction * width))))
    t = Text()
    t.append("█" * filled, style=color)
    t.append("░" * (width - filled), style="dim white")
    t.append(f" {fraction * 100:.1f}%", style="white")
    return t


def print_calculator_summary(
    cutoff_params: Any,
    *,
    model_type: Optional[str] = None,
    n_monomers: Optional[int] = None,
    n_atoms: Optional[int] = None,
    doML: bool = True,
    doMM: bool = True,
    doML_dimer: bool = True,
    complementary_handoff: Optional[bool] = None,
    ensemble: Optional[str] = None,
    checkpoint: Optional[str] = None,
    extra: Optional[dict] = None,
    console: Optional[Console] = None,
) -> None:
    """Print a rich, colored summary of the hybrid calculator configuration.

    Includes:
    * A property table with all cutoff/model parameters.
    * A colored ASCII ruler diagram showing where ML / MM / handoff zones
      are active along the COM-distance axis.
    """
    c = console or Console()

    ml_w = float(getattr(cutoff_params, "ml_switch_width", getattr(cutoff_params, "ml_cutoff", 1.5)))
    mm_on = float(getattr(cutoff_params, "mm_switch_on", 8.0))
    mm_w = float(getattr(cutoff_params, "mm_switch_width", getattr(cutoff_params, "mm_cutoff", 5.0)))
    if complementary_handoff is None:
        complementary_handoff = bool(getattr(cutoff_params, "complementary_handoff", True))

    # Key distances
    ml_full_end = mm_on - ml_w          # ML fully on  for r < ml_full_end
    mm_outer_end = mm_on + mm_w         # MM reaches 0 at mm_outer_end
    ruler_max = mm_outer_end * 1.15     # a bit of margin

    # ── ASCII ruler ──────────────────────────────────────────────────────────
    # Regions: (start, end, rich-style, label)
    ruler_regions = [
        (0.0,         ml_full_end, "bold bright_blue",   "ML=1"),
        (ml_full_end, mm_on,       "bold bright_yellow",  "handoff"),
        (mm_on,       mm_outer_end, "bold bright_red",   "MM↓"),
    ]
    ruler_line = _ascii_ruler(ruler_max, ruler_regions)

    # Tick marks below ruler
    tick_positions = {ml_full_end: f"{ml_full_end:.1f}", mm_on: f"{mm_on:.1f}", mm_outer_end: f"{mm_outer_end:.1f}"}
    tick_text = Text()
    prev = 0
    for pos in sorted(tick_positions):
        char_pos = int(round(pos / ruler_max * _BAR_WIDTH))
        pad = max(0, char_pos - prev)
        tick_text.append(" " * pad + "↑")
        prev = char_pos + 1

    label_text = Text()
    prev_pos = 0
    for pos in sorted(tick_positions):
        char_pos = int(round(pos / ruler_max * _BAR_WIDTH))
        label = tick_positions[pos]
        pad = max(0, char_pos - prev_pos)
        label_text.append(" " * pad + label)
        prev_pos = char_pos + len(label)

    # Legend
    legend = Text()
    legend.append("█", style="bold bright_blue")
    legend.append(" ML=1  ", style="white")
    legend.append("█", style="bold bright_yellow")
    legend.append(" handoff (ML↓,MM↑)  ", style="white")
    legend.append("█", style="bold bright_red")
    legend.append(" MM tail↓  ", style="white")
    legend.append("─", style="dim white")
    legend.append(" inactive", style="dim white")

    # ── Scale table ──────────────────────────────────────────────────────────
    table = Table(title="[bold green]Calculator Configuration[/bold green]", show_header=True)
    table.add_column("Parameter", style="bright_cyan", no_wrap=True)
    table.add_column("Value", style="white")

    if model_type:
        table.add_row("Model type", str(model_type))
    if checkpoint:
        table.add_row("Checkpoint", str(checkpoint))
    if ensemble:
        table.add_row("Ensemble", str(ensemble).upper())
    if n_monomers is not None:
        table.add_row("Monomers", str(n_monomers))
    if n_atoms is not None:
        table.add_row("Atoms", str(n_atoms))

    table.add_row("doML", "[green]✓[/green]" if doML else "[red]✗[/red]")
    table.add_row("doMM", "[green]✓[/green]" if doMM else "[red]✗[/red]")
    table.add_row("doML_dimer", "[green]✓[/green]" if doML_dimer else "[red]✗[/red]")
    table.add_row("Complementary handoff", "[green]✓[/green]" if complementary_handoff else "[yellow]legacy[/yellow]")
    table.add_row("─" * 22, "─" * 22)
    table.add_row("ml_switch_width (Å)", f"[bright_blue]{ml_w:.3f}[/bright_blue]")
    table.add_row("mm_switch_on (Å)", f"[bright_yellow]{mm_on:.3f}[/bright_yellow]")
    table.add_row("mm_switch_width (Å)", f"[bright_red]{mm_w:.3f}[/bright_red]")
    table.add_row("ML fully-on range (Å)", f"0 → [bright_blue]{ml_full_end:.3f}[/bright_blue]")
    table.add_row("ML/MM handoff range (Å)", f"[bright_blue]{ml_full_end:.3f}[/bright_blue] → [bright_yellow]{mm_on:.3f}[/bright_yellow]")
    table.add_row("MM tail range (Å)", f"[bright_yellow]{mm_on:.3f}[/bright_yellow] → [bright_red]{mm_outer_end:.3f}[/bright_red]")
    if extra:
        for k, v in extra.items():
            table.add_row(str(k), str(v))

    # Assemble panel content
    from rich.console import Group
    inner = Group(
        table,
        Text(""),
        Text("  COM-distance ruler (Å):", style="bold white"),
        Text("  ") + ruler_line,
        Text("  ") + tick_text,
        Text("  ") + label_text,
        Text("  ") + legend,
        Text(""),
        Text(f"  ruler scale: 0 → {ruler_max:.1f} Å", style="dim white"),
    )
    c.print(Panel(inner, title="[bold green]Calculator Summary[/bold green]", border_style="green"))


def print_neighbor_list_summary(
    *,
    n_atoms: int,
    n_monomers: Optional[int] = None,
    cell_L_A: Optional[float] = None,
    mm_cutoff_A: Optional[float] = None,
    capacity_pairs: Optional[int] = None,
    n_valid_pairs: Optional[int] = None,
    capacity_multiplier: Optional[float] = None,
    skin_distance_A: Optional[float] = None,
    update_interval_steps: Optional[int] = None,
    jax_md_capacity: Optional[int] = None,
    jax_md_n_valid: Optional[int] = None,
    extra: Optional[dict] = None,
    console: Optional[Console] = None,
) -> None:
    """Print a rich summary of neighbor-list configuration and initial capacities."""
    c = console or Console()

    table = Table(title="[bold magenta]Neighbor List Configuration[/bold magenta]", show_header=True)
    table.add_column("Property", style="bright_magenta", no_wrap=True)
    table.add_column("Value", style="white")

    table.add_row("Atoms", str(n_atoms))
    if n_monomers is not None:
        table.add_row("Monomers", str(n_monomers))
    if cell_L_A is not None:
        table.add_row("Cell side (Å)", f"{cell_L_A:.3f}")
    if mm_cutoff_A is not None:
        table.add_row("MM outer cutoff (Å)", f"{mm_cutoff_A:.3f}")
    if skin_distance_A is not None:
        table.add_row("Skin distance (Å)", f"{skin_distance_A:.3f}")
    if update_interval_steps is not None:
        table.add_row("Update interval (steps)", str(update_interval_steps))
    if capacity_multiplier is not None:
        table.add_row("Capacity multiplier", f"{capacity_multiplier:.2f}×")

    bars: list[tuple[str, int, int, str]] = []  # (label, n_valid, capacity, color)

    if capacity_pairs is not None and capacity_pairs > 0:
        table.add_row("─" * 22, "─" * 22)
        table.add_row("MM cell-list capacity", str(capacity_pairs))
        if n_valid_pairs is not None:
            table.add_row("MM valid pairs (init)", str(n_valid_pairs))
            fill = n_valid_pairs / capacity_pairs
            table.add_row("MM fill fraction", f"{fill * 100:.1f}%")
            bars.append(("MM pairs", n_valid_pairs, capacity_pairs, "bright_cyan"))

    if jax_md_capacity is not None and jax_md_capacity > 0:
        table.add_row("─" * 22, "─" * 22)
        table.add_row("JAX-MD NL capacity", str(jax_md_capacity))
        if jax_md_n_valid is not None:
            table.add_row("JAX-MD valid pairs (init)", str(jax_md_n_valid))
            fill_jax = jax_md_n_valid / jax_md_capacity
            table.add_row("JAX-MD fill fraction", f"{fill_jax * 100:.1f}%")
            bars.append(("JAX-MD NL", jax_md_n_valid, jax_md_capacity, "bright_green"))

    if extra:
        table.add_row("─" * 22, "─" * 22)
        for k, v in extra.items():
            table.add_row(str(k), str(v))

    # Build bar chart rows
    bar_lines: list[Text] = []
    if bars:
        bar_lines.append(Text(""))
        bar_lines.append(Text("  Capacity fill-fraction:", style="bold white"))
        for label, n_valid, capacity, color in bars:
            frac = n_valid / capacity
            bar = _scale_bar(frac, color=color)
            line = Text(f"  {label:<18} ")
            line.append_text(bar)
            line.append(f"  ({n_valid:,} / {capacity:,})", style="dim white")
            bar_lines.append(line)

    from rich.console import Group
    inner_parts: list[Any] = [table] + bar_lines
    c.print(Panel(Group(*inner_parts), title="[bold magenta]Neighbor Lists[/bold magenta]", border_style="magenta"))


def build_calculator_summary_dict(
    cutoff_params: Any,
    *,
    model_type: Optional[str] = None,
    n_monomers: Optional[int] = None,
    n_atoms: Optional[int] = None,
    doML: bool = True,
    doMM: bool = True,
    doML_dimer: bool = True,
    ensemble: Optional[str] = None,
    checkpoint: Optional[str] = None,
    nl_capacity_pairs: Optional[int] = None,
    nl_n_valid_pairs: Optional[int] = None,
    nl_capacity_multiplier: Optional[float] = None,
    nl_skin_distance_A: Optional[float] = None,
    nl_update_interval_steps: Optional[int] = None,
    jax_md_capacity: Optional[int] = None,
    jax_md_n_valid: Optional[int] = None,
    extra: Optional[dict] = None,
) -> dict:
    """Build a serialisable dict of calculator/NL configuration for JSON export."""
    ml_w = float(getattr(cutoff_params, "ml_switch_width", getattr(cutoff_params, "ml_cutoff", 1.5)))
    mm_on = float(getattr(cutoff_params, "mm_switch_on", 8.0))
    mm_w = float(getattr(cutoff_params, "mm_switch_width", getattr(cutoff_params, "mm_cutoff", 5.0)))
    comp = bool(getattr(cutoff_params, "complementary_handoff", True))
    d: dict = {
        "model_type": model_type,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "ensemble": ensemble,
        "n_monomers": n_monomers,
        "n_atoms": n_atoms,
        "doML": doML,
        "doMM": doMM,
        "doML_dimer": doML_dimer,
        "complementary_handoff": comp,
        "ml_switch_width_A": ml_w,
        "mm_switch_on_A": mm_on,
        "mm_switch_width_A": mm_w,
        "ml_fully_on_range_A": [0.0, mm_on - ml_w],
        "handoff_range_A": [mm_on - ml_w, mm_on],
        "mm_tail_range_A": [mm_on, mm_on + mm_w],
        "nl_capacity_pairs": nl_capacity_pairs,
        "nl_n_valid_pairs": nl_n_valid_pairs,
        "nl_capacity_multiplier": nl_capacity_multiplier,
        "nl_skin_distance_A": nl_skin_distance_A,
        "nl_update_interval_steps": nl_update_interval_steps,
        "jax_md_capacity": jax_md_capacity,
        "jax_md_n_valid": jax_md_n_valid,
    }
    if extra:
        d.update(extra)
    return d


def save_calculator_summary_json(
    path: Path | str,
    cutoff_params: Any,
    **kwargs,
) -> None:
    """Serialise calculator configuration to *path* as JSON."""
    d = build_calculator_summary_dict(cutoff_params, **kwargs)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(d, indent=2), encoding="utf-8")

