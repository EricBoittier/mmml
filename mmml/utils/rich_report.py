"""Shared Rich console helpers for MMML CLI and calculator setup (not for JAX-jitted code)."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from typing import Any, Iterable, Mapping, Sequence


def is_quiet() -> bool:
    return (os.environ.get("MMML_QUIET") or "").strip().lower() in ("1", "yes", "true")


def is_verbose() -> bool:
    return (os.environ.get("MMML_VERBOSE") or "").strip().lower() in ("1", "yes", "true")


def rich_enabled(*, quiet: bool = False) -> bool:
    if quiet or is_quiet():
        return False
    if (os.environ.get("MMML_NO_RICH") or "").strip().lower() in ("1", "yes", "true"):
        return False
    return True


def force_rich() -> bool:
    return (os.environ.get("MMML_RICH") or "").strip().lower() in ("1", "yes", "true")


@lru_cache(maxsize=1)
def _console(stderr: bool = False):
    from rich.console import Console

    return Console(
        stderr=stderr,
        force_terminal=force_rich() or None,
        no_color=not force_rich() and not sys.stdout.isatty(),
    )


def _emit_plain(message: str, *, stderr: bool = False) -> None:
    stream = sys.stderr if stderr else sys.stdout
    print(message, file=stream, flush=True)


def emit(message: str, *, quiet: bool = False, stderr: bool = False) -> None:
    """Print a line (Rich when enabled, plain otherwise)."""
    if quiet or is_quiet():
        return
    if not rich_enabled(quiet=quiet):
        _emit_plain(message, stderr=stderr)
        return
    try:
        _console(stderr=stderr).print(message)
    except Exception:
        _emit_plain(message, stderr=stderr)


def emit_tagged(
    tag: str,
    message: str,
    *,
    tag_style: str = "bold cyan",
    quiet: bool = False,
    stderr: bool = False,
) -> None:
    """``[tag] message`` with optional Rich styling."""
    plain = f"[{tag}] {message}"
    use_styled = rich_enabled(quiet=quiet) and (force_rich() or sys.stdout.isatty())
    if quiet or is_quiet() or not use_styled:
        _emit_plain(plain, stderr=stderr)
        return
    try:
        _console(stderr=stderr).print(f"[{tag_style}][{tag}][/{tag_style}] {message}")
    except Exception:
        _emit_plain(plain, stderr=stderr)


def emit_overlap_log(
    detail: str,
    *,
    context: str | None = None,
    quiet: bool = False,
) -> None:
    """Rich-tagged overlap / dynamics-guard note."""
    if context:
        emit_tagged(
            f"overlap ({context})",
            detail,
            tag_style="bold yellow",
            quiet=quiet,
        )
    else:
        emit_tagged("overlap", detail, tag_style="bold yellow", quiet=quiet)


def emit_panel(
    title: str,
    body: str,
    *,
    border_style: str = "blue",
    quiet: bool = False,
    stderr: bool = False,
) -> None:
    if quiet or is_quiet():
        return
    if not rich_enabled(quiet=quiet):
        _emit_plain(f"{title}\n{body}", stderr=stderr)
        return
    try:
        from rich.panel import Panel

        _console(stderr=stderr).print(
            Panel(body, title=f"[bold]{title}[/bold]", border_style=border_style)
        )
    except Exception:
        _emit_plain(f"{title}\n{body}", stderr=stderr)


def emit_table(
    title: str,
    rows: Sequence[tuple[str, Any]],
    *,
    border_style: str = "blue",
    quiet: bool = False,
    stderr: bool = False,
) -> None:
    if quiet or is_quiet():
        return
    plain_lines = [title, *(f"  {k}: {v}" for k, v in rows)]
    if not rich_enabled(quiet=quiet):
        _emit_plain("\n".join(plain_lines), stderr=stderr)
        return
    try:
        from rich.panel import Panel
        from rich.table import Table

        table = Table(show_header=True, header_style="bold", expand=True)
        table.add_column("Field", style="cyan", no_wrap=True)
        table.add_column("Value", style="white")
        for key, value in rows:
            table.add_row(str(key), _format_cell(value))
        _console(stderr=stderr).print(
            Panel(table, title=f"[bold]{title}[/bold]", border_style=border_style)
        )
    except Exception:
        _emit_plain("\n".join(plain_lines), stderr=stderr)


def _format_cell(value: Any) -> str:
    if isinstance(value, (list, tuple)) and len(value) > 12:
        head = ", ".join(repr(x) for x in value[:6])
        return f"[{head}, …] ({len(value)} items)"
    text = str(value)
    if len(text) > 240:
        return text[:237] + "…"
    return text


_HORIZONTAL_STYLES = (
    "cyan",
    "green",
    "yellow",
    "magenta",
    "blue",
    "bright_cyan",
    "bright_green",
    "bright_yellow",
    "bright_magenta",
    "bright_blue",
)


def _mapping_from_rows(rows: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    return {str(k): v for k, v in rows}


def _horizontal_table_from_mapping(
    mapping: Mapping[str, Any],
    *,
    title: str | None = None,
):
    from rich.table import Table

    table = Table(
        title=title,
        show_header=True,
        header_style="bold",
        expand=True,
        show_edge=True,
    )
    keys = list(mapping.keys())
    for i, key in enumerate(keys):
        table.add_column(str(key), style=_HORIZONTAL_STYLES[i % len(_HORIZONTAL_STYLES)])
    if keys:
        table.add_row(*[_format_cell(mapping[k]) for k in keys])
    return table


def emit_horizontal_table(
    title: str,
    mapping: Mapping[str, Any],
    *,
    quiet: bool = False,
    stderr: bool = False,
) -> None:
    """Model-Attributes style table: field names as columns, one value row."""
    if quiet or is_quiet() or not mapping:
        return
    plain = [title, "  " + "  ".join(f"{k}={_format_cell(v)}" for k, v in mapping.items())]
    if not rich_enabled(quiet=quiet):
        _emit_plain("\n".join(plain), stderr=stderr)
        return
    try:
        from rich.panel import Panel

        _console(stderr=stderr).print(
            Panel(
                _horizontal_table_from_mapping(mapping, title=None),
                title=f"[bold]{title}[/bold]",
                border_style="blue",
            )
        )
    except Exception:
        _emit_plain("\n".join(plain), stderr=stderr)


def _model_attributes_mapping(model: Any) -> dict[str, Any]:
    return _mapping_from_rows(_model_attribute_rows(model))


def emit_dashboard(
    title: str,
    sections: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    border_style: str = "cyan",
    quiet: bool = False,
) -> None:
    """Multi-section Rich panel (plain-text fallback when Rich is disabled)."""
    if quiet or is_quiet():
        return

    active = [(t, m) for t, m in sections if m]
    if not active:
        return

    if not rich_enabled(quiet=quiet):
        lines = [title]
        for section_title, mapping in active:
            lines.append(f"[{section_title}]")
            lines.extend(f"  {k}: {_format_cell(v)}" for k, v in mapping.items())
        _emit_plain("\n".join(lines))
        return

    try:
        from rich.console import Group
        from rich.panel import Panel

        blocks = []
        for section_title, mapping in active:
            blocks.append(
                Panel(
                    _horizontal_table_from_mapping(mapping),
                    title=f"[bold]{section_title}[/bold]",
                    border_style="dim",
                    padding=(0, 1),
                )
            )
        _console().print(
            Panel(
                Group(*blocks),
                title=f"[bold {border_style}]{title}[/bold {border_style}]",
                border_style=border_style,
            )
        )
    except Exception:
        lines = [title]
        for section_title, mapping in active:
            lines.append(f"[{section_title}]")
            lines.extend(f"  {k}: {_format_cell(v)}" for k, v in mapping.items())
        _emit_plain("\n".join(lines))


def emit_hybrid_ml_setup(
    *,
    system: Mapping[str, Any],
    handoff: Mapping[str, Any],
    neighbor_lists: Mapping[str, Any],
    model: Any,
    checkpoint: Mapping[str, Any] | None = None,
    ml_flags: Mapping[str, Any] | None = None,
    runtime: Mapping[str, Any] | None = None,
    long_range: Mapping[str, Any] | None = None,
    quiet: bool = False,
) -> None:
    """Single dashboard for hybrid calculator setup (replaces duplicate setup/model panels)."""
    sections: list[tuple[str, Mapping[str, Any]]] = [
        ("System", system),
        ("Handoff & cutoffs", handoff),
    ]
    if long_range:
        sections.append(("Long-range Coulomb", long_range))
    sections.extend(
        [
            ("Neighbor lists & ML batching", neighbor_lists),
            ("Model", _model_attributes_mapping(model)),
        ]
    )
    if runtime:
        sections.append(("Runtime threads", runtime))
    if ml_flags:
        sections.append(("ML/MM flags", ml_flags))
    if checkpoint:
        sections.append(("Checkpoint", checkpoint))
    emit_dashboard("Hybrid ML/MM setup", sections, border_style="cyan", quiet=quiet)


def collect_psf_topology_mapping(
    *,
    max_residue_rows: int = 6,
    max_type_samples: int = 8,
) -> dict[str, Any] | None:
    """Summarize in-memory CHARMM PSF when PyCHARMM is loaded."""
    try:
        from mmml.interfaces.pycharmmInterface.charmm_mpi import charmm_lib_available

        if not charmm_lib_available():
            return None
        import numpy as np
        import pycharmm.coor as coor
        import pycharmm.psf as psf
    except Exception:
        return None

    try:
        n_atom = int(coor.get_natom())
    except Exception:
        return None
    if n_atom <= 0:
        return None

    masses = np.asarray(psf.get_amass(), dtype=float)
    charges = np.asarray(psf.get_charges(), dtype=float)
    atom_names = [str(x) for x in np.asarray(psf.get_atype(), dtype=str)]
    iac = np.asarray(psf.get_iac(), dtype=int)

    unique_names, name_counts = np.unique(atom_names, return_counts=True)
    name_parts = [
        f"{t}×{int(c)}"
        for t, c in zip(unique_names[:max_type_samples], name_counts[:max_type_samples])
    ]
    if len(unique_names) > max_type_samples:
        name_parts.append(f"…+{len(unique_names) - max_type_samples} names")

    chem_type_parts: list[str] = []
    try:
        from pycharmm import atom_info

        chem_types = atom_info.get_chem_types(list(range(n_atom)))
        unique_chem, chem_counts = np.unique(chem_types, return_counts=True)
        chem_type_parts = [
            f"{t}×{int(c)}"
            for t, c in zip(unique_chem[:max_type_samples], chem_counts[:max_type_samples])
        ]
        if len(unique_chem) > max_type_samples:
            chem_type_parts.append(f"…+{len(unique_chem) - max_type_samples} types")
    except Exception:
        chem_type_parts = []

    n_res, res_label = _psf_residue_summary(
        psf,
        n_atom=n_atom,
        max_residue_rows=max_residue_rows,
    )

    mapping: dict[str, Any] = {
        "n_atoms": n_atom,
        "n_residues": n_res,
        "total_charge": f"{float(np.sum(charges)):.4f} e",
        "mass_range_amu": f"{float(masses.min()):.3f}–{float(masses.max()):.3f}",
        "atom_names": ", ".join(name_parts) if name_parts else "—",
        "iac_index_range": f"{int(iac.min())}–{int(iac.max())}",
        "residues": res_label,
    }
    if chem_type_parts:
        mapping["cgenff_types"] = ", ".join(chem_type_parts)
    return mapping


def _psf_residue_summary(
    psf: Any,
    *,
    n_atom: int,
    max_residue_rows: int,
) -> tuple[int | str, str]:
    """Return (n_residues, compact residue label) from in-memory PSF."""
    from collections import Counter

    try:
        n_res = int(psf.get_nres()) if hasattr(psf, "get_nres") else 0
    except Exception:
        n_res = 0

    if n_res <= 0:
        return "?", "—"

    resnames: list[str] = []
    if hasattr(psf, "get_res"):
        try:
            resnames = [str(x).strip() for x in psf.get_res()]
        except Exception:
            resnames = []

    resids: list[int] = []
    if hasattr(psf, "get_resid"):
        try:
            resids = [int(str(x).strip()) for x in psf.get_resid()]
        except Exception:
            resids = []

    parts: list[str] = []
    try:
        if len(resnames) == n_res:
            name_counts = Counter(resnames)
            seen_names: set[str] = set()
            ordered_names: list[str] = []
            for name in resnames:
                if name in seen_names:
                    continue
                seen_names.add(name)
                ordered_names.append(name)
            for name in ordered_names:
                parts.append(f"{name}×{name_counts[name]}")
                if len(parts) >= max_residue_rows:
                    if len(name_counts) > max_residue_rows:
                        parts.append("…")
                    break
        elif len(resids) == n_res:
            id_counts = Counter(resids)
            for rid, count in sorted(id_counts.items()):
                parts.append(f"res{rid}×{count}")
                if len(parts) >= max_residue_rows:
                    if len(id_counts) > max_residue_rows:
                        parts.append("…")
                    break
        elif len(resids) == n_atom and n_atom > 0:
            names_by_rid: list[str] = []
            for rid in sorted(set(resids)):
                if 0 < rid <= len(resnames):
                    names_by_rid.append(resnames[rid - 1])
                else:
                    names_by_rid.append(f"res{rid}")
            name_counts = Counter(names_by_rid)
            seen_names = set()
            ordered_names: list[str] = []
            for name in names_by_rid:
                if name in seen_names:
                    continue
                seen_names.add(name)
                ordered_names.append(name)
            for name in ordered_names:
                parts.append(f"{name}×{name_counts[name]}")
                if len(parts) >= max_residue_rows:
                    if len(name_counts) > max_residue_rows:
                        parts.append("…")
                    break
    except Exception:
        parts = []

    if parts:
        return n_res, ", ".join(parts)
    if resids and len(resids) == n_res:
        lo, hi = min(resids), max(resids)
        if lo == hi:
            return n_res, f"res{lo}"
        return n_res, f"res{lo}–res{hi}"
    return n_res, f"{n_res} residue(s)"


def emit_charmm_topology_summary(*, quiet: bool = False) -> bool:
    """Rich block for PSF atom types, charges, masses (no-op when PSF not loaded)."""
    mapping = collect_psf_topology_mapping()
    if not mapping:
        return False
    emit_horizontal_table("CHARMM topology (PSF)", mapping, quiet=quiet)
    return True


_MODEL_ATTR_LABELS: dict[str, str] = {
    "natoms": "max_padded_atoms",
    "n_res": "n_refinement_blocks",
    "num_iterations": "message_passing_steps",
    "runtime_natoms": "runtime_max_padded_atoms",
}


def _model_attr_label(name: str) -> str:
    return _MODEL_ATTR_LABELS.get(name, name)


def _model_attribute_rows(model: Any) -> list[tuple[str, Any]]:
    preferred = (
        "features",
        "max_degree",
        "num_iterations",
        "num_basis_functions",
        "cutoff",
        "max_atomic_number",
        "charges",
        "natoms",
        "max_padded_atoms",
        "total_charge",
        "n_res",
        "n_refinement_blocks",
        "zbl",
        "debug",
        "efa",
        "use_energy_bias",
        "use_pbc",
        "include_electrostatics",
    )
    seen_labels: set[str] = set()
    rows: list[tuple[str, Any]] = []
    for name in preferred:
        if not hasattr(model, name):
            continue
        label = _model_attr_label(name)
        if label in seen_labels:
            continue
        seen_labels.add(label)
        rows.append((label, getattr(model, name)))
    if rows:
        rows.insert(0, ("class", type(model).__name__))
        return rows
    return [("model", repr(model))]


def emit_model_loaded(
    model: Any,
    *,
    checkpoint: str | None = None,
    runtime_max_padded_atoms: int | None = None,
    runtime_natoms: int | None = None,
    quiet: bool = False,
) -> None:
    """Pretty-print a loaded PhysNet model summary (horizontal table)."""
    mapping = _model_attributes_mapping(model)
    if checkpoint is not None:
        mapping["checkpoint"] = checkpoint
    runtime = runtime_max_padded_atoms if runtime_max_padded_atoms is not None else runtime_natoms
    if runtime is not None:
        mapping["runtime_max_padded_atoms"] = runtime
    emit_horizontal_table("Model", mapping, quiet=quiet)


def emit_setup_calculator_summary(
    rows: Sequence[tuple[str, Any]],
    *,
    quiet: bool = False,
) -> None:
    """Legacy field/value panel — prefer :func:`emit_hybrid_ml_setup`."""
    emit_table("setup_calculator", list(rows), border_style="cyan", quiet=quiet)


def emit_charmm_block(summary: str, *, quiet: bool = False, verbose: bool = False) -> None:
    """One-line CHARMM BLOCK summary after a quiet script (verbose only by default)."""
    if quiet or is_quiet() or not (verbose or is_verbose()):
        return
    plain = summary if summary.startswith("CHARMM BLOCK:") else f"CHARMM BLOCK: {summary}"
    if not rich_enabled(quiet=quiet):
        _emit_plain(plain)
        return
    try:
        from rich.panel import Panel

        body = plain.removeprefix("CHARMM BLOCK:").strip()
        _console().print(
            Panel(body, title="[bold yellow]CHARMM BLOCK[/bold yellow]", border_style="yellow")
        )
    except Exception:
        _emit_plain(plain)


def emit_charmm_env(
    *,
    cgenff_rtf: str,
    cgenff_prm: str,
    charmm_home: str,
    charmm_lib_dir: str,
    quiet: bool = False,
) -> None:
    if quiet or is_quiet():
        return
    rows = [
        ("CGENFF RTF", cgenff_rtf),
        ("CGENFF PRM", cgenff_prm),
        ("CHARMM_HOME", charmm_home),
        ("CHARMM_LIB_DIR", charmm_lib_dir),
    ]
    emit_table("PyCHARMM environment", rows, border_style="dim", quiet=quiet)


def emit_jax_compile_pass(
    label: str,
    pass_index: int,
    wall_seconds: float,
    *,
    quiet: bool = False,
) -> None:
    phase = "compile+run" if pass_index == 0 else "run"
    message = (
        f"mmml: JAX compile timer [{label}] pass {pass_index + 1} ({phase}): "
        f"{wall_seconds:.2f}s"
    )
    use_styled = rich_enabled(quiet=quiet) and (force_rich() or sys.stdout.isatty())
    if quiet or is_quiet() or not use_styled:
        _emit_plain(message)
        return
    try:
        _console().print(
            f"[bold magenta]mmml[/bold magenta]: JAX compile timer "
            f"[cyan]{label}[/cyan] pass {pass_index + 1} "
            f"([dim]{phase}[/dim]): [bold]{wall_seconds:.2f}s[/bold]"
        )
    except Exception:
        _emit_plain(message)


def emit_jax_compile_label_summary(
    label: str,
    compile_s: float,
    run_s: float,
    *,
    quiet: bool = False,
) -> None:
    message = (
        f"mmml: JAX compile timer [{label}] summary: "
        f"compile≈{compile_s:.2f}s, run≈{run_s:.2f}s"
    )
    use_styled = rich_enabled(quiet=quiet) and (force_rich() or sys.stdout.isatty())
    if quiet or is_quiet() or not use_styled:
        _emit_plain(message)
        return
    try:
        _console().print(
            f"[bold magenta]mmml[/bold magenta]: JAX compile timer "
            f"[cyan]{label}[/cyan] summary: "
            f"compile≈[yellow]{compile_s:.2f}s[/yellow], "
            f"run≈[green]{run_s:.2f}s[/green]"
        )
    except Exception:
        _emit_plain(message)


def emit_jax_compile_session_summary(
    lines: Sequence[str],
    *,
    quiet: bool = False,
) -> None:
    if quiet or is_quiet() or not lines:
        return
    use_styled = rich_enabled(quiet=quiet) and (force_rich() or sys.stdout.isatty())
    if not use_styled:
        for line in lines:
            _emit_plain(line)
        return
    try:
        from rich.panel import Panel
        from rich.table import Table

        table = Table(show_header=True, header_style="bold")
        table.add_column("Kernel", style="cyan")
        table.add_column("Compile (s)", justify="right", style="yellow")
        table.add_column("Run (s)", justify="right", style="green")
        table.add_column("Pass 1 (s)", justify="right", style="dim")
        header = lines[0]
        for line in lines[1:]:
            if not line.strip():
                continue
            # "  label: compile≈X.XXs, run≈Y.YYs (pass1=Z.ZZs)"
            body = line.strip()
            if body.endswith(")"):
                body, pass1_part = body.rsplit("(pass1=", 1)
                pass1 = pass1_part.rstrip(")").rstrip("s")
            else:
                pass1 = "—"
            name, rest = body.split(":", 1)
            name = name.strip()
            compile_s = "—"
            run_s = "—"
            if "compile≈" in rest:
                try:
                    compile_s = rest.split("compile≈", 1)[1].split("s", 1)[0]
                except Exception:
                    pass
            if "run≈" in rest:
                try:
                    run_s = rest.split("run≈", 1)[1].split("s", 1)[0]
                except Exception:
                    pass
            table.add_row(name, compile_s, run_s, pass1)
        _console().print(
            Panel(
                table,
                title=f"[bold magenta]{header}[/bold magenta]",
                border_style="magenta",
            )
        )
    except Exception:
        for line in lines:
            _emit_plain(line)


def emit_status(ok: bool, message: str, *, quiet: bool = False) -> None:
    prefix = "PASS" if ok else "FAIL"
    if quiet or is_quiet():
        return
    if not rich_enabled(quiet=quiet):
        _emit_plain(f"{prefix}: {message}")
        return
    try:
        style = "bold green" if ok else "bold red"
        _console().print(f"[{style}]{prefix}[/]: {message}")
    except Exception:
        _emit_plain(f"{prefix}: {message}")


def emit_factory_summary(
    title: str,
    rows: Mapping[str, Any] | Iterable[tuple[str, Any]],
    *,
    quiet: bool = False,
) -> None:
    if isinstance(rows, Mapping):
        items = list(rows.items())
    else:
        items = list(rows)
    emit_table(title, items, border_style="blue", quiet=quiet)
