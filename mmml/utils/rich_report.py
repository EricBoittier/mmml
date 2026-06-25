"""Shared Rich console helpers for MMML CLI and calculator setup (not for JAX-jitted code)."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from typing import Any, Iterable, Mapping, Sequence


def is_quiet() -> bool:
    return (os.environ.get("MMML_QUIET") or "").strip().lower() in ("1", "yes", "true")


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
        "total_charge",
        "n_res",
        "zbl",
        "debug",
        "efa",
        "use_energy_bias",
        "use_pbc",
        "include_electrostatics",
    )
    rows: list[tuple[str, Any]] = []
    for name in preferred:
        if hasattr(model, name):
            rows.append((name, getattr(model, name)))
    if rows:
        rows.insert(0, ("class", type(model).__name__))
        return rows
    return [("model", repr(model))]


def emit_model_loaded(
    model: Any,
    *,
    checkpoint: str | None = None,
    runtime_natoms: int | None = None,
    quiet: bool = False,
) -> None:
    """Pretty-print a loaded EF/PhysNet model summary."""
    rows = _model_attribute_rows(model)
    if checkpoint is not None:
        rows.append(("checkpoint", checkpoint))
    if runtime_natoms is not None:
        rows.append(("runtime_natoms", runtime_natoms))
    emit_table("Model loaded", rows, border_style="green", quiet=quiet)


def emit_setup_calculator_summary(
  rows: Sequence[tuple[str, Any]],
  *,
  quiet: bool = False,
) -> None:
    emit_table("setup_calculator", list(rows), border_style="cyan", quiet=quiet)


def emit_charmm_block(summary: str, *, quiet: bool = False) -> None:
    """One-line CHARMM BLOCK summary after a quiet script."""
    if quiet or is_quiet():
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
