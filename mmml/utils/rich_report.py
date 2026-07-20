"""Shared Rich console helpers for MMML CLI and calculator setup (not for JAX-jitted code)."""

from __future__ import annotations

import os
import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Mapping, Sequence


_STATUS_STYLES = {
    "success": ("OK", "bold green"),
    "info": ("INFO", "bold cyan"),
    "warning": ("WARNING", "bold yellow"),
    "error": ("ERROR", "bold red"),
    "skipped": ("SKIPPED", "dim"),
}

_JSON_KEY_STYLE = "bold cyan"
_JSON_STRING_STYLE = "green"
_JSON_PATH_STYLE = "bold blue underline"
_JSON_NUMBER_STYLE = "bright_magenta"
_JSON_TRUE_STYLE = "bold green"
_JSON_FALSE_STYLE = "bold red"
_JSON_NULL_STYLE = "dim"
_JSON_EMPTY_STYLE = "green"
_JSON_ERROR_STYLE = "bold red"
_JSON_PATH_KEYS = frozenset(
    {
        "file",
        "filename",
        "path",
        "directory",
        "dir",
        "output",
        "output_dir",
        "summary",
        "trajectory",
        "checkpoint",
    }
)


def _looks_like_path(value: str, key: str | None) -> bool:
    key_lower = "" if key is None else key.lower()
    return (
        key_lower in _JSON_PATH_KEYS
        or key_lower.endswith(("_path", "_file", "_dir", "_directory"))
        or value.startswith(("/", "./", "../", "~/"))
    )


def _colored_json_text(value: Any, *, indent: int = 2):
    """Build a Rich ``Text`` rendering from an already-normalized JSON value."""

    from rich.text import Text

    out = Text()

    def whitespace(level: int) -> None:
        out.append(" " * (level * indent))

    def render(item: Any, level: int, key: str | None = None) -> None:
        if isinstance(item, dict):
            if not item:
                style = _JSON_EMPTY_STYLE if key != "errors" else _JSON_TRUE_STYLE
                out.append("{}", style=style)
                return
            container_style = _JSON_ERROR_STYLE if key == "errors" else None
            out.append("{", style=container_style)
            out.append("\n")
            entries = list(item.items())
            for index, (child_key, child) in enumerate(entries):
                whitespace(level + 1)
                out.append(json.dumps(child_key, ensure_ascii=False), style=_JSON_KEY_STYLE)
                out.append(": ")
                render(child, level + 1, child_key)
                if index + 1 < len(entries):
                    out.append(",")
                out.append("\n")
            whitespace(level)
            out.append("}", style=container_style)
            return
        if isinstance(item, list):
            if not item:
                out.append("[]", style=_JSON_EMPTY_STYLE)
                return
            out.append("[")
            out.append("\n")
            for index, child in enumerate(item):
                whitespace(level + 1)
                render(child, level + 1)
                if index + 1 < len(item):
                    out.append(",")
                out.append("\n")
            whitespace(level)
            out.append("]")
            return
        if isinstance(item, str):
            style = _JSON_PATH_STYLE if _looks_like_path(item, key) else _JSON_STRING_STYLE
            out.append(json.dumps(item, ensure_ascii=False), style=style)
            return
        if item is True:
            out.append("true", style=_JSON_TRUE_STYLE)
            return
        if item is False:
            out.append("false", style=_JSON_FALSE_STYLE)
            return
        if item is None:
            out.append("null", style=_JSON_NULL_STYLE)
            return
        if isinstance(item, (int, float)):
            out.append(json.dumps(item, allow_nan=False), style=_JSON_NUMBER_STYLE)
            return
        raise TypeError(f"unsupported normalized JSON value: {type(item).__name__}")

    render(value, 0)
    return out


def print_colored_json(
    value: Any,
    *,
    console: Any | None = None,
    indent: int = 2,
    sort_keys: bool = False,
    default: Any = None,
    quiet: bool = False,
    stderr: bool = False,
) -> None:
    """Print valid, indented JSON with compact semantic coloring.

    Paths are blue/underlined, keys cyan, strings green, numbers magenta,
    ``true`` green, ``false`` red, and ``null`` dim. Empty containers are green;
    a non-empty value under an ``errors`` key is red at its boundary. With Rich
    disabled, output is ordinary JSON suitable for copying or parsing.

    ``value`` must be JSON-serializable. Non-finite floats are rejected because
    ``NaN`` and infinities are not valid JSON.
    """

    if quiet or is_quiet():
        return
    if indent < 0:
        raise ValueError("indent must be non-negative")
    # Round-trip through the standard library to validate and normalize values
    # before rendering; this guarantees the colored text is also valid JSON.
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=sort_keys,
        default=default,
    )
    normalized = json.loads(serialized)
    if not rich_enabled(quiet=quiet):
        _emit_plain(
            json.dumps(normalized, ensure_ascii=False, indent=indent, sort_keys=sort_keys),
            stderr=stderr,
        )
        return
    target = console if console is not None else _console(stderr=stderr)
    target.print(_colored_json_text(normalized, indent=indent), soft_wrap=True)


def print_colored_python_repr(
    value: Any,
    *,
    console: Any | None = None,
    max_length: int | None = 20,
    max_string: int | None = 240,
    expand_all: bool = False,
    quiet: bool = False,
    stderr: bool = False,
) -> None:
    """Print a compact, colored Python representation of an object.

    This is intended for calculators, dataclasses, configuration objects, and
    other Python-native diagnostics where JSON would lose type information.
    Rich honors ``__rich_repr__`` when a class provides it and otherwise falls
    back to a bounded ``repr``. Plain mode prints ``repr(value)``.
    """

    if quiet or is_quiet():
        return
    if not rich_enabled(quiet=quiet):
        _emit_plain(repr(value), stderr=stderr)
        return
    from rich.pretty import Pretty

    target = console if console is not None else _console(stderr=stderr)
    target.print(
        Pretty(
            value,
            max_length=max_length,
            max_string=max_string,
            expand_all=expand_all,
        )
    )


@dataclass(frozen=True)
class CompactReporter:
    """Small, semantic Rich reports without panels or table borders.

    Choose the method from the information shape:

    - :meth:`status` for one event or outcome;
    - :meth:`summary` for key/value metadata;
    - :meth:`table` for repeated records with the same fields.

    Color is supplementary: plain output retains headings and status words.
    ``console`` is injectable for tests, capture, and callers using Rich Live.
    """

    console: Any | None = None
    quiet: bool = False
    stderr: bool = False

    def _active_console(self):
        return self.console if self.console is not None else _console(stderr=self.stderr)

    def _plain(self, text: str) -> None:
        _emit_plain(text, stderr=self.stderr)

    def _can_render(self) -> bool:
        return not self.quiet and not is_quiet() and rich_enabled(quiet=self.quiet)

    def status(
        self,
        level: str,
        message: str,
        *,
        detail: str | None = None,
    ) -> None:
        """Emit one compact, copy-friendly status line."""

        if self.quiet or is_quiet():
            return
        try:
            label, style = _STATUS_STYLES[level.lower()]
        except KeyError:
            raise ValueError(f"unknown status level {level!r}; expected {sorted(_STATUS_STYLES)}") from None
        suffix = f"  {detail}" if detail else ""
        if not self._can_render():
            self._plain(f"{label}  {message}{suffix}")
            return
        self._active_console().print(f"[{style}]{label:<7}[/{style}]  {message}{suffix}")

    def summary(self, title: str, rows: Mapping[str, Any] | Sequence[tuple[str, Any]]) -> None:
        """Emit a heading followed by a compact borderless key/value table."""

        if self.quiet or is_quiet():
            return
        items = list(rows.items() if isinstance(rows, Mapping) else rows)
        plain = "\n".join([title, *(f"{key}  {_format_cell(value)}" for key, value in items)])
        if not self._can_render():
            self._plain(plain)
            return
        table = make_compact_table(show_header=False)
        table.add_column(style="dim cyan", no_wrap=True)
        table.add_column()
        for key, value in items:
            table.add_row(str(key), _format_cell(value))
        console = self._active_console()
        console.print(f"[bold cyan]{title}[/bold cyan]")
        console.print(table)

    def table(
        self,
        title: str,
        columns: Sequence[str],
        rows: Iterable[Sequence[Any]],
        *,
        column_styles: Sequence[str | None] | None = None,
    ) -> None:
        """Emit repeated records as a compact borderless table."""

        if self.quiet or is_quiet():
            return
        materialized = [tuple(row) for row in rows]
        if any(len(row) != len(columns) for row in materialized):
            raise ValueError("every table row must have the same length as columns")
        if column_styles is not None and len(column_styles) != len(columns):
            raise ValueError("column_styles must have the same length as columns")
        if not self._can_render():
            lines = [title, "  ".join(str(column) for column in columns)]
            lines.extend("  ".join(_format_cell(value) for value in row) for row in materialized)
            self._plain("\n".join(lines))
            return
        table = make_compact_table(show_header=True)
        styles = column_styles or (None,) * len(columns)
        for column, style in zip(columns, styles, strict=True):
            table.add_column(str(column), style=style)
        for row in materialized:
            table.add_row(*(_format_cell(value) for value in row))
        console = self._active_console()
        console.print(f"[bold cyan]{title}[/bold cyan]")
        console.print(table)


def make_compact_table(*, show_header: bool = True):
    """Construct the single canonical copy-friendly Rich table style."""

    from rich.table import Table

    return Table(
        box=None,
        show_header=show_header,
        header_style="bold",
        show_edge=False,
        pad_edge=False,
        collapse_padding=True,
        padding=(0, 1),
        expand=False,
    )


def get_reporter(
    *,
    console: Any | None = None,
    quiet: bool = False,
    stderr: bool = False,
) -> CompactReporter:
    """Factory for the canonical compact CLI reporting interface."""

    return CompactReporter(console=console, quiet=quiet, stderr=stderr)


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


def _vertical_table_from_mapping(mapping: Mapping[str, Any]):
    from rich.table import Table

    table = Table(show_header=False, expand=True, show_edge=True)
    table.add_column("Field", style="cyan", no_wrap=True, ratio=1)
    table.add_column("Value", style="white", ratio=4)
    for key, value in mapping.items():
        table.add_row(str(key), _format_cell(value))
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

        vertical_sections = {"System", "Runtime threads", "Checkpoint"}
        blocks = []
        for section_title, mapping in active:
            table = (
                _vertical_table_from_mapping(mapping)
                if section_title in vertical_sections
                else _horizontal_table_from_mapping(mapping)
            )
            blocks.append(
                Panel(
                    table,
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


def collect_zbl_cutoff_mapping(model: Any) -> dict[str, Any] | None:
    """Extract ZBL repulsion cutoffs from a PhysNet/Spooky model when present.

    Returns ``None`` when the model has no ZBL flag (e.g. jax_mm_clone spoof with
    ``MODEL is None``). Pair-distance cutoffs (Å) are distinct from COM handoff.

    Fixed universal ZBL uses cuton/cutoff ≈ 0.1 / 0.6 Å.  Older trainable-ZBL
    checkpoints often omit those keys; loaders then infer ``cuton=None`` (switch
    from 0) and ``cutoff≈model cutoff`` (commonly 6 Å) with ``trainable=True``.
    """
    if model is None:
        return None
    if not hasattr(model, "zbl"):
        return None
    enabled = bool(getattr(model, "zbl", False))
    cuton = getattr(model, "zbl_cuton", None)
    cutoff = getattr(model, "zbl_cutoff", None)
    trainable = getattr(model, "trainable_zbl", None)
    # Some checkpoints store cutoffs only on the repulsion submodule.
    repulsion = getattr(model, "repulsion", None)
    if repulsion is not None:
        if cuton is None and not hasattr(model, "zbl_cuton"):
            cuton = getattr(repulsion, "cuton", None)
        if cutoff is None:
            cutoff = getattr(repulsion, "cutoff", None)
        if trainable is None:
            trainable = getattr(repulsion, "trainable", None)
    out: dict[str, Any] = {"enabled": enabled}
    # ``cuton is None`` means switch from 0 → cutoff (legacy trainable window).
    cuton_effective = 0.0 if cuton is None and enabled else cuton
    if cuton_effective is not None:
        try:
            out["cuton_Å"] = f"{float(cuton_effective):.4f}"
        except (TypeError, ValueError):
            out["cuton_Å"] = cuton_effective
    if cutoff is not None:
        try:
            out["cutoff_Å"] = f"{float(cutoff):.4f}"
        except (TypeError, ValueError):
            out["cutoff_Å"] = cutoff
    if trainable is not None:
        out["trainable"] = bool(trainable)
    try:
        cutoff_f = float(cutoff) if cutoff is not None else None
    except (TypeError, ValueError):
        cutoff_f = None
    if enabled and bool(trainable) and (cutoff_f is None or cutoff_f >= 1.0):
        out["mode"] = "legacy trainable (wide; not fixed 0.1–0.6)"
    elif enabled and not bool(trainable) and cutoff_f is not None and cutoff_f <= 1.0:
        out["mode"] = "fixed universal"
    out["distance"] = "pair r (Å), not COM"
    return out


def collect_short_range_wall_mapping(enabled: bool = True) -> dict[str, Any]:
    """Short-range inter-monomer wall settings, for the calculator summary.

    Reads the defaults from the single source of truth so the printout cannot
    drift from what the calculator actually evaluates.
    """
    from mmml.models.short_range_wall import (
        DEFAULT_WALL_K_EV_A2,
        DEFAULT_WALL_R_ON_A,
    )

    return {
        "enabled": bool(enabled),
        "r_on_Å": float(DEFAULT_WALL_R_ON_A),
        "k_eV_A2": float(DEFAULT_WALL_K_EV_A2),
    }


def collect_ml_energy_terms_mapping(
    model: Any,
    *,
    checkpoint_config: Mapping[str, Any] | None = None,
    mbd_loaded: bool = False,
    mbd_checkpoint: str | None = None,
    mbd_weight: float | None = None,
    mbd_missing_path: str | None = None,
) -> dict[str, Any]:
    """Human-readable which ML energy terms are active vs recorded in the ckpt."""
    cfg = dict(checkpoint_config or {})
    charges = False
    if model is not None and hasattr(model, "charges"):
        charges = bool(getattr(model, "charges"))
    elif "charges" in cfg or "predict_charges" in cfg:
        charges = bool(cfg.get("charges") or cfg.get("predict_charges"))
    if model is not None and hasattr(model, "include_electrostatics"):
        include_elec = bool(getattr(model, "include_electrostatics"))
    elif "include_electrostatics" in cfg:
        include_elec = bool(cfg["include_electrostatics"])
    else:
        include_elec = charges
    damp = getattr(model, "electrostatics_damping_sigma", None) if model is not None else None
    if damp is None:
        damp = cfg.get("electrostatics_damping_sigma")
    zbl = bool(getattr(model, "zbl", False)) if model is not None else bool(cfg.get("zbl", False))
    cgenff = not bool(cfg.get("no_cgenff_vdw", False)) if cfg else None
    cfg_mbd = cfg.get("mbd_checkpoint")
    out: dict[str, Any] = {
        "neural ML": "✓ (PhysNet/Spooky atomic)",
        "electrostatics": (
            f"✓ predicted charges (σ={float(damp):g} Å)"
            if include_elec and charges and damp is not None
            else ("✓ predicted charges" if include_elec and charges else "✗ off")
        ),
        "ZBL repulsion": "✓" if zbl else "✗ off",
    }
    if cgenff is not None:
        out["CGenFF LJ (training)"] = "✓ recorded" if cgenff else "✗ off"
    if mbd_loaded:
        w = 1.0 if mbd_weight is None else float(mbd_weight)
        path = mbd_checkpoint or (str(cfg_mbd) if cfg_mbd else "—")
        out["MBD dispersion"] = f"✓ loaded (weight={w:g})"
        out["MBD checkpoint"] = path
    elif cfg_mbd or mbd_missing_path:
        missing = mbd_missing_path or str(cfg_mbd)
        out["MBD dispersion"] = "✗ NOT loaded (checkpoint trained with MBD)"
        out["MBD checkpoint"] = f"missing: {missing}"
    else:
        out["MBD dispersion"] = "✗ not configured"
    return out


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
    zbl: Mapping[str, Any] | None = None,
    wall: Mapping[str, Any] | None = None,
    energy_terms: Mapping[str, Any] | None = None,
    quiet: bool = False,
) -> None:
    """Single dashboard for hybrid calculator setup (replaces duplicate setup/model panels)."""
    sections: list[tuple[str, Mapping[str, Any]]] = [
        ("System", system),
        ("Handoff & cutoffs", handoff),
    ]
    if energy_terms:
        sections.append(("ML energy terms", energy_terms))
    zbl_map = zbl if zbl is not None else collect_zbl_cutoff_mapping(model)
    if zbl_map:
        sections.append(("ZBL repulsion", zbl_map))
    if long_range:
        sections.append(("Long-range Coulomb", long_range))
    sections.extend(
        [
            ("Neighbor lists & ML batching", neighbor_lists),
            ("Model", _model_attributes_mapping(model) if model is not None else {"class": "—"}),
        ]
    )
    if runtime:
        sections.append(("Runtime threads", runtime))
    if ml_flags:
        sections.append(("ML/MM flags", ml_flags))
    if checkpoint:
        sections.append(("Checkpoint", checkpoint))
    emit_dashboard("Hybrid ML/MM setup", sections, border_style="cyan", quiet=quiet)


def emit_md_system_calculator_report(
    *,
    system: Mapping[str, Any] | None = None,
    handoff: Mapping[str, Any] | None = None,
    neighbor_lists: Mapping[str, Any] | None = None,
    model: Any = None,
    checkpoint: Mapping[str, Any] | None = None,
    ml_flags: Mapping[str, Any] | None = None,
    runtime: Mapping[str, Any] | None = None,
    long_range: Mapping[str, Any] | None = None,
    energy_terms: Mapping[str, Any] | None = None,
    cutoff_params: Any = None,
    model_type: str | None = None,
    n_monomers: int | None = None,
    n_atoms: int | None = None,
    doML: bool = True,
    doMM: bool = True,
    doML_dimer: bool = True,
    complementary_handoff: bool | None = None,
    ensemble: str | None = None,
    checkpoint_path: str | None = None,
    cell_L_A: float | None = None,
    mm_cutoff_A: float | None = None,
    capacity_pairs: int | None = None,
    n_valid_pairs: int | None = None,
    capacity_multiplier: float | None = None,
    skin_distance_A: float | None = None,
    update_interval_steps: int | None = None,
    jax_md_capacity: int | None = None,
    jax_md_n_valid: int | None = None,
    neighbor_extra: Mapping[str, Any] | None = None,
    calculator_extra: Mapping[str, Any] | None = None,
    zbl: Mapping[str, Any] | None = None,
    wall: Mapping[str, Any] | None = None,
    include_hybrid_setup: bool = True,
    include_calculator_summary: bool = True,
    include_neighbor_list_summary: bool = True,
    include_psf_topology: bool = True,
    quiet: bool = False,
) -> None:
    """Unified md-system calculator report: Track A dashboard + Track B ruler/NL + PSF.

    Track A (:func:`emit_hybrid_ml_setup`) covers system/handoff/model/runtime flags.
    Track B (:func:`mmml.cli.run.summaries.print_calculator_summary`) draws the
    COM-distance cutoff ruler and optional neighbor-list capacities.
    PSF/CGenFF topology is appended when CHARMM has a loaded PSF.
    """
    if quiet or is_quiet():
        return

    zbl_map = zbl if zbl is not None else collect_zbl_cutoff_mapping(model)
    wall_map = wall if wall is not None else collect_short_range_wall_mapping()
    energy_map = energy_terms
    if energy_map is None and model is not None:
        energy_map = collect_ml_energy_terms_mapping(model)

    if include_hybrid_setup and system is not None and handoff is not None:
        emit_hybrid_ml_setup(
            system=system,
            handoff=handoff,
            neighbor_lists=neighbor_lists or {},
            model=model if model is not None else object(),
            checkpoint=checkpoint,
            ml_flags=ml_flags,
            runtime=runtime,
            long_range=long_range,
            zbl=zbl_map,
            energy_terms=energy_map,
            quiet=quiet,
        )

    if include_calculator_summary and cutoff_params is not None:
        from mmml.cli.run.summaries import print_calculator_summary

        print_calculator_summary(
            cutoff_params,
            model_type=model_type,
            n_monomers=n_monomers,
            n_atoms=n_atoms,
            doML=doML,
            doMM=doMM,
            doML_dimer=doML_dimer,
            complementary_handoff=complementary_handoff,
            ensemble=ensemble,
            checkpoint=checkpoint_path,
            zbl=zbl_map,
            wall=wall_map,
            energy_terms=energy_map,
            extra=dict(calculator_extra) if calculator_extra else None,
        )

    if include_neighbor_list_summary and n_atoms is not None:
        has_nl_detail = any(
            v is not None
            for v in (
                cell_L_A,
                mm_cutoff_A,
                capacity_pairs,
                n_valid_pairs,
                capacity_multiplier,
                skin_distance_A,
                update_interval_steps,
                jax_md_capacity,
                jax_md_n_valid,
                neighbor_extra,
            )
        )
        if has_nl_detail or (neighbor_lists and any(neighbor_lists.values())):
            from mmml.cli.run.summaries import print_neighbor_list_summary

            extra: dict[str, Any] = {}
            if neighbor_lists:
                for key in (
                    "ml_sparse_dimers",
                    "dimers_total",
                    "max_active_dimers",
                    "ml_batch_size",
                    "ml_gpu_count",
                    "max_pairs",
                    "PBC",
                ):
                    if key in neighbor_lists:
                        extra[key] = neighbor_lists[key]
            if neighbor_extra:
                extra.update(neighbor_extra)
            print_neighbor_list_summary(
                n_atoms=int(n_atoms),
                n_monomers=n_monomers,
                cell_L_A=cell_L_A,
                mm_cutoff_A=mm_cutoff_A,
                capacity_pairs=capacity_pairs,
                n_valid_pairs=n_valid_pairs,
                capacity_multiplier=capacity_multiplier,
                skin_distance_A=skin_distance_A,
                update_interval_steps=update_interval_steps,
                jax_md_capacity=jax_md_capacity,
                jax_md_n_valid=jax_md_n_valid,
                extra=extra or None,
            )

    if include_psf_topology:
        emit_charmm_topology_summary(quiet=quiet)


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
    if model is None:
        return [("class", "—")]
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
        "zbl_cuton",
        "zbl_cutoff",
        "trainable_zbl",
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
