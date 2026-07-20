#!/usr/bin/env python3
"""Inventory and statically audit CHARMM's Fortran/Python C API surface.

This deliberately requires neither a CHARMM build nor a GPU.  It inspects every
``bind(c)`` routine, derived type, and enum under ``setup/charmm/source/api``
and every direct ``lib.charmm.<symbol>`` use in the vendored PyCHARMM package,
then writes a machine-readable JSON report and a navigable Markdown report.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import ctypes
import dataclasses
import json
from pathlib import Path
import re
from typing import Iterable


ROUTINE_RE = re.compile(
    r"^[ \t]*(?P<prefix>(?:(?:integer|real|complex|logical|character|type)"
    r"\s*(?:\([^)]*\))?\s+)?)"
    r"(?P<kind>subroutine|function)\s+(?P<name>[a-z][a-z0-9_]*)\s*"
    r"\((?P<args>[^)]*)\)\s*(?:result\s*\([^)]*\)\s*)?"
    r"bind\s*\(\s*c(?P<bind_opts>[^)]*)\)",
    re.IGNORECASE | re.DOTALL | re.MULTILINE,
)
PY_SYMBOL_RE = re.compile(r"\blib\.charmm\.([A-Za-z][A-Za-z0-9_]*)")
TYPE_RE = re.compile(
    r"^\s*type\s*,\s*bind\s*\(\s*c\s*\)\s*::\s*(?P<name>[a-z][a-z0-9_]*)\b",
    re.IGNORECASE,
)
ENUM_RE = re.compile(r"^\s*enum\s*,\s*bind\s*\(\s*c\s*\)\s*$", re.IGNORECASE)


@dataclasses.dataclass
class Argument:
    name: str
    declaration: str | None
    type_spec: str | None
    intent: str | None
    value: bool
    optional: bool
    dimension: str | None
    issues: list[dict[str, str]]


@dataclasses.dataclass
class Routine:
    symbol: str
    fortran_name: str
    kind: str
    source: str
    line: int
    arguments: list[Argument]
    python_wrappers: list[str]
    issues: list[dict[str, str]]


@dataclasses.dataclass
class InteroperableType:
    name: str
    source: str
    line: int
    components: list[Argument]
    issues: list[dict[str, str]]


@dataclasses.dataclass
class InteroperableEnum:
    name: str
    source: str
    line: int
    enumerators: list[str]
    issues: list[dict[str, str]]


def logical_lines(text: str) -> list[tuple[int, str]]:
    """Return comment-free Fortran logical lines with continuations joined."""
    rows: list[tuple[int, str]] = []
    start = 1
    buf = ""
    for lineno, raw in enumerate(text.splitlines(), 1):
        code = raw.split("!", 1)[0].rstrip()
        if not code.strip() or code.lstrip().startswith("#"):
            continue
        leading = code.lstrip().startswith("&")
        piece = code.strip()
        if leading:
            piece = piece[1:].lstrip()
        continued = piece.endswith("&")
        if continued:
            piece = piece[:-1].rstrip()
        if not buf:
            start = lineno
        buf = f"{buf} {piece}".strip()
        if not continued:
            rows.append((start, buf))
            buf = ""
    if buf:
        rows.append((start, buf))
    return rows


def routine_blocks(path: Path) -> Iterable[tuple[re.Match[str], int, str]]:
    rows = logical_lines(path.read_text(encoding="utf-8", errors="replace"))
    joined = "\n".join(line for _, line in rows)
    offsets: list[tuple[int, int]] = []
    pos = 0
    for lineno, line in rows:
        offsets.append((pos, lineno))
        pos += len(line) + 1
    matches = list(ROUTINE_RE.finditer(joined))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(joined)
        # Prefer the actual END marker when another non-bind routine lies between.
        tail = joined[match.end():end]
        end_match = re.search(
            rf"\bend\s+(?:subroutine|function)\s+{re.escape(match.group('name'))}\b",
            tail,
            flags=re.IGNORECASE,
        )
        if end_match:
            end = match.end() + end_match.end()
        line = max((ln for off, ln in offsets if off <= match.start()), default=1)
        yield match, line, joined[match.start():end]


def split_names(text: str) -> list[str]:
    return [x.strip().lower() for x in text.replace("&", " ").split(",") if x.strip()]


def declarations(block: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for line in block.splitlines():
        if "::" not in line:
            continue
        left, right = line.split("::", 1)
        for token in split_names(right):
            name = re.match(r"([a-z][a-z0-9_]*)", token, re.IGNORECASE)
            if name:
                found[name.group(1).lower()] = f"{left.strip()} :: {right.strip()}"
    return found


def data_type_blocks(path: Path) -> Iterable[tuple[str, int, list[str]]]:
    """Yield each C-interoperable derived type and its logical body lines."""
    rows = logical_lines(path.read_text(encoding="utf-8", errors="replace"))
    index = 0
    while index < len(rows):
        lineno, line = rows[index]
        match = TYPE_RE.match(line)
        if not match:
            index += 1
            continue
        body: list[str] = []
        index += 1
        while index < len(rows):
            _, candidate = rows[index]
            if re.match(r"^\s*end\s+type\b", candidate, re.IGNORECASE):
                break
            body.append(candidate)
            index += 1
        yield match.group("name").lower(), lineno, body
        index += 1


def enum_blocks(path: Path) -> Iterable[tuple[str, int, list[str]]]:
    """Yield each anonymous Fortran bind(c) enum with a stable report name."""
    rows = logical_lines(path.read_text(encoding="utf-8", errors="replace"))
    ordinal = 0
    index = 0
    while index < len(rows):
        lineno, line = rows[index]
        if not ENUM_RE.match(line):
            index += 1
            continue
        ordinal += 1
        values: list[str] = []
        index += 1
        while index < len(rows):
            _, candidate = rows[index]
            if re.match(r"^\s*end\s+enum\b", candidate, re.IGNORECASE):
                break
            if "::" in candidate and re.match(r"^\s*enumerator\b", candidate, re.IGNORECASE):
                values.extend(split_names(candidate.split("::", 1)[1]))
            index += 1
        name = f"{path.stem}_enum_{ordinal}"
        yield name, lineno, values
        index += 1


def issue(severity: str, code: str, message: str) -> dict[str, str]:
    return {"severity": severity, "code": code, "message": message}


def audit_argument(name: str, declaration: str | None) -> Argument:
    problems: list[dict[str, str]] = []
    if declaration is None:
        problems.append(issue("error", "missing_declaration", "argument declaration was not found"))
        return Argument(name, None, None, None, False, False, None, problems)
    left = declaration.split("::", 1)[0].lower()
    type_match = re.match(r"\s*(integer|real|complex|logical|character|type)\s*(\([^)]*\))?", left)
    procedure_match = re.match(r"\s*procedure\s*\([^)]*\)", left)
    type_spec = (
        type_match.group(0).strip() if type_match
        else procedure_match.group(0).strip() if procedure_match
        else None
    )
    intent_match = re.search(r"intent\s*\(\s*(inout|in|out)\s*\)", left)
    dim_match = re.search(r"dimension\s*\(([^)]*)\)", left)
    if dim_match is None:
        entity = declaration.split("::", 1)[1]
        dim_match = re.search(
            rf"(?:^|,)\s*{re.escape(name)}\s*\(([^)]*)\)", entity, re.IGNORECASE,
        )
    dimension = dim_match.group(1).strip() if dim_match else None
    value = bool(re.search(r"(?:^|,)\s*value\s*(?:,|$)", left))
    optional = "optional" in left
    dimension_parts = [part.strip() for part in dimension.split(",")] if dimension else []
    assumed_shape = any(
        part == ":" or (":" in part and not part.split(":", 1)[1].strip())
        for part in dimension_parts
    )
    if assumed_shape:
        problems.append(issue(
            "error", "assumed_shape_array",
            "bind(c) assumed-shape array requires a CFI descriptor; ctypes passes a raw pointer",
        ))
    if "allocatable" in left:
        problems.append(issue("error", "allocatable_dummy", "allocatable dummy is not a raw C pointer"))
    if re.search(r"(?:^|,)\s*pointer\s*(?:,|$)", left):
        problems.append(issue("error", "pointer_dummy", "Fortran POINTER dummy requires descriptor semantics"))
    if type_spec is not None and type_spec.startswith("procedure"):
        problems.append(issue(
            "info", "procedure_callback",
            "callback interoperability depends on the referenced bind(c) abstract interface",
        ))
    elif type_spec is None:
        problems.append(issue("error", "unknown_type", "could not determine interoperable type"))
    elif type_spec.startswith("character"):
        char_kind_ok = "c_char" in type_spec
        if not char_kind_ok:
            problems.append(issue("error", "non_c_character", "character dummy does not use kind=c_char"))
        if dimension is None and not re.search(r"len\s*=\s*1\b", type_spec):
            problems.append(issue("warning", "character_length", "scalar C character should have length 1"))
    elif type_spec.startswith("integer") and "c_" not in type_spec:
        problems.append(issue("warning", "default_integer", "default INTEGER kind is compiler-dependent"))
    elif type_spec.startswith("real") and "c_" not in type_spec:
        problems.append(issue("warning", "default_real", "default REAL kind is compiler-dependent"))
    elif type_spec.startswith("logical") and "c_bool" not in type_spec:
        problems.append(issue("warning", "default_logical", "LOGICAL should use c_bool at a C boundary"))
    if optional:
        problems.append(issue(
            "info", "optional_c_argument",
            "caller must pass NULL for absence and the compiler must support interoperable OPTIONAL",
        ))
    return Argument(
        name=name, declaration=declaration, type_spec=type_spec,
        intent=intent_match.group(1) if intent_match else None,
        value=value, optional=optional, dimension=dimension, issues=problems,
    )


def audit_components(body: list[str]) -> list[Argument]:
    """Audit fields whose binary layout is part of a bind(c) struct contract."""
    decls = declarations("\n".join(body))
    return [audit_argument(name, declaration) for name, declaration in decls.items()]


def python_symbol_map(root: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    for path in sorted(root.rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in PY_SYMBOL_RE.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            result[match.group(1)].append(f"{path}:{line}")
    return result


def scan(fortran_root: Path, python_root: Path) -> dict:
    wrappers = python_symbol_map(python_root)
    routines: list[Routine] = []
    data_types: list[InteroperableType] = []
    enums: list[InteroperableEnum] = []
    for path in sorted(fortran_root.glob("*.F90")):
        for match, line, block in routine_blocks(path):
            opts = match.group("bind_opts") or ""
            bind_name = re.search(r"name\s*=\s*['\"]([^'\"]+)", opts, re.IGNORECASE)
            symbol = bind_name.group(1) if bind_name else match.group("name")
            decls = declarations(block)
            args = [audit_argument(name, decls.get(name)) for name in split_names(match.group("args"))]
            routine_issues = [p for arg in args for p in arg.issues]
            routines.append(Routine(
                symbol=symbol, fortran_name=match.group("name"), kind=match.group("kind").lower(),
                source=str(path), line=line, arguments=args,
                python_wrappers=wrappers.get(symbol, []), issues=routine_issues,
            ))
        for name, line, body in data_type_blocks(path):
            components = audit_components(body)
            data_types.append(InteroperableType(
                name=name, source=str(path), line=line, components=components,
                issues=[problem for component in components for problem in component.issues],
            ))
        for name, line, enumerators in enum_blocks(path):
            problems = []
            if not enumerators:
                problems.append(issue("error", "empty_enum", "bind(c) enum has no enumerators"))
            enums.append(InteroperableEnum(
                name=name, source=str(path), line=line, enumerators=enumerators, issues=problems,
            ))
    symbols = Counter(row.symbol for row in routines)
    for row in routines:
        if symbols[row.symbol] > 1:
            row.issues.append(issue(
                "warning", "duplicate_symbol",
                f"C symbol appears {symbols[row.symbol]} times in source; inspect preprocessor branches",
            ))
    exported = {row.symbol for row in routines}
    unresolved = {name: locations for name, locations in wrappers.items() if name not in exported}
    all_rows = [*routines, *data_types, *enums]
    issue_counts = Counter(p["severity"] for row in all_rows for p in row.issues)
    wrapped = sum(bool(row.python_wrappers) for row in routines)
    return {
        "schema_version": 1,
        "fortran_root": str(fortran_root),
        "python_root": str(python_root),
        "summary": {
            "bind_c_routines": len(routines),
            "bind_c_types": len(data_types),
            "bind_c_enums": len(enums),
            "total_bind_c_surface_entries": len(routines) + len(data_types) + len(enums),
            "wrapped_exports": wrapped,
            "unwrapped_exports": len(routines) - wrapped,
            "python_referenced_symbols": len(wrappers),
            "python_symbols_not_in_api_directory": len(unresolved),
            "issues": dict(sorted(issue_counts.items())),
        },
        "routines": [dataclasses.asdict(row) for row in routines],
        "data_types": [dataclasses.asdict(row) for row in data_types],
        "enums": [dataclasses.asdict(row) for row in enums],
        "python_symbols_not_in_api_directory": unresolved,
        "runtime_symbol_probe": None,
    }


def probe_shared_library(library: Path, report: dict) -> dict:
    """Check that every inventoried routine symbol exists in a local build."""
    symbols = sorted({row["symbol"] for row in report["routines"]})
    try:
        handle = ctypes.CDLL(str(library.resolve()))
    except OSError as exc:
        return {
            "library": str(library), "expected_symbols": len(symbols), "found_symbols": 0,
            "missing_symbols": symbols, "load_error": str(exc),
        }
    missing = [symbol for symbol in symbols if not hasattr(handle, symbol)]
    return {
        "library": str(library), "expected_symbols": len(symbols),
        "found_symbols": len(symbols) - len(missing), "missing_symbols": missing,
        "load_error": None,
    }


def markdown(report: dict) -> str:
    s = report["summary"]
    lines = [
        "# CHARMM Fortran C API surface audit", "",
        "This report is generated statically; it requires no CHARMM build or GPU.", "",
        "## Summary", "",
        f"- `bind(c)` routines: **{s['bind_c_routines']}**",
        f"- `bind(c)` derived types: **{s['bind_c_types']}**",
        f"- `bind(c)` enums: **{s['bind_c_enums']}**",
        f"- Total C surface declarations: **{s['total_bind_c_surface_entries']}**",
        f"- Directly referenced by vendored PyCHARMM: **{s['wrapped_exports']}**",
        f"- Not directly referenced by vendored PyCHARMM: **{s['unwrapped_exports']}**",
        f"- Python symbols implemented outside `source/api` or unresolved: **{s['python_symbols_not_in_api_directory']}**",
        f"- Issues: `{json.dumps(s['issues'], sort_keys=True)}`", "",
        "The routine table includes the complete declared calling contract. Exact",
        "declaration text and every vendored-PyCHARMM call site are retained in the JSON report.", "",
        "## Exported routines", "",
        "| Symbol | Kind | Source | C contract | Python uses | ABI findings |",
        "|---|---|---|---|---:|---|",
    ]
    for row in report["routines"]:
        findings = "; ".join(f"**{x['severity']}** `{x['code']}`" for x in row["issues"]) or "—"
        source = f"`{Path(row['source']).name}:{row['line']}`"
        contract_parts = []
        for arg in row["arguments"]:
            shape = f"[{arg['dimension']}]" if arg["dimension"] is not None else ""
            qualifiers = [x for x in (arg["intent"], "value" if arg["value"] else None,
                                      "optional" if arg["optional"] else None) if x]
            qualifier_text = f" ({', '.join(qualifiers)})" if qualifiers else ""
            contract_parts.append(
                f"`{arg['name']}: {arg['type_spec'] or 'unknown'}{shape}{qualifier_text}`"
            )
        contract = "<br>".join(contract_parts) or "—"
        lines.append(
            f"| `{row['symbol']}` | {row['kind']} | {source} | {contract} | "
            f"{len(row['python_wrappers'])} | {findings} |"
        )
    lines += ["", "## Interoperable derived types", "",
              "| Type | Source | Components | ABI findings |", "|---|---|---:|---|"]
    for row in report["data_types"]:
        findings = "; ".join(f"**{x['severity']}** `{x['code']}`" for x in row["issues"]) or "—"
        source = f"`{Path(row['source']).name}:{row['line']}`"
        lines.append(f"| `{row['name']}` | {source} | {len(row['components'])} | {findings} |")
        component_text = ", ".join(
            f"`{component['name']}: {component['type_spec'] or 'unknown'}`"
            for component in row["components"]
        )
        lines.append(f"| ↳ fields |  |  | {component_text} |")
    lines += ["", "## Interoperable enums", "",
              "| Report name | Source | Enumerators | ABI findings |", "|---|---|---|---|"]
    for row in report["enums"]:
        findings = "; ".join(f"**{x['severity']}** `{x['code']}`" for x in row["issues"]) or "—"
        source = f"`{Path(row['source']).name}:{row['line']}`"
        values = ", ".join(f"`{value}`" for value in row["enumerators"])
        lines.append(f"| `{row['name']}` | {source} | {values} | {findings} |")
    probe = report.get("runtime_symbol_probe")
    if probe is not None:
        load_status = f"`{probe['load_error']}`" if probe["load_error"] else "none"
        missing = ", ".join(f"`{x}`" for x in probe["missing_symbols"]) or "none"
        lines += ["", "## Runtime shared-library symbol probe", "",
                  f"- Library: `{probe['library']}`",
                  f"- Expected symbols: **{probe['expected_symbols']}**",
                  f"- Found symbols: **{probe['found_symbols']}**",
                  f"- Load error: {load_status}",
                  f"- Missing symbols: {missing}"]
    lines += ["", "## Python symbols outside this API directory", ""]
    for symbol, locations in sorted(report["python_symbols_not_in_api_directory"].items()):
        lines.append(f"- `{symbol}` — {', '.join(f'`{x}`' for x in locations)}")
    lines += ["", "## Severity policy", "",
              "- **error**: known ABI-unsafe declaration or duplicate/missing contract.",
              "- **warning**: compiler-dependent declaration that should be reviewed.",
              "- **info**: interoperable feature with a caller/compiler obligation.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortran-root", type=Path, default=Path("setup/charmm/source/api"))
    parser.add_argument("--python-root", type=Path, default=Path("setup/charmm/tool/pycharmm/pycharmm"))
    parser.add_argument("--json", type=Path, default=Path("artifacts/diagnostics/charmm_fortran_api.json"))
    parser.add_argument("--markdown", type=Path, default=Path("artifacts/diagnostics/charmm_fortran_api.md"))
    parser.add_argument(
        "--library", type=Path,
        help="optional compatible libcharmm shared library for an exported-symbol probe",
    )
    parser.add_argument("--strict", action="store_true", help="exit nonzero when ABI errors are found")
    args = parser.parse_args()
    report = scan(args.fortran_root, args.python_root)
    if args.library is not None:
        report["runtime_symbol_probe"] = probe_shared_library(args.library, report)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.write_text(markdown(report), encoding="utf-8")
    from mmml.utils.rich_report import print_colored_json

    print_colored_json(report["summary"], sort_keys=True)
    print(f"JSON: {args.json}")
    print(f"Markdown: {args.markdown}")
    errors = int(report["summary"]["issues"].get("error", 0))
    probe_failed = bool(
        report["runtime_symbol_probe"] is not None
        and (report["runtime_symbol_probe"]["load_error"]
             or report["runtime_symbol_probe"]["missing_symbols"])
    )
    return 1 if args.strict and (errors or probe_failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
