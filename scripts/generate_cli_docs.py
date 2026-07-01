#!/usr/bin/env python3
"""Generate per-command CLI reference pages and mkdocs nav fragment.

Run from repo root::

    uv run python scripts/generate_cli_docs.py

Writes ``docs/cli/commands/<name>.md`` for every entry in ``COMMAND_REGISTRY`` and
updates the ``# CLI_NAV_START`` … ``# CLI_NAV_END`` block in ``mkdocs.yml``.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_CLI = REPO_ROOT / "docs" / "cli"
COMMANDS_DIR = DOCS_CLI / "commands"
MKDOCS = REPO_ROOT / "mkdocs.yml"
NAV_START = "# CLI_NAV_START"
NAV_END = "# CLI_NAV_END"

# Sidebar groups (order matters). Commands not listed fall into "Other utilities".
CLI_NAV_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Structure & boxes",
        ("make-res", "make-box", "build-crystal", "liquid-box"),
    ),
    (
        "MD & campaigns",
        (
            "md-system",
            "run",
            "run-pycharmm",
            "warmup-mlpot-jax",
            "mpi-check",
            "health-check",
            "lambda-mbar",
            "pycharmm-two-residue-sample",
        ),
    ),
    (
        "QM & data",
        (
            "pyscf-dft",
            "pyscf-mp2",
            "pyscf-evaluate",
            "pyscf-evaluate-mp2",
            "fix-and-split",
            "xml2npz",
            "validate",
            "verify-esp-alignment",
            "normal-mode-sample",
            "compare-npz",
            "cross-check",
        ),
    ),
    (
        "ML training & sampling",
        (
            "physnet-train",
            "physnet-evaluate",
            "physnet-md",
            "efield-train",
            "efield-evaluate",
            "efield-md",
            "active-learning",
            "kernel-fit",
            "sample-diverse-xyz",
            "interpolate-xyz",
            "train-joint",
            "extract-checkpoint-metrics",
            "diagnose-lc-outliers",
            "orbax-to-json",
        ),
    ),
    (
        "Workflow helpers",
        (
            "configure",
            "env",
            "commands",
            "examples",
            "completion",
            "gui",
            "unwrap-traj",
            "downstream",
        ),
    ),
    (
        "ORCA external",
        ("orca-server", "orca-client", "orca-external"),
    ),
    (
        "Deprecated & legacy",
        (
            "train",
            "evaluate",
            "ef-train",
            "ef-evaluate",
            "ef-md",
        ),
    ),
)

RELATED_DOCS: dict[str, list[tuple[str, str]]] = {
    "make-res": [
        ("Structure building guide", "../structure-building.md"),
    ],
    "make-box": [
        ("Structure building guide", "../structure-building.md"),
        ("Liquid box workflow", "../../liquid-box-workflow.md"),
    ],
    "build-crystal": [
        ("Structure building guide", "../structure-building.md"),
    ],
    "md-system": [
        ("md-system YAML configs", "../../md-system-configs.md"),
        ("Cross-backend handoff", "../../handoff.md"),
        ("PyCHARMM MPI", "../../pycharmm-mpi.md"),
    ],
    "liquid-box": [("Liquid box workflow", "../../liquid-box-workflow.md")],
    "mpi-check": [
        ("PyCHARMM MPI", "../../pycharmm-mpi.md"),
        ("Spatial ML MPI", "../../mlpot-spatial-mpi.md"),
    ],
    "warmup-mlpot-jax": [("MLpot settings", "../../mlpot-settings.md")],
    "health-check": [("MLpot settings", "../../mlpot-settings.md")],
    "cross-check": [("QC cross-check", "../../qc-cross-check.md")],
    "configure": [("md-system YAML configs", "../../md-system-configs.md")],
    "completion": [("Tab completion guide", "../completion.md")],
    "commands": [("CLI overview", "../index.md")],
    "examples": [("CLI overview", "../index.md")],
    "env": [("CLI overview", "../index.md")],
}

# Static figures under docs/images/ (see scripts/generate_docs_figures.py).
COMMAND_FIGURES: dict[str, list[tuple[str, str]]] = {
    "make-res": [
        ("Acetone monomer (ACO)", "../../images/structures/make-res-aco.png"),
    ],
    "make-box": [
        ("Packed acetone box (illustrative)", "../../images/structures/make-box-acetone.png"),
    ],
    "build-crystal": [
        ("DCM crystal / periodic cell (experimental Pbcn)", "../../images/structures/build-crystal.png"),
    ],
    "liquid-box": [
        ("Density prep ladder (schematic)", "../../images/plots/liquid-box-density-ladder.png"),
    ],
}

META_BODY: dict[str, str] = {
    "commands": """
`mmml commands` lists every subcommand grouped by task area — a browsable
alternative to the compact top-level `mmml -h`.

```bash
mmml commands
mmml commands --audit    # deprecated/legacy + tab-completion coverage
```

The grouped list is defined in `mmml/cli/help_text.py` and kept in sync with
`mmml/cli/registry.py`.
""",
    "examples": """
`mmml examples` prints copy-paste invocations for common workflows (boxes, MD
campaigns, QM pipelines). For interactive YAML setup, use `mmml configure`.

```bash
mmml examples
```
""",
    "completion": """
See the dedicated [Tab completion](../completion.md) page for bash/zsh/fish setup.

```bash
mmml completion bash
eval "$(mmml completion bash)"
```
""",
    "configure": """
Interactive wizard for `md-system` YAML, Snakemake scaffolds, and bundled
`cpu_tests` presets.

```bash
mmml configure
mmml configure --list-presets
mmml configure --non-interactive
```
""",
    "env": """
Resolve checkpoints, CHARMM paths, and shell export hints without importing
PyCHARMM.

```bash
mmml env
mmml env --json
```
""",
    "build-crystal": """
Build molecular crystals for MD. **Recommended for DCM and benzene:** literature
CIF + `make-res` atom names (`--literature dcm|benz`) — exact experimental unit
cell, tiled to a simulation supercell (≥28 Å edges by default) at literature ρ.

```bash
mmml make-res --res DCM --skip-energy-show
mmml build-crystal --literature dcm --monomer-pdb pdb/dcm.pdb -o pdb/dcm_crystal.pdb
mmml build-crystal --literature dcm --supercell 4,4,3 -o dcm_super.extxyz
```

PyXtal (`uv sync --extra chem`) is optional for random placement in the same
space group. DCM crystal: [COD 2100015](https://www.crystallography.net/2100015.html)
(Pbcn, ρ≈1.97 g/cm³). Benzene: [COD 4501704](https://www.crystallography.net/cod/4501704.html)
(P2₁/c, ρ≈1.20 g/cm³).

```bash
mmml build-crystal \\
  -m "$(python -c 'from mmml.paths import default_dcm_molecule_xyz; print(default_dcm_molecule_xyz())')" \\
  --spg 60 --z 4 --target-density-g-cm3 1.972 -o dcm_pyxtal.extxyz
mmml build-crystal -m benzene --spg 14 --z 2 --target-density-g-cm3 1.202 -o benzene.extxyz
```

Liquid DCM boxes use **1.326 g/cm³** (`liquid-box`, `md-system`).

Literature vs make-res+CIF vs PyXtal tables are in the
[structure building guide](../structure-building.md#literature-cross-check-auto-generated).
""",
}


def _import_registry():
    sys.path.insert(0, str(REPO_ROOT))
    from mmml.cli.registry import COMMAND_REGISTRY, command_by_name
    from mmml.cli.parser_utils import get_subcommand_parser, parser_available

    return COMMAND_REGISTRY, command_by_name, get_subcommand_parser, parser_available


def _parser_help(command: str, get_subcommand_parser) -> str | None:
    parser = get_subcommand_parser(command)
    if parser is None:
        return None
    parser.prog = f"mmml {command}"
    buf = io.StringIO()
    # Fixed width keeps generated docs stable across local vs CI terminals.
    with _temporary_columns("80"):
        parser.print_help(buf)
    return buf.getvalue().rstrip()


def _temporary_columns(width: str):
    import os
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        old = os.environ.get("COLUMNS")
        os.environ["COLUMNS"] = width
        try:
            yield
        finally:
            if old is None:
                os.environ.pop("COLUMNS", None)
            else:
                os.environ["COLUMNS"] = old

    return _ctx()


def _status_banner(spec) -> str:
    if spec.status == "active":
        return ""
    rep = f" Prefer **`mmml {spec.replacement}`**." if spec.replacement else ""
    note = f" {spec.note}" if spec.note else ""
    return f"!!! warning \"{spec.status}\"\n    {spec.status.capitalize()} command.{rep}{note}\n\n"


def _figures_section(name: str) -> str:
    figs = COMMAND_FIGURES.get(name)
    if not figs:
        return ""
    lines = ["## Example structures", ""]
    for caption, href in figs:
        lines.append(f"![{caption}]({href})")
        lines.append("")
    lines.append("More detail: [Structure building guide](../structure-building.md).")
    lines.append("")
    return "\n".join(lines)


def _related_section(name: str) -> str:
    links = RELATED_DOCS.get(name)
    if not links:
        return ""
    lines = ["## Related docs", ""]
    for title, href in links:
        lines.append(f"- [{title}]({href})")
    lines.append("")
    return "\n".join(lines)


def _render_command_page(spec, *, get_subcommand_parser, parser_available) -> str:
    name = spec.name
    lines = [
        f"# `mmml {name}`",
        "",
        spec.summary + ".",
        "",
    ]
    lines.append(_status_banner(spec))
    meta = META_BODY.get(name, "").strip()
    if meta:
        lines.append(meta)
        lines.append("")

    has_parser = parser_available(name, import_module=False)
    help_text = _parser_help(name, get_subcommand_parser)

    lines.extend(
        [
            "## Usage",
            "",
            "```bash",
            f"mmml {name} --help",
            "```",
            "",
        ]
    )

    if help_text:
        lines.extend(["## Options", "", "```text", help_text, "```", ""])
    elif has_parser:
        lines.extend(
            [
                "!!! note",
                "    This command defines `build_parser()` but help could not be loaded "
                "(optional deps missing in the doc build environment). Run "
                f"`mmml {name} --help` locally for flags.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "!!! note",
                "    No `build_parser()` hook — see module docstring or run the command "
                "without arguments for usage.",
                "",
                f"Implementation: `{spec.module}`",
                "",
            ]
        )

    lines.append(_figures_section(name))
    lines.append(_related_section(name))
    lines.append(
        f"---\n\n"
        f"[← CLI overview](../index.md) · "
        f"[All commands](../index.md#command-index)"
    )
    return "\n".join(lines).rstrip() + "\n"


def _nav_yaml_lines(registry_names: set[str]) -> list[str]:
    grouped: set[str] = set()
    lines = [f"      {NAV_START}"]
    for group, names in CLI_NAV_GROUPS:
        present = [n for n in names if n in registry_names]
        if not present:
            continue
        grouped.update(present)
        lines.append(f"      - {group}:")
        for name in present:
            lines.append(f"          - {name}: cli/commands/{name}.md")
    other = sorted(registry_names - grouped - {"completion"})
    if other:
        lines.append("      - Other utilities:")
        for name in other:
            lines.append(f"          - {name}: cli/commands/{name}.md")
    lines.append(f"      {NAV_END}")
    return lines


def _patch_mkdocs(nav_block_lines: list[str]) -> None:
    text = MKDOCS.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"^      {re.escape(NAV_START)}.*?^      {re.escape(NAV_END)}\n",
        re.MULTILINE | re.DOTALL,
    )
    replacement = "\n".join(nav_block_lines) + "\n"
    if not pattern.search(text):
        raise SystemExit(
            f"{MKDOCS} missing {NAV_START} / {NAV_END} markers — add them under the CLI nav section."
        )
    MKDOCS.write_text(pattern.sub(replacement, text), encoding="utf-8")


def generate(*, check: bool = False) -> int:
    COMMAND_REGISTRY, _, get_subcommand_parser, parser_available = _import_registry()
    COMMANDS_DIR.mkdir(parents=True, exist_ok=True)
    registry_names = {spec.name for spec in COMMAND_REGISTRY}
    changed = 0

    for spec in COMMAND_REGISTRY:
        path = COMMANDS_DIR / f"{spec.name}.md"
        body = _render_command_page(
            spec,
            get_subcommand_parser=get_subcommand_parser,
            parser_available=parser_available,
        )
        if not path.exists() or path.read_text(encoding="utf-8") != body:
            if check:
                print(f"stale: {path.relative_to(REPO_ROOT)}", file=sys.stderr)
                changed += 1
            else:
                path.write_text(body, encoding="utf-8")
                changed += 1

    nav_lines = _nav_yaml_lines(registry_names)
    old_mkdocs = MKDOCS.read_text(encoding="utf-8")
    _patch_mkdocs(nav_lines)
    if MKDOCS.read_text(encoding="utf-8") != old_mkdocs:
        if check:
            print("stale: mkdocs.yml (CLI nav block)", file=sys.stderr)
            changed += 1
        else:
            changed += 1

    # remove orphan command pages
    for path in COMMANDS_DIR.glob("*.md"):
        if path.stem not in registry_names:
            if check:
                print(f"orphan: {path.relative_to(REPO_ROOT)}", file=sys.stderr)
                changed += 1
            else:
                path.unlink()
                changed += 1

    if check:
        return 1 if changed else 0
    print(f"generate_cli_docs: wrote {len(COMMAND_REGISTRY)} command pages ({changed} updates)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if generated files would change (CI)",
    )
    args = parser.parse_args()
    return generate(check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
