#!/usr/bin/env python3
"""Generate per-command CLI reference pages and mkdocs nav fragment.

Run from repo root::

    uv run python scripts/generate_cli_docs.py

Writes ``docs/cli/commands/<name>.md`` for every entry in ``COMMAND_REGISTRY`` and
updates the ``# CLI_NAV_START: <group>`` … ``# CLI_NAV_END: <group>`` blocks in
``mkdocs.yml``.

One marker block per group, so hand-written guides can sit next to the generated
command pages inside the same nav section without being clobbered. Group names
track ``mmml.cli.help_text.COMMAND_GROUPS`` so the sidebar reads like
``mmml commands``.
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys
from pathlib import Path

# Rendering a page imports the command module, which initializes JAX. On a busy
# GPU that import raises, get_subcommand_parser() swallows the error, and the
# page silently regenerates as a "help could not be loaded" stub — which the
# pre-commit hook then stages over the real option dump. Docs need argparse only.
os.environ["JAX_PLATFORMS"] = "cpu"

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_CLI = REPO_ROOT / "docs" / "cli"
COMMANDS_DIR = DOCS_CLI / "commands"
MKDOCS = REPO_ROOT / "mkdocs.yml"
INDEX_MD = REPO_ROOT / "docs" / "index.md"
EXAMPLES_MD = REPO_ROOT / "docs" / "examples.md"
NAV_START = "# CLI_NAV_START"
NAV_END = "# CLI_NAV_END"
HELP_START = "MMML_TOP_HELP_START"
HELP_END = "MMML_TOP_HELP_END"

# Sidebar groups (order matters). Every registry command must land in exactly one
# group; unassigned commands are reported as an error rather than silently pooled,
# so a new subcommand cannot quietly disappear into an "Other" bucket.
#
# The first five names mirror ``mmml.cli.help_text.COMMAND_GROUPS`` — see
# ``tests/unit/test_generate_cli_docs.py`` for the drift guard.
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
            "md-embedding",
            "warmup-mlpot-jax",
            "mpi-check",
            "mpi-launch",
            "health-check",
            "lambda-mbar",
            "umbrella-sample",
            "umbrella-mbar",
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
            "prepare-mm-dataset",
            "xml2npz",
            "npz2traj",
            "validate",
            "verify-esp-alignment",
            "normal-mode-sample",
            "dimer-scan",
            "ic-scan",
            "mode-check",
            "compare-npz",
            "compare-charmm-ml",
            "cross-check",
        ),
    ),
    (
        "ORCA external",
        ("orca-server", "orca-client", "orca-external"),
    ),
    (
        "ML training & MD",
        (
            "physnet-train",
            "physnet-evaluate",
            "physnet-md",
            "neb",
            "dmc",
            "efield-train",
            "efield-evaluate",
            "efield-md",
            "kernnn-train",
            "kernnn-evaluate",
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
            "doctor",
            "commands",
            "examples",
            "completion",
            "gui",
            "unwrap-traj",
            "plot-restart-velocities",
            "downstream",
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
        (
            "Solid acetone & sublimation enthalpy",
            "../../acetone-crystal-sublimation.md",
        ),
        (
            "Solid dichloromethane & halogen contacts",
            "../../dcm-crystal-cohesion.md",
        ),
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
    "mpi-launch": [
        ("PyCHARMM MPI", "../../pycharmm-mpi.md"),
        ("Spatial ML MPI", "../../mlpot-spatial-mpi.md"),
        ("PyCHARMM threading", "../../pycharmm-threading.md"),
    ],
    "warmup-mlpot-jax": [("MLpot settings", "../../mlpot-settings.md")],
    "health-check": [("MLpot settings", "../../mlpot-settings.md")],
    "cross-check": [("QC cross-check", "../../qc-cross-check.md")],
    "configure": [("md-system YAML configs", "../../md-system-configs.md")],
    "completion": [("Tab completion guide", "../completion.md")],
    "commands": [("CLI overview", "../index.md")],
    "examples": [("CLI overview", "../index.md")],
    "env": [("CLI overview", "../index.md")],
    "dimer-scan": [
        ("1D dimer scan design", "../../dimer-scan-design.md"),
        ("Scientific code policy", "../../scientific-code.md"),
    ],
    "ic-scan": [
        ("Internal-coordinate scan design", "../../ic-scan-design.md"),
        ("Scientific code policy", "../../scientific-code.md"),
    ],
    "dmc": [
        ("Diffusion Monte Carlo guide", "../../dmc.md"),
    ],
}

# Static figures under docs/images/ (see scripts/generate_docs_figures.py).
COMMAND_FIGURES: dict[str, list[tuple[str, str]]] = {
    "make-res": [
        ("Acetone monomer (ACO)", "../../images/structures/make-res-aco.png"),
    ],
    "make-box": [
        ("Packed acetone box (Packmol)", "../../images/structures/make-box-acetone.png"),
    ],
    "build-crystal": [
        ("DCM crystal / periodic cell (experimental Pbcn)", "../../images/structures/build-crystal.png"),
    ],
    "liquid-box": [
        ("Density prep ladder (schematic)", "../../images/plots/liquid-box-density-ladder.png"),
    ],
}

META_BODY: dict[str, str] = {
    "md-system": """
ASE and JAX-MD command routing can be imported and inspected without a local
`libcharmm`. PyCHARMM is loaded lazily only when a CHARMM-backed builder,
minimizer, or backend operation is requested. Those operations require
`CHARMM_LIB_DIR` to point to the directory containing `libcharmm.so` on Linux
(`libcharmm.dylib` on macOS). Certified `--from-psf`/`--from-crd` geometry can
therefore be routed and unit-tested on ordinary CI runners without initializing
the native CHARMM runtime.
""",
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
Build molecular crystals for MD. **Recommended for DCM, benzene and acetone:**
literature CIF + `make-res` atom names (`--literature`) — exact experimental unit
cell, tiled to a simulation supercell (≥28 Å edges by default) at literature ρ.

```bash
mmml make-res --res DCM --skip-energy-show
mmml build-crystal --literature dcm --monomer-pdb pdb/dcm.pdb -o pdb/dcm_crystal.pdb
mmml build-crystal --literature dcm --supercell 4,4,3 -o dcm_super.extxyz
mmml build-crystal --literature aco -o acetone_pbca_150k.pdb
```

PyXtal (`uv sync --extra chem`) is optional for random placement in the same
space group.

## Bundled presets

| Preset | Residue | Structure | Source |
|---|---|---|---|
| `dcm` | `DCM` | Pbcn, 1.63 GPa, ρ≈1.97 g/cm³ | [COD 2100015](https://www.crystallography.net/cod/2100015.html) |
| `dcm133` | `DCM` | Pbcn, 1.33 GPa, ρ≈1.92 g/cm³ | [COD 2100014](https://www.crystallography.net/cod/2100014.html) |
| `benz` | `BENZ` | P2₁/c, ρ≈1.20 g/cm³ | [COD 4501704](https://www.crystallography.net/cod/4501704.html) |
| `aco` | `ACO` | Acetone Pbca, 150 K, Z=16 | [COD 7110464](https://www.crystallography.net/cod/7110464.html) |
| `aco110k` | `ACO` | Acetone Pbca, 110 K | [COD 7110466](https://www.crystallography.net/cod/7110466.html) |
| `aco5k` | `ACO` | Acetone Pbca, 5 K (neutron, d6) | [COD 7110465](https://www.crystallography.net/cod/7110465.html) |
| `acocmcm` | `ACO` | Acetone Cmcm, 160 K (metastable) | [COD 7110463](https://www.crystallography.net/cod/7110463.html) |

The acetone structures come from Allan et al., *Chem. Commun.* 1999, 751
([doi:10.1039/a900558g](https://doi.org/10.1039/a900558g)). The paper's fifth
structure — the 15 kbar Cmcm phase — is bundled but has no preset: its methyls
are rotationally disordered, so it has no single set of hydrogen positions to map
onto CGenFF. See
[Solid acetone & sublimation enthalpy](../../acetone-crystal-sublimation.md) for
validating a built acetone cell against the published contacts and computing its
sublimation enthalpy.

Both DCM presets are **high-pressure** structures, the two points of Podsiadło et
al., *Acta Crystallogr. B* 61, 595 (2005)
([doi:10.1107/S0108768105017374](https://doi.org/10.1107/S0108768105017374)), and
the only pure CH₂Cl₂ entries in COD. They are fine as packing references and as
starting densities, but they are compressed 13% and 11% below the
ambient-pressure cell, so their static lattice energies are not cohesive
energies. See
[Solid dichloromethane & halogen contacts](../../dcm-crystal-cohesion.md) for
relaxing to ambient pressure and for the H···Cl versus Cl···Cl decomposition.

!!! warning "Non-cubic cells and `--write-charmm`"

    `--write-charmm` installs a **cubic** CHARMM IMAGE. The acetone Pbca cell is
    9.17 × 7.53 × 21.25 Å, which no cubic box represents, so MD started that way
    would run a differently shaped cell than the one you built. For a static
    periodic energy on the true cell use `mmml.analysis.lattice_energy` instead.

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
    "dmc": """
Diffusion Monte Carlo on a PhysNetJax potential. Walker energies are evaluated
in parallel with `jax.vmap` (chunked by `--max-batch`).

## Example (acetone dimer)

Bundled geometry: `mmml/generate/dmc/examples/acetone_dmc.extxyz` (20 atoms).

Smoke run (short equilibration, few production steps):

```bash
mmml env   # resolve $MMML_CKPT if you use the bundled checkpoint

mmml dmc \\
  --natm 20 \\
  --nwalker 64 \\
  --stepsize 5e-4 \\
  --nstep 200 \\
  --eqstep 50 \\
  --alpha 1200.0 \\
  --max-batch 64 \\
  --seed 0 \\
  --checkpoint "$MMML_CKPT" \\
  --input mmml/generate/dmc/examples/acetone_dmc.extxyz \\
  --output-dir runs/dmc_acetone_smoke
```

Production-style settings (more walkers / longer averaging):

```bash
mmml dmc \\
  --natm 20 \\
  --nwalker 512 \\
  --stepsize 5e-4 \\
  --nstep 5000 \\
  --eqstep 1000 \\
  --alpha 1200.0 \\
  --max-batch 512 \\
  --seed 0 \\
  --checkpoint "$MMML_CKPT" \\
  --input mmml/generate/dmc/examples/acetone_dmc.extxyz \\
  --output-dir runs/dmc_acetone
```

Outputs under `--output-dir` (or CWD):

- `acetone_dmc.pot` — reference energy vs step (hartree and cm⁻¹)
- `acetone_dmc.log` — run metadata + average energy
- `configs_acetone_dmc.traj` — last 10 steps of surviving walkers
- `defective_acetone_dmc.xyz` — geometries flagged below the reference minimum

See the [Diffusion Monte Carlo guide](../../dmc.md) for inputs, units, and
memory tips.
""",
    "orbax-to-json": """
## SpookyPhysNet / SO3LR checkpoints

The JSON written by `evaluate_so3lr_spooky_extxyz.py --output` contains
evaluation metrics; it is **not** a model checkpoint. Export the trained Orbax
epoch separately:

```bash
uv run mmml orbax-to-json \\
  ~/mmml/artifacts/spooky_so3lr/epoch-0002 \\
  --output ~/mmml/artifacts/spooky_so3lr/epoch-0002_params.json
```

The exporter combines the checkpoint's training `config` with its
`model_attributes`. The latter identifies the model as `spooky` and records
constructor values such as `features`, `cutoff`, and `max_padded_atoms`.

Use the resulting JSON anywhere MMML accepts `--checkpoint`. For the JAX-MD
backend (the `cg_jaxmd` path):

```bash
uv run mmml md-system \\
  --backend jaxmd \\
  --setup free_nvt \\
  --checkpoint ~/mmml/artifacts/spooky_so3lr/epoch-0002_params.json \\
  --composition "RES:1" \\
  --template-pdb /path/to/monomer.pdb \\
  --temperature 300 \\
  --ps 1 \\
  --output-dir ~/runs/spooky_epoch2_jaxmd
```

For periodic MD, use a periodic setup such as `pbc_nvt`, provide the normal
box/build inputs, and keep the same JSON checkpoint:

```bash
uv run mmml md-system \\
  --backend jaxmd \\
  --setup pbc_nvt \\
  --checkpoint ~/mmml/artifacts/spooky_so3lr/epoch-0002_params.json \\
  --composition "RES:20" \\
  --template-pdb /path/to/monomer.pdb \\
  --box-size 30 \\
  --temperature 300 \\
  --ps 1 \\
  --output-dir ~/runs/spooky_epoch2_pbc
```

Current `md-system` SpookyPhysNet calculator construction uses neutral-singlet
conditioning (`Q=0`, spin multiplicity `1`). Do not use this route for charged
or open-shell systems until those conditioning values are exposed by the MD
CLI.

To check that the portable file restores as SpookyPhysNet before a long run:

```bash
uv run mmml health-check \\
  --checkpoint ~/mmml/artifacts/spooky_so3lr/epoch-0002_params.json
```
""",
}


def _import_registry():
    sys.path.insert(0, str(REPO_ROOT))
    from mmml.cli.registry import COMMAND_REGISTRY, command_by_name
    from mmml.cli.parser_utils import get_subcommand_parser, parser_available

    return COMMAND_REGISTRY, command_by_name, get_subcommand_parser, parser_available


def _parser_help(command: str, get_subcommand_parser) -> str | None:
    import argparse

    parser = get_subcommand_parser(command)
    if parser is None:
        return None
    parser.prog = f"mmml {command}"
    parser.formatter_class = lambda prog: argparse.HelpFormatter(prog, width=80)
    # md-system defaults to a short category index for -h; docs need the full dump.
    if hasattr(parser, "_mmml_help_mode"):
        parser._mmml_help_mode = "all"
    buf = io.StringIO()
    parser.print_help(buf)
    return buf.getvalue().rstrip()


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

    usage_lines = [f"mmml {name} --help"]
    if name == "md-system":
        usage_lines = [
            "mmml md-system -h              # category index",
            "mmml md-system -h4             # category by number",
            "mmml md-system -hpycharmm      # same via alias",
            "mmml md-system --help-all      # full option dump",
        ]
    lines.extend(
        [
            "## Usage",
            "",
            "```bash",
            *usage_lines,
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


def _render_examples_page() -> str:
    """``docs/examples.md`` — the ``mmml examples`` output, verbatim."""
    from mmml.cli.help_text import EXAMPLE_BLOCKS

    lines = [
        "# Examples",
        "",
        "Copy-paste invocations, grouped the same way as `mmml examples`.",
        "Run `mmml <command> --help` for the full flag list of any of these.",
        "",
        "!!! note",
        "    This page is generated from `mmml.cli.help_text.EXAMPLE_BLOCKS`,",
        "    so it always matches what `mmml examples` prints.",
        "",
    ]
    for title, examples in EXAMPLE_BLOCKS:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("```bash")
        lines.extend(examples)
        lines.append("```")
        lines.append("")
    lines.append("Interactive setup for YAML and Snakemake scaffolds: `mmml configure`.")
    lines.append("")
    lines.append("See also: [How the CLI is organized](cli/index.md).")
    return "\n".join(lines).rstrip() + "\n"


def _update_top_level_help(text: str) -> str:
    """Refresh the ``mmml -h`` transcript embedded in ``docs/index.md``."""
    from mmml.cli.help_text import format_top_level_help

    pattern = re.compile(
        rf"^([ \t]*)<!-- {HELP_START} -->$.*?^[ \t]*<!-- {HELP_END} -->$\n",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(text)
    if match is None:
        raise SystemExit(
            f"{INDEX_MD} missing <!-- {HELP_START} --> / <!-- {HELP_END} --> markers"
        )
    indent = match.group(1)
    body = "\n".join(
        [
            f"{indent}<!-- {HELP_START} -->",
            f"{indent}```console",
            f"{indent}$ mmml -h",
            *(f"{indent}{line}".rstrip() for line in format_top_level_help().splitlines()),
            f"{indent}```",
            f"{indent}<!-- {HELP_END} -->",
        ]
    )
    return pattern.sub(lambda _m: body + "\n", text, count=1)


def _check_group_coverage(registry_names: set[str]) -> None:
    """Fail loudly when a registry command has no nav group, or is in two."""
    seen: dict[str, str] = {}
    for group, names in CLI_NAV_GROUPS:
        for name in names:
            if name in seen:
                raise SystemExit(
                    f"command {name!r} listed in both {seen[name]!r} and {group!r} "
                    "in CLI_NAV_GROUPS"
                )
            seen[name] = group
    missing = sorted(registry_names - set(seen))
    if missing:
        raise SystemExit(
            "commands missing from CLI_NAV_GROUPS in scripts/generate_cli_docs.py: "
            + ", ".join(missing)
        )


def _nav_block_pattern(group: str) -> re.Pattern[str]:
    return re.compile(
        rf"^([ \t]*){re.escape(NAV_START)}: {re.escape(group)}[ \t]*$"
        rf".*?"
        rf"^[ \t]*{re.escape(NAV_END)}: {re.escape(group)}[ \t]*$\n",
        re.MULTILINE | re.DOTALL,
    )


def _render_nav_block(group: str, names: list[str], indent: str) -> str:
    lines = [f"{indent}{NAV_START}: {group}"]
    for name in names:
        lines.append(f"{indent}- {name}: cli/commands/{name}.md")
    lines.append(f"{indent}{NAV_END}: {group}")
    return "\n".join(lines) + "\n"


def _update_nav(text: str, registry_names: set[str]) -> str:
    """Rewrite each per-group marker block in ``mkdocs.yml`` in place."""
    _check_group_coverage(registry_names)
    for group, names in CLI_NAV_GROUPS:
        present = [n for n in names if n in registry_names]
        pattern = _nav_block_pattern(group)
        match = pattern.search(text)
        if match is None:
            raise SystemExit(
                f"{MKDOCS} missing '{NAV_START}: {group}' / '{NAV_END}: {group}' markers"
            )
        if not present:
            # An empty block would leave a null-valued nav key and break the build.
            raise SystemExit(
                f"nav group {group!r} matches no registry command — drop it from "
                "CLI_NAV_GROUPS and remove its section from mkdocs.yml"
            )
        indent = match.group(1)
        text = pattern.sub(
            lambda _m, g=group, n=present, i=indent: _render_nav_block(g, n, i),
            text,
            count=1,
        )
    return text


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

    for path, body in (
        (MKDOCS, _update_nav(MKDOCS.read_text(encoding="utf-8"), registry_names)),
        (INDEX_MD, _update_top_level_help(INDEX_MD.read_text(encoding="utf-8"))),
        (EXAMPLES_MD, _render_examples_page()),
    ):
        if not path.exists() or path.read_text(encoding="utf-8") != body:
            if check:
                print(f"stale: {path.relative_to(REPO_ROOT)}", file=sys.stderr)
                changed += 1
            else:
                path.write_text(body, encoding="utf-8")
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
    from mmml.cli.help_style import install_colored_argparse

    install_colored_argparse()
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
