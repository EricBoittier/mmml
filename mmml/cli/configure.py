"""Interactive wizard: ``mmml configure`` for MD/ML YAML and Snakemake workflows."""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path
from typing import Any

import yaml

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmml configure",
        description=(
            "Interactive, multiple-choice setup for md-system YAML, PhysNet training "
            "configs, and Snakemake workflow scaffolding."
        ),
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Print menu only (for tests); do not read stdin",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory to write generated files (default: current directory)",
    )
    parser.add_argument(
        "--workflow",
        choices=("md-single", "md-campaign", "physnet-train", "snakemake-md"),
        default=None,
        help="Skip menu and run a specific wizard (still prompts for details)",
    )
    return parser


# ---------------------------------------------------------------------------
# Prompt helpers (numbered multiple choice)
# ---------------------------------------------------------------------------


def _prompt_line(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        raw = input(f"{prompt}{suffix}: ").strip()
        if raw:
            return raw
        if default is not None:
            return default
        print("  (required)")


def _prompt_choice(
    title: str,
    options: list[tuple[str, str, str]],
    *,
    default_index: int = 0,
) -> str:
    """Return the key from ``options`` (key, label, hint)."""
    print()
    print(title)
    for i, (_key, label, hint) in enumerate(options, start=1):
        mark = "*" if i - 1 == default_index else " "
        print(f"  {mark} [{i}] {label}")
        if hint:
            for line in textwrap.wrap(hint, width=70, initial_indent="        ", subsequent_indent="        "):
                print(line)
    default_num = default_index + 1
    while True:
        raw = input(f"Choice [1-{len(options)}] (default {default_num}): ").strip()
        if not raw:
            return options[default_index][0]
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx][0]
        print("  Invalid choice — enter a number from the list.")


def _prompt_yes_no(prompt: str, *, default: bool = True) -> bool:
    d = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{prompt} [{d}]: ").strip().lower()
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False


def _resolve_checkpoint(raw: str) -> str:
    if raw.startswith("$") or raw.startswith("/"):
        return raw
    env = os.environ.get("MMML_CKPT")
    if env:
        return env
    return "${MMML_CKPT}"


def _write_yaml(path: Path, data: dict[str, Any], *, header: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(data, sort_keys=False, default_flow_style=False)
    with path.open("w", encoding="utf-8") as fh:
        if header:
            fh.write(f"# {header}\n#\n")
        fh.write(body)


# ---------------------------------------------------------------------------
# Wizards
# ---------------------------------------------------------------------------


def wizard_md_single(out_dir: Path) -> list[Path]:
    goal = _prompt_choice(
        "What are you trying to do?",
        [
            ("liquid", "Liquid / condensed phase", "NPT equilibration, density, solvation"),
            ("vacuum", "Cluster / gas phase", "Free-boundary NVE/NVT with flat-bottom restraint"),
            ("minimize", "Minimize only", "PyCHARMM SD + bonded-MM strain relief"),
            ("lambda", "Lambda TI / free energy", "Alchemical coupling; run mmml lambda-mbar after"),
        ],
        default_index=0,
    )

    setup_map = {
        "liquid": ("pbc_npt", "pycharmm", 260.0, 32.0),
        "vacuum": ("free_nvt", "jaxmd", 300.0, None),
        "minimize": ("pycharmm_minimize", "pycharmm", 300.0, None),
        "lambda": ("lambda_ti", "pycharmm", 300.0, 28.0),
    }
    setup, backend_default, temp_default, box_default = setup_map[goal]

    setup = _prompt_choice(
        "MD setup preset",
        [
            ("free_nve", "free_nve", "Microcanonical, no thermostat"),
            ("free_nvt", "free_nvt", "NVT in open boundary (flat-bottom)"),
            ("pbc_nve", "pbc_nve", "Fixed-volume periodic NVE"),
            ("pbc_nvt", "pbc_nvt", "Fixed-volume periodic NVT"),
            ("pbc_npt", "pbc_npt", "NPT — recommended for liquids"),
            ("pycharmm_minimize", "pycharmm_minimize", "Minimize only (PyCHARMM)"),
            ("lambda_ti", "lambda_ti", "Lambda thermodynamic integration"),
        ],
        default_index={
            "free_nve": 0,
            "free_nvt": 1,
            "pbc_nve": 2,
            "pbc_nvt": 3,
            "pbc_npt": 4,
            "pycharmm_minimize": 5,
            "lambda_ti": 6,
        }[setup],
    )

    backend = _prompt_choice(
        "MD backend",
        [
            ("pycharmm", "pycharmm", "CHARMM + MLpot — production MD, MPI/GPU cluster"),
            ("jaxmd", "jaxmd", "JAX-MD — fast NPT/NVT on GPU"),
            ("ase", "ase", "ASE — prototyping, vacuum/PBC NVE"),
            ("auto", "auto", "Pick ASE/JAX-MD from setup (no PyCHARMM)"),
        ],
        default_index={"pycharmm": 0, "jaxmd": 1, "ase": 2, "auto": 3}[backend_default],
    )

    composition = _prompt_line("Composition (RES:N, e.g. DCM:20 or MEOH:5,TIP3:5)", "DCM:20")
    checkpoint = _resolve_checkpoint(
        _prompt_line("Checkpoint path or $MMML_CKPT", "${MMML_CKPT}")
    )
    temperature = float(_prompt_line("Temperature (K)", str(temp_default)))
    output = _prompt_line("Output directory", "artifacts/md_run")

    cfg: dict[str, Any] = {
        "setup": setup,
        "backend": backend,
        "composition": composition,
        "checkpoint": checkpoint,
        "output_dir": output,
        "temperature": temperature,
        "seed": 42,
    }

    if setup.startswith("pbc_") or goal == "liquid":
        box = _prompt_line("Box side length (Å)", str(box_default or 32.0))
        cfg["box_size"] = float(box)
        cfg.setdefault("spacing", 5.0)
        cfg.setdefault("pressure", 1.0)

    if setup == "pycharmm_minimize":
        cfg["md_stages"] = "mini"
        cfg["mini_nstep"] = 500
        cfg["bonded_mm_mini"] = True
    elif backend == "pycharmm" and setup not in ("lambda_ti",):
        cfg.setdefault("md_stages", "mini,heat,nve")
        cfg["bonded_mm_mini"] = True
        cfg["no_echeck"] = True

    if setup == "lambda_ti":
        cfg["couple_residues"] = _prompt_line("Couple residues (comma-separated IDs)", "1")
        cfg["lambda_md_mode"] = "free_nve"
        cfg["pre_min_steps"] = 50

    path = out_dir / "md_system.yaml"
    _write_yaml(path, cfg, header="Generated by mmml configure (single MD run)")
    print()
    print(f"Wrote {path}")
    print(f"Run:  mmml md-system --config {path}")
    return [path]


def wizard_md_campaign(out_dir: Path) -> list[Path]:
    print()
    print("Campaign wizard — chained jobs with handoffs (prep → equil → prod).")
    composition = _prompt_line("Composition", "DCM:20")
    checkpoint = _resolve_checkpoint(_prompt_line("Checkpoint", "${MMML_CKPT}"))
    box = float(_prompt_line("Box size (Å)", "32.0"))
    campaign_root = _prompt_line("Campaign output root", "artifacts/md_campaign")

    cfg: dict[str, Any] = {
        "campaign_output": campaign_root,
        "defaults": {
            "composition": composition,
            "checkpoint": checkpoint,
            "box_size": box,
            "spacing": 5.0,
            "temperature": 260.0,
            "pressure": 1.0,
            "seed": 42,
            "backend": "pycharmm",
            "bonded_mm_mini": True,
            "no_echeck": True,
        },
        "runs": {},
    }

    stages = _prompt_choice(
        "Campaign template",
        [
            ("mini-equil", "Mini → NPT equil", "Liquid prep: minimize strain then short NPT"),
            ("full", "Mini → heat → NVE → NPT", "Full PyCHARMM pipeline"),
            ("custom", "Mini only (smoke)", "Quick MPI/GPU smoke test"),
        ],
        default_index=0,
    )

    if stages == "mini-equil":
        cfg["runs"] = {
            "prep": {
                "description": "Minimize + bonded-MM strain relief",
                "setup": "pycharmm_minimize",
                "md_stages": "mini",
                "mini_nstep": 500,
                "output_dir": "results/prep",
            },
            "equil": {
                "description": "NPT equilibration",
                "setup": "pbc_npt",
                "depends_on": "prep",
                "ps": 50.0,
                "md_stages": "mini,heat,nve",
                "output_dir": "results/equil",
            },
        }
    elif stages == "full":
        cfg["runs"] = {
            "prep": {
                "description": "Minimize",
                "setup": "pycharmm_minimize",
                "md_stages": "mini",
                "output_dir": "results/prep",
            },
            "heat_nve": {
                "description": "Heat + short NVE",
                "setup": "pbc_nve",
                "depends_on": "prep",
                "ps": 5.0,
                "md_stages": "mini,heat,nve",
                "output_dir": "results/heat_nve",
            },
            "prod": {
                "description": "NPT production",
                "setup": "pbc_npt",
                "depends_on": "heat_nve",
                "ps": 200.0,
                "md_stages": "equi,prod",
                "output_dir": "results/prod",
            },
        }
    else:
        cfg["runs"] = {
            "smoke": {
                "description": "Mini smoke (spatial MPI friendly)",
                "setup": "pbc_npt",
                "md_stages": "mini",
                "mini_nstep": 20,
                "output_dir": "results/smoke",
            },
        }

    path = out_dir / "md_campaign.yaml"
    _write_yaml(path, cfg, header="Generated by mmml configure (MD campaign)")
    print()
    print(f"Wrote {path}")
    print(f"Plan:  mmml md-system --config {path} --run-all")
    print(f"Single job: mmml md-system --config {path} --job-id <name>")
    return [path]


def wizard_physnet_train(out_dir: Path) -> list[Path]:
    train_npz = _prompt_line("Training NPZ", "splits/train.npz")
    valid_npz = _prompt_line("Validation NPZ (optional)", "splits/valid.npz")
    ckpt_dir = _prompt_line("Checkpoint directory", "./ckpts/run")
    tag = _prompt_line("Run tag", "my_run")

    scale = _prompt_choice(
        "Training scale",
        [
            ("smoke", "Smoke (fast)", "32 batch, 50 epochs — sanity check"),
            ("medium", "Medium", "32 batch, 500 epochs — typical molecule"),
            ("production", "Production", "32 batch, 2000 epochs — publication quality"),
        ],
        default_index=1,
    )
    epochs = {"smoke": 50, "medium": 500, "production": 2000}[scale]

    cfg: dict[str, Any] = {
        "data": train_npz,
        "valid_data": valid_npz,
        "ckpt_dir": ckpt_dir,
        "tag": tag,
        "seed": 42,
        "batch_size": 32,
        "num_epochs": epochs,
        "learning_rate": 0.001,
        "energy_weight": 1.0,
        "forces_weight": 52.91,
        "dipole_weight": 27.21,
        "max_atomic_number": 35,
        "features": 64,
        "num_basis_functions": 32,
        "num_iterations": 2,
        "cutoff": 8.0,
    }

    path = out_dir / "physnet_train.yaml"
    _write_yaml(path, cfg, header="Generated by mmml configure (PhysNet training)")
    print()
    print(f"Wrote {path}")
    print(f"Run:  mmml physnet-train --config {path}")
    return [path]


_SNAKEFILE_TEMPLATE = '''configfile: "config.yaml"

_JOB_IDS = list(config["jobs"].keys())
_OPTIONAL_JOBS = set(config.get("optional_jobs") or [])
_REQUIRED_JOB_IDS = [j for j in _JOB_IDS if j not in _OPTIONAL_JOBS]


def _needs_mpi(wildcards) -> int:
    return 1 if config["jobs"][wildcards.job]["backend"] == "pycharmm" else 0


rule all:
    input:
        "results/benchmark_summary.csv",


rule run_job:
    output:
        touch("results/{job}/done.txt"),
    log:
        stdout="results/{job}/stdout.log",
    params:
        job=lambda wildcards: wildcards.job,
        workflow_dir=workflow.basedir,
    resources:
        gpu=1,
        mpi=_needs_mpi,
    shell:
        """
        mkdir -p results/{wildcards.job}
        bash {params.workflow_dir}/scripts/job_shell.sh {params.job} > {log.stdout} 2>&1
        touch {output}
        """


rule collect:
    input:
        jobs=expand("results/{job}/done.txt", job=_REQUIRED_JOB_IDS),
    output:
        csv="results/benchmark_summary.csv",
    params:
        workflow_dir=workflow.basedir,
    shell:
        """
        bash {params.workflow_dir}/scripts/collect_shell.sh \\
            {params.workflow_dir} {output.csv}
        """
'''

_JOB_SHELL_TEMPLATE = '''#!/usr/bin/env bash
# Run one benchmark job from config.yaml (generated by mmml configure).
set -euo pipefail
JOB="${1:?job id}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${ROOT}/config.yaml"
OUT="${ROOT}/results/${JOB}"
mkdir -p "${OUT}"

# Requires MMML_CKPT and mmml on PATH (or: uv run mmml ...)
mmml md-system \\
  --config "${CONFIG}" \\
  --job-id "${JOB}" \\
  --output-dir "${OUT}" \\
  --quiet
'''

_COLLECT_SHELL_TEMPLATE = '''#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:?workflow dir}"
CSV="${2:?output csv}"
mkdir -p "$(dirname "${CSV}")"
echo "job,status" > "${CSV}"
for d in "${ROOT}"/results/*/; do
  job="$(basename "${d}")"
  if [[ -f "${d}/done.txt" ]]; then
    echo "${job},done" >> "${CSV}"
  fi
done
'''


def wizard_snakemake_md(out_dir: Path) -> list[Path]:
    wf_name = _prompt_line("Workflow directory name", "my_md_workflow")
    wf_dir = out_dir / wf_name
    composition = _prompt_line("Composition", "DCM:5")
    checkpoint = _resolve_checkpoint("${MMML_CKPT}")

    include_pycharmm = _prompt_yes_no("Include PyCHARMM MLpot job?", default=True)
    include_jaxmd = _prompt_yes_no("Include JAX-MD PBC NVE job?", default=True)

    cfg: dict[str, Any] = {
        "composition": composition,
        "ps": 2.0,
        "dt_fs": 0.25,
        "temperature": 300.0,
        "seed": 123,
        "spacing": 5.0,
        "box_size": 25.0,
        "checkpoint": checkpoint,
        "output_root": "results",
        "mm_switch_on": 7.0,
        "mm_switch_width": 5.0,
        "ml_switch_width": 0.1,
        "jobs": {},
    }

    if include_jaxmd:
        cfg["jobs"]["jaxmd_pbc_nve"] = {
            "backend": "jaxmd",
            "setup": "pbc_nve",
            "pbc": True,
            "integrator": "nve",
        }
    if include_pycharmm:
        cfg["jobs"]["pycharmm_pbc_npt"] = {
            "backend": "pycharmm",
            "setup": "pbc_npt",
            "pbc": True,
            "pressure": 1.0,
            "md_stages": "mini,heat,nve",
        }

    written: list[Path] = []
    wf_dir.mkdir(parents=True, exist_ok=True)
    (wf_dir / "Snakefile").write_text(_SNAKEFILE_TEMPLATE, encoding="utf-8")
    written.append(wf_dir / "Snakefile")

    config_path = wf_dir / "config.yaml"
    _write_yaml(config_path, cfg, header="Snakemake workflow — generated by mmml configure")
    written.append(config_path)

    scripts = wf_dir / "scripts"
    scripts.mkdir(exist_ok=True)
    job_sh = scripts / "job_shell.sh"
    job_sh.write_text(_JOB_SHELL_TEMPLATE, encoding="utf-8")
    job_sh.chmod(0o755)
    written.append(job_sh)

    collect_sh = scripts / "collect_shell.sh"
    collect_sh.write_text(_COLLECT_SHELL_TEMPLATE, encoding="utf-8")
    collect_sh.chmod(0o755)
    written.append(collect_sh)

    readme = wf_dir / "README.configure.md"
    readme.write_text(
        textwrap.dedent(
            f"""\
            # {wf_name}

            Generated by ``mmml configure``.

            ## Prerequisites

            - ``export MMML_CKPT=/path/to/checkpoint.json``
            - ``pip install snakemake`` (or ``uv run --with snakemake snakemake``)

            ## Dry run

            ```bash
            cd {wf_name}
            snakemake -n
            ```

            ## Local run

            ```bash
            snakemake -j2 --resources gpu=1 mpi=1 --keep-going
            ```
            """
        ),
        encoding="utf-8",
    )
    written.append(readme)

    print()
    print(f"Wrote workflow under {wf_dir}/")
    for p in written:
        print(f"  {p.relative_to(out_dir)}")
    print()
    print(f"Next: cd {wf_name} && snakemake -n")
    return written


_WIZARDS = {
    "md-single": wizard_md_single,
    "md-campaign": wizard_md_campaign,
    "physnet-train": wizard_physnet_train,
    "snakemake-md": wizard_snakemake_md,
}


def run_wizard(workflow: str, out_dir: Path) -> list[Path]:
    fn = _WIZARDS[workflow]
    return fn(out_dir)


def configure_main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = args.output_dir.expanduser().resolve()

    if args.non_interactive:
        print("Workflows: md-single, md-campaign, physnet-train, snakemake-md")
        return 0

    workflow = args.workflow
    if workflow is None:
        workflow = _prompt_choice(
            "Configure what?",
            [
                ("md-single", "Single MD run", "One md-system YAML (liquid, vacuum, minimize, λ)"),
                ("md-campaign", "MD campaign", "Multi-stage YAML with depends_on handoffs"),
                ("physnet-train", "PhysNet training", "physnet-train YAML from NPZ splits"),
                ("snakemake-md", "Snakemake MD workflow", "config.yaml + Snakefile + job scripts"),
            ],
            default_index=0,
        )

    run_wizard(workflow, out_dir)
    return 0
