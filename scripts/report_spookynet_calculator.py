#!/usr/bin/env python3
"""Write JSON and Markdown manifests for a SpookyNet ASE calculator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmml.models.spookynet_calc import SpookyNetCalculator


def _markdown(report: dict) -> str:
    lj = report["cgenff_lennard_jones"]
    electrostatics = report["electrostatics"]
    short = report["short_range"]
    mbd = report["mbd"]
    precision = report["precision"]
    lines = [
        "# Calculator energy-function report",
        "",
        f"- Calculator: `{report['calculator']}`",
        f"- Model: `{report['model_class']}`",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Precision: `{precision['compute_dtype']}`; JAX x64 = `{precision['jax_enable_x64']}`",
        f"- Output units: `{report['energy_units']}` and `{report['force_units']}`",
        "",
        "## Included terms",
        "",
        f"- Neural atomic energy: `{short['neural_atomic_energy']}` (cutoff `{short['cutoff_angstrom']}` Å)",
        f"- Predicted-charge electrostatics: `{electrostatics['predicted_atomic_charges']}`",
        f"- ZBL repulsion: `{short['zbl_repulsion']}`",
        f"- CGenFF LJ used during training: `{lj['enabled_during_training']}`",
        f"- Annotated CGenFF inputs supported by adapter: `{lj['annotated_atoms_supported']}`",
        f"- CGenFF LJ inputs supplied now: `{lj['inputs_supplied_at_inference']}`",
        f"- Companion MBD loaded: `{mbd['loaded']}` (weight `{mbd['weight']}`)",
        "",
        "## CHARMM LJ convention",
        "",
        f"- Parameter radius field: `{lj['parameter_file_radius_field']}`",
        f"- Combination: `{lj['pair_combination']}`",
        f"- Potential: `{lj['charmm_form']}`",
        f"- Conventional conversion: `{lj['conventional_sigma_conversion']}`",
    ]
    if report["warnings"]:
        lines += ["", "## Warnings", ""]
        lines += [f"- **{warning}**" for warning in report["warnings"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--no-mbd", action="store_true", help="Disable a configured companion MBD model"
    )
    args = parser.parse_args()

    calc = SpookyNetCalculator(
        args.checkpoint,
        mbd_checkpoint=False if args.no_mbd else None,
    )
    report = calc.energy_function_report()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "calculator_energy_function.json"
    md_path = args.output_dir / "calculator_energy_function.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json_path)
    print(md_path)
    if report["warnings"]:
        for warning in report["warnings"]:
            print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
