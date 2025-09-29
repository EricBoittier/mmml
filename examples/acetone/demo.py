#!/usr/bin/env python3
"""Compare the hybrid MM/ML calculator against the pure ML ASE calculator.

This example loads one configuration from the acetone dimer dataset and
initialises both calculators so that only the ML contribution is active.  The
energies/forces reported by the hybrid calculator (via
``mmml.pycharmmInterface.mmml_calculator.setup_calculator``) are compared with
those from the lightweight ASE helper ``get_ase_calc``.  Agreement between the
numbers is a quick sanity check that the calculator plumbing is consistent.

The script expects the optional runtime dependencies (JAX, e3x, ASE, PyCHARMM)
needed by ``setup_calculator``.  Use the ``MMML_DATA`` and ``MMML_CKPT``
environment variables to point to the acetone dataset and checkpoint
respectively, or rely on the default repository paths.  Energies/forces are
reported in eV by default; pass ``--units kcal/mol`` to convert the outputs.

For additional functionality, see the specialized demo scripts:
- demo_test_minimize.py: Test minimization capabilities
- demo_pdbfile.py: Process PDB files and run MD simulations
- demo_packmol.py: Create molecular boxes using Packmol
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

from demo_base import (
    compute_force_metrics,
    flatten_array,
    get_conversion_factors,
    get_unit_labels,
    load_configuration,
    load_model_parameters,
    parse_base_args,
    resolve_checkpoint_paths,
    resolve_dataset_path,
    setup_ase_imports,
    setup_mmml_imports,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate that the hybrid MM/ML calculator reproduces the pure "
            "ML ASE calculator when MM is disabled."
        )
    )
    
    # Add base arguments
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=(
            "Path to the acetone dataset (.npz). Defaults to $MMML_DATA or "
            "mmml/data/fixed-acetone-only_MP2_21000.npz."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Checkpoint directory used for the ML model. Defaults to $MMML_CKPT "
            "or mmml/physnetjax/ckpts."
        ),
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Index of the configuration to evaluate (default: 0).",
    )
    parser.add_argument(
        "--n-monomers",
        type=int,
        default=2,
        help="Number of monomers in the system (default: 2).",
    )
    parser.add_argument(
        "--atoms-per-monomer",
        type=int,
        default=None,
        help=(
            "Number of atoms per monomer. Defaults to total_atoms/n_monomers "
            "derived from the dataset."
        ),
    )
    parser.add_argument(
        "--ml-cutoff",
        type=float,
        default=2.0,
        help="ML cutoff distance passed to the calculator factory (default: 2.0 Å).",
    )
    parser.add_argument(
        "--mm-switch-on",
        type=float,
        default=5.0,
        help="MM switch-on distance for the hybrid calculator (default: 5.0 Å).",
    )
    parser.add_argument(
        "--mm-cutoff",
        type=float,
        default=1.0,
        help="MM cutoff width for the hybrid calculator (default: 1.0 Å).",
    )
    parser.add_argument(
        "--include-mm",
        action="store_true",
        help="Keep MM contributions enabled when evaluating the hybrid calculator.",
    )
    parser.add_argument(
        "--skip-ml-dimers",
        action="store_true",
        help="If set, skip the ML dimer correction in the hybrid calculator.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug output inside the calculator factory.",
    )
    parser.add_argument(
        "--units",
        choices=("eV", "kcal/mol"),
        default="eV",
        help=(
            "Output units for energies/forces. Use 'kcal/mol' to apply the "
            "ASE conversion factor."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save a JSON report containing the comparison results.",
    )
    
    return parser.parse_args()


def print_component_comparison(name: str, hybrid: np.ndarray, reference: np.ndarray | None, 
                              component_units: Dict[str, str], component_reports: Dict[str, Dict[str, Any]]) -> None:
    """Print component comparison results."""
    header = f"{name:<12}"
    unit = component_units.get(name)
    unit_suffix = f" {unit}" if unit else ""

    hybrid_list = hybrid.tolist()
    reference_list = reference.tolist() if reference is not None else None
    stats: Dict[str, Any] = {}

    if hybrid.size == 1:
        value = float(hybrid[0])
        line = f"{value: .8f}{unit_suffix}"
        stats["value"] = value
        if reference is not None and reference.size == 1:
            ref_val = float(reference[0])
            delta = value - ref_val
            line += f" | ref {ref_val: .8f}{unit_suffix} | Δ={delta: .8e}{unit_suffix}"
            stats["reference"] = ref_val
            stats["delta"] = delta
    else:
        rms = float(np.sqrt(np.mean(hybrid**2)))
        max_abs = float(np.abs(hybrid).max())
        line = f"rms={rms: .8e}, max|.|={max_abs: .8e}"
        stats["rms"] = rms
        stats["max_abs"] = max_abs
        if reference is not None and reference.shape == hybrid.shape:
            diff = hybrid - reference
            diff_rms, diff_max = compute_force_metrics(diff)
            line += f" | Δrms={diff_rms: .8e}, Δmax={diff_max: .8e}"
            stats["delta_rms"] = diff_rms
            stats["delta_max"] = diff_max
        if unit:
            line += f" [{unit}]"
    
    component_reports[name] = {
        "unit": unit,
        "hybrid": hybrid_list,
        "reference": reference_list,
        "stats": stats,
    }

    print(f"{header}: {line}")
    if hybrid.size <= 12:
        hybrid_repr = np.array2string(hybrid, precision=6, separator=", ")
        label = f" ({unit})" if unit else ""
        print(f"    hybrid{label}: {hybrid_repr}")
        if reference is not None and reference.shape == hybrid.shape:
            ref_repr = np.array2string(reference, precision=6, separator=", ")
            diff_repr = np.array2string(hybrid - reference, precision=6, separator=", ")
            print(f"    ref{label}:    {ref_repr}")
            print(f"    delta{label}:  {diff_repr}")


def main() -> int:
    """Main function for hybrid vs pure ML comparison demo."""
    args = parse_args()
    dataset_report: Dict[str, Any] = {}
    dataset_path = resolve_dataset_path(args.dataset)
    base_ckpt_dir, epoch_dir = resolve_checkpoint_paths(args.checkpoint)

    Atoms = setup_ase_imports()
    CutoffParameters, ev2kcalmol, setup_calculator, get_ase_calc = setup_mmml_imports()
    
    n_monomers = args.n_monomers
    natoms = n_monomers * args.atoms_per_monomer

    Z, R, references = load_configuration(dataset_path, args.sample_index)
    natoms = len(Z)
    atoms_per_monomer = args.atoms_per_monomer
    params, model = load_model_parameters(epoch_dir, natoms)
    atoms = Atoms(numbers=Z, positions=R)
    cutoff = CutoffParameters(
        ml_cutoff=args.ml_cutoff,
        mm_switch_on=args.mm_switch_on,
        mm_cutoff=args.mm_cutoff,
    )
    
    energy_factor, force_factor = get_conversion_factors(args.units)
    energy_unit_label, force_unit_label = get_unit_labels(args.units)

    from mmml.pycharmmInterface import setupRes
    from mmml.pycharmmInterface.import_pycharmm import ic, coor
    setupRes.generate_residue("ACO ACO")
    ic.build()
    coor.show()
    
    calculator_factory = setup_calculator(
        ATOMS_PER_MONOMER=atoms_per_monomer,
        N_MONOMERS=2,
        ml_cutoff_distance=args.ml_cutoff,
        mm_switch_on=args.mm_switch_on,
        mm_cutoff=args.mm_cutoff,
        doML=True,
        doMM=args.include_mm,
        doML_dimer=not args.skip_ml_dimers,
        debug=args.debug,
        model_restart_path=base_ckpt_dir,
        MAX_ATOMS_PER_SYSTEM=natoms,
        ml_energy_conversion_factor=energy_factor,
        ml_force_conversion_factor=force_factor,
    )

    hybrid_calc, _ = calculator_factory(
        atomic_numbers=Z,
        atomic_positions=R,
        n_monomers=2,
        cutoff_params=cutoff,
        doML=True,
        doMM=args.include_mm,
        doML_dimer=not args.skip_ml_dimers,
        backprop=True,
        debug=args.debug,
        energy_conversion_factor=energy_factor,
        force_conversion_factor=force_factor,
    )

    atoms.calc = hybrid_calc
    hybrid_energy = float(atoms.get_potential_energy())
    hybrid_forces = np.asarray(atoms.get_forces())

    ml_atoms = Atoms(numbers=Z, positions=R)
    ml_calc = get_ase_calc(
        params,
        model,
        ml_atoms,
        conversion={"energy": energy_factor, "forces": force_factor},
        implemented_properties=["energy", "forces"],
    )
    ml_atoms.calc = ml_calc
    ml_energy = float(ml_atoms.get_potential_energy())
    ml_forces = np.asarray(ml_atoms.get_forces())

    energy_delta = hybrid_energy - ml_energy
    force_delta = hybrid_forces - ml_forces
    force_rmsd, force_max = compute_force_metrics(force_delta)

    component_keys = [
        "dH",
        "energy",
        "forces",
        "internal_E",
        "internal_F",
        "ml_2b_E",
        "ml_2b_F",
        "mm_E",
        "mm_F",
    ]
    component_units = {
        "dH": energy_unit_label,
        "energy": energy_unit_label,
        "internal_E": energy_unit_label,
        "ml_2b_E": energy_unit_label,
        "mm_E": energy_unit_label,
        "forces": force_unit_label,
        "internal_F": force_unit_label,
        "ml_2b_F": force_unit_label,
        "mm_F": force_unit_label,
    }
    hybrid_components: Dict[str, np.ndarray] = {}
    component_reports: Dict[str, Dict[str, Any]] = {}
    hybrid_out = atoms.calc.results.get("out")
    if hybrid_out is not None:
        for key in component_keys:
            if hasattr(hybrid_out, key):
                hybrid_components[key] = flatten_array(getattr(hybrid_out, key))

    ml_component_refs: Dict[str, np.ndarray] = {
        "energy": np.array([ml_energy]),
        "internal_E": np.array([ml_energy]),
        "forces": ml_forces.reshape(-1),
        "internal_F": ml_forces.reshape(-1),
    }
    if not args.include_mm:
        zeros = np.zeros_like(ml_component_refs["forces"])
        ml_component_refs["mm_E"] = np.array([0.0])
        ml_component_refs["mm_F"] = zeros

    print(f"Energy difference (hybrid - ML):           {energy_delta: .8e} {energy_unit_label}")
    print(f"Force RMSD:                                {force_rmsd: .8e} {force_unit_label}")
    print(f"Max |ΔF|:                                  {force_max: .8e} {force_unit_label}")

    if hybrid_components:
        print("\nComponent comparison (hybrid vs pure ML reference):")
        for key in component_keys:
            hybrid_val = hybrid_components.get(key)
            if hybrid_val is None:
                continue
            reference_val = ml_component_refs.get(key)
            print_component_comparison(key, hybrid_val, reference_val, component_units, component_reports)

    if references:
        if "E" in references:
            dataset_energy = float(references["E"]) * energy_factor
            dataset_report["energy"] = dataset_energy
            print(f"Dataset reference energy ({energy_unit_label}):   {dataset_energy: .8f}")
        if "F" in references:
            ref_forces = np.asarray(references["F"]) * force_factor
            ref_force_rmsd, ref_force_max = compute_force_metrics(
                hybrid_forces - ref_forces
            )
            dataset_report["force_metrics"] = {
                "rms": ref_force_rmsd,
                "max": ref_force_max,
            }
            print(
                f"RMSD vs dataset forces ({force_unit_label}):    {ref_force_rmsd: .8e}"
            )
            print(
                f"Max |ΔF| vs dataset ({force_unit_label}):       {ref_force_max: .8e}"
            )
    
    print(f"Hybrid calculator energy ({energy_unit_label}):      {hybrid_energy: .8f}")
    print(f"Pure ML calculator energy ({energy_unit_label}):     {ml_energy: .8f}")
    
    report: Dict[str, Any] = {
        "settings": {
            "dataset": str(dataset_path),
            "checkpoint_root": str(base_ckpt_dir),
            "checkpoint_epoch": str(epoch_dir),
            "sample_index": args.sample_index,
            "n_monomers": n_monomers,
            "atoms_per_monomer": atoms_per_monomer,
            "include_mm": args.include_mm,
            "skip_ml_dimers": args.skip_ml_dimers,
        },
        "units": {
            "energy": energy_unit_label,
            "force": force_unit_label,
        },
        "energies": {
            "hybrid": hybrid_energy,
            "pure_ml": ml_energy,
            "difference": energy_delta,
        },
        "forces": {
            "hybrid": hybrid_forces.tolist(),
            "pure_ml": ml_forces.tolist(),
            "difference_metrics": {
                "rms": force_rmsd,
                "max": force_max,
            },
        },
        "components": component_reports,
    }

    if dataset_report:
        report["dataset"] = dataset_report

    if args.output is not None:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=4)

    return 0




if __name__ == "__main__":
    raise SystemExit(main())
