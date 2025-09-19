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
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate that the hybrid MM/ML calculator reproduces the pure "
            "ML ASE calculator when MM is disabled."
        )
    )
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


    parser.add_argument(
        "--pdbfile",
        type=Path,
        default=None,
        help="Path to the PDB file to load for pycharmm [requires correct atom names and types].",
    )
    return parser.parse_args()


def resolve_dataset_path(arg: Path | None) -> Path:
    candidate = arg or Path(os.environ.get("MMML_DATA", "mmml/data/fixed-acetone-only_MP2_21000.npz"))
    if not candidate.exists():
        sys.exit(f"Dataset not found: {candidate}")
    return candidate


def resolve_checkpoint_paths(arg: Path | None) -> Tuple[Path, Path]:
    """Return (factory_base_dir, epoch_dir) for the supplied checkpoint."""

    from mmml.physnetjax.physnetjax.restart.restart import get_last

    candidate = arg or Path(os.environ.get("MMML_CKPT", "mmml/physnetjax/ckpts"))
    if not candidate.exists():
        sys.exit(f"Checkpoint directory not found: {candidate}")

    candidate = candidate.resolve()
    if not candidate.is_dir():
        sys.exit(f"Checkpoint path is not a directory: {candidate}")

    def last_dir(path: Path) -> Path:
        return Path(get_last(str(path)))

    # Allow pointing directly at an epoch directory containing manifest files
    if (candidate / "manifest.ocdbt").exists():
        return candidate.parent, candidate

    children = [child for child in candidate.iterdir() if child.is_dir()]
    if not children:
        sys.exit("Checkpoint path must contain epoch-* subdirectories or point to an epoch directory.")

    subdir = last_dir(candidate)
    if (subdir / "manifest.ocdbt").exists():
        # The user provided an experiment directory with epoch-* children
        return candidate, subdir

    epoch_dir = last_dir(subdir)
    if (epoch_dir / "manifest.ocdbt").exists():
        # Two-level hierarchy: ckpts/<experiment>/epoch-*
        return subdir, epoch_dir

    sys.exit("Could not locate an epoch directory under the supplied checkpoint path.")


def load_configuration(npz_path: Path, index: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    data = np.load(npz_path)
    n_samples = data["R"].shape[0]
    if index < 0 or index >= n_samples:
        sys.exit(f"Sample index {index} out of range (0..{n_samples - 1}).")

    Z = np.asarray(data["Z"][index], dtype=np.int32)
    R = np.asarray(data["R"][index], dtype=np.float64)

    references: Dict[str, np.ndarray] = {}
    for key in ("E", "F"):
        if key in data:
            references[key] = np.asarray(data[key][index])
    return Z, R, references


def load_model_parameters(epoch_dir: Path, natoms: int):
    from mmml.physnetjax.physnetjax.restart.restart import get_params_model

    params, model = get_params_model(str(epoch_dir), natoms=natoms)
    if model is None:
        sys.exit(
            "Checkpoint does not contain model attributes; cannot construct PhysNetJax model."
        )
    model.natoms = natoms
    return params, model


def compute_force_metrics(delta_forces: np.ndarray) -> Tuple[float, float]:
    rms = float(np.sqrt(np.mean(delta_forces**2)))
    max_abs = float(np.abs(delta_forces).max())
    return rms, max_abs


def flatten_array(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value)
    return arr.reshape(-1)


def main() -> int:
    args = parse_args()

    dataset_path = resolve_dataset_path(args.dataset)
    base_ckpt_dir, epoch_dir = resolve_checkpoint_paths(args.checkpoint)

    try:
        from ase import Atoms
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        sys.exit(f"ASE is required for this example: {exc}")

    try:
        from mmml.pycharmmInterface.mmml_calculator import (
            CutoffParameters,
            ev2kcalmol,
            setup_calculator,
        )
        from mmml.physnetjax.physnetjax.calc.helper_mlp import get_ase_calc
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        sys.exit(f"Required MMML modules not available: {exc}")

    Z, R, references = load_configuration(dataset_path, args.sample_index)
    natoms = len(Z)

    n_monomers = args.n_monomers
    if natoms % n_monomers != 0:
        sys.exit(
            f"Cannot evenly divide {natoms} atoms into {n_monomers} monomers; "
            "specify --atoms-per-monomer explicitly."
        )

    atoms_per_monomer = args.atoms_per_monomer or natoms // n_monomers
    # if atoms_per_monomer * n_monomers != natoms:
    #     sys.exit(
    #         "Provided --atoms-per-monomer does not match the dataset configuration."
    #     )

    params, model = load_model_parameters(epoch_dir, natoms)

    atoms = Atoms(numbers=Z, positions=R)

    cutoff = CutoffParameters(
        ml_cutoff=args.ml_cutoff,
        mm_switch_on=args.mm_switch_on,
        mm_cutoff=args.mm_cutoff,
    )

    if args.units == "eV":
        energy_factor = 1.0
        force_factor = 1.0
    else:
        energy_factor = ev2kcalmol
        force_factor = ev2kcalmol

    energy_unit_label = args.units
    force_unit_label = "eV/Å" if args.units == "eV" else "kcal/mol/Å"
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

    def print_component_comparison(name: str, hybrid: np.ndarray, reference: np.ndarray | None) -> None:
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
            print_component_comparison(key, hybrid_val, reference_val)

    dataset_report: Dict[str, Any] = {}
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


    # minimize in ase
    print("Minimizing in ase")
    import ase.optimize as ase_opt
    print("starting minimization")
    _ = ase_opt.BFGS(ml_atoms).run(fmax=0.01, steps=100)
    print("minimization done")
    print("starting minimization")
    _ = ase_opt.BFGS(atoms).run(fmax=0.01, steps=100)
    print("minimization done")

    if args.output:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nSaved report to {output_path}")

    if args.pdbfile:
        import pycharmm
        from mmml.pycharmmInterface.import_pycharmm import coor
        pdbfilename = str(args.pdbfile)

    #import pycharmm
    #from mmml.pycharmmInterface.import_pycharmm import coor
    #_ = pycharmm.lingo.charmm_script("ENER")
    #print(_)
    #print(coor.show())
        # pycharmm.read.sequence_string("ACO ACO")
        from mmml.pycharmmInterface.setupBox import setup_box_generic
        setup_box_generic(pdbfilename, side_length=1000)
        import ase
        import ase.io
        pdb_ase_atoms = ase.io.read(pdbfilename)
        print(pdb_ase_atoms)
        print(coor.get_positions())
        print(coor.show())
        params, model = load_model_parameters(epoch_dir, 200)
        model.natoms = 200
        print(model)
        calculator_factory = setup_calculator(
            ATOMS_PER_MONOMER=10,
            N_MONOMERS=20,
            ml_cutoff_distance=args.ml_cutoff,
            mm_switch_on=args.mm_switch_on,
            mm_cutoff=args.mm_cutoff,
            doML=True,
            doMM=args.include_mm,
            doML_dimer=not args.skip_ml_dimers,
            debug=args.debug,
            model_restart_path=base_ckpt_dir,
            MAX_ATOMS_PER_SYSTEM=len(pdb_ase_atoms),
            ml_energy_conversion_factor=energy_factor,
            ml_force_conversion_factor=force_factor,
        )
        hybrid_calc, _ = calculator_factory(
            atomic_numbers=Z,
            atomic_positions=R,
            n_monomers=20,
            cutoff_params=cutoff,
            doML=True,
            doMM=args.include_mm,
            doML_dimer=not args.skip_ml_dimers,
            backprop=True,
            debug=args.debug,
            energy_conversion_factor=energy_factor,
            force_conversion_factor=force_factor,
        )
        print(hybrid_calc)
        atoms = pdb_ase_atoms
        print(atoms)
        atoms.calc = hybrid_calc
        hybrid_energy = float(atoms.get_potential_energy())
        hybrid_forces = np.asarray(atoms.get_forces())
        print("Minimizing in hybrid calculator")
        _ = ase_opt.BFGS(atoms).run(fmax=0.05, steps=1000)


        print("\nAgreement within numerical noise indicates the calculators are consistent.")
        import io
        import ase
        import ase.calculators.calculator as ase_calc
        import ase.io as ase_io
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
        from ase.md.verlet import VelocityVerlet
        import ase.optimize as ase_opt
        import matplotlib.pyplot as plt
        import py3Dmol
        # Parameters.
        temperature = 200.0
        timestep_fs = 0.01
        num_steps = 100_000
        ase_atoms = atoms
        # Draw initial momenta.

        import pandas as pd
        xyz = pd.DataFrame(ase_atoms.get_positions(), columns=["x", "y", "z"])
        coor.set_positions(xyz)
        print(coor.show())
        from mmml.pycharmmInterface.import_pycharmm import minimize
        minimize.run_abnr(nstep=1000, tolenr=1e-4, tolgrd=1e-4)
        pycharmm.lingo.charmm_script("ENER")
        print(coor.show())
        ase_atoms.set_positions(coor.get_positions())
        _ = ase_opt.BFGS(atoms).run(fmax=0.001, steps=100)
        MaxwellBoltzmannDistribution(ase_atoms, temperature_K=temperature)
        Stationary(ase_atoms)  # Remove center of mass translation.
        ZeroRotation(ase_atoms)  # Remove rotations.

        MaxwellBoltzmannDistribution(ase_atoms, temperature_K=temperature)
        Stationary(ase_atoms)  # Remove center of mass translation.
        ZeroRotation(ase_atoms)  # Remove rotations.

        # Initialize Velocity Verlet integrator.
        integrator = VelocityVerlet(ase_atoms, timestep=timestep_fs*ase.units.fs)

        # open a trajectory file
        traj = ase_io.Trajectory(f'{args.output}trajectory_{temperature}K_{num_steps}steps.xyz', 'w')

        # Run molecular dynamics.
        frames = np.zeros((num_steps, len(ase_atoms), 3))
        potential_energy = np.zeros((num_steps,))
        kinetic_energy = np.zeros((num_steps,))
        total_energy = np.zeros((num_steps,))

        breakcount = 0
        for i in range(num_steps):
            # Run 1 time step.
            integrator.run(1)
            # Save current frame and keep track of energies.
            frames[i] = ase_atoms.get_positions()
            potential_energy[i] = ase_atoms.get_potential_energy()
            kinetic_energy[i] = ase_atoms.get_kinetic_energy()
            total_energy[i] = ase_atoms.get_total_energy()
            traj.write(ase_atoms)
            if kinetic_energy[i] > 20:
                pycharmm.lingo.charmm_script("ENER")
                import pandas as pd
                xyz = pd.DataFrame(ase_atoms.get_positions(), columns=["x", "y", "z"])
                coor.set_positions(xyz)
                from mmml.pycharmmInterface.import_pycharmm import minimize
                minimize.run_abnr(nstep=1000, tolenr=1e-2, tolgrd=1e-2)
                pycharmm.lingo.charmm_script("ENER")
                ase_atoms.set_positions(coor.get_positions())
                _ = ase_opt.BFGS(atoms).run(fmax=0.001, steps=100)
                MaxwellBoltzmannDistribution(ase_atoms, temperature_K=temperature)
                Stationary(ase_atoms)  # Remove center of mass translation.
                ZeroRotation(ase_atoms)  # Remove rotations.
                breakcount += 1
                if breakcount > 100:
                    print("Maximum number of breaks reached")
                    break
            # Occasionally print progress.
            if i % 10000 == 0:
                temperature += 1
                Stationary(ase_atoms)  # Remove center of mass translation.
                ZeroRotation(ase_atoms)  # Remove rotations.
                MaxwellBoltzmannDistribution(ase_atoms, temperature_K=temperature)
                print(f"temperature: {temperature}")

            if i % 100 == 0:
                print(f"step {i:5d} epot {potential_energy[i]: 5.3f} ekin {kinetic_energy[i]: 5.3f} etot {total_energy[i]: 5.3f}")



        # plot the time series of the energy
        plt.plot(total_energy)
        plt.xlabel('time [fs]')
        plt.ylabel('energy [eV]')
        plt.title('Total energy')
        plt.savefig(f'{args.output}total_energy_{temperature}K_{num_steps}steps.png')
        # plt.show()


    return 0




if __name__ == "__main__":
    raise SystemExit(main())
