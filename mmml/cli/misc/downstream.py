#!/usr/bin/env python3
"""Downstream task CLI for MMML models.

This command generalises the CO2 downstream workflow so that any dataset
stored in MMML NPZ format (e.g. glycol) can be analysed with arbitrary
PhysNet/DCMNet checkpoints.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from ase import Atoms
from ase.optimize import BFGS

# Ensure example modules are importable (for legacy training utilities)
REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = REPO_ROOT / "examples" / "co2" / "dcmnet_physnet_train"
sys.path.insert(0, str(EXAMPLE_DIR))

from mmml.calculators.simple_inference import create_calculator_from_checkpoint
from dynamics_calculator import (  # type: ignore
    calculate_frequencies,
    calculate_ir_spectrum,
    run_molecular_dynamics,
    compute_ir_from_md,
)
from raman_calculator import (  # type: ignore
    calculate_raman_spectrum,
)


@dataclass
class SinglePointMetrics:
    energy: float
    dipole: list[float]
    max_force: float
    total_charge: float


@dataclass
class HarmonicMetrics:
    optimized_energy: float
    optimization_steps: int
    max_force: float
    frequencies_cm1: list[float]
    ir_intensity_physnet: list[float]
    ir_intensity_dcmnet: Optional[list[float]]
    raman_iso: Optional[list[float]] = None


@dataclass
class MDMetrics:
    average_temperature: float
    energy_std: float
    ir_frequencies_cm1: Optional[list[float]]
    ir_intensity_physnet: Optional[list[float]]
    runtime_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mmml downstream",
        description="Run harmonic / MD downstream analyses on an MMML dataset",
    )
    parser.add_argument("--dataset", required=True, type=Path, help="Path to dataset NPZ file")
    parser.add_argument("--checkpoint-dcm", required=True, type=Path, help="Equivariant checkpoint (best_params.pkl)")
    parser.add_argument("--checkpoint-noneq", required=True, type=Path, help="Non-equivariant checkpoint")
    parser.add_argument("--sample-index", type=int, default=0, help="Configuration index in dataset")
    parser.add_argument(
        "--mode",
        choices=["check", "quick", "full"],
        default="quick",
        help="Analysis mode: 'check' (single-point), 'quick' (harmonic), 'full' (+MD)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("downstream_output"))
    parser.add_argument("--temperature", type=float, default=300.0, help="MD temperature in Kelvin")
    parser.add_argument("--md-steps", type=int, default=5000, help="Number of MD steps for full mode")
    parser.add_argument("--timestep", type=float, default=0.5, help="MD timestep in femtoseconds")
    parser.add_argument("--ir-delta", type=float, default=0.01, help="Displacement for IR finite differences (Å)")
    parser.add_argument("--freq-delta", type=float, default=0.01, help="Displacement for numerical Hessian (Å)")
    parser.add_argument("--opt-fmax", type=float, default=0.01, help="Geometry optimisation force threshold (eV/Å)")
    parser.add_argument("--opt-steps", type=int, default=200, help="Maximum optimisation steps")
    parser.add_argument("--raman", action="store_true", help="Compute Raman spectrum using finite-field polarizability")
    parser.add_argument("--raman-delta", type=float, default=0.01, help="Displacement for Raman derivatives (Å)")
    parser.add_argument("--raman-field", type=float, default=1e-4, help="Electric field strength for Raman finite-field (V/Å)")
    return parser.parse_args()


def load_dataset_sample(npz_path: Path, index: int) -> tuple[Atoms, Dict[str, Any]]:
    data = np.load(npz_path, allow_pickle=True)
    n_samples = data["R"].shape[0]
    if index < 0 or index >= n_samples:
        raise IndexError(f"Sample index {index} out of range (0..{n_samples - 1})")

    natoms_total = data["R"].shape[1]
    natoms_real = int(data["N"][index])

    Z_full = np.asarray(data["Z"][index], dtype=np.int32)
    R_full = np.asarray(data["R"][index], dtype=np.float64)

    Z = Z_full[:natoms_real]
    R = R_full[:natoms_real]

    atoms = Atoms(numbers=Z, positions=R)

    metadata: Dict[str, Any] = {
        "natoms_total": int(natoms_total),
        "natoms_real": natoms_real,
    }
    for key in ("E", "F", "D", "metadata"):
        if key in data:
            value = data[key][index]
            if isinstance(value, np.ndarray):
                metadata[key] = value.tolist()
            else:
                metadata[key] = value
    return atoms, metadata


def compute_single_point(atoms: Atoms, calculator) -> SinglePointMetrics:
    atoms_sp = atoms.copy()
    atoms_sp.calc = calculator
    energy = float(atoms_sp.get_potential_energy())
    forces = np.array(atoms_sp.get_forces())
    dipole = calculator.results.get("dipole", atoms_sp.get_dipole_moment())
    charges = calculator.results.get("charges", np.zeros(len(atoms_sp)))

    return SinglePointMetrics(
        energy=energy,
        dipole=np.asarray(dipole).tolist(),
        max_force=float(np.abs(forces).max()),
        total_charge=float(np.sum(charges)),
    )


def run_harmonic_workflow(
    label: str,
    atoms: Atoms,
    calculator,
    output_dir: Path,
    opt_fmax: float,
    opt_steps: int,
    freq_delta: float,
    ir_delta: float,
    raman: bool,
    raman_delta: float,
    raman_field: float,
) -> HarmonicMetrics:
    work_dir = output_dir / label
    work_dir.mkdir(parents=True, exist_ok=True)

    atoms_opt = atoms.copy()
    atoms_opt.calc = calculator
    opt = BFGS(atoms_opt, logfile=str(work_dir / "optimization.log"))
    opt.run(fmax=opt_fmax, steps=opt_steps)
    opt_energy = float(atoms_opt.get_potential_energy())
    max_force = float(np.abs(atoms_opt.get_forces()).max())

    freqs, vib = calculate_frequencies(atoms_opt.copy(), calculator, delta=freq_delta, output_dir=work_dir)
    freqs_real = np.real(freqs)

    ir_data = calculate_ir_spectrum(
        atoms_opt.copy(),
        calculator,
        vib,
        delta=ir_delta,
        output_dir=work_dir,
    )

    ir_physnet = np.asarray(ir_data.get("intensities_physnet", [])).tolist()
    ir_dcmnet = (
        np.asarray(ir_data.get("intensities_dcmnet", [])).tolist()
        if "intensities_dcmnet" in ir_data
        else None
    )

    raman_iso: Optional[list[float]] = None
    if raman:
        raman_data = calculate_raman_spectrum(
            atoms_opt.copy(),
            calculator,
            vib,
            delta=raman_delta,
            field_strength=raman_field,
            output_dir=work_dir,
        )
        if "isotropic" in raman_data:
            raman_iso = np.asarray(raman_data["isotropic"]).tolist()

    vib.clean()

    return HarmonicMetrics(
        optimized_energy=opt_energy,
        optimization_steps=int(opt.nsteps),
        max_force=max_force,
        frequencies_cm1=freqs_real.tolist(),
        ir_intensity_physnet=ir_physnet,
        ir_intensity_dcmnet=ir_dcmnet,
        raman_iso=raman_iso,
    )


def run_md_workflow(
    label: str,
    atoms: Atoms,
    calculator,
    output_dir: Path,
    ensemble: str,
    temperature: float,
    timestep: float,
    nsteps: int,
) -> MDMetrics:
    work_dir = output_dir / label
    work_dir.mkdir(parents=True, exist_ok=True)

    md_atoms = atoms.copy()
    md_atoms.calc = calculator

    start = time.time()
    md_results = run_molecular_dynamics(
        md_atoms,
        calculator,
        ensemble=ensemble,
        temperature=temperature,
        timestep=timestep,
        nsteps=nsteps,
        save_dipoles=True,
        output_dir=work_dir,
    )
    runtime = time.time() - start

    ir_results = compute_ir_from_md(md_results, output_dir=work_dir)
    md_results.update(ir_results)

    temps = np.asarray(md_results.get("temperatures", []))
    energies = np.asarray(md_results.get("energies_tot", []))

    freqs = (
        np.asarray(md_results.get("ir_frequencies", [])).tolist()
        if "ir_frequencies" in md_results
        else None
    )
    intensities = (
        np.asarray(md_results.get("ir_intensities_physnet", [])).tolist()
        if "ir_intensities_physnet" in md_results
        else None
    )

    return MDMetrics(
        average_temperature=float(temps.mean()) if temps.size else 0.0,
        energy_std=float(energies.std()) if energies.size else 0.0,
        ir_frequencies_cm1=freqs,
        ir_intensity_physnet=intensities,
        runtime_seconds=runtime,
    )


def main():
    args = parse_args()

    atoms, metadata = load_dataset_sample(args.dataset, args.sample_index)
    print(f"Loaded sample {args.sample_index} with {len(atoms)} atoms from {args.dataset}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    calc_dcm = create_calculator_from_checkpoint(args.checkpoint_dcm)
    calc_noneq = create_calculator_from_checkpoint(args.checkpoint_noneq, is_noneq=True)

    summary: Dict[str, Any] = {
        "dataset": str(args.dataset),
        "sample_index": args.sample_index,
        "metadata": metadata,
        "atoms": {
            "numbers": atoms.get_atomic_numbers().tolist(),
            "positions": atoms.get_positions().tolist(),
        },
        "calculators": {
            "dcmnet": str(args.checkpoint_dcm),
            "noneq": str(args.checkpoint_noneq),
        },
        "mode": args.mode,
    }

    if args.mode == "check":
        summary["single_point"] = {
            "dcmnet": asdict(compute_single_point(atoms, calc_dcm)),
            "noneq": asdict(compute_single_point(atoms, calc_noneq)),
        }
    else:
        print("\nRunning harmonic analysis (DCMNet)...")
        harmonic_dcm = run_harmonic_workflow(
            "dcmnet",
            atoms,
            calc_dcm,
            output_dir,
            args.opt_fmax,
            args.opt_steps,
            args.freq_delta,
            args.ir_delta,
            args.raman,
            args.raman_delta,
            args.raman_field,
        )

        print("\nRunning harmonic analysis (Non-Equivariant)...")
        harmonic_noneq = run_harmonic_workflow(
            "noneq",
            atoms,
            calc_noneq,
            output_dir,
            args.opt_fmax,
            args.opt_steps,
            args.freq_delta,
            args.ir_delta,
            args.raman,
            args.raman_delta,
            args.raman_field,
        )

        summary["harmonic"] = {
            "dcmnet": asdict(harmonic_dcm),
            "noneq": asdict(harmonic_noneq),
        }

        summary["single_point"] = {
            "dcmnet": asdict(compute_single_point(atoms, calc_dcm)),
            "noneq": asdict(compute_single_point(atoms, calc_noneq)),
        }

        if args.mode == "full":
            print("\nRunning MD analysis (DCMNet)...")
            md_dcm = run_md_workflow(
                "dcmnet",
                atoms,
                calc_dcm,
                output_dir,
                ensemble="nvt",
                temperature=args.temperature,
                timestep=args.timestep,
                nsteps=args.md_steps,
            )

            print("\nRunning MD analysis (Non-Equivariant)...")
            md_noneq = run_md_workflow(
                "noneq",
                atoms,
                calc_noneq,
                output_dir,
                ensemble="nvt",
                temperature=args.temperature,
                timestep=args.timestep,
                nsteps=args.md_steps,
            )

            summary["md"] = {
                "dcmnet": asdict(md_dcm),
                "noneq": asdict(md_noneq),
            }

    summary_path = output_dir / "downstream_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Summary written to {summary_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
