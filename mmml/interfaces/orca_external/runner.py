"""MMML potential evaluation for ORCA external-tool jobs."""

from __future__ import annotations

import warnings
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers

from mmml.cli.misc.fix_and_split import (
    convert_energy_ev_to_hartree,
    convert_forces_ev_angstrom_to_hartree_bohr,
)
from ase.calculators.calculator import Calculator

from mmml.interfaces.calculators.simple_inference import create_calculator_from_checkpoint
from mmml.interfaces.orca_external.protocol import (
    ExtInpData,
    natoms_from_xyz,
    read_extinp,
    read_xyz,
    write_engrad,
)
from mmml.interfaces.orca_external.settings import (
    MmmlOrcaSettings,
    add_model_arguments,
    settings_from_namespace,
)

_CALCULATOR_CACHE: dict[tuple[Any, ...], SimpleInferenceCalculator] = {}


def _cache_key(settings: MmmlOrcaSettings) -> tuple[Any, ...]:
    return (
        str(settings.checkpoint.resolve()),
        settings.cutoff,
        settings.is_noneq,
        settings.use_dcmnet_dipole,
        settings.disable_physnet_point_coulomb,
    )


def get_calculator(settings: MmmlOrcaSettings) -> Calculator:
    """Return a cached MMML calculator for the given settings."""
    key = _cache_key(settings)
    cached = _CALCULATOR_CACHE.get(key)
    if cached is not None:
        return cached

    kwargs: dict[str, Any] = {
        "is_noneq": settings.is_noneq,
        "use_dcmnet_dipole": settings.use_dcmnet_dipole,
        "disable_physnet_point_coulomb": settings.disable_physnet_point_coulomb,
    }
    if settings.cutoff is not None:
        kwargs["cutoff"] = settings.cutoff

    calculator = create_calculator_from_checkpoint(settings.checkpoint, **kwargs)
    _CALCULATOR_CACHE[key] = calculator
    return calculator


def clear_calculator_cache() -> None:
    """Drop cached calculators (useful in tests)."""
    _CALCULATOR_CACHE.clear()


def atoms_from_xyz(xyz_file: str | Path) -> Atoms:
    """Build an ASE ``Atoms`` object from an ORCA XYZ file."""
    symbols, coordinates = read_xyz(xyz_file)
    numbers = [atomic_numbers[symbol.capitalize()] for symbol in symbols]
    return Atoms(numbers=numbers, positions=np.asarray(coordinates, dtype=float))


def mmml_forces_to_orca_gradient(forces_ev_angstrom: np.ndarray) -> np.ndarray:
    """Convert ASE forces (eV/Å) to ORCA gradients (Eh/bohr)."""
    gradient_ev_angstrom = -np.asarray(forces_ev_angstrom, dtype=float)
    return convert_forces_ev_angstrom_to_hartree_bohr(gradient_ev_angstrom).reshape(-1)


def evaluate_structure(
    atoms: Atoms,
    calculator: Calculator,
    *,
    do_gradient: bool,
) -> tuple[float, list[float]]:
    """Evaluate energy (Eh) and optional gradient (Eh/bohr) for ``atoms``."""
    atoms.calc = calculator
    properties = ["energy", "forces"] if do_gradient else ["energy"]
    calculator.calculate(atoms, properties=properties)
    energy_hartree = convert_energy_ev_to_hartree(np.asarray(atoms.get_potential_energy(), dtype=float))

    gradient: list[float] = []
    if do_gradient:
        forces = atoms.get_forces()
        gradient = mmml_forces_to_orca_gradient(forces).tolist()

    return float(energy_hartree), gradient


def _warn_unsupported_extinp_fields(extinp: ExtInpData) -> None:
    if extinp.charge != 0:
        warnings.warn(
            "MMML ORCA external tool does not apply the requested total charge to the "
            "potential; ensure your checkpoint was trained for this charge state.",
            UserWarning,
            stacklevel=2,
        )
    if extinp.multiplicity != 1:
        warnings.warn(
            "MMML ORCA external tool ignores multiplicity (models are closed-shell).",
            UserWarning,
            stacklevel=2,
        )
    if extinp.pointcharges_path is not None:
        warnings.warn(
            "MMML ORCA external tool does not incorporate ORCA point charges.",
            UserWarning,
            stacklevel=2,
        )


class MmmlOrcaExternalRunner:
    """Run a single ORCA external-tool request with an MMML checkpoint."""

    def __init__(self, settings: MmmlOrcaSettings) -> None:
        self.settings = settings

    def run(self, inputfile: str | Path) -> Path:
        """Parse ``inputfile``, evaluate the structure, and write ``*.engrad``."""
        input_path = Path(inputfile).resolve()
        extinp = read_extinp(input_path)
        _warn_unsupported_extinp_fields(extinp)

        if not extinp.xyz_path.is_file():
            raise FileNotFoundError(f"XYZ file not found: {extinp.xyz_path}")

        atoms = atoms_from_xyz(extinp.xyz_path)
        calculator = get_calculator(self.settings)
        energy, gradient = evaluate_structure(
            atoms,
            calculator,
            do_gradient=extinp.do_gradient,
        )

        basename = extinp.xyz_path.name.removesuffix(".xyz")
        engrad_path = input_path.parent / f"{basename}.engrad"
        write_engrad(
            engrad_path,
            natoms=natoms_from_xyz(extinp.xyz_path),
            energy_hartree=energy,
            gradient_hartree_bohr=gradient or None,
        )
        return engrad_path


def build_runner_parser() -> ArgumentParser:
    """CLI argument parser for standalone ORCA external-tool runs."""
    parser = ArgumentParser(
        prog="mmml-orca-external",
        description="MMML ML potential wrapper for ORCA's external-tool interface.",
    )
    parser.add_argument("inputfile", help="ORCA *.extinp.tmp file")
    add_model_arguments(parser)
    return parser


def parse_runner_arguments(
    arguments: list[str] | None = None,
    *,
    default_settings: MmmlOrcaSettings | None = None,
) -> tuple[Path, MmmlOrcaSettings]:
    """Parse ORCA/client argument vectors into an input path and settings."""
    parser = build_runner_parser()
    args = parser.parse_args(arguments)
    settings = settings_from_namespace(args, default_settings=default_settings)
    return Path(args.inputfile), settings


def main(argv: list[str] | None = None) -> None:
    try:
        inputfile, settings = parse_runner_arguments(argv)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc
    MmmlOrcaExternalRunner(settings).run(inputfile)
