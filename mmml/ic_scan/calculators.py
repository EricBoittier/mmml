"""Explicit calculator selection for internal-coordinate scans.

Reuses the dimer-scan calculator implementations where possible; IC scans always
request total molecular energies (no fragment interaction decomposition).
"""

from __future__ import annotations

from collections.abc import Callable

from ase.calculators.calculator import Calculator

from mmml.analysis.dimer_scans import make_dftb3_d4_calculator, make_xtb_calculator
from mmml.dimer_scan.calculators import PySCFDimerCalculator

from .config import IcScanConfig

CalculatorFactory = Callable[[], Calculator]


def calculator_factory(config: IcScanConfig) -> CalculatorFactory:
    """Validate calculator requirements and return an ASE calculator factory."""

    if config.calculator is None:
        raise ValueError("calculator is required for energy evaluation")
    if config.calculator == "xtb":
        if config.checkpoint is not None:
            raise ValueError("the xtb calculator does not accept a checkpoint")
        return lambda: make_xtb_calculator(method=config.method or "GFN2-xTB")
    if config.calculator == "spookynet":
        if config.checkpoint is None:
            raise ValueError("the spookynet calculator requires checkpoint")

        def create_spookynet() -> Calculator:
            from mmml.models.spookynet_calc import SpookyNetCalculator

            return SpookyNetCalculator(
                config.checkpoint,
                charge=float(config.charge or 0.0),
                spin_multiplicity=float(config.spin or 1.0),
            )

        return create_spookynet
    if config.calculator == "mbd":
        if config.checkpoint is None:
            raise ValueError("the mbd calculator requires checkpoint")

        def create_mbd() -> Calculator:
            from mmml.models.mbd import QCMLMBDCalculator

            return QCMLMBDCalculator(
                config.checkpoint,
                charge=float(config.charge or 0.0),
                multiplicity=float(config.spin or 1.0),
            )

        return create_mbd
    if config.calculator == "multipoles":
        if config.checkpoint is None:
            raise ValueError("the multipoles calculator requires checkpoint")

        def create_multipoles() -> Calculator:
            from mmml.models.multipoles import LearnedMolecularMultipoleElectrostatics

            return LearnedMolecularMultipoleElectrostatics(
                config.checkpoint,
                softening_bohr=0.5,
                force_step_angstrom=config.multipole_force_step_angstrom,
            )

        return create_multipoles
    if config.calculator == "efield":
        if config.checkpoint is None:
            raise ValueError("the efield calculator requires checkpoint")
        if config.electric_field_au is None:
            raise ValueError("the efield calculator requires electric_field_au")

        def create_efield() -> Calculator:
            from mmml.models.efield.ase_calc_EF import EFieldCalculator

            return EFieldCalculator(
                config.checkpoint,
                config_path=config.calculator_config,
                electric_field=config.electric_field_au,
                field_scale=1.0,
            )

        return create_efield
    if config.calculator == "dftb3-d4":
        if config.slako_dir is None:
            raise ValueError("dftb3-d4 requires slako_dir")
        if config.workdir is None:
            raise ValueError("dftb3-d4 requires workdir")
        return lambda: make_dftb3_d4_calculator(
            slako_dir=config.slako_dir,
            workdir=config.workdir,
            command=config.executable or "dftb+",
        )
    if config.calculator == "pyscf":
        return lambda: PySCFDimerCalculator(
            method=config.method or "dft",
            basis=config.basis or "def2-svp",
            xc=config.xc or "pbe0",
            charge=int(config.charge or 0),
            multiplicity=int(config.spin or 1),
        )
    if config.calculator == "physnet":
        if config.checkpoint is None:
            raise ValueError("the physnet calculator requires checkpoint")
        checkpoint = config.checkpoint.resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")

        def create() -> Calculator:
            from mmml.interfaces.calculators.simple_inference import (
                create_calculator_from_checkpoint,
            )

            return create_calculator_from_checkpoint(
                checkpoint,
                charge=config.charge,
                spin=config.spin,
            )

        return create
    raise ValueError(
        f"unsupported calculator {config.calculator!r}; see mmml ic-scan --help"
    )
