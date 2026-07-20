"""Explicit calculator selection for supported dimer scans."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from ase import units

from ase.calculators.calculator import Calculator

from mmml.analysis.dimer_scans import make_dftb3_d4_calculator, make_xtb_calculator

from .config import DimerScanConfig


class PySCFDimerCalculator(Calculator):
    """Lazy molecular PySCF HF/DFT calculator for scan geometries."""

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        *,
        method: str = "dft",
        basis: str = "def2-svp",
        xc: str = "pbe0",
        charge: int = 0,
        multiplicity: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.method = method.lower()
        self.basis = basis
        self.xc = xc
        self.charge = int(charge)
        self.multiplicity = int(multiplicity)
        if self.method not in {"hf", "dft"}:
            raise ValueError("canonical dimer PySCF currently supports method=hf or dft")
        if self.multiplicity < 1:
            raise ValueError("multiplicity must be at least 1")

    def calculate(self, atoms=None, properties=None, system_changes=None):
        from ase.calculators.calculator import all_changes

        super().calculate(
            atoms,
            properties or ("energy", "forces"),
            all_changes if system_changes is None else system_changes,
        )
        try:
            from pyscf import dft, gto, scf
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "PySCF dimer scans require the mmml quantum dependencies"
            ) from exc
        atom_spec = [
            (symbol, tuple(position))
            for symbol, position in zip(
                self.atoms.get_chemical_symbols(),
                self.atoms.get_positions(),
                strict=True,
            )
        ]
        molecule = gto.M(
            atom=atom_spec,
            unit="Angstrom",
            basis=self.basis,
            charge=self.charge,
            spin=self.multiplicity - 1,
            verbose=0,
        )
        unrestricted = molecule.spin != 0
        if self.method == "hf":
            mean_field = scf.UHF(molecule) if unrestricted else scf.RHF(molecule)
        else:
            mean_field = dft.UKS(molecule) if unrestricted else dft.RKS(molecule)
            mean_field.xc = self.xc
        energy_hartree = float(mean_field.kernel())
        if not mean_field.converged:
            raise RuntimeError("PySCF SCF did not converge")
        gradient_hartree_bohr = np.asarray(mean_field.nuc_grad_method().kernel())
        self.results = {
            "energy": energy_hartree * units.Hartree,
            "forces": -gradient_hartree_bohr * units.Hartree / units.Bohr,
        }


def calculator_factory(config: DimerScanConfig) -> Callable[[], Calculator]:
    """Validate calculator requirements and return an ASE calculator factory."""

    if config.calculator == "xtb":
        if config.checkpoint is not None:
            raise ValueError("the xtb calculator does not accept a checkpoint")
        return lambda: make_xtb_calculator(method=config.method or "GFN2-xTB")
    if config.calculator == "spookynet":
        if config.checkpoint is None:
            raise ValueError("the spookynet calculator requires --checkpoint")

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
            raise ValueError("the mbd calculator requires --checkpoint")

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
            raise ValueError("the multipoles calculator requires --checkpoint")

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
            raise ValueError("the efield calculator requires --checkpoint")
        if config.electric_field_au is None:
            raise ValueError("the efield calculator requires --electric-field EX EY EZ")

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
            raise ValueError("dftb3-d4 requires --slako-dir")
        if config.workdir is None:
            raise ValueError("dftb3-d4 requires --calculator-workdir")
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
            raise ValueError("the physnet calculator requires --checkpoint")
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
        f"unsupported calculator {config.calculator!r}; see mmml dimer-scan --help"
    )
