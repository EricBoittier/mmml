"""Custom ASE calculators for hybrid ML/MM monomer and intermolecular simulations."""

from __future__ import annotations

from typing import Any, Callable, Sequence
import numpy as np

from ase.calculators.calculator import Calculator, all_changes
from mmml.data.units import KCAL_MOL_TO_EV
from mmml.interfaces.pycharmmInterface.mm_system_energy import nonbonded_energy_and_forces


class MolecularPhysNetCalculator(Calculator):
    """ASE calculator that computes intramolecular energies and forces for a peptide and solvent waters.

    Splits the atoms object into a peptide (single monomer) and individual water molecules,
    calculates their potential energies and forces independently using separate calculators,
    and aggregates them.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        peptide_calc: Calculator,
        water_calc: Calculator,
        peptide_indices: Sequence[int],
        water_indices: Sequence[Sequence[int]],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.peptide_calc = peptide_calc
        self.water_calc = water_calc
        self.peptide_indices = np.asarray(peptide_indices, dtype=int)
        self.water_indices = [np.asarray(idx, dtype=int) for idx in water_indices]

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)

        if atoms is None:
            raise ValueError("Atoms object is required")

        energy = 0.0
        forces = np.zeros((len(atoms), 3), dtype=np.float64)

        # Peptide (Trialanine)
        peptide = atoms[self.peptide_indices].copy()
        peptide.pbc = False
        peptide.calc = self.peptide_calc

        energy += peptide.get_potential_energy()
        forces[self.peptide_indices] += peptide.get_forces()

        # Waters
        for idx in self.water_indices:
            water = atoms[idx].copy()
            water.pbc = False
            water.calc = self.water_calc

            energy += water.get_potential_energy()
            forces[idx] += water.get_forces()

        self.results = {
            "energy": float(energy),
            "forces": forces,
        }


class MonomerSumCalculator(Calculator):
    """General ASE calculator that sums the intramolecular energies and forces of multiple monomers.

    For each group of atom indices specified in `monomer_indices`, a separate calculator
    is used to compute its isolated potential energy and forces (with periodic boundary
    conditions disabled, as is typical for molecular ML potentials).
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        monomer_indices: Sequence[Sequence[int]],
        calculators: Sequence[Calculator] | None = None,
        calculator_factory: Callable[[], Calculator] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.monomer_indices = [np.asarray(indices, dtype=int) for indices in monomer_indices]

        if calculators is not None:
            if len(calculators) != len(self.monomer_indices):
                raise ValueError("Need exactly one calculator per monomer.")
            self.calculators = list(calculators)
        elif calculator_factory is not None:
            self.calculators = [calculator_factory() for _ in self.monomer_indices]
        else:
            raise ValueError("Provide calculators or calculator_factory.")

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)

        if atoms is None:
            raise ValueError("Atoms object is required")

        energy = 0.0
        forces = np.zeros((len(atoms), 3), dtype=np.float64)

        for indices, calc in zip(self.monomer_indices, self.calculators):
            monomer = atoms[indices].copy()
            # Usually desirable for a molecular ML potential:
            monomer.pbc = False
            monomer.calc = calc

            energy += monomer.get_potential_energy()
            forces[indices] += monomer.get_forces()

        self.results = {
            "energy": float(energy),
            "forces": forces,
        }


class JAXIntermolecularCalculator(Calculator):
    """ASE calculator that computes intermolecular nonbonded (electrostatic, Lennard-Jones) interactions.

    Delegates nonbonded calculations to JAX via `nonbonded_energy_and_forces`.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        nbond_data: Any,
        nb_settings: Any,
        energy_factor: float = KCAL_MOL_TO_EV,
        force_factor: float = KCAL_MOL_TO_EV,
        molecule_id: np.ndarray | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.nbond_data = nbond_data
        self.nb_settings = nb_settings
        self.energy_factor = float(energy_factor)
        self.force_factor = float(force_factor)
        self.molecule_id = molecule_id

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)

        if atoms is None:
            raise ValueError("Atoms object is required")

        pos = np.asarray(atoms.get_positions(), dtype=np.float64)
        cell = np.asarray(atoms.cell.array, dtype=np.float64)

        terms_raw, forces = nonbonded_energy_and_forces(
            pos,
            self.nbond_data,
            cell,
            self.nb_settings,
            molecule_id=self.molecule_id,
        )

        terms = {key: float(value) for key, value in terms_raw.items()}

        if "total" in terms:
            energy = terms["total"]
        else:
            energy = sum(terms.values())

        self.results = {
            "energy": energy * self.energy_factor,
            "forces": np.asarray(forces, dtype=np.float64) * self.force_factor,
        }

        # Optional diagnostic information
        self.results["jax_terms"] = terms
