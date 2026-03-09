# from jax import config
# config.update('jax_enable_x64', True)
import ase
import ase.calculators.calculator as ase_calc
import ase.io as ase_io
import e3x
import jax
import numpy as np


@jax.jit
def evaluate_energies_and_forces(atomic_numbers, positions, dst_idx, src_idx):
    return model.apply(
        params,
        atomic_numbers=atomic_numbers,
        positions=positions,
        dst_idx=dst_idx,
        src_idx=src_idx,
    )


class MessagePassingCalculator(ase_calc.Calculator):
    implemented_properties = ["energy", "forces", "dipole"]

    def calculate(
        self, atoms, properties, system_changes=ase.calculators.calculator.all_changes
    ):
        ase_calc.Calculator.calculate(self, atoms, properties, system_changes)
        dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(len(atoms))
        output = evaluate_energies_and_forces(
            atomic_numbers=atoms.get_atomic_numbers(),
            positions=atoms.get_positions(),
            dst_idx=dst_idx,
            src_idx=src_idx,
        )
        if model.charges:
            self.results["dipole"] = output["dipoles"]
        self.results["energy"] = output[
            "energy"
        ].squeeze()  # * (ase.units.kcal/ase.units.mol)
        self.results["forces"] = output[
            "forces"
        ]  # * (ase.units.kcal/ase.units.mol) #/ase.units.Angstrom
