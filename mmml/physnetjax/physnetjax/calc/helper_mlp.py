# from jax import config
# config.update('jax_enable_x64', True)
import ase
import ase.calculators.calculator as ase_calc
import ase.io as ase_io
import e3x
import jax
import numpy as np

# import numpy as np
from  mmml.physnetjax.physnetjax.calc.pycharmm_calculator import PyCharmm_Calculator

conversion = {
    "energy": 1,
    "forces": 1,
    "dipole": 1,
}
implemented_properties = ["energy", "forces", "dipole"]

def get_ase_calc(params, model, ase_mol, 
conversion=conversion, 
implemented_properties = implemented_properties):
    """Ase calculator implementation for physnetjax model

    Args:
    params: params of the physnetjax model
    model: physnetjax model
    ase_mol: ase molecule
    conversion: conversion factor for the energy, forces, and dipole
    implemented_properties: implemented properties for the ase calculator

    Returns:
    Ase calculator implementation for physnetjax model
    """
    Implemented_properties = implemented_properties
    print(implemented_properties)

    assert model.natoms == len(ase_mol.get_atomic_numbers())

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
        implemented_properties = Implemented_properties

        def calculate(
            self,
            atoms,
            properties,
            system_changes=ase.calculators.calculator.all_changes,
        ):
            ase_calc.Calculator.calculate(self, atoms, properties, system_changes)
            dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(len(atoms))
            output = evaluate_energies_and_forces(
                atomic_numbers=atoms.get_atomic_numbers(),
                positions=atoms.get_positions(),
                dst_idx=dst_idx,
                src_idx=src_idx,
            )
            if model.charges and "dipoles" in properties:
                self.results["dipole"] = output["dipoles"] * conversion["dipole"]
            self.results["energy"] = output["energy"].squeeze() * conversion["energy"]
            self.results["forces"] = output["forces"] * conversion["forces"]
            atoms.info["output"] = output

    return MessagePassingCalculator()


try:
    _kcal_to_ev = 1 / (ase.units.kcal / ase.units.mol)
except Exception:  # pragma: no cover - triggered when ASE units mocked
    _kcal_to_ev = 1.0

pycharmm_conversion = {
    "energy": _kcal_to_ev,
    "forces": _kcal_to_ev,
    "charge": 1,
}


def get_pyc(params, model, ase_mol, conversion=pycharmm_conversion):
    """PyCharmm calculator implementation for physnetjax model

    Args:
    params: params of the physnetjax model
    model: physnetjax model
    ase_mol: ase molecule
    conversion: conversion factor for the energy, forces, and dipole

    Returns:
    PyCharmm calculator implementation for physnetjax model
    """
    Z = ase_mol.get_atomic_numbers()
    Z = [_ if _ < 9 else 6 for _ in Z]
    NATOMS = len(Z)
    assert model.natoms == NATOMS

    @jax.jit
    def model_calc(batch):
        output = model.apply(
            params,
            atomic_numbers=jax.numpy.array(batch["atomic_numbers"]),
            positions=jax.numpy.array(batch["positions"]),
            dst_idx=jax.numpy.array(batch["dst_idx"]),
            src_idx=jax.numpy.array(batch["src_idx"]),
        )
        output["energy"] = output["energy"].squeeze()
        output["forces"] = output["forces"].squeeze()
        output["energy"] *= conversion["energy"]
        output["forces"] *= conversion["forces"]
        return output

    pyc = PyCharmm_Calculator(
        model_calc,
        ml_atom_indices=np.arange(model.natoms),
        ml_atomic_numbers=Z,
        ml_charge=None,
        # ml_fluctuating_charges = model.charges
    )

    if __name__ == "__main__":
        blah = np.array(list(range(NATOMS)))
        blah1 = np.array(list(range(10000)))
        blah2 = np.arange(NATOMS) * 1.0
        print("...", dir(pyc)), pyc, "pyc?"
        _ = pyc.calculate_charmm(
            Natom=NATOMS,
            Ntrans=0,
            Natim=0,
            idxp=blah,
            x=blah2,
            y=blah2,
            z=blah2,
            dx=blah2,
            dy=blah2,
            dz=blah2,
            Nmlp=NATOMS,
            Nmlmmp=NATOMS,
            idxi=blah1,
            idxj=blah1,
            idxjp=blah,
            idxu=blah,
            idxv=blah,
            idxup=blah,
            idxvp=blah,
        )

    class pyCModel:
        def __init__():
            pass

        def get_pycharmm_calculator(
            ml_atom_indices=None,
            ml_atomic_numbers=None,
            ml_charge=None,
            ml_fluctuating_charges=False,
            mlmm_atomic_charges=None,
            mlmm_cutoff=None,
            mlmm_cuton=None,
            **kwargs,
        ):
            """Dummy function to return the PyCharmm_Calculator object"""
            return pyc

    return pyCModel
