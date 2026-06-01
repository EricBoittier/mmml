# from jax import config
# config.update('jax_enable_x64', True)
import ase
import ase.calculators.calculator as ase_calc
import e3x
import jax
import numpy as np

# import numpy as np
from  mmml.models.physnetjax.physnetjax.calc.pycharmm_calculator import PyCharmm_Calculator

conversion = {
    "energy": 1,
    "forces": 1,
    "dipole": 1,
}
implemented_properties = ["energy", "forces", "dipole"]

def get_ase_calc(
    params,
    model,
    ase_mol,
    conversion=conversion,
    implemented_properties=implemented_properties,
    *,
    spooky_charge: float = 0.0,
    spooky_multiplicity: float = 1.0,
):
    """Ase calculator implementation for physnetjax model

    Args:
    params: params of the physnetjax model
    model: physnetjax model
    ase_mol: ase molecule
    conversion: conversion factor for the energy, forces, and dipole
    implemented_properties: implemented properties for the ase calculator
    spooky_charge: total system charge (broadcast per atom) for spooky EF models.
    spooky_multiplicity: spin multiplicity (broadcast per atom) for spooky EF models.

    Returns:
    Ase calculator implementation for physnetjax model
    """
    Implemented_properties = implemented_properties
    print(implemented_properties)

    assert model.natoms == len(ase_mol.get_atomic_numbers())
    is_spooky_model = "spooky_model" in type(model).__module__

    @jax.jit
    def evaluate_energies_and_forces(atomic_numbers, positions, dst_idx, src_idx):
        if is_spooky_model:
            n_atoms = atomic_numbers.shape[0]
            z = atomic_numbers
            atom_mask = (z > 0).astype(jax.numpy.float32)
            batch_segments = jax.numpy.zeros((n_atoms,), dtype=jax.numpy.int32)
            atom_mask_2d = (z > 0).astype(jax.numpy.float32)[None, :]
            valid_pairs = (atom_mask_2d[:, dst_idx] > 0) & (
                atom_mask_2d[:, src_idx] > 0
            )
            batch_mask = valid_pairs.astype(jax.numpy.float32).reshape(-1)
            q_atoms = jax.numpy.full(
                (n_atoms, 1), spooky_charge, dtype=jax.numpy.float32
            )
            s_atoms = jax.numpy.full(
                (n_atoms, 1), spooky_multiplicity, dtype=jax.numpy.float32
            )
            return model.apply(
                params,
                atomic_numbers=atomic_numbers,
                charges=q_atoms,
                spins=s_atoms,
                positions=positions,
                dst_idx=dst_idx,
                src_idx=src_idx,
                batch_segments=batch_segments,
                batch_size=1,
                batch_mask=batch_mask,
                atom_mask=atom_mask,
            )
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
            self.results["forces"] = np.array(output["forces"] * conversion["forces"])
            atoms.info["output"] = output

    return MessagePassingCalculator()


try:
    _ev_to_kcalmol = 1 / (ase.units.kcal / ase.units.mol)
except Exception:  # pragma: no cover - triggered when ASE units mocked
    _ev_to_kcalmol = 23.060548867

# PhysNet EF outputs eV and eV/Å; CHARMM MLpot expects kcal/mol and kcal/mol/Å.
pycharmm_conversion = {
    "energy": _ev_to_kcalmol,
    "forces": _ev_to_kcalmol,
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
        atomic_numbers = jax.numpy.array(batch["atomic_numbers"])
        positions = jax.numpy.array(batch["positions"])
        dst_idx = jax.numpy.array(batch["dst_idx"])
        src_idx = jax.numpy.array(batch["src_idx"])
        if "spooky_model" in type(model).__module__:
            n_atoms = atomic_numbers.shape[0]
            atom_mask = jax.numpy.ones((n_atoms,), dtype=jax.numpy.float32)
            batch_segments = jax.numpy.zeros((n_atoms,), dtype=jax.numpy.int32)
            batch_mask = jax.numpy.ones_like(dst_idx, dtype=jax.numpy.float32)
            q_atoms = jax.numpy.zeros((n_atoms, 1), dtype=jax.numpy.float32)
            s_atoms = jax.numpy.ones((n_atoms, 1), dtype=jax.numpy.float32)
            output = model.apply(
                params,
                atomic_numbers=atomic_numbers,
                charges=q_atoms,
                spins=s_atoms,
                positions=positions,
                dst_idx=dst_idx,
                src_idx=src_idx,
                batch_segments=batch_segments,
                batch_size=1,
                batch_mask=batch_mask,
                atom_mask=atom_mask,
            )
        else:
            output = model.apply(
                params,
                atomic_numbers=atomic_numbers,
                positions=positions,
                dst_idx=dst_idx,
                src_idx=src_idx,
            )
        output["energy"] = output["energy"].squeeze()
        output["forces"] = output["forces"].squeeze()
        output["energy"] *= conversion["energy"]
        output["forces"] *= conversion["forces"]
        return output

    default_indices = list(np.arange(NATOMS, dtype=int))

    class pyCModel:
        """Wrapper required by ``pycharmm.MLpot`` (``get_pycharmm_calculator``)."""

        def get_pycharmm_calculator(
            self,
            ml_atom_indices=None,
            ml_atomic_numbers=None,
            ml_charge=None,
            ml_fluctuating_charges=False,
            mlmm_atomic_charges=None,
            mlmm_cutoff=None,
            mlmm_cuton=None,
            **kwargs,
        ):
            indices = (
                list(ml_atom_indices)
                if ml_atom_indices is not None
                else default_indices
            )
            if ml_atomic_numbers is not None:
                z_ml = np.asarray(ml_atomic_numbers, dtype=int).tolist()
            else:
                z_ml = [int(Z[i]) for i in indices]
            graph_cutoff = float(getattr(model, "cutoff", 6.0))
            return PyCharmm_Calculator(
                model_calc,
                ml_atom_indices=indices,
                ml_atomic_numbers=z_ml,
                ml_charge=ml_charge,
                ml_fluctuating_charges=ml_fluctuating_charges,
                mlmm_atomic_charges=mlmm_atomic_charges,
                mlmm_cutoff=mlmm_cutoff if mlmm_cutoff is not None else 12.0,
                mlmm_cuton=mlmm_cuton if mlmm_cuton is not None else 10.0,
                ml_graph_cutoff=graph_cutoff,
                use_e3x_pair_list=True,
                **kwargs,
            )

    return pyCModel()
