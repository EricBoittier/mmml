"""Brief: SimpleHarmonic torch module with multiple forces.

Verifies a torch.nn.Module-based custom restraint implementation matches
CHARMM's cons_harm absolute restraint energies and forces.

Skips when torch / openmm / openmmtorch are not importable.
"""

from math import isclose

import pytest

# A leftover torch directory under site-packages with no __init__.py is a
# namespace package, so `import torch' succeeds while torch.nn does not
# exist.  Ask for the submodule, which such a directory cannot provide.
pytest.importorskip("torch.nn")
pt = pytest.importorskip("torch")
openmm = pytest.importorskip("openmm")
pytest.importorskip("openmmtorch")

# CPU/GPU selection (referenced by classes below)
device = "cuda" if pt.cuda.is_available() else "cpu"
import numpy as np  # noqa: E402
from _torch_helpers import SimpleHarmonic  # noqa: E402
from openmm.unit import kilocalorie_per_mole, nanometer  # noqa: E402
from openmmtorch import TorchForce  # noqa: E402

from pycharmm import (  # noqa: E402
    SelectAtoms,
    cons_harm,
    coor,
    energy,
    lingo,
    omm,
    psf,
)


@pytest.fixture
def _torch_seed():
    np.random.seed(6270)


@pytest.mark.stateful
def test_torch_force_multiple(dummy_peptide_15, scratch_dir):
    """Module-level body of the original script, wrapped as a pytest test."""
    coors_ref = dummy_peptide_15.coors_ref
    atoms_restrained = dummy_peptide_15.atoms_restrained
    atom_mask = dummy_peptide_15.atom_mask
    coors_perturbed = dummy_peptide_15.coors_perturbed
    coor.show()

    # Select a SECOND random set of atoms to restrain, drawn from
    # the complement of the first set.
    complement = np.arange(psf.get_natom())[~atom_mask]
    print(f"Complement {complement}")
    atoms_restrained_list_2 = np.random.choice(complement, size=5, replace=False)
    atoms_restrained_2 = SelectAtoms().by_atom_nums(atoms_restrained_list_2)
    atom_mask_2 = np.array(list(atoms_restrained_2))
    print(f"Atoms restraind list 2 {atoms_restrained_list_2}")

    # Turn on abolute harmonic restraints with comparison as reference
    cons_harm.setup_absolute(force_const=1 / 418.4, comparison=True, selection=atoms_restrained)
    energy.show()
    c_energy = energy.get_total()
    c_forces = coor.get_forces().to_numpy()[atom_mask]
    lingo.charmm_script("energy omm")
    c_omm_energy = energy.get_total()
    c_omm_forces = coor.get_forces().to_numpy()[atom_mask]

    print(f"Harmonic Restraint energy from CHARMM/OpenMM: {c_omm_energy}")
    print(f"Harmonic restriant energy from CHARMM {c_energy:}")
    assert isclose(c_energy, c_omm_energy, rel_tol=1e-6)

    print("Forces CHARMM:\n", c_forces)
    print("Forces CHARMM/OpenMM:\n", c_omm_forces)
    assert np.sum(~np.isclose(c_forces, c_omm_forces, rtol=1e-7)) == 0

    # Turn on abolute harmonic restraints with comparison as reference_2
    cons_harm.setup_absolute(force_const=1 / 418.4, comparison=True, selection=atoms_restrained_2)
    energy.show()
    c_energy_2 = energy.get_total()
    c_forces_2 = coor.get_forces().to_numpy()[atom_mask_2]
    lingo.charmm_script("energy omm")
    c_omm_energy_2 = energy.get_total()
    c_omm_forces_2 = coor.get_forces().to_numpy()[atom_mask_2]
    # CHARMM-direct vs CHARMM-via-OMM: both should match for set 2,
    # exactly as the symmetric block below verifies for set 1.
    assert isclose(c_energy_2, c_omm_energy_2, rel_tol=1e-6), (
        f"set 2 CHARMM={c_energy_2} vs CHARMM/OpenMM={c_omm_energy_2}"
    )
    assert np.sum(~np.isclose(c_forces_2, c_omm_forces_2, rtol=1e-7)) == 0
    cons_harm.turn_off()

    # Render the compute graph to a TorchScript module
    module = pt.jit.script(SimpleHarmonic(atom_mask, coors_ref[atom_mask], device))

    # Serialize the compute graph to a file
    torch_force_filename = str(scratch_dir / "simpleharmonic.pt")
    module.save(torch_force_filename)

    # Implement in Torch-OpenMM
    system = openmm.System()
    for _ in range(psf.get_natom()):
        system.addParticle(1.0)

    print("***********SimpleHarmonic***********")
    tforce = TorchForce(torch_force_filename)

    system.addForce(tforce)
    context = openmm.Context(system, openmm.VerletIntegrator(1.0))
    context.setPositions(coors_perturbed / 10)
    state = context.getState(getForces=True, getEnergy=True)
    omm_energy = state.getPotentialEnergy().value_in_unit(kilocalorie_per_mole)
    omm_forces = (
        -state.getForces(asNumpy=True).value_in_unit(kilocalorie_per_mole / nanometer)[atom_mask]
        / 10
    )

    print(f"Energy from OpenMM: {omm_energy}")
    print(f"Harmonic Restraint energy from CHARMM: {c_energy}")
    print(f"Harmonic restriant energy from Torch {omm_energy:}")
    assert isclose(c_energy, omm_energy, rel_tol=1e-5)

    print("Forces CHARMM:\n", c_forces)
    print("Forces from OpenMM:\n", omm_forces)
    assert np.sum(~np.isclose(c_forces, omm_forces, rtol=1e-7)) == 0

    # Check restrained torch force in CHARMM/OpenMM
    print("***********SimpleHarmonic***********")
    omm.torch_add_force(torch_force_filename)

    lingo.charmm_script("energy omm")
    c_omm_torch = energy.get_total()
    c_omm_torch_forces = coor.get_forces().to_numpy()[atom_mask]
    print(f"Harmonic Restraint energy from CHARMM/OpenMM-Torch: {c_omm_torch}")
    print(f"Harmonic restriant energy from Openmm-Torch {omm_energy:}")
    assert isclose(c_omm_torch, omm_energy, rel_tol=1e-6)

    print("Forces CHARMM:\n", c_omm_torch_forces)
    print("Forces from OpenMM:\n", omm_forces)
    assert np.sum(~np.isclose(c_omm_torch_forces, omm_forces, rtol=1e-7)) == 0

    # Render the compute graph to a TorchScript module
    module = pt.jit.script(SimpleHarmonic(atom_mask_2, coors_ref[atom_mask_2], device))

    torch_force_filename = str(scratch_dir / "simpleharmonic_2.pt")
    module.save(torch_force_filename)
    omm.torch_add_force(torch_force_filename)
    lingo.charmm_script("energy omm")
    c_omm_torch_2 = energy.get_total()
    c_omm_torch_forces_2 = coor.get_forces().to_numpy()[atom_mask_2]
    print(f"Harmonic Restraint energy from CHARMM/OpenMM-Torch: {c_omm_torch_2}")
    print(f"Harmonic restriant energy from Openmm-Torch {c_omm_energy_2:}")
    assert isclose(c_omm_torch, omm_energy, rel_tol=1e-6)

    print("Forces CHARMM:\n", c_omm_torch_forces_2)
    print("Forces from OpenMM:\n", c_omm_forces_2)
    assert np.sum(~np.isclose(c_omm_torch_forces_2, c_omm_forces_2, rtol=1e-7)) == 0
