"""Brief: ScaledHarmonic with explicit force return.

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
from _torch_helpers import ScaledHarmonicwForce  # noqa: E402
from openmm.unit import kilocalorie_per_mole, nanometer  # noqa: E402
from openmmtorch import TorchForce  # noqa: E402

from pycharmm import (  # noqa: E402
    cons_harm,
    coor,
    energy,
    lingo,
    omm,
    psf,
)


@pytest.fixture
def _torch_seed():
    np.random.seed(8692)


@pytest.mark.stateful
def test_torch_scaled_harmonic_with_forces(dummy_peptide_15, scratch_dir):
    """Module-level body of the original script, wrapped as a pytest test."""
    coors_ref = dummy_peptide_15.coors_ref
    atoms_restrained = dummy_peptide_15.atoms_restrained
    atom_mask = dummy_peptide_15.atom_mask
    coors_perturbed = dummy_peptide_15.coors_perturbed
    coor.show()
    # Turn on abolute harmonic restraints with comparison as reference
    cons_harm.setup_absolute(force_const=1 / 418.4, comparison=True, selection=atoms_restrained)
    energy.show()
    c_energy = energy.get_total()
    c_forces = coor.get_forces().to_numpy()[atom_mask]
    lingo.charmm_script("energy omm")
    c_omm_energy = energy.get_total()
    c_omm_forces = coor.get_forces().to_numpy()[atom_mask]
    cons_harm.turn_off()

    print(f"Harmonic Restraint energy from CHARMM/OpenMM: {c_omm_energy}")
    print(f"Harmonic restriant energy from CHARMM {c_energy:}")
    assert isclose(c_energy, c_omm_energy, rel_tol=1e-6)

    print("Forces CHARMM:\n", c_forces)
    print("Forces CHARMM/OpenMM:\n", c_omm_forces)
    assert np.sum(~np.isclose(c_forces, c_omm_forces, rtol=1e-7)) == 0

    # Render the compute graph to a TorchScript module
    module = pt.jit.script(ScaledHarmonicwForce(atom_mask, coors_ref[atom_mask], device))

    # Serialize the compute graph to a file
    torch_module_filename = str(scratch_dir / "scaledharmonicwforce.pt")
    module.save(torch_module_filename)

    torch_global_params = {"k": 1.0}

    # Implement in Torch-OpenMM
    system = openmm.System()
    for _ in range(psf.get_natom()):
        system.addParticle(1.0)

    print("***********ScaledHarmonicwForce***********")
    tforce = TorchForce(torch_module_filename)
    tforce.setOutputsForces(True)
    for k, v in torch_global_params.items():
        tforce.addGlobalParameter(k, v)

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
    print("***********ScaledHarmonicwForce***********")
    force_index = omm.torch_add_force(torch_module_filename)
    omm.torch_outputs_forces(force_index)
    # Track the param indices so we can update values in place below.
    torch_param_indices = {}
    for k, v in torch_global_params.items():
        torch_param_indices[k] = omm.torch_add_global_param(force_index, k, v)

    lingo.charmm_script("energy omm")
    c_omm_torch = energy.get_total()
    c_omm_torch_forces = coor.get_forces().to_numpy()[atom_mask]

    print(f"Harmonic Restraint energy from CHARMM/OpenMM-Torch: {c_omm_torch}")
    print(f"Harmonic restriant energy from Openmm-Torch {omm_energy:}")
    assert isclose(c_omm_torch, omm_energy, rel_tol=1e-6)

    print("Forces CHARMM:\n", c_omm_torch_forces)
    print("Forces from OpenMM:\n", omm_forces)
    assert np.sum(~np.isclose(c_omm_torch_forces, omm_forces, rtol=1e-7)) == 0

    print("***********Set force constant to a new value***********")
    torch_global_params = {"k": 50.0}

    for k, v in torch_global_params.items():
        context.setParameter(k, v)
        # Update in place by (force_index, param_index, value). The
        # api_torch_set_global_param shim sets the OpenMM default and
        # invalidates CHARMM's cached context so the next `energy omm`
        # call rebuilds with the new value.
        omm.torch_set_global_param(force_index, torch_param_indices[k], v)

    state = context.getState(getForces=True, getEnergy=True)
    omm_energy = state.getPotentialEnergy().value_in_unit(kilocalorie_per_mole)
    omm_forces = (
        -state.getForces(asNumpy=True).value_in_unit(kilocalorie_per_mole / nanometer)[atom_mask]
        / 10
    )

    lingo.charmm_script("energy omm")
    c_omm_torch = energy.get_total()
    c_omm_torch_forces = coor.get_forces().to_numpy()[atom_mask]

    print(f"Harmonic Restraint energy from CHARMM/OpenMM-Torch: {c_omm_torch}")
    print(f"Harmonic restriant energy from Openmm-Torch {omm_energy:}")
    assert isclose(c_omm_torch, omm_energy, rel_tol=1e-6)

    print("Forces CHARMM:\n", c_omm_torch_forces)
    print("Forces from OpenMM:\n", omm_forces)
    assert np.sum(~np.isclose(c_omm_torch_forces, omm_forces, rtol=1e-7)) == 0
