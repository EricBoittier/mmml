"""Integration test for OpenMM-Torch alongside custom forces infrastructure.

Creates a minimal TorchScript module that returns a constant energy and
constant forces, adds it via omm.torch_add_force(), runs dynamics, and
verifies the forces apply correctly.

Run with:
    CHARMM_LIB_DIR=.../lib python -m pytest test_custom_forces_torch.py -v

----

openmm-torch 1.5.1 CUDA + ``setOutputsForces(True)`` device-mismatch bug
=======================================================================

openmm-torch 1.5.1 ships a CUDA-platform bug in ``CudaTorchKernels.cpp``
(addForcesKernel launch) that fires ``CUDA_ERROR_ILLEGAL_ADDRESS (700)``
when a TorchScript module declared with ``setOutputsForces(True)``
returns a ``forces`` tensor that lives on a **different device** than
the ``positions`` input it received.  The kernel reads the returned
tensor's data pointer as a device pointer without checking residency,
so any CPU-resident output crashes the context the first time it
synchronizes.

Concretely: when OpenMM runs on the CUDA platform it hands the module a
CUDA tensor for ``positions``.  Modules whose ``forward`` writes to a
freshly created ``pt.zeros(n, 3)`` (no device argument) put the result
on CPU.  That mismatch is the trigger.  The bug is **not** about
``setOutputsForces(True)`` per se -- modules that return tensors on the
same device as positions (e.g. ``_torch_helpers.SimpleHarmonicwForce``,
which calls ``.to(device)`` in ``__init__``) run cleanly on CUDA with
``setOutputsForces(True)``.

The fix on our side is one line in ``ConstantForce.forward``: allocate
``energy`` and ``forces`` with
``device=positions.device, dtype=positions.dtype``.  With that, the
test runs on whichever OpenMM platform the build chose and exercises
the CHARMM <-> openmm-torch bridge end-to-end on the GPU (strictly more
coverage than the prior CPU-platform pin).

The underlying kernel bug is still upstream; ``CudaTorchKernels.cpp``
should do an explicit ``cudaMemcpy`` when the returned tensor isn't
already on the active CUDA device, rather than assuming it.
"""

import os

import pytest

import pycharmm
import pycharmm.omm as omm
import pycharmm.psf as psf
from pycharmm import coor, generate, lingo, read

# torch may be absent (e.g. the CI env has no torch). Skip the whole module
# cleanly in that case -- matching the other torch tests
# (test_torch_harmonic_scaled.py, _torch_helpers.py) -- instead of letting the
# module-scope `class ConstantForce(pt.nn.Module)` below raise NameError at
# collection, which aborts the ENTIRE pytest session (not just this file).
# A leftover torch directory under site-packages with no __init__.py is a
# namespace package, so `import torch' succeeds while torch.nn does not
# exist.  Ask for the submodule, which such a directory cannot provide.
pytest.importorskip("torch.nn")
pt = pytest.importorskip("torch")

# CHARMM builds without --with-ommtorch route api_torch_* calls into a
# Fortran stub that calls wrndie(-1, ...) -> _gfortran_exit, which kills
# the process silently from pytest's perspective.  Skip rather than crash.
HAS_OMMTORCH = omm.has_ommtorch()


class ConstantForce(pt.nn.Module):
    """A TorchScript module that applies a constant force in +x.

    Returned tensors are allocated on the same device (and with the same
    dtype) as ``positions``.  This is required for correctness with
    openmm-torch 1.5.1 on the CUDA platform: returning CPU-resident
    tensors when positions live on the GPU triggers
    ``CUDA_ERROR_ILLEGAL_ADDRESS`` inside the CudaTorch addForcesKernel
    (see module docstring at the top of this file).
    """

    def __init__(self, force_x):
        super().__init__()
        self.force_x = force_x

    def forward(self, positions):
        """Return constant energy and forces.

        Parameters
        ----------
        positions : torch.Tensor with shape (nparticles, 3)

        Returns
        -------
        energy : torch.Scalar (kJ/mol)
        forces : torch.Tensor with shape (nparticles, 3) (kJ/mol/nm)
        """
        n = positions.shape[0]
        energy = pt.zeros(1, device=positions.device, dtype=positions.dtype)
        forces = pt.zeros(n, 3, device=positions.device, dtype=positions.dtype)
        forces[:, 0] = self.force_x
        return energy, forces


def setup_single_atom_system():
    """Create a minimal single-atom CHARMM system."""
    omm.clear()
    if psf.get_natom() > 0:
        lingo.charmm_script("delete atom sele all end")

    lingo.charmm_script("""
    read rtf card
* Single atom topology file
*
   20    1
MASS     -1 X     10.0

RESI TEST       0.0
GROUP
ATOM A    X     0.0
PATC  FIRS NONE LAST NONE
END
    """)
    lingo.charmm_script("""
    read param card
* dummy parameters for testing
*
NONBONDED   ATOM CDIEL SWITCH VATOM VDISTANCE VSWITCH -
     CUTNB 8.0  CTOFNB 7.5  CTONNB 6.5  EPS 1.0  E14FAC 1.0  WMIN 1.5
X        0.0440    1.0       0.8000

END
    """)
    read.sequence_string("TEST")
    generate.new_segment(seg_name="MOL")
    pos = coor.get_positions()
    pos.iloc[0] = [0.0, 0.0, 0.0]
    coor.set_positions(pos)


@pytest.fixture(scope="module", autouse=True)
def single_atom_system():
    """Ensure pytest runs see the same initialized system as main()."""
    setup_single_atom_system()


@pytest.mark.skipif(not HAS_OMMTORCH, reason="CHARMM built without --with-ommtorch")
def test_torch_constant_force():
    """Test that a TorchScript constant force produces displacement.

    Runs on whichever OpenMM platform the build defaults to (typically
    CUDA when present).  ``ConstantForce.forward`` returns tensors on
    ``positions.device`` so it does not trip the openmm-torch 1.5.1
    CUDA device-mismatch bug documented at the top of this file.
    """
    print("Test: Torch constant force displacement...", flush=True)

    # Save initial position
    pos_before = coor.get_positions().to_numpy().copy()
    x_before = pos_before[0, 0]

    # Create and save TorchScript module
    force_x = 5.0  # kJ/mol/nm in +x
    module = pt.jit.script(ConstantForce(force_x=force_x))
    model_path = "test_constant_force.pt"
    module.save(model_path)

    try:
        # Add torch force
        force_idx = omm.torch_add_force(model_path)
        omm.torch_outputs_forces(force_idx)

        # Run dynamics
        pycharmm.DynamicsScript(
            start=True,
            lang=False,
            nstep=25,
            timestep=0.001,
            iasors=1,
            iasvel=1,
            nprint=25,
            echeck=1000,
            omm=True,
        ).run()

        # Check displacement
        pos_after = coor.get_positions().to_numpy()
        x_after = pos_after[0, 0]
        dx = x_after - x_before

        print(f"  x_before={x_before:.6f}, x_after={x_after:.6f}, dx={dx:.6f}", flush=True)

        assert dx > 0, f"Expected positive x-displacement, got {dx}"
        print("  PASSED", flush=True)
    finally:
        if os.path.exists(model_path):
            os.remove(model_path)


@pytest.mark.skipif(not HAS_OMMTORCH, reason="CHARMM built without --with-ommtorch")
def test_torch_with_custom_force():
    """Test that torch forces and custom forces can coexist."""
    print("Test: Torch force + CustomExternalForce coexistence...", flush=True)

    # Create a custom external force (just to verify no conflicts)
    ext_f = omm.CustomExternalForce("0*x")
    ext_f.add_particle(0)

    # Create a torch force
    module = pt.jit.script(ConstantForce(force_x=1.0))
    model_path = "test_coexist.pt"
    module.save(model_path)

    try:
        force_idx = omm.torch_add_force(model_path)
        omm.torch_outputs_forces(force_idx)

        # Both forces should be created without error
        assert ext_f.index >= 0
        assert force_idx >= 0
        print("  PASSED", flush=True)
    finally:
        if os.path.exists(model_path):
            os.remove(model_path)
