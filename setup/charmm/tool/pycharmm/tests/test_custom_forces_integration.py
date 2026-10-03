"""Integration tests for Custom*Force types with real OpenMM simulations.

These tests create minimal CHARMM systems, add custom forces, run dynamics
via OpenMM, and verify physical behavior.

Run with:
    CHARMM_LIB_DIR=.../lib python -m pytest test_custom_forces_integration.py -v
"""

import pytest

import pycharmm
import pycharmm.omm as omm
import pycharmm.psf as psf
from pycharmm import coor, generate, lingo, read


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


def test_external_force_displacement():
    """Test that CustomExternalForce produces displacement under dynamics.

    Apply a constant force fx*x in the +x direction and run 25 steps.
    The atom should move in +x.
    """
    print("Test 1: CustomExternalForce displacement...", flush=True)

    # Save initial position
    pos_before = coor.get_positions().to_numpy().copy()
    x_before = pos_before[0, 0]

    # Create custom external force: constant force in +x
    # fx = 100 kJ/mol/nm, energy = -fx*x so force = +fx in x
    f = omm.CustomExternalForce("-(fx*x)")
    f.add_per_particle_parameter("fx")
    f.add_particle(0, [100.0])  # 100 kJ/mol/nm in +x

    # Run dynamics with OpenMM
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


def test_bond_force_equilibrium():
    """Test that CustomBondForce with harmonic potential works.

    Create a bond with k*(r-r0)^2, add it, and verify the force
    object was created and configured correctly.
    """
    print("Test 2: CustomBondForce creation and configuration...", flush=True)

    f = omm.CustomBondForce("k*(r-r0)^2")
    f.add_per_bond_parameter("k")
    f.add_per_bond_parameter("r0")

    # Add bond between particle 0 and itself (single atom system)
    bond_idx = f.add_bond(0, 0, [100.0, 0.1])  # k=100, r0=0.1 nm

    assert bond_idx == 0
    assert f.get_num_bonds() == 1

    # Verify we can read back parameters
    p1, p2, params = f.get_bond_parameters(0, 2)
    assert p1 == 0
    assert p2 == 0
    assert abs(params[0] - 100.0) < 1e-10
    assert abs(params[1] - 0.1) < 1e-10

    # Update parameters and verify
    f.set_bond_parameters(0, 0, 0, [200.0, 0.2])
    p1, p2, params = f.get_bond_parameters(0, 2)
    assert abs(params[0] - 200.0) < 1e-10
    assert abs(params[1] - 0.2) < 1e-10

    print("  PASSED", flush=True)
