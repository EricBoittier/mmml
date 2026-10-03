"""Tests for the class-based Custom*Force API in pycharmm.omm: collective-variable forces.

Covers compound-bond, centroid-bond, CV, volume, RMSD, and Rg
forces -- the OpenMM custom forces that act on collective
variables or groups of atoms rather than individual particles
or pairs. Includes their getter/introspection coverage.

This file was split out of the original 1247-line test_custom_forces.py.
The shared `single_atom_system` setup lives in
`tests/_custom_forces_helpers.py`.

Run with:
    cd tool/pycharmm
    pytest tests/test_custom_forces_collective.py -v
"""

import numpy as np
import pytest
from _custom_forces_helpers import setup_single_atom_system

import pycharmm.omm as omm
import pycharmm.psf as psf
from pycharmm import coor, energy, lingo, read
from pycharmm import generate as gen


@pytest.fixture(scope="module", autouse=True)
def single_atom_system():
    """Build the minimal one-atom system; see _custom_forces_helpers."""
    setup_single_atom_system()


# ============================================================
# Test classes
# ============================================================


class TestCustomCompoundBondForce:
    def test_create(self):
        f = omm.CustomCompoundBondForce(2, "k*distance(p1,p2)^2")
        assert f.index >= 0

    def test_add_per_bond_param(self):
        f = omm.CustomCompoundBondForce(2, "k*distance(p1,p2)^2")
        assert f.add_per_bond_parameter("k") == 0

    def test_add_bond(self):
        f = omm.CustomCompoundBondForce(2, "k*distance(p1,p2)^2")
        f.add_per_bond_parameter("k")
        idx = f.add_bond([0, 0], [100.0])
        assert idx == 0

    def test_get_num_bonds(self):
        f = omm.CustomCompoundBondForce(2, "k*distance(p1,p2)^2")
        f.add_per_bond_parameter("k")
        assert f.get_num_bonds() == 0
        f.add_bond([0, 0], [100.0])
        assert f.get_num_bonds() == 1

    def test_set_bond_parameters(self):
        f = omm.CustomCompoundBondForce(2, "k*distance(p1,p2)^2")
        f.add_per_bond_parameter("k")
        f.add_bond([0, 0], [100.0])
        f.set_bond_parameters(0, [0, 0], [200.0])

    def test_get_bond_parameters(self):
        f = omm.CustomCompoundBondForce(2, "k*distance(p1,p2)^2")
        f.add_per_bond_parameter("k")
        f.add_bond([0, 0], [100.0])
        particles, params = f.get_bond_parameters(0, 2, 1)
        assert particles == [0, 0]
        assert abs(params[0] - 100.0) < 1e-10


class TestCustomCentroidBondForce:
    def test_create(self):
        f = omm.CustomCentroidBondForce(2, "k*distance(g1,g2)^2")
        assert f.index >= 0

    def test_add_per_bond_param(self):
        f = omm.CustomCentroidBondForce(2, "k*distance(g1,g2)^2")
        assert f.add_per_bond_parameter("k") == 0

    def test_add_group(self):
        f = omm.CustomCentroidBondForce(2, "k*distance(g1,g2)^2")
        g0 = f.add_group([0])
        assert g0 == 0

    def test_add_group_with_weights(self):
        f = omm.CustomCentroidBondForce(2, "k*distance(g1,g2)^2")
        g0 = f.add_group([0], [1.0])
        assert g0 == 0

    def test_add_bond(self):
        f = omm.CustomCentroidBondForce(2, "k*distance(g1,g2)^2")
        f.add_per_bond_parameter("k")
        g0 = f.add_group([0])
        g1 = f.add_group([0])
        bond_idx = f.add_bond([g0, g1], [100.0])
        assert bond_idx == 0

    def test_get_num_groups(self):
        f = omm.CustomCentroidBondForce(2, "k*distance(g1,g2)^2")
        assert f.get_num_groups() == 0
        f.add_group([0])
        assert f.get_num_groups() == 1

    def test_get_num_bonds(self):
        f = omm.CustomCentroidBondForce(2, "k*distance(g1,g2)^2")
        f.add_per_bond_parameter("k")
        g0 = f.add_group([0])
        g1 = f.add_group([0])
        assert f.get_num_bonds() == 0
        f.add_bond([g0, g1], [100.0])
        assert f.get_num_bonds() == 1


class TestCustomCentroidBondGetters:
    def test_group_roundtrip(self):
        f = omm.CustomCentroidBondForce(2, "distance(g1,g2)")
        f.add_group([0], [1.0])
        particles, weights = f.get_group_parameters(0)
        assert particles == [0]
        assert abs(weights[0] - 1.0) < 1e-10

    def test_set_group_parameters(self):
        f = omm.CustomCentroidBondForce(2, "distance(g1,g2)")
        f.add_group([0], [1.0])
        f.set_group_parameters(0, [0], [2.5])
        particles, weights = f.get_group_parameters(0)
        assert particles == [0]
        assert abs(weights[0] - 2.5) < 1e-10

    def test_bond_roundtrip(self):
        f = omm.CustomCentroidBondForce(2, "k*distance(g1,g2)")
        f.add_per_bond_parameter("k")
        f.add_group([0])
        f.add_group([0])
        f.add_bond([0, 1], [4.0])
        groups, params = f.get_bond_parameters(0)
        assert groups == [0, 1]
        assert abs(params[0] - 4.0) < 1e-10

    def test_set_bond_parameters(self):
        f = omm.CustomCentroidBondForce(2, "k*distance(g1,g2)")
        f.add_per_bond_parameter("k")
        f.add_group([0])
        f.add_group([0])
        f.add_bond([0, 1], [1.0])
        f.set_bond_parameters(0, [0, 1], [8.8])
        groups, params = f.get_bond_parameters(0)
        assert groups == [0, 1]
        assert abs(params[0] - 8.8) < 1e-10

    def test_per_bond_param_introspection(self):
        f = omm.CustomCentroidBondForce(2, "k*distance(g1,g2)")
        f.add_per_bond_parameter("k")
        assert f.get_num_per_bond_parameters() == 1
        assert f.get_per_bond_parameter_name(0) == "k"


# ============================================================
# Item 4: Force groups
# ============================================================


class TestCustomCVForce:
    def test_create(self):
        f = omm.CustomCVForce("cv1^2")
        assert f.index >= 0

    def test_add_collective_variable_from_object(self):
        """Add a CV using another CustomForce object."""
        bond = omm.CustomBondForce("r")
        cv = omm.CustomCVForce("cv1^2")
        idx = cv.add_collective_variable("cv1", bond)
        assert idx == 0

    def test_add_collective_variable_from_index(self):
        """Add a CV using a raw store index."""
        bond = omm.CustomBondForce("r")
        cv = omm.CustomCVForce("cv1^2")
        idx = cv.add_collective_variable("cv1", bond.index)
        assert idx == 0

    def test_global_parameter(self):
        f = omm.CustomCVForce("lam*cv1^2")
        idx = f.add_global_parameter("lam", 1.0)
        assert idx == 0


@pytest.mark.skipif(omm.omm_version() < 84, reason="CustomVolumeForce requires OpenMM 8.4+")
class TestCustomVolumeForce:
    def test_create(self):
        f = omm.CustomVolumeForce("k*V")
        assert f.index >= 0

    def test_global_parameter(self):
        f = omm.CustomVolumeForce("k*V")
        idx = f.add_global_parameter("k", 1.0)
        assert idx == 0
        assert f.get_num_global_parameters() == 1
        f.set_global_parameter_default_value(0, 2.0)


# ============================================================
# Test tabulated functions
# ============================================================


class TestRMSDForce:
    def test_create(self):
        ref = np.array([[0.0, 0.0, 0.0]])  # 1 atom, nanometers
        f = omm.RMSDForce(ref)
        assert f.index >= 0

    def test_create_with_particles(self):
        ref = np.array([[0.0, 0.0, 0.0]])
        f = omm.RMSDForce(ref, particles=[0])
        assert f.index >= 0

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError):
            omm.RMSDForce(np.array([1.0, 2.0, 3.0]))  # 1D, not (N,3)

    def test_set_reference_positions(self):
        ref = np.array([[0.0, 0.0, 0.0]])
        f = omm.RMSDForce(ref)
        new_ref = np.array([[0.1, 0.0, 0.0]])
        f.set_reference_positions(new_ref)  # should not raise

    def test_set_particles(self):
        ref = np.array([[0.0, 0.0, 0.0]])
        f = omm.RMSDForce(ref, particles=[0])
        f.set_particles([0])  # should not raise

    def test_as_cv(self):
        """RMSDForce as a collective variable in CustomCVForce."""
        ref = np.array([[0.0, 0.0, 0.0]])
        rmsd = omm.RMSDForce(ref, particles=[0])
        cv = omm.CustomCVForce("k * rmsd^2")
        cv.add_global_parameter("k", 100.0)
        idx = cv.add_collective_variable("rmsd", rmsd)
        assert idx == 0


@pytest.mark.skipif(omm.omm_version() < 84, reason="RGForce requires OpenMM 8.4+")
class TestRGForce:
    def test_create(self):
        f = omm.RGForce()
        assert f.index >= 0

    def test_create_with_particles(self):
        f = omm.RGForce(particles=[0])
        assert f.index >= 0

    def test_as_cv(self):
        """RGForce as a collective variable in CustomCVForce."""
        rg = omm.RGForce(particles=[0])
        cv = omm.CustomCVForce("k * (rg - rg0)^2")
        cv.add_global_parameter("k", 100.0)
        cv.add_global_parameter("rg0", 1.0)
        idx = cv.add_collective_variable("rg", rg)
        assert idx == 0


# ============================================================
# Test CVForce repeated energy (regression for segfault bug)
# ============================================================


class TestCVForceRepeatedEnergy:
    """Regression test: calling energy.from_omm() multiple times with a
    CustomCVForce used to segfault because the ForcesStore shallow-copied
    the CV force pointers, and the NBONDS command unconditionally tore
    down the OpenMM context on every energy call."""

    @pytest.fixture(autouse=True)
    def two_atom_argon(self):
        """Set up a two-atom Argon system with a CVForce distance restraint."""
        import pandas as pd

        omm.clear()
        if psf.get_natom() > 0:
            lingo.charmm_script("delete atom sele all end")
        lingo.charmm_script("""
        read rtf card
        * Argon rtf
        *
          36 1
        mass -1 ar 40 ar
        resi ar 0
        atom ar ar 0
        end
        read param card
        * parameters for Argon
        *
        nonbonded group cdiel switch vgroup vdistance vswitch -
          cutnb 10 ctofnb 8 ctonnb 8 eps 1 e14fac 1 wmin 1.5
        ar      0    -0     1.908
        end
        """)
        read.sequence_string("AR AR")
        gen.new_segment("ARTEST", setup=True)
        xyz = pd.DataFrame({"x": [4, 0], "y": [0, 0], "z": [0, 0]})
        coor.set_positions(xyz)

        # IC constraint for reference energy
        lingo.charmm_script("ic generate")
        from pycharmm import cons_methods, ic

        ic.edit_dist(1, "AR", 2, "AR", 4.5)
        cons_methods.ic(bond=100)

        # CVForce distance restraint
        distance_bond = omm.CustomBondForce("r")
        distance_bond.add_bond(0, 1)
        cv = omm.CustomCVForce("k * (d - d0)^2")
        cv.add_global_parameter(
            "k", 100 * omm.KJ_PER_KCAL / omm.NM_PER_ANGSTROM / omm.NM_PER_ANGSTROM
        )
        cv.add_global_parameter("d0", 4.5 * omm.NM_PER_ANGSTROM)
        cv.add_collective_variable("d", distance_bond)

        lingo.charmm_script("skipe vdw cic")

    def test_repeated_energy_no_crash(self):
        """Three consecutive energy calls must all succeed."""
        energy.from_omm()
        e1 = lingo.get_energy_value("ENER")
        energy.from_omm()
        e2 = lingo.get_energy_value("ENER")
        energy.from_omm()
        e3 = lingo.get_energy_value("ENER")
        assert abs(e1 - e2) < 1e-4, f"e1={e1} e2={e2}"
        assert abs(e2 - e3) < 1e-4, f"e2={e2} e3={e3}"

    def test_no_spurious_rebuild(self, capsys):
        """The OpenMM context should not be rebuilt between identical
        energy calls (performance regression)."""
        energy.from_omm()  # first call builds context
        energy.from_omm()  # second call should reuse it
        captured = capsys.readouterr()
        # Count how many times the context was initialized
        init_count = captured.out.count("Setup_OpenMM: Initializing OpenMM context")
        assert init_count <= 1, (
            f"OpenMM context was rebuilt {init_count} times (expected at most 1)"
        )


# ============================================================
# Test unit conversion constants and convenience methods
# ============================================================
