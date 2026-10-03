"""Tests for the class-based Custom*Force API in pycharmm.omm: canonical force types.

Covers the standard OpenMM custom-force types corresponding to
single force terms in a force field: bond, angle, torsion,
external, nonbonded, GB, hbond, and many-particle. Tests construct
instances, set per-particle/global parameters, and verify the
Python -> Fortran -> C++ bridge round-trips correctly.

This file was split out of the original 1247-line test_custom_forces.py.
The shared `single_atom_system` setup lives in
`tests/_custom_forces_helpers.py`.

Run with:
    cd tool/pycharmm
    pytest tests/test_custom_forces_basic.py -v
"""

import pytest
from _custom_forces_helpers import setup_single_atom_system

import pycharmm.omm as omm


@pytest.fixture(scope="module", autouse=True)
def single_atom_system():
    """Build the minimal one-atom system; see _custom_forces_helpers."""
    setup_single_atom_system()


# ============================================================
# Test classes
# ============================================================


class TestCustomBondForce:
    def test_create(self):
        f = omm.CustomBondForce("k*(r-r0)^2")
        assert f.index >= 0

    def test_add_per_bond_param(self):
        f = omm.CustomBondForce("k*(r-r0)^2")
        idx_k = f.add_per_bond_parameter("k")
        idx_r0 = f.add_per_bond_parameter("r0")
        assert idx_k == 0
        assert idx_r0 == 1

    def test_add_bond(self):
        f = omm.CustomBondForce("k*(r-r0)^2")
        f.add_per_bond_parameter("k")
        f.add_per_bond_parameter("r0")
        # particle indices are 0-based OpenMM indices
        bond_idx = f.add_bond(0, 0, [100.0, 0.1])
        assert bond_idx == 0

    def test_add_global_parameter(self):
        f = omm.CustomBondForce("scale*k*(r-r0)^2")
        idx = f.add_global_parameter("scale", 1.0)
        assert idx == 0
        assert f.get_num_global_parameters() == 1

    def test_set_global_parameter_default_value(self):
        f = omm.CustomBondForce("scale*k*(r-r0)^2")
        f.add_global_parameter("scale", 1.0)
        # Should not raise
        f.set_global_parameter_default_value(0, 2.0)

    def test_add_energy_parameter_derivative(self):
        f = omm.CustomBondForce("lam*k*(r-r0)^2")
        f.add_global_parameter("lam", 1.0)
        # Should not raise
        f.add_energy_parameter_derivative("lam")

    def test_set_uses_pbc(self):
        f = omm.CustomBondForce("k*(r-r0)^2")
        f.set_uses_periodic_boundary_conditions(True)
        f.set_uses_periodic_boundary_conditions(False)

    def test_turn_on_off(self):
        f = omm.CustomBondForce("k*(r-r0)^2")
        f.turn_off()
        f.turn_on()

    def test_get_num_bonds(self):
        f = omm.CustomBondForce("k*(r-r0)^2")
        f.add_per_bond_parameter("k")
        f.add_per_bond_parameter("r0")
        assert f.get_num_bonds() == 0
        f.add_bond(0, 0, [100.0, 0.1])
        assert f.get_num_bonds() == 1

    def test_set_bond_parameters(self):
        f = omm.CustomBondForce("k*(r-r0)^2")
        f.add_per_bond_parameter("k")
        f.add_per_bond_parameter("r0")
        f.add_bond(0, 0, [100.0, 0.1])
        f.set_bond_parameters(0, 0, 0, [200.0, 0.2])

    def test_get_bond_parameters(self):
        f = omm.CustomBondForce("k*(r-r0)^2")
        f.add_per_bond_parameter("k")
        f.add_per_bond_parameter("r0")
        f.add_bond(0, 0, [100.0, 0.15])
        p1, p2, params = f.get_bond_parameters(0, 2)
        assert p1 == 0
        assert p2 == 0
        assert abs(params[0] - 100.0) < 1e-10
        assert abs(params[1] - 0.15) < 1e-10

    def test_get_global_param_default_value(self):
        f = omm.CustomBondForce("scale*k*(r-r0)^2")
        f.add_global_parameter("scale", 3.14)
        val = f.get_global_parameter_default_value(0)
        assert abs(val - 3.14) < 1e-10


class TestCustomAngleForce:
    def test_create(self):
        f = omm.CustomAngleForce("k*(theta-theta0)^2")
        assert f.index >= 0

    def test_add_per_angle_param(self):
        f = omm.CustomAngleForce("k*(theta-theta0)^2")
        idx = f.add_per_angle_parameter("k")
        assert idx == 0

    def test_add_angle(self):
        f = omm.CustomAngleForce("k*(theta-theta0)^2")
        f.add_per_angle_parameter("k")
        f.add_per_angle_parameter("theta0")
        idx = f.add_angle(0, 0, 0, [100.0, 1.91])
        assert idx == 0

    def test_get_num_angles(self):
        f = omm.CustomAngleForce("k*(theta-theta0)^2")
        f.add_per_angle_parameter("k")
        assert f.get_num_angles() == 0
        f.add_angle(0, 0, 0, [100.0])
        assert f.get_num_angles() == 1

    def test_set_angle_parameters(self):
        f = omm.CustomAngleForce("k*(theta-theta0)^2")
        f.add_per_angle_parameter("k")
        f.add_angle(0, 0, 0, [100.0])
        f.set_angle_parameters(0, 0, 0, 0, [200.0])

    def test_get_angle_parameters(self):
        f = omm.CustomAngleForce("k*(theta-theta0)^2")
        f.add_per_angle_parameter("k")
        f.add_per_angle_parameter("theta0")
        f.add_angle(0, 0, 0, [100.0, 1.91])
        p1, p2, p3, params = f.get_angle_parameters(0, 2)
        assert p1 == 0 and p2 == 0 and p3 == 0
        assert abs(params[0] - 100.0) < 1e-10
        assert abs(params[1] - 1.91) < 1e-10


class TestCustomTorsionForce:
    def test_create(self):
        f = omm.CustomTorsionForce("k*(1+cos(n*theta-delta))")
        assert f.index >= 0

    def test_add_per_torsion_param(self):
        f = omm.CustomTorsionForce("k*(1+cos(n*theta-delta))")
        assert f.add_per_torsion_parameter("k") == 0
        assert f.add_per_torsion_parameter("n") == 1
        assert f.add_per_torsion_parameter("delta") == 2

    def test_add_torsion(self):
        f = omm.CustomTorsionForce("k*(1+cos(n*theta-delta))")
        f.add_per_torsion_parameter("k")
        f.add_per_torsion_parameter("n")
        f.add_per_torsion_parameter("delta")
        idx = f.add_torsion(0, 0, 0, 0, [10.0, 2.0, 3.14])
        assert idx == 0

    def test_get_num_torsions(self):
        f = omm.CustomTorsionForce("k*(1+cos(n*theta-delta))")
        f.add_per_torsion_parameter("k")
        assert f.get_num_torsions() == 0
        f.add_torsion(0, 0, 0, 0, [10.0])
        assert f.get_num_torsions() == 1

    def test_set_torsion_parameters(self):
        f = omm.CustomTorsionForce("k*(1+cos(n*theta-delta))")
        f.add_per_torsion_parameter("k")
        f.add_torsion(0, 0, 0, 0, [10.0])
        f.set_torsion_parameters(0, 0, 0, 0, 0, [20.0])

    def test_get_torsion_parameters(self):
        f = omm.CustomTorsionForce("k*(1+cos(n*theta-delta))")
        f.add_per_torsion_parameter("k")
        f.add_per_torsion_parameter("n")
        f.add_per_torsion_parameter("delta")
        f.add_torsion(0, 0, 0, 0, [10.0, 2.0, 3.14])
        p1, p2, p3, p4, params = f.get_torsion_parameters(0, 3)
        assert p1 == 0 and p2 == 0 and p3 == 0 and p4 == 0
        assert abs(params[0] - 10.0) < 1e-10
        assert abs(params[1] - 2.0) < 1e-10
        assert abs(params[2] - 3.14) < 1e-10


class TestCustomExternalForce:
    def test_create(self):
        f = omm.CustomExternalForce("k*x")
        assert f.index >= 0

    def test_add_per_particle_param(self):
        f = omm.CustomExternalForce("k*x")
        assert f.add_per_particle_parameter("k") == 0

    def test_add_particle(self):
        f = omm.CustomExternalForce("k*x")
        f.add_per_particle_parameter("k")
        idx = f.add_particle(0, [100.0])
        assert idx == 0

    def test_add_particle_no_params(self):
        f = omm.CustomExternalForce("x^2")
        idx = f.add_particle(0)
        assert idx == 0

    def test_get_num_particles(self):
        f = omm.CustomExternalForce("k*x")
        f.add_per_particle_parameter("k")
        assert f.get_num_particles() == 0
        f.add_particle(0, [100.0])
        assert f.get_num_particles() == 1

    def test_set_particle_parameters(self):
        f = omm.CustomExternalForce("k*x")
        f.add_per_particle_parameter("k")
        f.add_particle(0, [100.0])
        f.set_particle_parameters(0, 0, [200.0])

    def test_get_particle_parameters(self):
        f = omm.CustomExternalForce("k*x")
        f.add_per_particle_parameter("k")
        f.add_particle(0, [100.0])
        particle, params = f.get_particle_parameters(0, 1)
        assert particle == 0
        assert abs(params[0] - 100.0) < 1e-10


class TestCustomNonbondedForce:
    def test_create(self):
        f = omm.CustomNonbondedForce("epsilon*((sigma/r)^12-2*(sigma/r)^6)")
        assert f.index >= 0

    def test_add_per_particle_param(self):
        f = omm.CustomNonbondedForce("epsilon*((sigma/r)^12-2*(sigma/r)^6)")
        assert f.add_per_particle_parameter("epsilon") == 0
        assert f.add_per_particle_parameter("sigma") == 1

    def test_add_particle(self):
        f = omm.CustomNonbondedForce("epsilon*((sigma/r)^12)")
        f.add_per_particle_parameter("epsilon")
        f.add_per_particle_parameter("sigma")
        idx = f.add_particle([1.0, 0.3])
        assert idx == 0

    def test_add_exclusion(self):
        f = omm.CustomNonbondedForce("1/r")
        p0 = f.add_particle()
        p1 = f.add_particle()
        exc_idx = f.add_exclusion(p0, p1)
        assert exc_idx == 0

    def test_set_nonbonded_method(self):
        f = omm.CustomNonbondedForce("1/r")
        f.set_nonbonded_method(omm.CustomNonbondedForce.NoCutoff)
        f.set_nonbonded_method(omm.CustomNonbondedForce.CutoffNonPeriodic)

    def test_set_cutoff(self):
        f = omm.CustomNonbondedForce("1/r")
        f.set_cutoff_distance(1.0)  # 1 nm

    def test_switching_function(self):
        f = omm.CustomNonbondedForce("1/r")
        f.set_nonbonded_method(omm.CustomNonbondedForce.CutoffNonPeriodic)
        f.set_cutoff_distance(1.2)
        f.set_use_switching_function(True)
        f.set_switching_distance(1.0)

    def test_add_interaction_group(self):
        f = omm.CustomNonbondedForce("1/r")
        f.add_particle()
        f.add_particle()
        idx = f.add_interaction_group([0], [1])
        assert idx == 0

    def test_global_parameter(self):
        f = omm.CustomNonbondedForce("lam/r")
        idx = f.add_global_parameter("lam", 1.0)
        assert idx == 0
        f.set_global_parameter_default_value(0, 0.5)
        assert f.get_num_global_parameters() == 1

    def test_get_num_particles(self):
        f = omm.CustomNonbondedForce("1/r")
        assert f.get_num_particles() == 0
        f.add_particle()
        assert f.get_num_particles() == 1

    def test_set_particle_parameters(self):
        f = omm.CustomNonbondedForce("epsilon/r")
        f.add_per_particle_parameter("epsilon")
        f.add_particle([1.0])
        f.set_particle_parameters(0, [2.0])

    def test_get_nonbonded_method(self):
        f = omm.CustomNonbondedForce("1/r")
        f.set_nonbonded_method(omm.CustomNonbondedForce.CutoffNonPeriodic)
        assert f.get_nonbonded_method() == omm.CustomNonbondedForce.CutoffNonPeriodic

    def test_get_cutoff_distance(self):
        f = omm.CustomNonbondedForce("1/r")
        f.set_cutoff_distance(1.5)
        assert abs(f.get_cutoff_distance() - 1.5) < 1e-10

    def test_get_particle_parameters(self):
        f = omm.CustomNonbondedForce("epsilon*((sigma/r)^12)")
        f.add_per_particle_parameter("epsilon")
        f.add_per_particle_parameter("sigma")
        f.add_particle([1.0, 0.3])
        params = f.get_particle_parameters(0, 2)
        assert abs(params[0] - 1.0) < 1e-10
        assert abs(params[1] - 0.3) < 1e-10


class TestCustomGBForce:
    def test_create(self):
        f = omm.CustomGBForce()
        assert f.index >= 0

    def test_add_per_particle_param(self):
        f = omm.CustomGBForce()
        assert f.add_per_particle_parameter("charge") == 0
        assert f.add_per_particle_parameter("radius") == 1

    def test_add_particle(self):
        f = omm.CustomGBForce()
        f.add_per_particle_parameter("charge")
        idx = f.add_particle([0.5])
        assert idx == 0

    def test_add_computed_value(self):
        f = omm.CustomGBForce()
        f.add_per_particle_parameter("radius")
        idx = f.add_computed_value(
            "I",
            "step(r+sr2-or1)*0.5*(1/L-1/U+0.25*(r-sr2^2/r)*(1/(U^2)-1/(L^2))+0.5*log(L/U)/r)",
            omm.CustomGBForce.ParticlePairNoExclusions,
        )
        assert idx == 0

    def test_add_energy_term(self):
        f = omm.CustomGBForce()
        f.add_per_particle_parameter("charge")
        idx = f.add_energy_term("charge1*charge2/r", omm.CustomGBForce.ParticlePair)
        assert idx == 0

    def test_set_nonbonded_method(self):
        f = omm.CustomGBForce()
        f.set_nonbonded_method(omm.CustomGBForce.NoCutoff)

    def test_set_cutoff(self):
        f = omm.CustomGBForce()
        f.set_nonbonded_method(omm.CustomGBForce.CutoffNonPeriodic)
        f.set_cutoff_distance(1.2)

    def test_get_num_particles(self):
        f = omm.CustomGBForce()
        f.add_per_particle_parameter("charge")
        assert f.get_num_particles() == 0
        f.add_particle([0.5])
        assert f.get_num_particles() == 1

    def test_set_particle_parameters(self):
        f = omm.CustomGBForce()
        f.add_per_particle_parameter("charge")
        f.add_particle([0.5])
        f.set_particle_parameters(0, [1.0])

    def test_get_particle_parameters(self):
        f = omm.CustomGBForce()
        f.add_per_particle_parameter("charge")
        f.add_per_particle_parameter("radius")
        f.add_particle([0.5, 0.15])
        params = f.get_particle_parameters(0, 2)
        assert abs(params[0] - 0.5) < 1e-10
        assert abs(params[1] - 0.15) < 1e-10


class TestCustomHbondForce:
    def test_create(self):
        f = omm.CustomHbondForce("k*distance(d1,a1)^2")
        assert f.index >= 0

    def test_add_per_donor_param(self):
        f = omm.CustomHbondForce("k*distance(d1,a1)^2")
        assert f.add_per_donor_parameter("k") == 0

    def test_add_per_acceptor_param(self):
        f = omm.CustomHbondForce("k*distance(d1,a1)^2")
        assert f.add_per_acceptor_parameter("scale") == 0

    def test_add_donor(self):
        f = omm.CustomHbondForce("k*distance(d1,a1)^2")
        f.add_per_donor_parameter("k")
        idx = f.add_donor(0, parameters=[100.0])
        assert idx == 0

    def test_add_acceptor(self):
        f = omm.CustomHbondForce("k*distance(d1,a1)^2")
        f.add_per_acceptor_parameter("k")
        idx = f.add_acceptor(0, parameters=[100.0])
        assert idx == 0

    def test_add_exclusion(self):
        f = omm.CustomHbondForce("distance(d1,a1)^2")
        d = f.add_donor(0)
        a = f.add_acceptor(0)
        exc = f.add_exclusion(d, a)
        assert exc == 0

    def test_set_nonbonded_method(self):
        f = omm.CustomHbondForce("distance(d1,a1)^2")
        f.set_nonbonded_method(omm.CustomHbondForce.NoCutoff)

    def test_set_cutoff(self):
        f = omm.CustomHbondForce("distance(d1,a1)^2")
        f.set_cutoff_distance(0.5)


class TestCustomManyParticleForce:
    def test_create(self):
        f = omm.CustomManyParticleForce(3, "k*(r12+r13+r23)")
        assert f.index >= 0

    def test_add_per_particle_param(self):
        f = omm.CustomManyParticleForce(3, "k*(r12+r13+r23)")
        assert f.add_per_particle_parameter("k") == 0

    def test_add_particle(self):
        f = omm.CustomManyParticleForce(3, "k*(r12+r13+r23)")
        f.add_per_particle_parameter("k")
        idx = f.add_particle([1.0], particle_type=0)
        assert idx == 0

    def test_add_exclusion(self):
        f = omm.CustomManyParticleForce(3, "r12+r13+r23")
        f.add_particle()
        f.add_particle()
        exc = f.add_exclusion(0, 1)
        assert exc == 0

    def test_set_nonbonded_method(self):
        f = omm.CustomManyParticleForce(3, "r12+r13+r23")
        f.set_nonbonded_method(omm.CustomManyParticleForce.NoCutoff)

    def test_set_cutoff(self):
        f = omm.CustomManyParticleForce(3, "r12+r13+r23")
        f.set_cutoff_distance(1.0)


class TestCustomHbondForceGetters:
    def test_get_num_donors(self):
        f = omm.CustomHbondForce("1/r")
        f.add_donor(0)
        assert f.get_num_donors() == 1

    def test_get_num_acceptors(self):
        f = omm.CustomHbondForce("1/r")
        f.add_acceptor(0)
        assert f.get_num_acceptors() == 1

    def test_donor_roundtrip(self):
        f = omm.CustomHbondForce("k*r")
        f.add_per_donor_parameter("k")
        f.add_donor(0, 0, -1, [3.5])
        d1, d2, d3, params = f.get_donor_parameters(0, 1)
        assert d1 == 0
        assert d2 == 0
        assert d3 == -1
        assert abs(params[0] - 3.5) < 1e-10

    def test_set_donor_parameters(self):
        f = omm.CustomHbondForce("k*r")
        f.add_per_donor_parameter("k")
        f.add_donor(0, -1, -1, [1.0])
        f.set_donor_parameters(0, 0, 0, -1, [9.9])
        d1, d2, d3, params = f.get_donor_parameters(0, 1)
        assert d1 == 0
        assert d2 == 0
        assert abs(params[0] - 9.9) < 1e-10

    def test_acceptor_roundtrip(self):
        f = omm.CustomHbondForce("s*r")
        f.add_per_acceptor_parameter("s")
        f.add_acceptor(0, 0, -1, [2.5])
        a1, a2, a3, params = f.get_acceptor_parameters(0, 1)
        assert a1 == 0
        assert a2 == 0
        assert a3 == -1
        assert abs(params[0] - 2.5) < 1e-10

    def test_set_acceptor_parameters(self):
        f = omm.CustomHbondForce("s*r")
        f.add_per_acceptor_parameter("s")
        f.add_acceptor(0, -1, -1, [1.0])
        f.set_acceptor_parameters(0, 0, 0, -1, [7.7])
        a1, a2, a3, params = f.get_acceptor_parameters(0, 1)
        assert a1 == 0
        assert a2 == 0
        assert abs(params[0] - 7.7) < 1e-10

    def test_per_donor_param_introspection(self):
        f = omm.CustomHbondForce("k*r")
        f.add_per_donor_parameter("k")
        f.add_per_donor_parameter("sigma")
        assert f.get_num_per_donor_parameters() == 2
        assert f.get_per_donor_parameter_name(0) == "k"
        assert f.get_per_donor_parameter_name(1) == "sigma"

    def test_per_acceptor_param_introspection(self):
        f = omm.CustomHbondForce("q*r")
        f.add_per_acceptor_parameter("q")
        assert f.get_num_per_acceptor_parameters() == 1
        assert f.get_per_acceptor_parameter_name(0) == "q"


# ============================================================
# Item 2: CustomManyParticleForce getters/setters
# ============================================================


class TestCustomManyParticleGetters:
    def test_get_num_particles(self):
        f = omm.CustomManyParticleForce(2, "r^2")
        f.add_particle()
        f.add_particle()
        assert f.get_num_particles() == 2

    def test_particle_roundtrip(self):
        f = omm.CustomManyParticleForce(2, "k*r")
        f.add_per_particle_parameter("k")
        f.add_particle([3.0], particle_type=1)
        params, ptype = f.get_particle_parameters(0, 1)
        assert abs(params[0] - 3.0) < 1e-10
        assert ptype == 1

    def test_set_particle_parameters(self):
        f = omm.CustomManyParticleForce(2, "k*r")
        f.add_per_particle_parameter("k")
        f.add_particle([1.0], particle_type=0)
        f.set_particle_parameters(0, [5.5], particle_type=2)
        params, ptype = f.get_particle_parameters(0, 1)
        assert abs(params[0] - 5.5) < 1e-10
        assert ptype == 2

    def test_per_particle_param_introspection(self):
        f = omm.CustomManyParticleForce(2, "k*r")
        f.add_per_particle_parameter("k")
        f.add_per_particle_parameter("sigma")
        assert f.get_num_per_particle_parameters() == 2
        assert f.get_per_particle_parameter_name(0) == "k"
        assert f.get_per_particle_parameter_name(1) == "sigma"


# ============================================================
# Item 3: CustomCentroidBondForce setters/getters
# ============================================================


class TestManyParticleExtras:
    def test_type_filter_roundtrip(self):
        f = omm.CustomManyParticleForce(2, "r^2")
        f.add_particle(particle_type=0)
        f.add_particle(particle_type=1)
        f.set_type_filter(0, [0, 1])
        types = f.get_type_filter(0)
        assert sorted(types) == [0, 1]

    def test_type_filter_empty(self):
        f = omm.CustomManyParticleForce(2, "r^2")
        f.add_particle()
        types = f.get_type_filter(0)
        assert types == []

    def test_set_type_filter_replaces(self):
        f = omm.CustomManyParticleForce(2, "r^2")
        f.add_particle()
        f.set_type_filter(0, [0, 1, 2])
        f.set_type_filter(0, [3])
        types = f.get_type_filter(0)
        assert types == [3]

    def test_permutation_mode_default(self):
        f = omm.CustomManyParticleForce(2, "r^2")
        assert f.get_permutation_mode() == omm.CustomManyParticleForce.SinglePermutation

    def test_set_permutation_mode(self):
        f = omm.CustomManyParticleForce(3, "r^2")
        f.set_permutation_mode(omm.CustomManyParticleForce.UniqueCentralParticle)
        assert f.get_permutation_mode() == omm.CustomManyParticleForce.UniqueCentralParticle
