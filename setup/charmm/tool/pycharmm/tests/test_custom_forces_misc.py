"""Tests for the class-based Custom*Force API in pycharmm.omm: tabulated, introspection, and helpers.

Covers the cross-cutting machinery: tabulated functions (1D,
2D, 3D), `updateParametersInContext`, unit conversions,
standalone helper functions, force groups, and introspection.

This file was split out of the original 1247-line test_custom_forces.py.
The shared `single_atom_system` setup lives in
`tests/_custom_forces_helpers.py`.

Run with:
    cd tool/pycharmm
    pytest tests/test_custom_forces_misc.py -v
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


class TestTabulatedFunctions:
    def test_continuous1d_nonbonded(self):
        f = omm.CustomNonbondedForce("tabfunc(r)")
        idx = f.add_tabulated_function_continuous1d(
            "tabfunc", [0.0, 1.0, 0.0], 0.0, 2.0, periodic=False
        )
        assert idx == 0

    def test_discrete1d_nonbonded(self):
        f = omm.CustomNonbondedForce("tabfunc(r)")
        idx = f.add_tabulated_function_discrete1d("tabfunc", [0.0, 1.0, 2.0, 3.0])
        assert idx == 0

    def test_continuous1d_compound(self):
        f = omm.CustomCompoundBondForce(2, "tabfunc(distance(p1,p2))")
        idx = f.add_tabulated_function_continuous1d("tabfunc", [0.0, 1.0, 0.0], 0.0, 2.0)
        assert idx == 0

    def test_continuous1d_gb(self):
        f = omm.CustomGBForce()
        idx = f.add_tabulated_function_continuous1d("tabfunc", [0.0, 1.0, 0.0], 0.0, 2.0)
        assert idx == 0

    def test_continuous1d_hbond(self):
        f = omm.CustomHbondForce("tabfunc(distance(d1,a1))")
        idx = f.add_tabulated_function_continuous1d("tabfunc", [0.0, 1.0, 0.0], 0.0, 2.0)
        assert idx == 0

    def test_continuous1d_many_particle(self):
        f = omm.CustomManyParticleForce(3, "tabfunc(r12)")
        idx = f.add_tabulated_function_continuous1d("tabfunc", [0.0, 1.0, 0.0], 0.0, 2.0)
        assert idx == 0

    def test_continuous1d_cv(self):
        bond = omm.CustomBondForce("r")
        f = omm.CustomCVForce("tabfunc(cv1)")
        f.add_collective_variable("cv1", bond)
        idx = f.add_tabulated_function_continuous1d("tabfunc", [0.0, 1.0, 0.0], 0.0, 2.0)
        assert idx == 0

    def test_continuous1d_centroid(self):
        f = omm.CustomCentroidBondForce(2, "tabfunc(distance(g1,g2))")
        idx = f.add_tabulated_function_continuous1d("tabfunc", [0.0, 1.0, 0.0], 0.0, 2.0)
        assert idx == 0

    def test_unsupported_bond(self):
        f = omm.CustomBondForce("r^2")
        with pytest.raises(TypeError):
            f.add_tabulated_function_continuous1d("tabfunc", [0.0, 1.0], 0.0, 1.0)

    def test_unsupported_angle(self):
        f = omm.CustomAngleForce("theta^2")
        with pytest.raises(TypeError):
            f.add_tabulated_function_discrete1d("tabfunc", [0.0, 1.0])

    def test_unsupported_external(self):
        f = omm.CustomExternalForce("x^2")
        with pytest.raises(TypeError):
            f.add_tabulated_function_continuous1d("tabfunc", [0.0, 1.0], 0.0, 1.0)

    @pytest.mark.skipif(omm.omm_version() < 84, reason="CustomVolumeForce requires OpenMM 8.4+")
    def test_unsupported_volume(self):
        f = omm.CustomVolumeForce("V")
        with pytest.raises(TypeError):
            f.add_tabulated_function_continuous1d("tabfunc", [0.0, 1.0], 0.0, 1.0)


# ============================================================
# Test updateParametersInContext validation
# ============================================================


class TestTabulatedFunctions2D3D:
    def test_continuous2d_nonbonded(self):
        f = omm.CustomNonbondedForce("tab2d(r,r)")
        # 3x3 grid
        values = [float(i) for i in range(9)]
        idx = f.add_tabulated_function_continuous2d("tab2d", values, 3, 3, 0.0, 1.0, 0.0, 1.0)
        assert idx >= 0

    def test_discrete2d_nonbonded(self):
        f = omm.CustomNonbondedForce("tab2d(r,r)")
        values = [float(i) for i in range(6)]
        idx = f.add_tabulated_function_discrete2d("tab2d", values, 2, 3)
        assert idx >= 0

    def test_continuous3d_nonbonded(self):
        f = omm.CustomNonbondedForce("tab3d(r,r,r)")
        values = [float(i) for i in range(27)]
        idx = f.add_tabulated_function_continuous3d(
            "tab3d", values, 3, 3, 3, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0
        )
        assert idx >= 0

    def test_discrete3d_nonbonded(self):
        f = omm.CustomNonbondedForce("tab3d(r,r,r)")
        values = [float(i) for i in range(8)]
        idx = f.add_tabulated_function_discrete3d("tab3d", values, 2, 2, 2)
        assert idx >= 0

    def test_2d_compound(self):
        f = omm.CustomCompoundBondForce(2, "tab2d(r,r)")
        values = [float(i) for i in range(4)]
        idx = f.add_tabulated_function_continuous2d("tab2d", values, 2, 2, 0.0, 1.0, 0.0, 1.0)
        assert idx >= 0

    def test_2d_unsupported_bond(self):
        f = omm.CustomBondForce("r^2")
        with pytest.raises(TypeError):
            f.add_tabulated_function_continuous2d("tab", [0.0] * 4, 2, 2, 0.0, 1.0, 0.0, 1.0)

    def test_3d_unsupported_external(self):
        f = omm.CustomExternalForce("x^2")
        with pytest.raises(TypeError):
            f.add_tabulated_function_continuous3d(
                "tab", [0.0] * 8, 2, 2, 2, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0
            )


# ============================================================
# Item 6: Introspection
# ============================================================


class TestUpdateParametersInContext:
    def test_unsupported_cv(self):
        f = omm.CustomCVForce("cv1^2")
        with pytest.raises(TypeError):
            f.update_parameters_in_context()

    @pytest.mark.skipif(omm.omm_version() < 84, reason="CustomVolumeForce requires OpenMM 8.4+")
    def test_unsupported_volume(self):
        f = omm.CustomVolumeForce("V")
        with pytest.raises(TypeError):
            f.update_parameters_in_context()


# ============================================================
# Test RMSDForce and RGForce
# ============================================================


class TestUnitConversions:
    def test_constants(self):
        assert omm.NM_PER_ANGSTROM == 0.1
        assert omm.ANGSTROM_PER_NM == 10.0
        assert omm.KJ_PER_KCAL == 4.184
        assert abs(omm.KCAL_PER_KJ - 1.0 / 4.184) < 1e-10

    def test_nb_cutoff_angstrom(self):
        f = omm.CustomNonbondedForce("1/r")
        f.set_cutoff_distance_angstrom(12.0)  # 12 A -> 1.2 nm

    def test_nb_switching_distance_angstrom(self):
        f = omm.CustomNonbondedForce("1/r")
        f.set_nonbonded_method(omm.CustomNonbondedForce.CutoffNonPeriodic)
        f.set_cutoff_distance_angstrom(12.0)
        f.set_use_switching_function(True)
        f.set_switching_distance_angstrom(10.0)

    def test_gb_cutoff_angstrom(self):
        f = omm.CustomGBForce()
        f.set_nonbonded_method(omm.CustomGBForce.CutoffNonPeriodic)
        f.set_cutoff_distance_angstrom(12.0)

    def test_hbond_cutoff_angstrom(self):
        f = omm.CustomHbondForce("distance(d1,a1)^2")
        f.set_cutoff_distance_angstrom(5.0)

    def test_many_cutoff_angstrom(self):
        f = omm.CustomManyParticleForce(3, "r12+r13+r23")
        f.set_cutoff_distance_angstrom(10.0)


# ============================================================
# Test backward-compatible standalone functions
# ============================================================


class TestStandaloneFunctions:
    def test_custom_force_add(self):
        idx = omm.custom_force_add(omm.CustomForceType.BOND, "r^2")
        assert idx >= 0

    def test_custom_force_add_int_kind(self):
        idx = omm.custom_force_add(2, "r^2")  # 2 = BOND
        assert idx >= 0

    def test_customnb_add_force(self):
        idx = omm.customnb_add_force("1/r")
        assert idx >= 0

    def test_force_turn_on_off(self):
        f = omm.CustomBondForce("r^2")
        omm.force_turn_off(f.index)
        omm.force_turn_on(f.index)


# ============================================================
# Item 1: CustomHbondForce getters/setters
# ============================================================


class TestForceGroups:
    """Force groups are stored, but CHARMM reassigns them at system build.

    CHARMM gives each energy term its own OpenMM force group so it can report
    terms separately, overwriting any group set here. These tests therefore
    check the stored value round-trips *and* that setting it warns, since the
    setting cannot reach OpenMM. Use ``set_eterm`` to choose a force's energy
    term.
    """

    def test_set_get_force_group(self):
        f = omm.CustomBondForce("r^2")
        with pytest.warns(DeprecationWarning, match="no lasting effect"):
            f.set_force_group(3)
        assert f.get_force_group() == 3

    def test_default_force_group(self):
        f = omm.CustomAngleForce("theta^2")
        assert f.get_force_group() == 0

    def test_force_group_all_types(self):
        forces = [
            omm.CustomBondForce("r^2"),
            omm.CustomAngleForce("theta^2"),
            omm.CustomTorsionForce("theta^2"),
            omm.CustomExternalForce("x^2"),
            omm.CustomNonbondedForce("1/r"),
            omm.CustomCompoundBondForce(2, "r^2"),
            omm.CustomCentroidBondForce(2, "distance(g1,g2)"),
            omm.CustomGBForce(),
            omm.CustomHbondForce("1/r"),
            omm.CustomManyParticleForce(2, "r^2"),
            omm.CustomCVForce("cv1^2"),
        ]
        for i, f in enumerate(forces):
            group = (i + 1) % 32
            with pytest.warns(DeprecationWarning):
                f.set_force_group(group)
            assert f.get_force_group() == group


# ============================================================
# Item 5: 2D/3D tabulated functions
# ============================================================


class TestIntrospection:
    def test_energy_expression_bond(self):
        f = omm.CustomBondForce("k*(r-r0)^2")
        assert f.get_energy_expression() == "k*(r-r0)^2"

    def test_energy_expression_angle(self):
        f = omm.CustomAngleForce("0.5*k*theta^2")
        assert f.get_energy_expression() == "0.5*k*theta^2"

    def test_energy_expression_nonbonded(self):
        f = omm.CustomNonbondedForce("epsilon*(sigma/r)^12")
        assert f.get_energy_expression() == "epsilon*(sigma/r)^12"

    def test_energy_expression_cv(self):
        f = omm.CustomCVForce("cv1^2+cv2")
        assert f.get_energy_expression() == "cv1^2+cv2"

    def test_global_param_name(self):
        f = omm.CustomBondForce("k*r^2")
        f.add_global_parameter("k", 100.0)
        f.add_global_parameter("temp", 300.0)
        assert f.get_global_parameter_name(0) == "k"
        assert f.get_global_parameter_name(1) == "temp"

    def test_num_per_params_bond(self):
        f = omm.CustomBondForce("k*(r-r0)^2")
        f.add_per_bond_parameter("k")
        f.add_per_bond_parameter("r0")
        assert f.get_num_per_parameters() == 2

    def test_per_param_name_bond(self):
        f = omm.CustomBondForce("k*(r-r0)^2")
        f.add_per_bond_parameter("k")
        f.add_per_bond_parameter("r0")
        assert f.get_per_parameter_name(0) == "k"
        assert f.get_per_parameter_name(1) == "r0"

    def test_num_per_params_external(self):
        f = omm.CustomExternalForce("k*x^2")
        f.add_per_particle_parameter("k")
        assert f.get_num_per_parameters() == 1
        assert f.get_per_parameter_name(0) == "k"

    def test_num_per_params_nonbonded(self):
        f = omm.CustomNonbondedForce("sigma/r")
        f.add_per_particle_parameter("sigma")
        assert f.get_num_per_parameters() == 1
        assert f.get_per_parameter_name(0) == "sigma"

    @pytest.mark.skipif(omm.omm_version() < 84, reason="CustomVolumeForce requires OpenMM 8.4+")
    def test_global_param_name_volume(self):
        f = omm.CustomVolumeForce("k*V")
        f.add_global_parameter("k", 1.0)
        assert f.get_global_parameter_name(0) == "k"


# ============================================================
# Item 7: ManyParticle extras (type filter, permutation mode)
# ============================================================
