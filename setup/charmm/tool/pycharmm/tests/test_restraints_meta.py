"""Tests for pycharmm restraints module: cross-cutting -- backend compat, state, namespace API.

Covers the cross-cutting machinery -- SCAT wrappers,
backend-compatibility checks, the restraints state machine,
the namespace API, deprecation warnings, get_active_restraints,
and module-level logging.

This file was split out of the original 1325-line test_restraints.py.
The shared `_wipe_psf` helper lives in
`tests/_restraints_helpers.py`.

Run with:
    cd tool/pycharmm
    pytest tests/test_restraints_meta.py -v
"""

from unittest import mock

import pytest
from _restraints_helpers import wipe_psf as _wipe_psf

# ============================================================
# Test classes
# ============================================================


class TestSCATWrappers:
    """Test SCAT wrapper functions that delegate to block module."""

    @pytest.fixture(autouse=True)
    def _setup(self, alanine_dipeptide_with_nbonds):
        import pycharmm.block as block

        block.clear()
        yield
        try:
            block.clear()
        except Exception:
            pass
        _wipe_psf()

    def test_scat_enable_disable(self):
        """Test SCAT enable/disable through restraints module."""
        import pycharmm.block as block
        import pycharmm.restraints as restraints

        block.initialize(2)

        restraints.scat_enable(mode="on")

        state = restraints.scat_get_state()
        assert state["enabled"]

        restraints.scat_disable()

        state = restraints.scat_get_state()
        assert not state["enabled"]

        block.clear()

    def test_scat_with_force_constant(self):
        """Test SCAT with specific force constant."""
        import pycharmm.block as block
        import pycharmm.restraints as restraints

        block.initialize(2)

        restraints.scat_enable(mode="k", k=300.0)

        state = restraints.scat_get_state()
        assert state["enabled"]
        assert state["k"] == 300.0

        block.clear()

    def test_scat_define_atoms(self):
        """Test SCAT atom definition through restraints module."""
        import pycharmm.block as block
        import pycharmm.restraints as restraints

        block.initialize(2)
        restraints.scat_enable(mode="on")

        restraints.scat_define_atoms("segid ADP .and. type CA")

        block.clear()


class TestBackendCompatibility:
    """Test backend compatibility checking."""

    def test_check_noe_standard(self):
        """Test NOE support check for standard backend."""
        import pycharmm.restraints as restraints

        result = restraints.check_backend_support("NOE", "standard")
        assert result["supported"]

    def test_check_noe_openmm(self):
        """Test NOE support check for OpenMM backend."""
        import pycharmm.restraints as restraints

        result = restraints.check_backend_support("NOE", "openmm")
        assert not result["supported"]

    def test_check_noe_blade(self):
        """Test NOE support check for BLaDE backend."""
        import pycharmm.restraints as restraints

        result = restraints.check_backend_support("NOE", "blade")
        assert result["supported"]

    def test_check_scat_all_backends(self):
        """Test SCAT support check for all backends."""
        import pycharmm.restraints as restraints

        for backend in ["standard", "domdec", "blade", "openmm"]:
            result = restraints.check_backend_support("SCAT", backend)
            assert result["supported"], f"SCAT should be supported on {backend}"

    def test_unknown_facility(self):
        """Test unknown facility raises ValueError."""
        import pycharmm.restraints as restraints

        with pytest.raises(ValueError):
            restraints.check_backend_support("UNKNOWN", "standard")

    def test_unknown_backend(self):
        """Test unknown backend raises ValueError."""
        import pycharmm.restraints as restraints

        with pytest.raises(ValueError):
            restraints.check_backend_support("NOE", "unknown_backend")


class TestStateManagement:
    """Test state management without requiring CHARMM."""

    def test_state_reset(self):
        """Test state reset clears all restraint state."""
        import pycharmm.restraints as restraints

        restraints._state.reset()

        assert not restraints._state.noe["active"]
        assert restraints._state.noe["count"] == 0
        assert restraints._state.noe["scale"] == 1.0
        assert restraints._state.noe["restraints"] == []
        assert not restraints._state.noe["charmm_synced"]
        assert restraints._state._noe_command_buffer == []
        assert not restraints._state._in_noe_context

    def test_backend_support_matrix(self):
        """Test backend support matrix is properly defined."""
        import pycharmm.restraints as restraints

        matrix = restraints.BACKEND_SUPPORT

        assert "NOE" in matrix
        assert "SCAT" in matrix

        noe_support = matrix["NOE"]
        assert "standard" in noe_support
        assert "domdec" in noe_support
        assert "blade" in noe_support
        assert "openmm" in noe_support

        scat_support = matrix["SCAT"]
        assert "location" in scat_support
        assert scat_support["location"] == "block.py"

        assert "HARMONIC" in matrix
        harm_support = matrix["HARMONIC"]
        assert "location" in harm_support
        assert harm_support["location"] == "cons_harm.py"


class TestNamespaceAPI:
    """Test the new namespace-based API (restraints.atoms, restraints.distances, etc.)."""

    def test_namespace_objects_exist(self):
        """Test that namespace objects are exported at module level."""
        import pycharmm.restraints as restraints

        assert hasattr(restraints, "atoms")
        assert hasattr(restraints, "distances")
        assert hasattr(restraints, "angles")
        assert hasattr(restraints, "positions")
        assert hasattr(restraints, "internal_coords")

    def test_atoms_namespace_methods(self):
        """Test that atoms namespace has expected methods."""
        import pycharmm.restraints as restraints

        assert hasattr(restraints.atoms, "fix")
        assert hasattr(restraints.atoms, "fix_turn_off")
        assert hasattr(restraints.atoms, "harmonic_absolute")
        assert hasattr(restraints.atoms, "harmonic_best_fit")
        assert hasattr(restraints.atoms, "harmonic_relative")
        assert hasattr(restraints.atoms, "harmonic_turn_off")

    def test_distances_namespace_methods(self):
        """Test that distances namespace has expected methods."""
        import pycharmm.restraints as restraints

        assert hasattr(restraints.distances, "NOE")
        assert hasattr(restraints.distances, "noe_reset")
        assert hasattr(restraints.distances, "noe_scale")
        assert hasattr(restraints.distances, "resd")
        assert hasattr(restraints.distances, "resd_reset")
        assert hasattr(restraints.distances, "resd_scale")

    def test_angles_namespace_methods(self):
        """Test that angles namespace has expected methods."""
        import pycharmm.restraints as restraints

        assert hasattr(restraints.angles, "dihedral")
        assert hasattr(restraints.angles, "dihedral_clear")

    def test_positions_namespace_methods(self):
        """Test that positions namespace has expected methods."""
        import pycharmm.restraints as restraints

        assert hasattr(restraints.positions, "droplet")
        assert hasattr(restraints.positions, "harmonic_pca")

    def test_internal_coords_namespace_methods(self):
        """Test that internal_coords namespace has expected methods."""
        import pycharmm.restraints as restraints

        assert hasattr(restraints.internal_coords, "all")
        assert hasattr(restraints.internal_coords, "bond")
        assert hasattr(restraints.internal_coords, "angle")
        assert hasattr(restraints.internal_coords, "dihedral")
        assert hasattr(restraints.internal_coords, "improper")

    def test_state_management_functions(self):
        """Test module-level state management functions."""
        import pycharmm.restraints as restraints

        assert hasattr(restraints, "get_active_restraints")
        assert hasattr(restraints, "set_backend")
        assert hasattr(restraints, "get_backend")
        assert hasattr(restraints, "reset_state")


class TestNamespaceAPIFunctional:
    """Functional tests for namespace-based API with CHARMM library."""

    @pytest.fixture(autouse=True)
    def _setup(self, alanine_dipeptide_with_nbonds):
        yield
        _wipe_psf()

    def test_atoms_harmonic_absolute_via_namespace(self):
        """Test harmonic_absolute through namespace API."""
        import pycharmm.restraints as restraints

        restraints.atoms.harmonic_absolute(force_const=10.0)
        result = restraints.atoms.harmonic_turn_off()
        assert result

    def test_atoms_fix_via_namespace(self):
        """Test fix through namespace API."""
        import pycharmm
        import pycharmm.restraints as restraints

        ca_sel = pycharmm.SelectAtoms(atom_type="CA")
        restraints.atoms.fix(selection=ca_sel)
        result = restraints.atoms.fix_turn_off()
        assert result

    def test_positions_harmonic_pca_via_namespace(self):
        """Test harmonic_pca through namespace API."""
        import pycharmm.restraints as restraints

        restraints.positions.harmonic_pca(force_const=5.0)
        result = restraints.atoms.harmonic_turn_off()
        assert result

    def test_distances_noe_context_manager_via_namespace(self):
        """Test NOE context manager through namespace API."""
        import pycharmm.restraints as restraints

        restraints.noe_reset()

        with restraints.distances.NOE() as noe:
            noe.assign(
                selection1="segid ADP .and. type CA",
                selection2="segid ADP .and. type C",
                kmax=10.0,
                rmax=3.0,
            )

        assert restraints.noe_get_count() > 0
        restraints.noe_reset()


class TestIncompatibilityErrors:
    """Test incompatibility detection and error handling."""

    def test_exception_classes_exist(self):
        """Test that exception classes are defined."""
        import pycharmm.restraints as restraints

        assert hasattr(restraints, "RestraintError")
        assert hasattr(restraints, "IncompatibleBackendError")
        assert hasattr(restraints, "IncompatibleRestraintError")
        assert hasattr(restraints, "RestraintStateError")

    def test_exception_hierarchy(self):
        """Test exception inheritance hierarchy."""
        import pycharmm.restraints as restraints

        assert issubclass(restraints.IncompatibleBackendError, restraints.RestraintError)
        assert issubclass(restraints.IncompatibleRestraintError, restraints.RestraintError)
        assert issubclass(restraints.RestraintStateError, restraints.RestraintError)

        assert issubclass(restraints.RestraintError, Exception)

    def test_incompatible_backend_error_message(self):
        """Test IncompatibleBackendError creates proper message."""
        import pycharmm.restraints as restraints

        error = restraints.IncompatibleBackendError("NOE", "openmm")
        assert "NOE" in str(error)
        assert "openmm" in str(error)

    def test_incompatible_backend_error_with_reason(self):
        """Test IncompatibleBackendError with custom reason."""
        import pycharmm.restraints as restraints

        error = restraints.IncompatibleBackendError(
            "NOE", "openmm", reason="Not implemented in OpenMM interface"
        )
        assert "Not implemented" in str(error)

    def test_incompatible_restraint_error_message(self):
        """Test IncompatibleRestraintError creates proper message."""
        import pycharmm.restraints as restraints

        error = restraints.IncompatibleRestraintError("FIX", "HARMONIC_ABSOLUTE")
        assert "FIX" in str(error)
        assert "HARMONIC_ABSOLUTE" in str(error)

    def test_backend_incompatibility_matrix_defined(self):
        """Test BACKEND_INCOMPATIBILITY matrix is defined."""
        import pycharmm.restraints as restraints

        assert hasattr(restraints, "BACKEND_INCOMPATIBILITY")

        matrix = restraints.BACKEND_INCOMPATIBILITY
        assert "openmm" in matrix
        assert "blade" in matrix
        assert "domdec" in matrix
        assert "standard" in matrix

    def test_openmm_disallowed_restraints(self):
        """Test OpenMM backend has correct disallowed restraints."""
        import pycharmm.restraints as restraints

        openmm_rules = restraints.BACKEND_INCOMPATIBILITY["openmm"]
        assert "NOE" in openmm_rules["disallowed"]
        assert "PNOE" in openmm_rules["disallowed"]
        assert "RESD" in openmm_rules["disallowed"]

    def test_mutual_exclusion_rules_defined(self):
        """Test MUTUAL_EXCLUSION_RULES is defined."""
        import pycharmm.restraints as restraints

        assert hasattr(restraints, "MUTUAL_EXCLUSION_RULES")

        rules = restraints.MUTUAL_EXCLUSION_RULES
        assert ("FIX", "HARMONIC_ABSOLUTE") in rules
        assert ("FIX", "HARMONIC_BEST_FIT") in rules

    def test_backend_set_get(self):
        """Test backend setting and getting."""
        import pycharmm.restraints as restraints

        restraints.reset_state()
        assert restraints.get_backend() == "standard"

        restraints.set_backend("blade")
        assert restraints.get_backend() == "blade"

        restraints.set_backend("standard")


class TestModuleLevelBackendTracking:
    """Regression tests for module-level wrapper state tracking."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        import pycharmm.restraints as restraints

        restraints.reset_state()
        yield
        restraints.reset_state()

    def test_ic_restraint_updates_state_for_blade_checks(self):
        """Module-level IC wrappers must participate in BLaDE checks."""
        import pycharmm.blade as blade
        import pycharmm.restraints as restraints

        with mock.patch.object(restraints, "_ic_restraint") as ic_impl:
            restraints.ic_restraint(bond=100.0, angle=50.0)

        ic_impl.assert_called_once()
        assert "IC" in blade.check_restraints()
        active = restraints.get_active_restraints()
        assert active["ic_bond"]["enabled"]
        assert active["ic_angle"]["enabled"]

    def test_droplet_restraint_updates_state_for_blade_checks(self):
        """Module-level DROPLET wrappers must participate in BLaDE checks."""
        import pycharmm.blade as blade
        import pycharmm.restraints as restraints

        with mock.patch.object(restraints, "_droplet_restraint") as droplet_impl:
            restraints.droplet_restraint(force=1.0, exponent=4)

        droplet_impl.assert_called_once()
        assert "DROPLET" in blade.check_restraints()
        active = restraints.get_active_restraints()
        assert active["droplet"]["settings"]["force"] == 1.0

    def test_blade_enable_rejects_module_level_ic_restraint(self):
        """BLaDE enable() must reject module-level IC restraints."""
        import pycharmm.blade as blade
        import pycharmm.restraints as restraints

        with mock.patch.object(restraints, "_ic_restraint"):
            restraints.ic_restraint(bond=100.0)

        with mock.patch.object(blade, "charmm_script") as charmm_script:
            with pytest.raises(blade.BladeEngineError):
                blade.enable(raise_on_incompatible=True)

        charmm_script.assert_not_called()

    def test_blade_enable_rejects_module_level_droplet_restraint(self):
        """BLaDE enable() must reject module-level droplet restraints."""
        import pycharmm.blade as blade
        import pycharmm.restraints as restraints

        with mock.patch.object(restraints, "_droplet_restraint"):
            restraints.droplet_restraint(force=2.0)

        with mock.patch.object(blade, "charmm_script") as charmm_script:
            with pytest.raises(blade.BladeEngineError):
                blade.enable(raise_on_incompatible=True)

        charmm_script.assert_not_called()

    def test_bond_alias_tracks_ic_state(self):
        """Convenience aliases should route through the fixed IC path."""
        import pycharmm.blade as blade
        import pycharmm.restraints as restraints

        with mock.patch.object(restraints, "_ic_restraint") as ic_impl:
            restraints.bond(75.0)

        ic_impl.assert_called_once()
        assert "IC" in blade.check_restraints()
        assert restraints.get_active_restraints()["ic_bond"]["enabled"]

    def test_mmfp_dihedral_openmm_raises_backend_error(self):
        """OpenMM MMFP dihedral rejection should raise the expected exception."""
        import pycharmm.restraints as restraints

        restraints.set_backend("openmm")

        with pytest.raises(restraints.IncompatibleBackendError) as exc_info:
            restraints.mmfp_dihedral(
                "type A", "type B", "type C", "type D", force=100.0, target_angle=-60.0
            )

        assert "Use dihe_restraint()" in str(exc_info.value)


class TestDeprecationWarnings:
    """Test that deprecation warnings are emitted for old API."""

    def test_cons_harm_setup_absolute_warning(self):
        """Test cons_harm.setup_absolute emits deprecation warning."""
        import warnings

        import pycharmm.cons_harm as cons_harm

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                cons_harm.setup_absolute(force_const=10.0)
            except Exception:
                pass

            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) > 0
            assert "deprecated" in str(deprecation_warnings[0].message).lower()

    def test_cons_harm_turn_off_warning(self):
        """Test cons_harm.turn_off emits deprecation warning."""
        import warnings

        import pycharmm.cons_harm as cons_harm

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                cons_harm.turn_off()
            except Exception:
                pass

            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) > 0

    def test_cons_fix_setup_warning(self):
        """Test cons_fix.setup emits deprecation warning."""
        import warnings

        import pycharmm
        import pycharmm.cons_fix as cons_fix

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                sel = pycharmm.SelectAtoms()
                cons_fix.setup(sel)
            except Exception:
                pass

            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) > 0

    def test_cons_methods_dihe_warning(self):
        """Test cons_methods.dihe emits deprecation warning."""
        import warnings

        import pycharmm.cons_methods as cons_methods

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                cons_methods.dihe(selection="bynum 1 2 3 4")
            except Exception:
                pass

            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) > 0

    def test_cons_methods_ic_warning(self):
        """Test cons_methods.ic emits deprecation warning."""
        import warnings

        import pycharmm.cons_methods as cons_methods

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                cons_methods.ic(bond=100.0)
            except Exception:
                pass

            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) > 0

    def test_cons_methods_droplet_warning(self):
        """Test cons_methods.droplet emits deprecation warning."""
        import warnings

        import pycharmm.cons_methods as cons_methods

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                cons_methods.droplet(force=10.0)
            except Exception:
                pass

            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) > 0


class TestGetActiveRestraints:
    """Test get_active_restraints function."""

    def test_get_active_restraints_returns_dict(self):
        """Test get_active_restraints returns a dictionary."""
        import pycharmm.restraints as restraints

        restraints.reset_state()
        result = restraints.get_active_restraints()

        assert isinstance(result, dict)

    def test_get_active_restraints_empty_after_reset(self):
        """Test get_active_restraints returns empty dict after reset."""
        import pycharmm.restraints as restraints

        restraints.reset_state()
        result = restraints.get_active_restraints()

        assert result == {}


class TestModuleLogging:
    """Test that module-level logging is configured."""

    def test_logger_exists(self):
        """Test that module has logger configured."""
        import pycharmm.restraints as restraints

        assert hasattr(restraints, "logger")

    def test_logger_name(self):
        """Test that logger has correct name."""
        import pycharmm.restraints as restraints

        assert restraints.logger.name == "pycharmm.restraints"
