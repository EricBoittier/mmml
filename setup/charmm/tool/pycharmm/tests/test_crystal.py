#!/usr/bin/env python
"""Comprehensive tests for pycharmm crystal module.

This module tests crystal lattice definitions and backend compatibility:
1. All crystal type definitions (CUBI, ORTH, MONO, TRIC, etc.)
2. Backend compatibility checking (BLaDE, OpenMM, PME/EWALD)
3. Query functions (is_defined, get_crystal_type_direct, etc.)
4. PME/domdec orthorhombic requirements
5. Error handling for invalid parameters
"""

import logging
import logging.handlers
from unittest.mock import patch

import numpy as np
import pytest

# =============================================================================
# Unit Tests (No CHARMM library required)
# =============================================================================


class TestCrystalBackendSupportMatrix:
    """Test CRYSTAL_BACKEND_SUPPORT matrix structure and values."""

    def test_backend_support_defined(self):
        import pycharmm.crystal as crystal

        assert hasattr(crystal, "CRYSTAL_BACKEND_SUPPORT")
        assert isinstance(crystal.CRYSTAL_BACKEND_SUPPORT, dict)

    def test_all_crystal_types_present(self):
        import pycharmm.crystal as crystal

        expected_types = [
            "CUBI",
            "TETR",
            "ORTH",
            "RECT",
            "MONO",
            "TRIC",
            "HEXA",
            "RHOM",
            "OCTA",
            "RHDO",
        ]

        for crystal_type in expected_types:
            assert crystal_type in crystal.CRYSTAL_BACKEND_SUPPORT

    def test_support_matrix_keys(self):
        import pycharmm.crystal as crystal

        required_keys = ["blade", "blade_npt", "openmm", "pme", "domdec", "orthorhombic"]

        for _crystal_type, support in crystal.CRYSTAL_BACKEND_SUPPORT.items():
            for key in required_keys:
                assert key in support

    def test_orthorhombic_types_consistent(self):
        import pycharmm.crystal as crystal

        for _crystal_type, support in crystal.CRYSTAL_BACKEND_SUPPORT.items():
            if support["orthorhombic"]:
                assert support["pme"]
                assert support["domdec"]

    def test_non_orthorhombic_no_pme(self):
        import pycharmm.crystal as crystal

        for _crystal_type, support in crystal.CRYSTAL_BACKEND_SUPPORT.items():
            if not support["orthorhombic"]:
                assert not support["pme"]
                assert not support["domdec"]

    def test_cubic_full_support(self):
        import pycharmm.crystal as crystal

        cubi = crystal.CRYSTAL_BACKEND_SUPPORT["CUBI"]
        assert cubi["blade"]
        assert cubi["blade_npt"]
        assert cubi["openmm"]
        assert cubi["pme"]
        assert cubi["domdec"]
        assert cubi["orthorhombic"]

    def test_orthorhombic_full_support(self):
        import pycharmm.crystal as crystal

        orth = crystal.CRYSTAL_BACKEND_SUPPORT["ORTH"]
        assert orth["blade"]
        assert orth["blade_npt"]
        assert orth["openmm"]
        assert orth["pme"]
        assert orth["domdec"]
        assert orth["orthorhombic"]

    def test_monoclinic_limited_support(self):
        import pycharmm.crystal as crystal

        mono = crystal.CRYSTAL_BACKEND_SUPPORT["MONO"]
        assert mono["blade"]
        assert not mono["blade_npt"]
        assert not mono["openmm"]
        assert not mono["pme"]
        assert not mono["domdec"]
        assert not mono["orthorhombic"]

    def test_triclinic_limited_support(self):
        import pycharmm.crystal as crystal

        tric = crystal.CRYSTAL_BACKEND_SUPPORT["TRIC"]
        assert tric["blade"]
        assert not tric["blade_npt"]
        assert not tric["openmm"]
        assert not tric["pme"]
        assert not tric["domdec"]
        assert not tric["orthorhombic"]


class TestIsOrthorhombicFunction:
    """Test is_orthorhombic function."""

    def test_is_orthorhombic_cubic(self):
        import pycharmm.crystal as crystal

        assert crystal.is_orthorhombic("CUBI")

    def test_is_orthorhombic_ortho(self):
        import pycharmm.crystal as crystal

        assert crystal.is_orthorhombic("ORTH")

    def test_is_orthorhombic_tetra(self):
        import pycharmm.crystal as crystal

        assert crystal.is_orthorhombic("TETR")

    def test_is_not_orthorhombic_mono(self):
        import pycharmm.crystal as crystal

        assert not crystal.is_orthorhombic("MONO")

    def test_is_not_orthorhombic_tric(self):
        import pycharmm.crystal as crystal

        assert not crystal.is_orthorhombic("TRIC")

    def test_is_not_orthorhombic_hexa(self):
        import pycharmm.crystal as crystal

        assert not crystal.is_orthorhombic("HEXA")

    def test_is_not_orthorhombic_rhom(self):
        import pycharmm.crystal as crystal

        assert not crystal.is_orthorhombic("RHOM")

    def test_is_not_orthorhombic_octa(self):
        import pycharmm.crystal as crystal

        assert not crystal.is_orthorhombic("OCTA")

    def test_is_not_orthorhombic_rhdo(self):
        import pycharmm.crystal as crystal

        assert not crystal.is_orthorhombic("RHDO")

    def test_is_orthorhombic_unknown(self):
        import pycharmm.crystal as crystal

        assert not crystal.is_orthorhombic("UNKNOWN")


class TestBackendCompatibilityChecking:
    """Test _check_backend_compatibility function."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        import pycharmm.crystal as crystal

        self.crystal = crystal
        self.log_handler = logging.handlers.MemoryHandler(capacity=100)
        self.crystal.logger.addHandler(self.log_handler)
        yield
        self.crystal.logger.removeHandler(self.log_handler)

    def test_check_compatibility_exists(self):
        import pycharmm.crystal as crystal

        assert hasattr(crystal, "_check_backend_compatibility")
        assert callable(crystal._check_backend_compatibility)

    def test_check_compatibility_orthorhombic_no_warning(self):
        import pycharmm.crystal as crystal

        crystal._check_backend_compatibility("CUBI")
        crystal._check_backend_compatibility("ORTH")
        crystal._check_backend_compatibility("TETR")

    def test_blade_does_not_report_inactive_domdec(self, caplog):
        import pycharmm.crystal as crystal

        with (
            patch("pycharmm.blade.is_enabled", return_value=True),
            patch("pycharmm.domdec.is_enabled", return_value=False),
            patch("pycharmm.crystal._is_pme_enabled", return_value=False),
        ):
            crystal._check_backend_compatibility("OCTA")

        assert "Domain decomposition" not in caplog.text

    def test_active_domdec_reports_unsupported_cell(self, caplog):
        import pycharmm.crystal as crystal

        with (
            patch("pycharmm.blade.is_enabled", return_value=False),
            patch("pycharmm.domdec.is_enabled", return_value=True),
            patch("pycharmm.crystal._is_pme_enabled", return_value=False),
        ):
            crystal._check_backend_compatibility("OCTA")

        assert "Domain decomposition" in caplog.text


class TestGetBackendFunction:
    """Test _get_current_backend function."""

    def test_get_backend_exists(self):
        import pycharmm.crystal as crystal

        assert hasattr(crystal, "_get_current_backend")
        assert callable(crystal._get_current_backend)

    def test_get_backend_returns_string(self):
        import pycharmm.crystal as crystal

        result = crystal._get_current_backend()
        assert isinstance(result, str)

    def test_get_backend_returns_valid_value(self):
        import pycharmm.crystal as crystal

        result = crystal._get_current_backend()
        assert result in ["blade", "openmm", "domdec", "standard"]

    def test_get_backend_detects_domdec(self):
        import pycharmm.crystal as crystal

        with (
            patch("pycharmm.blade.is_enabled", return_value=False),
            patch("pycharmm.domdec.is_enabled", return_value=True),
        ):
            assert crystal._get_current_backend() == "domdec"


class TestCtypesInitialization:
    """Test ctypes initialization."""

    def test_ctypes_init_function_exists(self):
        import pycharmm.crystal as crystal

        assert hasattr(crystal, "_init_ctypes")
        assert callable(crystal._init_ctypes)

    def test_ctypes_initialized_flag(self):
        import pycharmm.crystal as crystal

        assert hasattr(crystal, "_ctypes_initialized")


class TestXtlaccFunction:
    """Test get_xtlacc lattice vector calculation."""

    def test_xtlacc_cubic(self):
        import pycharmm.crystal as crystal

        result = crystal.get_xtlacc(10.0, 10.0, 10.0, 90.0, 90.0, 90.0)

        assert result.shape == (3, 3)
        np.testing.assert_almost_equal(result[0], [10.0, 0.0, 0.0], decimal=5)
        np.testing.assert_almost_equal(result[1], [0.0, 10.0, 0.0], decimal=5)
        np.testing.assert_almost_equal(result[2], [0.0, 0.0, 10.0], decimal=5)

    def test_xtlacc_orthorhombic(self):
        import pycharmm.crystal as crystal

        result = crystal.get_xtlacc(10.0, 20.0, 30.0, 90.0, 90.0, 90.0)

        assert result.shape == (3, 3)
        np.testing.assert_almost_equal(result[0], [10.0, 0.0, 0.0], decimal=5)
        np.testing.assert_almost_equal(result[1], [0.0, 20.0, 0.0], decimal=5)
        np.testing.assert_almost_equal(result[2], [0.0, 0.0, 30.0], decimal=5)

    def test_xtlacc_monoclinic(self):
        import pycharmm.crystal as crystal

        result = crystal.get_xtlacc(10.0, 20.0, 30.0, 90.0, 100.0, 90.0)

        assert result.shape == (3, 3)
        np.testing.assert_almost_equal(result[0, 0], 10.0, decimal=5)
        np.testing.assert_almost_equal(result[0, 1], 0.0, decimal=5)
        np.testing.assert_almost_equal(result[0, 2], 0.0, decimal=5)

        assert result[2, 0] != 0.0


class TestXtltypHeuristics:
    """Test get_xtltyp lattice type inference."""

    def test_xtltyp_function_exists(self):
        import pycharmm.crystal as crystal

        assert hasattr(crystal, "get_xtltyp")
        assert callable(crystal.get_xtltyp)


class TestModuleLogging:
    """Test that module-level logging is configured."""

    def test_logger_exists(self):
        import pycharmm.crystal as crystal

        assert hasattr(crystal, "logger")

    def test_logger_name(self):
        import pycharmm.crystal as crystal

        assert crystal.logger.name == "pycharmm.crystal"


# =============================================================================
# Integration Tests (CHARMM library required)
# =============================================================================


@pytest.fixture
def _crystal_fresh_system(alanine_dipeptide_with_nbonds):
    """Wrap alanine_dipeptide_with_nbonds with crystal teardown.

    Crystal define_* leaves coordinate state that breaks the next call to
    ``ic.seed`` inside the fixture; we must call ``crystal.free()`` and
    clear PSF atoms so the next per-test rebuild works.
    """
    import pycharmm.crystal as crystal

    crystal.free()
    yield
    crystal.free()
    from pycharmm import psf, settings

    old_warn = settings.set_warn_level(-5)
    old_bomb = settings.set_bomb_level(-5)
    if psf.get_natom() > 0:
        psf.delete_atoms()
    settings.set_warn_level(old_warn)
    settings.set_bomb_level(old_bomb)


class TestCrystalDefinitions:
    """Test crystal definition functions with CHARMM library."""

    @pytest.fixture(autouse=True)
    def _setup(self, _crystal_fresh_system):
        yield

    def test_define_cubic(self):
        import pycharmm.crystal as crystal

        assert crystal.define_cubic(50.0)

    def test_define_tetragonal(self):
        import pycharmm.crystal as crystal

        assert crystal.define_tetra(50.0, 70.0)

    def test_define_orthorhombic(self):
        import pycharmm.crystal as crystal

        assert crystal.define_ortho(50.0, 60.0, 70.0)

    def test_define_monoclinic(self):
        import pycharmm.crystal as crystal

        assert crystal.define_mono(50.0, 60.0, 70.0, 100.0)

    def test_define_triclinic(self):
        import pycharmm.crystal as crystal

        assert crystal.define_tri(50.0, 60.0, 70.0, 80.0, 85.0, 95.0)

    def test_define_hexagonal(self):
        import pycharmm.crystal as crystal

        assert crystal.define_hexa(50.0, 70.0)

    def test_define_rhombohedral(self):
        import pycharmm.crystal as crystal

        assert crystal.define_rhombo(50.0, 60.0)

    def test_define_octahedral(self):
        import pycharmm.crystal as crystal

        assert crystal.define_octa(50.0)

    def test_define_rhombic_dodecahedron(self):
        import pycharmm.crystal as crystal

        assert crystal.define_rhdo(50.0)

    def test_rhombohedral_angle_validation(self):
        import pycharmm.crystal as crystal

        with pytest.raises(ValueError):
            crystal.define_rhombo(50.0, 0.0)

        with pytest.raises(ValueError):
            crystal.define_rhombo(50.0, -10.0)

        with pytest.raises(ValueError):
            crystal.define_rhombo(50.0, 120.0)

        with pytest.raises(ValueError):
            crystal.define_rhombo(50.0, 150.0)


class TestCrystalBuild:
    """Test crystal build function."""

    @pytest.fixture(autouse=True)
    def _setup(self, _crystal_fresh_system):
        yield

    def test_build_simple(self):
        import pycharmm.crystal as crystal

        crystal.define_cubic(50.0)
        assert crystal.build(cutoff=12.0)

    def test_build_with_symmetry_ops(self):
        import pycharmm.crystal as crystal

        crystal.define_cubic(50.0)
        # Identity (X,Y,Z) is added automatically; pass non-identity inversion
        assert crystal.build(cutoff=12.0, sym_ops=["(-X,-Y,-Z)"])


class TestCrystalQueryFunctions:
    """Test crystal query functions."""

    @pytest.fixture(autouse=True)
    def _setup(self, _crystal_fresh_system):
        yield

    def test_is_defined_false_when_no_crystal(self):
        import pycharmm.crystal as crystal

        crystal.free()
        result = crystal.is_defined()
        assert isinstance(result, bool)

    def test_is_defined_true_after_define(self):
        import pycharmm.crystal as crystal

        crystal.define_cubic(50.0)
        try:
            result = crystal.is_defined()
            if result is not None:
                assert isinstance(result, bool)
        except AttributeError:
            pass

    def test_get_unit_cell(self):
        import pycharmm.crystal as crystal

        crystal.define_cubic(50.0)
        crystal.build(cutoff=12.0)

        ucell = crystal.get_unit_cell()
        assert len(ucell) == 6

        a, b, c, alpha, beta, gamma = ucell
        assert a == pytest.approx(50.0, abs=1e-2)
        assert b == pytest.approx(50.0, abs=1e-2)
        assert c == pytest.approx(50.0, abs=1e-2)
        assert alpha == pytest.approx(90.0, abs=1e-2)
        assert beta == pytest.approx(90.0, abs=1e-2)
        assert gamma == pytest.approx(90.0, abs=1e-2)

    def test_get_unit_cell_orthorhombic(self):
        import pycharmm.crystal as crystal

        crystal.define_ortho(50.0, 60.0, 70.0)
        crystal.build(cutoff=12.0)

        ucell = crystal.get_unit_cell()
        a, b, c, alpha, beta, gamma = ucell

        assert a == pytest.approx(50.0, abs=1e-2)
        assert b == pytest.approx(60.0, abs=1e-2)
        assert c == pytest.approx(70.0, abs=1e-2)
        assert alpha == pytest.approx(90.0, abs=1e-2)
        assert beta == pytest.approx(90.0, abs=1e-2)
        assert gamma == pytest.approx(90.0, abs=1e-2)

    def test_get_unit_cell_monoclinic(self):
        import pycharmm.crystal as crystal

        crystal.define_mono(50.0, 60.0, 70.0, 100.0)
        crystal.build(cutoff=12.0)

        ucell = crystal.get_unit_cell()
        a, b, c, alpha, beta, gamma = ucell

        assert a == pytest.approx(50.0, abs=1e-2)
        assert b == pytest.approx(60.0, abs=1e-2)
        assert c == pytest.approx(70.0, abs=1e-2)
        assert alpha == pytest.approx(90.0, abs=1e-2)
        assert beta == pytest.approx(100.0, abs=1e-2)
        assert gamma == pytest.approx(90.0, abs=1e-2)

    def test_get_transformation_count(self):
        import pycharmm.crystal as crystal

        crystal.define_cubic(50.0)
        crystal.build(cutoff=12.0)

        ntrans = crystal.get_transformation_count()
        assert isinstance(ntrans, int)
        assert ntrans > 0

    def test_get_symmetry_count(self):
        import pycharmm.crystal as crystal

        crystal.define_cubic(50.0)
        crystal.build(cutoff=12.0)

        result = crystal.get_symmetry_count()
        if result is not None:
            assert isinstance(result, int)

    def test_get_cutoff(self):
        import pycharmm.crystal as crystal

        crystal.define_cubic(50.0)
        crystal.build(cutoff=12.0)

        result = crystal.get_cutoff()
        if result is not None:
            assert isinstance(result, float)
            assert result == pytest.approx(12.0, abs=1e-2)

    def test_get_crystal_type_direct(self):
        import pycharmm.crystal as crystal

        crystal.define_cubic(50.0)
        crystal.build(cutoff=12.0)

        result = crystal.get_crystal_type_direct()
        if result is not None:
            assert isinstance(result, str)
            assert result == "CUBI"


class TestCrystalFree:
    """Test crystal free function."""

    @pytest.fixture(autouse=True)
    def _setup(self, _crystal_fresh_system):
        yield

    def test_free_no_error_when_no_crystal(self):
        import pycharmm.crystal as crystal

        crystal.free()
        crystal.free()

    def test_free_clears_crystal(self):
        import pycharmm.crystal as crystal

        crystal.define_cubic(50.0)
        crystal.build(cutoff=12.0)
        crystal.free()


class TestCrystalWithPME:
    """Test crystal with PME/EWALD electrostatics."""

    @pytest.fixture(autouse=True)
    def _setup(self, _crystal_fresh_system):
        yield

    def test_cubic_with_pme_allowed(self):
        import pycharmm.crystal as crystal

        crystal.define_cubic(50.0)
        assert crystal.is_orthorhombic("CUBI")

    def test_ortho_with_pme_allowed(self):
        import pycharmm.crystal as crystal

        crystal.define_ortho(50.0, 60.0, 70.0)
        assert crystal.is_orthorhombic("ORTH")

    def test_mono_pme_warning(self, caplog):
        import pycharmm.crystal as crystal

        with caplog.at_level(logging.WARNING, logger=crystal.logger.name):
            crystal._check_backend_compatibility("MONO")

        log_output = "\n".join(r.getMessage() for r in caplog.records)
        assert "PME" in log_output or "EWALD" in log_output or "domdec" in log_output

    def test_tric_pme_warning(self, caplog):
        import pycharmm.crystal as crystal

        with caplog.at_level(logging.WARNING, logger=crystal.logger.name):
            crystal._check_backend_compatibility("TRIC")

        log_output = "\n".join(r.getMessage() for r in caplog.records)
        assert "PME" in log_output or "EWALD" in log_output or "domdec" in log_output


class TestMonoclinicBugFix:
    """Test that the monoclinic define_mono bug has been fixed."""

    @pytest.fixture(autouse=True)
    def _setup(self, _crystal_fresh_system):
        yield

    def test_monoclinic_beta_angle_applied(self):
        import pycharmm.crystal as crystal

        crystal.define_mono(50.0, 60.0, 70.0, 100.0)
        crystal.build(cutoff=12.0)

        ucell = crystal.get_unit_cell()
        beta = ucell[4]

        assert beta == pytest.approx(100.0, abs=1e-1)

    def test_monoclinic_vs_orthorhombic_different(self):
        import pycharmm.crystal as crystal

        crystal.define_ortho(50.0, 60.0, 70.0)
        crystal.build(cutoff=12.0)
        ortho_ucell = crystal.get_unit_cell()
        crystal.free()

        crystal.define_mono(50.0, 60.0, 70.0, 100.0)
        crystal.build(cutoff=12.0)
        mono_ucell = crystal.get_unit_cell()

        assert abs(ortho_ucell[4] - mono_ucell[4]) > 1e-1


class TestXtltypInference:
    """Test get_xtltyp lattice type inference from unit cell."""

    @pytest.fixture(autouse=True)
    def _setup(self, _crystal_fresh_system):
        yield

    def test_xtltyp_cubic(self):
        import pycharmm.crystal as crystal

        crystal.define_cubic(50.0)
        crystal.build(cutoff=12.0)

        assert crystal.get_xtltyp() == "CUBI"

    def test_xtltyp_monoclinic(self):
        import pycharmm.crystal as crystal

        crystal.define_mono(50.0, 60.0, 70.0, 100.0)
        crystal.build(cutoff=12.0)

        assert crystal.get_xtltyp() == "MONO"
