"""Tests for pycharmm restraints module: harmonic and fixed-atom constraints.

Covers per-atom restraint primitives: harmonic positional
restraints (cons_harm) and fixed-atom constraints (cons_fix).

This file was split out of the original 1325-line test_restraints.py.
The shared `_wipe_psf` helper lives in
`tests/_restraints_helpers.py`.

Run with:
    cd tool/pycharmm
    pytest tests/test_restraints_atoms.py -v
"""

import pytest
from _restraints_helpers import wipe_psf as _wipe_psf

# ============================================================
# Test classes
# ============================================================


class TestHarmonicRestraints:
    """Test harmonic restraints wrapper functions."""

    @pytest.fixture(autouse=True)
    def _setup(self, alanine_dipeptide_with_nbonds):
        yield
        _wipe_psf()

    def test_harmonic_absolute_simple(self):
        """Test basic harmonic absolute restraints - calls don't raise."""
        import pycharmm.restraints as restraints

        restraints.harmonic_absolute(force_const=10.0)
        result = restraints.harmonic_turn_off()
        assert result

    def test_harmonic_absolute_with_selection(self):
        """Test harmonic absolute restraints with selection - no exceptions."""
        import pycharmm
        import pycharmm.restraints as restraints

        ca_sel = pycharmm.SelectAtoms(atom_type="CA")
        restraints.harmonic_absolute(selection=ca_sel, force_const=5.0)
        restraints.harmonic_turn_off()

    def test_harmonic_absolute_with_scaling(self):
        """Test harmonic absolute restraints with axis scaling - no exceptions."""
        import pycharmm.restraints as restraints

        restraints.harmonic_absolute(force_const=10.0, x_scale=0.0, y_scale=0.0, z_scale=1.0)
        restraints.harmonic_turn_off()

    def test_harmonic_best_fit(self):
        """Test best-fit harmonic restraints - no exceptions."""
        import pycharmm
        import pycharmm.restraints as restraints

        bb_sel = pycharmm.SelectAtoms(atom_type="CA")
        restraints.harmonic_best_fit(selection=bb_sel, force_const=1.0)
        restraints.harmonic_turn_off()

    def test_harmonic_pca(self):
        """Test PCA-style harmonic restraints - no exceptions."""
        import pycharmm.restraints as restraints

        restraints.harmonic_pca(force_const=5.0, expo=2)
        restraints.harmonic_turn_off()

    def test_harmonic_turn_off(self):
        """Test turning off harmonic restraints returns True."""
        import pycharmm.restraints as restraints

        restraints.harmonic_absolute(force_const=10.0)
        result = restraints.harmonic_turn_off()
        assert result

    def test_check_harmonic_backend_support(self):
        """Test harmonic backend compatibility checking."""
        import pycharmm.restraints as restraints

        for backend in ["standard", "domdec", "blade", "openmm"]:
            result = restraints.check_backend_support("HARMONIC", backend)
            assert result["supported"], f"HARMONIC should be supported on {backend}"


class TestFixedAtomConstraints:
    """Test fixed atom constraints wrapper functions."""

    @pytest.fixture(autouse=True)
    def _setup(self, alanine_dipeptide_with_nbonds):
        yield
        _wipe_psf()

    def test_fix_setup_and_turn_off(self):
        """Test fix setup and turn_off."""
        import pycharmm
        import pycharmm.restraints as restraints

        ca_sel = pycharmm.SelectAtoms(atom_type="CA")
        restraints.fix_atoms(ca_sel)

        result = restraints.fix_turn_off()
        assert result

    def test_check_fix_backend_support(self):
        """Test FIX backend compatibility checking."""
        import pycharmm.restraints as restraints

        for backend in ["standard", "domdec", "blade", "openmm"]:
            result = restraints.check_backend_support("FIX", backend)
            assert result["supported"], f"FIX should be supported on {backend}"
