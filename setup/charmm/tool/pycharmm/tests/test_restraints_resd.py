"""Tests for pycharmm restraints module: RESD distance/dihedral restraints.

Covers the RESD family of distance and dihedral restraints,
their validation, scale-factor handling, and error codes.

This file was split out of the original 1325-line test_restraints.py.
The shared `_wipe_psf` helper lives in
`tests/_restraints_helpers.py`.

Run with:
    cd tool/pycharmm
    pytest tests/test_restraints_resd.py -v
"""

import pytest
from _restraints_helpers import wipe_psf as _wipe_psf

# ============================================================
# Test classes
# ============================================================


class TestRESD:
    """Test RESD (Restrained Distances) functionality.

    Note: RESD uses atom specification format 'SEGID RESID ATOMNAME',
    not selection syntax like NOE.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, alanine_dipeptide_with_nbonds):
        import pycharmm.restraints as restraints

        restraints.resd_reset()
        yield
        _wipe_psf()

    def test_resd_reset(self):
        """Test RESD reset clears state."""
        import pycharmm.restraints as restraints

        restraints.resd_reset()

        state = restraints.resd_get_state()
        assert not state["active"]
        assert state["count"] == 0

    def test_resd_add_simple(self):
        """Test adding a simple distance restraint."""
        import pycharmm.restraints as restraints

        distances = [(1.0, ("ADP", 1, "CA"), ("ADP", 1, "C"))]

        idx = restraints.resd_add(distances, kval=100.0, rval=1.5)

        assert idx == 1
        assert restraints.resd_get_count() == 1
        assert restraints.resd_is_active()

    def test_resd_add_string_format(self):
        """Test adding restraint using string atom specification."""
        import pycharmm.restraints as restraints

        distances = [(1.0, "ADP 1 CA", "ADP 1 C")]

        idx = restraints.resd_add(distances, kval=100.0, rval=1.5)

        assert idx == 1
        assert restraints.resd_get_count() == 1

    def test_resd_add_reaction_coord(self):
        """Test reaction coordinate (linear combination of distances)."""
        import pycharmm.restraints as restraints

        atom1 = ("ADP", 1, "CA")
        atom2 = ("ADP", 1, "CB")
        atom3 = ("ADP", 1, "C")

        distances = [(1.0, atom1, atom2), (-1.0, atom2, atom3)]

        idx = restraints.resd_add(distances, kval=500.0, rval=0.0)

        assert idx == 1
        assert restraints.resd_get_count() == 1

    def test_resd_multiple_restraints(self):
        """Test adding multiple RESD restraints."""
        import pycharmm.restraints as restraints

        restraints.resd_add([(1.0, ("ADP", 1, "CA"), ("ADP", 1, "C"))], kval=100.0, rval=1.5)

        restraints.resd_add([(1.0, ("ADP", 1, "N"), ("ADP", 1, "C"))], kval=200.0, rval=2.0)

        assert restraints.resd_get_count() == 2

    def test_resd_scale(self):
        """Test RESD scale factor setting."""
        import pycharmm.restraints as restraints

        restraints.resd_add([(1.0, ("ADP", 1, "CA"), ("ADP", 1, "C"))], kval=100.0, rval=1.5)

        restraints.resd_scale(0.5)

        state = restraints.resd_get_state()
        assert state["scale"] == pytest.approx(0.5, abs=1e-7)

    def test_resd_print(self):
        """Test RESD print does not raise errors."""
        import pycharmm.restraints as restraints

        restraints.resd_add([(1.0, ("ADP", 1, "CA"), ("ADP", 1, "C"))], kval=100.0, rval=1.5)

        restraints.resd_print()

    def test_resd_with_options(self):
        """Test RESD with EVAL, IVAL, POSITIVE options."""
        import pycharmm.restraints as restraints

        restraints.resd_add(
            [(1.0, ("ADP", 1, "CA"), ("ADP", 1, "C"))],
            kval=100.0,
            rval=1.5,
            eval_exp=4,
            ival=2,
            positive=True,
        )

        restraints_list = restraints.resd_get_restraints()
        assert len(restraints_list) == 1
        assert restraints_list[0]["eval"] == 4
        assert restraints_list[0]["ival"] == 2
        assert restraints_list[0]["positive"]

    def test_resd_empty_distances_error(self):
        """Test that empty distances list raises ValueError."""
        import pycharmm.restraints as restraints

        with pytest.raises(ValueError):
            restraints.resd_add([], kval=100.0, rval=1.5)


class TestResdValidation:
    """Test RESD parameter validation.

    These tests verify that invalid parameters are properly rejected
    before being passed to CHARMM, ensuring consistent behavior
    between direct API and script-based approaches.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, alanine_dipeptide_with_nbonds):
        import pycharmm.restraints as restraints

        restraints.resd_reset()
        yield
        _wipe_psf()

    def test_kval_zero_rejected(self):
        """Test that kval=0 raises ValueError."""
        import pycharmm.restraints as restraints

        with pytest.raises(ValueError) as exc_info:
            restraints.resd_add(
                distances=[(1.0, ("ADP", 1, "CA"), ("ADP", 1, "C"))], kval=0.0, rval=1.5
            )
        assert "force constant" in str(exc_info.value).lower()

    def test_eval_negative_rejected(self):
        """Test that eval_exp <= 0 raises ValueError."""
        import pycharmm.restraints as restraints

        with pytest.raises(ValueError) as exc_info:
            restraints.resd_add(
                distances=[(1.0, ("ADP", 1, "CA"), ("ADP", 1, "C"))],
                kval=100.0,
                rval=1.5,
                eval_exp=-1,
            )
        assert "exponent" in str(exc_info.value).lower()

    def test_eval_zero_rejected(self):
        """Test that eval_exp=0 raises ValueError."""
        import pycharmm.restraints as restraints

        with pytest.raises(ValueError) as exc_info:
            restraints.resd_add(
                distances=[(1.0, ("ADP", 1, "CA"), ("ADP", 1, "C"))],
                kval=100.0,
                rval=1.5,
                eval_exp=0,
            )
        assert "exponent" in str(exc_info.value).lower()

    def test_positive_negative_mutual_exclusion(self):
        """Test that both positive=True and negative=True raises ValueError."""
        import pycharmm.restraints as restraints

        with pytest.raises(ValueError) as exc_info:
            restraints.resd_add(
                distances=[(1.0, ("ADP", 1, "CA"), ("ADP", 1, "C"))],
                kval=100.0,
                rval=1.5,
                positive=True,
                negative=True,
            )
        assert "both" in str(exc_info.value).lower()

    def test_valid_params_accepted(self):
        """Test that valid parameters are accepted."""
        import pycharmm.restraints as restraints

        idx = restraints.resd_add(
            distances=[(1.0, ("ADP", 1, "CA"), ("ADP", 1, "C"))],
            kval=100.0,
            rval=1.5,
            eval_exp=2,
            ival=1,
        )
        assert idx > 0

    def test_positive_only_accepted(self):
        """Test that positive=True alone is accepted."""
        import pycharmm.restraints as restraints

        idx = restraints.resd_add(
            distances=[(1.0, ("ADP", 1, "CA"), ("ADP", 1, "C"))],
            kval=100.0,
            rval=1.5,
            positive=True,
        )
        assert idx > 0

    def test_negative_only_accepted(self):
        """Test that negative=True alone is accepted."""
        import pycharmm.restraints as restraints

        idx = restraints.resd_add(
            distances=[(1.0, ("ADP", 1, "CA"), ("ADP", 1, "C"))],
            kval=100.0,
            rval=1.5,
            negative=True,
        )
        assert idx > 0


class TestResdScale:
    """Test RESD scale function with validation."""

    @pytest.fixture(autouse=True)
    def _setup(self, alanine_dipeptide_with_nbonds):
        import pycharmm.restraints as restraints

        restraints.resd_reset()
        yield
        _wipe_psf()

    def test_negative_scale_rejected(self):
        """Test that negative scale factor raises ValueError."""
        import pycharmm.restraints as restraints

        with pytest.raises(ValueError) as exc_info:
            restraints.resd_scale(-0.5)
        assert "non-negative" in str(exc_info.value).lower()

    def test_zero_scale_accepted(self):
        """Test that zero scale factor is accepted."""
        import pycharmm.restraints as restraints

        restraints.resd_add(
            distances=[(1.0, ("ADP", 1, "CA"), ("ADP", 1, "C"))], kval=100.0, rval=1.5
        )

        restraints.resd_scale(0.0)
        state = restraints.resd_get_state()
        assert state["scale"] == 0.0


class TestResdErrorCodes:
    """Test RESD error code constants and interpretation."""

    def test_error_code_constants_defined(self):
        """Test that error code constants are defined."""
        import pycharmm.restraints as restraints

        assert hasattr(restraints, "RESD_ERR_NPAIRS")
        assert hasattr(restraints, "RESD_ERR_REDMAX")
        assert hasattr(restraints, "RESD_ERR_REDMX2")
        assert hasattr(restraints, "RESD_ERR_DISABLED")

    def test_error_code_values(self):
        """Test that error code values match Fortran API."""
        import pycharmm.restraints as restraints

        assert restraints.RESD_ERR_NPAIRS == -10
        assert restraints.RESD_ERR_REDMAX == -11
        assert restraints.RESD_ERR_REDMX2 == -12
        assert restraints.RESD_ERR_DISABLED == -20

    def test_interpret_error_function_exists(self):
        """Test that _interpret_resd_error function exists."""
        import pycharmm.restraints as restraints

        assert hasattr(restraints, "_interpret_resd_error")
        assert callable(restraints._interpret_resd_error)

    def test_interpret_npairs_error(self):
        """Test error message for npairs error."""
        import pycharmm.restraints as restraints

        msg = restraints._interpret_resd_error(restraints.RESD_ERR_NPAIRS)
        assert "atom pairs" in msg.lower()

    def test_interpret_redmax_error(self):
        """Test error message for REDMAX error."""
        import pycharmm.restraints as restraints

        msg = restraints._interpret_resd_error(restraints.RESD_ERR_REDMAX)
        assert "restraint" in msg.lower()

    def test_interpret_redmx2_error(self):
        """Test error message for REDMX2 error."""
        import pycharmm.restraints as restraints

        msg = restraints._interpret_resd_error(restraints.RESD_ERR_REDMX2)
        assert "pair" in msg.lower()

    def test_interpret_disabled_error(self):
        """Test error message for DISABLED error."""
        import pycharmm.restraints as restraints

        msg = restraints._interpret_resd_error(restraints.RESD_ERR_DISABLED)
        assert "not compiled" in msg.lower()

    def test_interpret_unknown_error(self):
        """Test error message for unknown error code."""
        import pycharmm.restraints as restraints

        msg = restraints._interpret_resd_error(-999)
        assert "unknown" in msg.lower()
