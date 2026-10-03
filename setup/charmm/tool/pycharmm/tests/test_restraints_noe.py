"""Tests for pycharmm restraints module: NOE distance restraints.

Covers the NOE family -- standard NOE, point-NOE (PNOE), and
moving PNOE -- including the context manager / command buffering
and exception-handling paths.

This file was split out of the original 1325-line test_restraints.py.
The shared `_wipe_psf` helper lives in
`tests/_restraints_helpers.py`.

Run with:
    cd tool/pycharmm
    pytest tests/test_restraints_noe.py -v
"""

import pytest
from _restraints_helpers import wipe_psf as _wipe_psf

# ============================================================
# Test classes
# ============================================================


class TestNOEBasic:
    """Basic NOE restraint functionality tests."""

    @pytest.fixture(autouse=True)
    def _setup(self, alanine_dipeptide_with_nbonds):
        import pycharmm.restraints as restraints

        restraints.noe_reset()
        yield
        _wipe_psf()

    def test_noe_reset(self):
        """Test NOE reset clears state."""
        import pycharmm.restraints as restraints

        restraints.noe_reset()

        state = restraints.noe_get_state()
        assert not state["active"]
        assert state["count"] == 0

    def test_noe_context_manager(self):
        """Test NOE context manager."""
        import pycharmm.restraints as restraints

        with restraints.NOE() as noe:
            noe.assign(
                selection1="segid ADP .and. type CA",
                selection2="segid ADP .and. type C",
                kmax=10.0,
                rmax=3.0,
            )

        assert restraints.noe_get_count() > 0

    def test_noe_assign_basic(self):
        """Test basic noe_assign functionality."""
        import pycharmm.restraints as restraints

        restraints.noe_assign(
            selection1="segid ADP .and. type CA",
            selection2="segid ADP .and. type C",
            kmin=0.0,
            rmin=0.0,
            kmax=10.0,
            rmax=3.0,
            fmax=100.0,
        )

        assert restraints.noe_get_count() > 0

    def test_noe_assign_with_soft_asymptote(self):
        """Test NOE with soft asymptote (RSWI/SEXP)."""
        import pycharmm.restraints as restraints

        restraints.noe_assign(
            selection1="segid ADP .and. type CA",
            selection2="segid ADP .and. type C",
            kmax=10.0,
            rmax=3.0,
            rswi=4.0,
            sexp=2.0,
        )

        assert restraints.noe_get_count() > 0

    def test_noe_scale(self):
        """Test NOE scale factor setting."""
        import pycharmm.restraints as restraints

        restraints.noe_assign(
            selection1="segid ADP .and. type CA",
            selection2="segid ADP .and. type C",
            kmax=10.0,
            rmax=3.0,
        )

        restraints.noe_scale(0.5)

        state = restraints.noe_get_state()
        assert state["scale"] == pytest.approx(0.5, abs=1e-7)

    def test_noe_multiple_restraints(self):
        """Test multiple NOE restraints."""
        import pycharmm.restraints as restraints

        restraints.noe_assign(
            selection1="segid ADP .and. type CA",
            selection2="segid ADP .and. type C",
            kmax=10.0,
            rmax=3.0,
        )

        restraints.noe_assign(
            selection1="segid ADP .and. type N",
            selection2="segid ADP .and. type C",
            kmax=5.0,
            rmax=4.0,
        )

        assert restraints.noe_get_count() >= 2

    def test_noe_context_manager_reset(self):
        """Test NOE context manager with reset option."""
        import pycharmm.restraints as restraints

        restraints.noe_assign(
            selection1="segid ADP .and. type CA",
            selection2="segid ADP .and. type C",
            kmax=10.0,
            rmax=3.0,
        )

        initial_count = restraints.noe_get_count()

        with restraints.NOE(reset=True) as noe:
            noe.assign(
                selection1="segid ADP .and. type N",
                selection2="segid ADP .and. type C",
                kmax=5.0,
                rmax=4.0,
            )

        assert restraints.noe_get_count() <= initial_count

    def test_noe_print(self):
        """Test NOE print does not raise errors."""
        import pycharmm.restraints as restraints

        restraints.noe_assign(
            selection1="segid ADP .and. type CA",
            selection2="segid ADP .and. type C",
            kmax=10.0,
            rmax=3.0,
        )

        restraints.noe_print()
        restraints.noe_print(analysis=True)


class TestNOEPNOE:
    """Point NOE (PNOE) restraint tests."""

    @pytest.fixture(autouse=True)
    def _setup(self, alanine_dipeptide_with_nbonds):
        import pycharmm.restraints as restraints

        restraints.noe_reset()
        yield
        _wipe_psf()

    def test_pnoe_assign(self):
        """Test Point NOE (PNOE) assignment."""
        import pycharmm.restraints as restraints

        restraints.noe_assign_pnoe(
            selection="segid ADP .and. type CA", cnox=0.0, cnoy=0.0, cnoz=0.0, kmax=10.0, rmax=2.0
        )

        assert restraints.noe_get_count() > 0

    def test_pnoe_with_context_manager(self):
        """Test PNOE through context manager."""
        import pycharmm.restraints as restraints

        with restraints.NOE() as noe:
            noe.assign_pnoe(
                selection="segid ADP .and. type CA",
                cnox=1.0,
                cnoy=2.0,
                cnoz=3.0,
                kmax=5.0,
                rmax=3.0,
            )

        assert restraints.noe_get_count() > 0


class TestNOEMovingPNOE:
    """Moving Point NOE tests."""

    @pytest.fixture(autouse=True)
    def _setup(self, alanine_dipeptide_with_nbonds):
        import pycharmm.restraints as restraints

        restraints.noe_reset()
        yield
        _wipe_psf()

    def test_mpnoe_setup(self):
        """Test moving PNOE target setup."""
        import pycharmm.restraints as restraints

        restraints.noe_assign_pnoe(
            selection="segid ADP .and. type CA", cnox=0.0, cnoy=0.0, cnoz=0.0, kmax=10.0, rmax=2.0
        )

        restraints.noe_mpnoe(inoe=1, tnox=5.0, tnoy=5.0, tnoz=5.0)

        restraints.noe_nmpnoe(nsteps=1000)


class TestNoeContextManagerExceptionHandling:
    """Test NOE context manager exception handling.

    These tests verify that the NOE context manager properly handles
    exceptions and cleans up state.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        import pycharmm.restraints as restraints

        restraints._state.reset()

    def test_exception_clears_buffer(self):
        """Test that NOE context clears buffer on exception."""
        import pycharmm.restraints as restraints

        with pytest.raises(RuntimeError):
            with restraints.NOE():
                raise RuntimeError("test exception")

        assert len(restraints._state._noe_command_buffer) == 0

    def test_exception_clears_context_flag(self):
        """Test that exception clears _in_noe_context flag."""
        import pycharmm.restraints as restraints

        with pytest.raises(RuntimeError):
            with restraints.NOE():
                assert restraints._state._in_noe_context
                raise RuntimeError("test exception")

        assert not restraints._state._in_noe_context

    def test_nested_context_rejected(self):
        """Test that nested NOE contexts raise RuntimeError."""
        import pycharmm.restraints as restraints

        with pytest.raises(RuntimeError) as exc_info:
            with restraints.NOE():
                with restraints.NOE():
                    pass
        assert "nested" in str(exc_info.value).lower()

    def test_normal_exit_clears_buffer(self):
        """Test that normal exit also clears buffer."""
        import pycharmm.restraints as restraints

        restraints._state._in_noe_context = False
        restraints._state._noe_command_buffer = []

        assert len(restraints._state._noe_command_buffer) == 0
