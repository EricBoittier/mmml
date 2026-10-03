"""Tests for pycharmm.set_mpi_comm — the base-communicator hook.

These exercise the pure pre-init guard/bookkeeping logic and never
initialize the CHARMM library.  Each test snapshots and restores the
loader's private state so it is order-independent (some other test in
the session may already have initialized the library, which would
otherwise make set_mpi_comm raise).
"""

import pytest

from pycharmm.loader import _loader, set_mpi_comm


class _FakeComm:
    """Minimal stand-in for an mpi4py communicator."""

    def __init__(self, handle):
        self._handle = handle

    def py2f(self):
        return self._handle


@pytest.fixture(autouse=True)
def _restore_loader_state():
    saved = (_loader._lib, _loader._user_comm, _loader._user_comm_handle)
    # Pretend the library is not yet initialized so set_mpi_comm is allowed.
    _loader._lib = None
    _loader._user_comm = None
    _loader._user_comm_handle = None
    yield
    (_loader._lib, _loader._user_comm, _loader._user_comm_handle) = saved


def test_stores_handle_and_retains_comm():
    comm = _FakeComm(42)
    set_mpi_comm(comm)
    assert _loader._user_comm_handle == 42
    # The Comm object must be retained so mpi4py cannot free it before the
    # deferred library init consumes the handle.
    assert _loader._user_comm is comm


def test_none_resets_selection():
    set_mpi_comm(_FakeComm(7))
    set_mpi_comm(None)
    assert _loader._user_comm_handle is None
    assert _loader._user_comm is None


def test_non_comm_argument_raises_typeerror():
    with pytest.raises(TypeError, match="py2f"):
        set_mpi_comm(object())


def test_raises_after_initialization():
    # Simulate an already-initialized library.
    _loader._lib = object()
    with pytest.raises(RuntimeError, match="before CHARMM is initialized"):
        set_mpi_comm(_FakeComm(1))
