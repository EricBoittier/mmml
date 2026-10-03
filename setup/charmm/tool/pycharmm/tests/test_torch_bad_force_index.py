"""Brief: NULL-guard regression for the torch global-parameter C API.

``omm.torch_add_force`` returns the 0-based index of the new force in the
ForcesStore.  Passing any *other* index to the torch global-parameter
entry points means ``fstore_get`` returns NULL (out of range), which the
C layer used to dereference -- crashing CHARMM with a hard SIGSEGV inside
``api_torch_add_global_param`` (see source/openmm/torch.cpp).

The guarded C code now returns -1 (for the int-valued add) or a no-op
(for the void setters), matching the parallel customForces path
(``cf_add_global_param``: "if (!f) return -1;").  This test locks that in:
an out-of-range index must be reported, never crash.
"""

import pytest

pytest.importorskip("openmm")

from pycharmm import omm  # noqa: E402


def test_torch_global_param_out_of_range_index_does_not_crash():
    if not omm.has_ommtorch():
        pytest.skip("CHARMM built without OMMTORCH")

    # No torch force at index 999 (the store is far shorter than that),
    # so fstore_get returns NULL.  The guard must turn that into a -1
    # return rather than a segmentation fault.
    assert omm.torch_add_global_param(999, "k", 1.0) == -1

    # The void setters must likewise no-op on a bad index instead of
    # dereferencing NULL.  Reaching the assert at all proves no crash.
    omm.torch_set_global_param(999, 0, 1.0)
