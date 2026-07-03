"""PyCHARMM pytest marker selection rules."""

from __future__ import annotations

import tests.conftest as root_conftest


def test_trialanine_water_box_is_serial_under_mpirun_smoke():
    rel = "functionality/charmm/test_trialanine_water_box_mm.py"

    assert root_conftest._matches_any(
        rel,
        root_conftest._PYCHARMM_PATH_PREFIXES,
    )
    assert root_conftest._matches_any(
        rel,
        root_conftest._CHARMM_SERIAL_PATH_PREFIXES,
    )

