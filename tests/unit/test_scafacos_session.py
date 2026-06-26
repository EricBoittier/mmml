"""Unit tests for ScaFaCoS ctypes wrapper (mocked library)."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from mmml.interfaces.scafacosInterface.scafacos_session import (
    ScaFaCoSUnavailable,
    compute_scafacos_coulomb,
    have_scafacos,
    resolve_scafacos_library_path,
)


def test_have_scafacos_false_when_lib_missing():
    with mock.patch(
        "mmml.interfaces.scafacosInterface.scafacos_session.load_scafacos_library",
        side_effect=ScaFaCoSUnavailable("missing"),
    ):
        assert have_scafacos() is False


def test_resolve_scafacos_library_path_from_env(tmp_path):
    lib = tmp_path / "libfcs.so"
    lib.write_bytes(b"\x00")
    with mock.patch.dict("os.environ", {"SCAFACOS_LIB": str(lib)}):
        assert resolve_scafacos_library_path() == lib.resolve()


def test_compute_scafacos_coulomb_calls_session(tmp_path):
    pos = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    chg = np.array([1.0, -1.0])
    forces = np.zeros((2, 3))

    class FakeSession:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def configure_cubic_box(self, **kwargs):
            return None

        def set_parameter(self, key, value):
            return None

        def run_coulomb(self, positions, charges):
            from mmml.interfaces.scafacosInterface.scafacos_session import CoulombFieldResult

            return CoulombFieldResult(energy_kcalmol=-5.0, forces_kcalmol_A=forces)

    with mock.patch(
        "mmml.interfaces.scafacosInterface.scafacos_session.ScaFaCoSSession",
        FakeSession,
    ):
        out = compute_scafacos_coulomb(pos, chg, box_length_A=20.0, method="p3m")
    assert out.energy_kcalmol == pytest.approx(-5.0)
    assert out.forces_kcalmol_A.shape == (2, 3)
