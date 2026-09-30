"""Restart crystal files keep lattice type and shear."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.charmm_restart_io import (
    format_rest_header,
    lattice_type_for_cell,
    parse_restart_crystal,
    rest_header_lattice_token,
    set_rest_header_lattice_token,
    write_charmm_restart_from_memory,
)


def test_sheared_cell_round_trips_as_triclinic(tmp_path) -> None:
    cell = np.array(
        [
            [20.0, 1.5, 0.2],
            [0.0, 21.0, -0.4],
            [0.0, 0.0, 22.0],
        ],
        dtype=float,
    )
    path = tmp_path / "shear.res"
    write_charmm_restart_from_memory(
        path,
        positions=np.zeros((2, 3)),
        include_velocities=False,
        cell=cell,
    )
    text = path.read_text(encoding="ascii")
    token, parsed = parse_restart_crystal(text)
    assert token == "TRIC"
    assert rest_header_lattice_token(text.splitlines()[0]) == "TRIC"
    np.testing.assert_allclose(parsed, cell, rtol=0, atol=1e-12)
    assert "1.500000000000000D+00" in text


def test_cubic_restart_header_stays_cubi(tmp_path) -> None:
    cell = np.diag([30.0, 30.0, 30.0])
    path = tmp_path / "cube.res"
    write_charmm_restart_from_memory(
        path,
        positions=np.zeros((1, 3)),
        include_velocities=False,
        cell=cell,
    )
    token, parsed = parse_restart_crystal(path.read_text(encoding="ascii"))
    assert lattice_type_for_cell(cell) == "CUBI"
    assert token == "CUBI"
    np.testing.assert_allclose(parsed, cell, rtol=0, atol=1e-12)


def test_set_rest_header_lattice_token_keeps_version() -> None:
    header = "REST    48     1                "
    updated = set_rest_header_lattice_token(header, "TRIC")
    assert updated[4:16] == header[4:16]
    assert rest_header_lattice_token(updated) == "TRIC"
    assert set_rest_header_lattice_token("REST  SYNTHETIC-HANDOFF      0", "TRIC") == (
        "REST  SYNTHETIC-HANDOFF      0"
    )
    assert format_rest_header(xtltyp="CUBI").startswith("REST")


def test_synthetic_handoff_writes_off_diagonal(tmp_path) -> None:
    from mmml.cli.run.md_handoff import MdHandoffState, _write_synthetic_charmm_restart

    cell = np.array([[18.0, 0.8, 0.0], [0.0, 19.0, 0.3], [0.0, 0.0, 17.5]])
    handoff = MdHandoffState(
        positions=np.zeros((2, 3)),
        atomic_numbers=np.array([18, 18], dtype=np.int32),
        cell=cell,
        pbc=True,
    )
    path = tmp_path / "handoff.res"
    _write_synthetic_charmm_restart(handoff, path)
    token, parsed = parse_restart_crystal(path.read_text(encoding="ascii"))
    assert token == "TRIC"
    np.testing.assert_allclose(parsed, cell, rtol=0, atol=1e-12)


def _byref_values(call) -> list[float]:
    return [float(arg._obj.value) for arg in call.args]


def test_define_tri_calls_triclinic_entry(monkeypatch) -> None:
    try:
        import pycharmm.crystal as crystal
        import pycharmm.lib as lib
    except Exception as exc:
        pytest.skip(f"pycharmm crystal import unavailable: {exc}")
    fake = MagicMock()
    fake.crystal_define_tri.return_value = 1
    fake.crystal_define_mono.return_value = 1
    monkeypatch.setattr(lib, "charmm", fake)

    assert crystal.define_tri(10.0, 11.0, 12.0, 80.0, 85.0, 75.0)
    assert crystal.define_mono(10.0, 11.0, 12.0, 100.0)
    assert _byref_values(fake.crystal_define_tri.call_args) == pytest.approx(
        [10.0, 11.0, 12.0, 80.0, 85.0, 75.0]
    )
    assert _byref_values(fake.crystal_define_mono.call_args) == pytest.approx(
        [10.0, 11.0, 12.0, 100.0]
    )
    fake.crystal_define_ortho.assert_not_called()


def test_define_tri_round_trip_gamma_when_charmm_imports() -> None:
    try:
        import pycharmm.crystal as crystal
        import pycharmm.lib as lib
    except Exception as exc:
        pytest.skip(f"pycharmm crystal import unavailable: {exc}")

    if not hasattr(lib.charmm, "crystal_define_tri"):
        pytest.skip("libcharmm has no crystal_define_tri")
    assert crystal.define_tri(20.0, 21.0, 22.0, 80.0, 85.0, 75.0)
    cell = crystal.get_unit_cell()
    assert float(cell[5]) == pytest.approx(75.0, abs=1.0e-3)
    assert float(cell[3]) == pytest.approx(80.0, abs=1.0e-3)
    assert float(cell[4]) == pytest.approx(85.0, abs=1.0e-3)
