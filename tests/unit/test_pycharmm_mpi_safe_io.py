"""Source-level checks for MPI-safe PyCHARMM file I/O (no WriteScript on hot paths)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_read_rtf_uses_c_api_not_command_script():
    block = _read("pycharmm/read.py").split("def rtf(")[1].split("\ndef ")[0]
    assert "read_rtf_file" in block
    assert "CommandScript" not in block
    assert "c_api_path_buffer" in block
    assert "byref(append)" in block


def test_read_prm_uses_c_api_not_command_script():
    block = _read("pycharmm/read.py").split("def prm(")[1].split("\ndef ")[0]
    assert "read_param_file" in block
    assert "CommandScript" not in block


def test_write_coor_pdb_uses_c_api_not_write_script():
    block = _read("pycharmm/write.py").split("def coor_pdb(")[1].split("\ndef ")[0]
    assert "write_coor_pdb" in block
    assert "WriteScript" not in block
    assert "c_api_path_buffer" in block


def test_write_coor_card_avoids_write_script_when_mmml_available():
    block = _read("pycharmm/write.py").split("def coor_card(")[1].split("\ndef ")[0]
    assert "write_charmm_crd_from_charmm" in block
    assert "WriteScript" in block  # fallback when mmml not importable


def test_write_charmm_crd_from_charmm_avoids_charmm_write():
    block = (
        _read("mmml/interfaces/pycharmmInterface/mlpot/setup.py")
        .split("def write_charmm_crd_from_charmm(")[1]
        .split("\ndef ")[0]
    )
    assert "coor_card" not in block
    assert "WriteScript" not in block
    assert " EXT" in block


def test_write_charmm_restart_from_memory_roundtrip(tmp_path):
    from mmml.interfaces.pycharmmInterface.charmm_restart_io import (
        write_charmm_restart_from_memory,
    )

    pos = np.array(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        dtype=float,
    )
    res = tmp_path / "mini_box_equil.res"
    write_charmm_restart_from_memory(
        res,
        positions=pos,
        title="seed",
        include_crystal=False,
        include_velocities=False,
    )
    text = res.read_text(encoding="ascii")
    assert text.startswith("REST")
    assert " !X, Y, Z" in text
    assert "1.000000000000000D-01" in text
    assert "9.000000000000000D-01" in text


def test_rewrite_dynamics_restart_avoids_charmm_script():
    block = (
        _read("mmml/interfaces/pycharmmInterface/mlpot/bonded_mm_recovery.py")
        .split("def rewrite_dynamics_restart_from_current_state(")[1]
        .split("\ndef ")[0]
    )
    assert "write_charmm_restart_from_memory" in block
    assert "charmm_restart_io" in block
    assert "charmm_script" not in block
    assert "write restart" not in block


def test_dynamics_setters_use_c_api_path_buffer():
    block = _read("pycharmm/dynamics.py").split("def set_iunwri(")[1].split(
        "\ndef "
    )[0]
    assert "c_api_path_buffer" in block or "_dynamics_path_ctypes" in block
    block = _read("mmml/interfaces/pycharmmInterface/mlpot/dynamics.py").split(
        "def run_dynamics("
    )[1].split("\ndef ")[0]
    assert "_apply_dynamics_io_setters" in block
    assert "DynamicsScript" in block


def test_apply_dynamics_io_setters_uses_path_strings():
    import sys
    from types import SimpleNamespace
    from unittest.mock import patch

    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        _apply_dynamics_io_setters,
    )

    fake_dyn = SimpleNamespace(
        set_iunrea=lambda path: True,
        set_iunwri=lambda path: True,
        set_iuncrd=lambda path: True,
    )
    calls: list[tuple[str, str]] = []
    fake_dyn.set_iunrea = lambda path: calls.append(("iunrea", path)) or True
    fake_dyn.set_iunwri = lambda path: calls.append(("iunwri", path)) or True
    fake_dyn.set_iuncrd = lambda path: calls.append(("iuncrd", path)) or True

    kw = {
        "restart": True,
        "iunrea": "/tmp/in.res",
        "iunwri": "/tmp/out.res",
        "iuncrd": "/tmp/out.dcd",
        "nstep": 10,
    }
    with patch.dict(
        sys.modules,
        {"pycharmm": SimpleNamespace(dynamics=fake_dyn), "pycharmm.dynamics": fake_dyn},
    ):
        _apply_dynamics_io_setters(kw)
    assert calls == [
        ("iunrea", "/tmp/in.res"),
        ("iunwri", "/tmp/out.res"),
        ("iuncrd", "/tmp/out.dcd"),
    ]
    assert "iunrea" not in kw
    assert "iunwri" not in kw
    assert "iuncrd" not in kw


def test_apply_dynamics_io_setters_keeps_integer_units():
    import sys
    from types import SimpleNamespace
    from unittest.mock import patch

    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        _apply_dynamics_io_setters,
    )

    fake_dyn = SimpleNamespace(
        set_iunrea=lambda path: True,
        set_iunwri=lambda path: True,
        set_iuncrd=lambda path: True,
    )
    kw = {"iuncrd": 1, "iunwri": 2, "iunrea": -1, "nstep": 10}
    with patch.dict(
        sys.modules,
        {"pycharmm": SimpleNamespace(dynamics=fake_dyn), "pycharmm.dynamics": fake_dyn},
    ):
        _apply_dynamics_io_setters(kw)
    assert kw["iuncrd"] == 1
    assert kw["iunwri"] == 2
    assert kw["iunrea"] == -1
