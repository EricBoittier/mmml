"""Dynamics KEY_LIBRARY C API routing (no ``dynamics`` script command)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import sys


def test_flatten_dynamics_script_logic():
    """Mirror of ``pycharmm.dynamics.flatten_dynamics_script`` (no libcharmm import)."""

    def flatten_dynamics_script(script: str) -> str:
        body = script.replace("-\n", " ").replace("\n", " ").strip()
        if body.lower().startswith("dynamics "):
            body = body[len("dynamics ") :]
        return " ".join(body.split())

    script = (
        "dynamics timestep 0.00025 -\n"
        " nstep 8000 -\n"
        " leap -\n"
        " hoover reft 300.0 -\n"
        " tmass 160\n"
    )
    flat = flatten_dynamics_script(script)
    assert flat.startswith("timestep 0.00025")
    assert "leap" in flat
    assert "hoover reft 300.0" in flat
    assert "tmass 160" in flat


def test_run_dynamics_uses_c_api_when_available():
    block = (
        Path("mmml/interfaces/pycharmmInterface/mlpot/dynamics.py")
        .read_text(encoding="utf-8")
        .split("def run_dynamics(")[1]
        .split("\ndef ")[0]
    )
    assert "_dynamics_c_api_available" in block
    assert "_run_dynamics_via_c_api" in block


def test_run_dynamics_c_api_path_invoked():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import run_dynamics

    kw = {
        "nstep": 10,
        "timestep": 0.00025,
        "nsavc": 5,
        "leap": True,
        "start": True,
        "iasvel": 1,
        "echeck": -1.0,
    }
    fake_dyn = MagicMock()
    fake_pycharmm = MagicMock()
    fake_pycharmm.DynamicsScript = MagicMock(return_value=fake_dyn)
    with (
        patch.dict("sys.modules", {"pycharmm": fake_pycharmm}),
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.dynamics._dynamics_c_api_available",
            return_value=True,
        ),
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.dynamics._run_dynamics_via_c_api",
            return_value=fake_dyn,
        ) as run_c_api,
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.dynamics._apply_dynamics_io_setters",
        ),
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.dynamics._release_charmm_dynamics_api_buffers",
        ),
    ):
        out = run_dynamics(kw)
    assert out is fake_dyn
    run_c_api.assert_called_once()
