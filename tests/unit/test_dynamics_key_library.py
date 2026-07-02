"""Dynamics KEY_LIBRARY C API routing (no ``dynamics`` script command)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
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


def test_configure_known_only_skips_integer_io_units():
    block = (
        Path("pycharmm/dynamics.py")
        .read_text(encoding="utf-8")
        .split("def _configure_known_only(")[1]
        .split("\ndef ")[0]
    )
    assert 'not isinstance(v, str)' in block
    assert '("iunwri", "iuncrd", "iunrea")' in block


def test_resolve_dynamics_init_velocities_uses_bussi_rescale_ladder():
    from unittest.mock import patch

    import numpy as np

    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        _resolve_dynamics_init_velocities,
    )

    v = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities._resolve_bussi_rescale_velocities",
        return_value=v,
    ) as rescale, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.velocities_are_cold",
        return_value=False,
    ):
        out = _resolve_dynamics_init_velocities(
            {"start": False, "iasvel": 0, "firstt": 12.0},
            restart_read_path="/tmp/heat.a.res",
        )
    rescale.assert_called_once()
    assert out is not None
    assert list(out.keys()) == ["vx", "vy", "vz"]
    assert out["vx"].shape == (2,)


def test_run_dynamics_passes_init_velocities_for_iasvel_zero_continuation():
    import sys
    from unittest.mock import MagicMock, patch

    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import run_dynamics

    v = np.ones((4, 3), dtype=float) * 42.0
    init = {"vx": v[:, 0], "vy": v[:, 1], "vz": v[:, 2]}
    fake_dyn = MagicMock()
    fake_pycharmm = MagicMock()
    fake_pycharmm.DynamicsScript = MagicMock(return_value=fake_dyn)
    with (
        patch.dict(sys.modules, {"pycharmm": fake_pycharmm}),
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.dynamics._dynamics_c_api_available",
            return_value=True,
        ),
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.dynamics._resolve_dynamics_init_velocities",
            return_value=init,
        ) as resolve_init,
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.dynamics._run_dynamics_via_c_api",
            return_value=fake_dyn,
        ) as run_capi,
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.dynamics._apply_dynamics_io_setters",
        ),
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.mirror_comparison_velocities_for_dynamics",
        ),
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.refresh_bussi_comp_velocity_handoff",
        ),
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.dynamics._init_velocities_handoff_looks_valid",
            return_value=True,
        ),
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.dynamics._validate_init_velocities_handoff",
        ),
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.dynamics._release_charmm_dynamics_api_buffers",
        ),
    ):
        run_dynamics(
            {
                "nstep": 50,
                "start": False,
                "iasvel": 0,
                "_skip_ase_cold_velocity_assign": True,
            }
        )
    resolve_init.assert_called_once()
    run_capi.assert_called_once()
    assert run_capi.call_args.kwargs.get("init_velocities") is not None
    passed = run_capi.call_args.kwargs["init_velocities"]
    assert np.allclose(passed["vx"], init["vx"])


def test_run_dynamics_bussi_passes_init_velocities_after_comp_refresh():
    import sys
    from unittest.mock import MagicMock, patch

    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import run_dynamics

    v = np.ones((4, 3), dtype=float) * 42.0
    init = {"vx": v[:, 0], "vy": v[:, 1], "vz": v[:, 2]}
    fake_dyn = MagicMock()
    fake_pycharmm = MagicMock()
    fake_pycharmm.DynamicsScript = MagicMock(return_value=fake_dyn)
    with (
        patch.dict(sys.modules, {"pycharmm": fake_pycharmm}),
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.dynamics._dynamics_c_api_available",
            return_value=True,
        ),
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.dynamics._resolve_dynamics_init_velocities",
            return_value=init,
        ),
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.dynamics._run_dynamics_via_c_api",
            return_value=fake_dyn,
        ) as run_capi,
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.dynamics._apply_dynamics_io_setters",
        ),
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.mirror_comparison_velocities_for_dynamics",
        ),
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.refresh_bussi_comp_velocity_handoff",
        ) as refresh_comp,
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.dynamics._init_velocities_handoff_looks_valid",
            return_value=True,
        ),
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.dynamics._validate_init_velocities_handoff",
        ),
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.dynamics._release_charmm_dynamics_api_buffers",
        ),
    ):
        run_dynamics(
            {
                "nstep": 50,
                "start": False,
                "iasvel": 0,
                "_skip_ase_cold_velocity_assign": True,
                "_bussi_ramp": {
                    "firstt": 20.0,
                    "finalt": 100.0,
                    "teminc": 0.08,
                    "ihtfrq": 50,
                },
            }
        )
    refresh_comp.assert_called_once()
    passed = run_capi.call_args.kwargs.get("init_velocities")
    assert passed is not None
    assert np.allclose(passed["vx"], init["vx"])
    assert np.allclose(passed["vy"], init["vy"])
    assert np.allclose(passed["vz"], init["vz"])


def test_apply_bussi_in_memory_continuation_uses_iasvel_one_without_c_api():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        _apply_bussi_in_memory_continuation_kw,
    )

    kw = {
        "iasvel": 0,
        "start": False,
        "_bussi_ramp": {
            "firstt": 0.0,
            "finalt": 100.0,
            "teminc": 0.5,
            "ihtfrq": 50,
        },
        "_bussi_global_step": 50,
    }
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._dynamics_c_api_available",
        return_value=False,
    ):
        _apply_bussi_in_memory_continuation_kw(kw)
    assert kw["iasvel"] == 1
    assert kw["firstt"] == 0.5
    assert kw.get("_skip_ase_cold_velocity_assign") is None


def test_validate_init_velocities_handoff_rejects_position_like_arrays():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        _validate_init_velocities_handoff,
    )

    pos_like = {
        "vx": np.full(100, 9500.0),
        "vy": np.zeros(100),
        "vz": np.zeros(100),
    }
    with pytest.raises(RuntimeError, match="Cartesian coordinates"):
        _validate_init_velocities_handoff(pos_like, quiet=True)


def test_finalize_init_velocities_handoff_falls_back_to_iasvel_one():
    from unittest.mock import patch

    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        _finalize_init_velocities_handoff,
    )

    pos_like = {
        "vx": np.full(4, 9500.0),
        "vy": np.zeros(4),
        "vz": np.zeros(4),
    }
    kw = {
        "iasvel": 0,
        "start": False,
        "_bussi_ramp": {
            "firstt": 10.0,
            "finalt": 50.0,
            "teminc": 0.08,
            "ihtfrq": 50,
        },
        "_bussi_global_step": 50,
    }
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities._resolve_bussi_rescale_velocities",
        return_value=np.full((4, 3), 9500.0),
    ):
        out = _finalize_init_velocities_handoff(
            kw,
            pos_like,
            handoff_vel=np.column_stack(
                [pos_like["vx"], pos_like["vy"], pos_like["vz"]]
            ),
            quiet=True,
        )
    assert out is None
    assert kw["iasvel"] == 1
    assert kw["firstt"] == pytest.approx(10.08)


def test_run_dynamics_c_api_path_invoked():
    from unittest.mock import MagicMock, patch

    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import run_dynamics

    kw = {
        "nstep": 10,
        "timestep": 0.00025,
        "nsavc": 5,
        "leap": True,
        "start": True,
        "iasvel": 1,
        "echeck": -1.0,
        "iunrea": -1,
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
