"""NPT CPT stability chunking and post-dynamics finite-state checks."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd
import pytest

from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
    DEFAULT_CPT_DYNAMICS_CHUNK_NSTEP,
    _cpt_stability_chunk_nstep,
    _cpt_subchunk_use_in_memory_handoff,
    _dynamics_chunk_state_corrupt,
)
from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
    charmm_coordinates_are_finite,
    charmm_dynamics_energy_is_finite,
    charmm_dynamics_state_is_finite,
    validate_charmm_dynamics_state_after_chunk,
)
from tests.unit.pycharmm_stubs import fake_pycharmm_modules


def test_default_cpt_chunk_nstep_is_250():
    assert DEFAULT_CPT_DYNAMICS_CHUNK_NSTEP == 250


def test_cpt_stability_chunk_nstep_skips_short_and_non_cpt():
    assert _cpt_stability_chunk_nstep({"cpt": True}, 200) is None
    assert _cpt_stability_chunk_nstep({"cpt": False}, 5000) is None


def test_cpt_stability_chunk_nstep_for_long_npt():
    assert _cpt_stability_chunk_nstep({"cpt": True}, 1000) == 250


def test_cpt_stability_chunk_nstep_env_override(monkeypatch):
    monkeypatch.setenv("MMML_CPT_DYNAMICS_CHUNK_NSTEP", "100")
    assert _cpt_stability_chunk_nstep({"cpt": True}, 500) == 100


def test_cpt_subchunk_defaults_to_in_memory_handoff(monkeypatch):
    monkeypatch.delenv("MMML_CPT_READYN_SUBCHUNK", raising=False)
    assert _cpt_subchunk_use_in_memory_handoff() is True


def test_cpt_subchunk_readyn_handoff_opt_in(monkeypatch):
    monkeypatch.setenv("MMML_CPT_READYN_SUBCHUNK", "1")
    assert _cpt_subchunk_use_in_memory_handoff() is False


def test_charmm_coordinates_are_finite_detects_nan():
    pos = pd.DataFrame({"x": [0.0, np.nan], "y": [0.0, 0.0], "z": [0.0, 0.0]})
    fake_coor = mock.MagicMock()
    fake_coor.get_positions.return_value = pos
    with fake_pycharmm_modules(coor=fake_coor):
        assert not charmm_coordinates_are_finite()


def test_charmm_dynamics_energy_is_finite_detects_nan():
    row = pd.DataFrame([{"USER": -1.0, "TOTE": np.nan}])
    fake_energy = mock.MagicMock()
    fake_energy.get_energy.return_value = row
    with fake_pycharmm_modules(energy=fake_energy):
        assert not charmm_dynamics_energy_is_finite()


def test_charmm_dynamics_state_is_finite_requires_both():
    pos = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 0.0], "z": [0.0, 0.0]})
    row = pd.DataFrame([{"USER": -1.0}])
    fake_coor = mock.MagicMock()
    fake_coor.get_positions.return_value = pos
    fake_energy = mock.MagicMock()
    fake_energy.get_energy.return_value = row
    with fake_pycharmm_modules(coor=fake_coor, energy=fake_energy):
        assert charmm_dynamics_state_is_finite()


def test_validate_charmm_dynamics_state_raises_on_corruption():
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation.charmm_coordinates_are_finite",
        return_value=False,
    ):
        with pytest.raises(RuntimeError, match="non-finite"):
            validate_charmm_dynamics_state_after_chunk(context="EQUI")


def test_dynamics_chunk_state_corrupt_checks_memory_and_restart(tmp_path):
    bad_restart = tmp_path / "bad.res"
    bad_restart.write_text("REST\n!X, Y, Z\nNAN\n", encoding="utf-8")
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation.charmm_dynamics_state_is_finite",
        return_value=True,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation.restart_has_nonfinite_coordinates",
        return_value=True,
    ):
        assert _dynamics_chunk_state_corrupt(
            overlap_context="EQUI",
            restart_path=bad_restart,
        )


def test_materialize_cpt_subchunk_skips_nstep0_velocity_assign(tmp_path):
    """CPT sub-chunk handoff must snapshot in-memory barostat state, not nstep=0 assign."""
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        _materialize_cpt_subchunk_restart_handoff,
    )

    write_path = tmp_path / "heat.cptsc_a.res"
    chunk_kw = {"cpt": True, "hoover reft": 300.0, "timestep": 0.00025}

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics.assign_velocities_at_temperature",
    ) as assign, mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery.rewrite_dynamics_restart_validated",
        return_value=True,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._valid_restart_file",
        return_value=write_path,
    ):
        _materialize_cpt_subchunk_restart_handoff(
            write_path,
            global_step=250,
            overlap_context="HEAT",
            mlpot_ctx=mock.Mock(),
            chunk_kw=chunk_kw,
        )

    assign.assert_not_called()
