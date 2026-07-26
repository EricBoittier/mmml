"""ADUMB traced-RC preflight vs umbrella max (UM1RXN guard)."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.mlpot.restraints import (
    AdumbRcGuard,
    adumb_rc_wall_droff,
    check_adumb_rc_before_overlap_chunk,
    measure_adumb_rc_distances,
    prepare_adumb_rc_before_overlap_chunk,
)


def test_adumb_rc_walls_backend_defaults_to_noe(monkeypatch: pytest.MonkeyPatch) -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.restraints import adumb_rc_walls_backend

    monkeypatch.delenv("MMML_ADUMB_RC_WALL_BACKEND", raising=False)
    monkeypatch.delenv("MMML_ADUMB_RC_MMFP_WALLS", raising=False)
    assert adumb_rc_walls_backend() == "noe"


def test_adumb_rc_walls_backend_legacy_mmfp_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.restraints import adumb_rc_walls_backend

    monkeypatch.setenv("MMML_ADUMB_RC_MMFP_WALLS", "1")
    assert adumb_rc_walls_backend() == "mmfp"


def test_adumb_rc_wall_droff_below_max() -> None:
    assert adumb_rc_wall_droff(8.0, margin=0.75) == pytest.approx(7.25)


def test_measure_adumb_rc_distances_from_coords() -> None:
    x = np.array([0.0, 3.0, 0.0], dtype=np.float64)
    y = np.zeros(3, dtype=np.float64)
    z = np.zeros(3, dtype=np.float64)
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.restraints._unique_atom_index_by_name",
        side_effect=[1, 2],
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.restraints._positions_xyz",
        return_value=(x, y, z),
    ):
        dists = measure_adumb_rc_distances((("CL1", "C1"),))
    assert dists["CL1-C1"] == pytest.approx(3.0)


def test_check_adumb_rc_raises_at_hard_limit() -> None:
    guard = AdumbRcGuard(rcmax=8.0, rcwall=500.0, pairs=(("CL1", "C1"),))
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.restraints.prepare_adumb_rc_before_overlap_chunk",
        return_value=True,
    ):
        with pytest.raises(RuntimeError, match="internal: prepare_adumb_rc"):
            check_adumb_rc_before_overlap_chunk(
                guard,
                overlap_context="HEAT",
                chunk_index=43,
                n_chunks=400,
            )


def test_prepare_adumb_rc_rewinds_from_numbered_restart(tmp_path) -> None:
    guard = AdumbRcGuard(rcmax=8.0, rcwall=500.0, pairs=(("CL1", "C1"),))
    stage = tmp_path / "heat.res"
    stage.touch()
    good = tmp_path / "heat.0021.res"
    good.write_text("!X\n", encoding="utf-8")
    dists = iter([{"C1-N1": 8.475}, {"C1-N1": 7.5}])
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.restraints.measure_adumb_rc_distances",
        side_effect=lambda *_a, **_k: next(dists),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.restraints.install_adumb_rxncor_distance_walls",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery.restore_charmm_state_from_restart",
    ) as restore:
        retry = prepare_adumb_rc_before_overlap_chunk(
            guard,
            overlap_context="HEAT",
            chunk_index=22,
            n_chunks=400,
            final_restart=stage,
        )
    assert retry is True
    restore.assert_called_once_with(good)
