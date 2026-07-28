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


def test_adumb_rc_walls_backend_defaults_to_resd(monkeypatch: pytest.MonkeyPatch) -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.restraints import adumb_rc_walls_backend

    monkeypatch.delenv("MMML_ADUMB_RC_WALL_BACKEND", raising=False)
    monkeypatch.delenv("MMML_ADUMB_RC_MMFP_WALLS", raising=False)
    assert adumb_rc_walls_backend() == "resd"


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
        side_effect=[0, 1],
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


def test_measure_adumb_bond_difference_xi() -> None:
    x = np.array([0.0, 2.0, 5.0], dtype=np.float64)  # N1, C1, CL1 along x
    y = np.zeros(3, dtype=np.float64)
    z = np.zeros(3, dtype=np.float64)

    def _idx(name: str) -> int:
        return {"N1": 0, "C1": 1, "CL1": 2}[name]

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.restraints._unique_atom_index_by_name",
        side_effect=_idx,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.restraints._positions_xyz",
        return_value=(x, y, z),
    ):
        from mmml.interfaces.pycharmmInterface.mlpot.restraints import (
            measure_adumb_bond_difference_xi,
        )

        # r(Cl-C)=3, r(C-N)=2 → ξ = 1
        assert measure_adumb_bond_difference_xi() == pytest.approx(1.0)


def test_prepare_adumb_rc_rewinds_when_xi_out_of_window(tmp_path) -> None:
    guard = AdumbRcGuard(
        rcmax=8.0,
        rcwall=500.0,
        pairs=(("CL1", "C1"), ("C1", "N1")),
        umb_min=-3.0,
        umb_max=3.0,
    )
    stage = tmp_path / "heat.res"
    stage.touch()
    good = tmp_path / "heat.0021.res"
    good.write_text("!X\n", encoding="utf-8")
    # First call: ξ out of window; after restore: ξ ok and distances ok.
    xi_vals = iter([-3.5, -1.0])
    dists = iter(
        [
            {"CL1-C1": 1.8, "C1-N1": 5.3},
            {"CL1-C1": 1.8, "C1-N1": 2.8},
        ]
    )
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.restraints.measure_adumb_bond_difference_xi",
        side_effect=lambda *_a, **_k: next(xi_vals),
    ), mock.patch(
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


def test_prepare_adumb_near_wall_triggers_force_rewind(tmp_path) -> None:
    """RC at RESD onset must rewind before the next iasvel=1 chunk."""
    from mmml.interfaces.pycharmmInterface.mlpot.restraints import (
        AdumbRcGuard,
        prepare_adumb_rc_before_overlap_chunk,
    )

    guard = AdumbRcGuard(
        rcmax=12.0,
        rcwall=2000.0,
        pairs=(("CL1", "C1"), ("C1", "N1")),
        wall_margin=2.5,  # onset at 9.5 Å
    )
    stage = tmp_path / "heat.res"
    stage.write_text("heat\n", encoding="utf-8")
    good = tmp_path / "heat.0020.res"
    good.write_text("ok\n", encoding="utf-8")
    # First measure: at wall; after restore: safe.
    dists = iter(
        [
            {"CL1-C1": 9.6, "C1-N1": 3.0},
            {"CL1-C1": 4.0, "C1-N1": 3.0},
        ]
    )
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.restraints.measure_adumb_bond_difference_xi",
        return_value=None,
    ), mock.patch(
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
            chunk_index=21,
            n_chunks=400,
            final_restart=stage,
        )
    assert retry is True
    restore.assert_called()


def test_prepare_adumb_force_rewind_returns_false_without_raise(tmp_path) -> None:
    """force_rewind must not raise when no usable restart exists (warn mode)."""
    from mmml.interfaces.pycharmmInterface.mlpot.restraints import (
        AdumbRcGuard,
        prepare_adumb_rc_before_overlap_chunk,
    )

    guard = AdumbRcGuard(
        rcmax=8.0,
        rcwall=500.0,
        pairs=(("CL1", "C1"), ("C1", "N1")),
    )
    stage = tmp_path / "heat.res"
    # No numbered / baseline files.
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.restraints.measure_adumb_bond_difference_xi",
        return_value=None,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.restraints.measure_adumb_rc_distances",
        return_value={"CL1-C1": 3.0, "C1-N1": 3.0},
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.restraints.install_adumb_rxncor_distance_walls",
    ):
        retry = prepare_adumb_rc_before_overlap_chunk(
            guard,
            overlap_context="HEAT",
            chunk_index=3,
            n_chunks=400,
            final_restart=stage,
            force_rewind=True,
        )
    assert retry is False


def test_prepare_adumb_force_rewind_falls_back_to_baseline(tmp_path) -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.restraints import (
        AdumbRcGuard,
        prepare_adumb_rc_before_overlap_chunk,
    )

    guard = AdumbRcGuard(
        rcmax=8.0,
        rcwall=500.0,
        pairs=(("CL1", "C1"), ("C1", "N1")),
    )
    stage = tmp_path / "heat.res"
    stage.write_text("heat\n", encoding="utf-8")
    baseline = tmp_path / "baseline.res"
    baseline.write_text("base\n", encoding="utf-8")
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.restraints.measure_adumb_bond_difference_xi",
        return_value=None,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.restraints.measure_adumb_rc_distances",
        return_value={"CL1-C1": 2.0, "C1-N1": 2.5},
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.restraints.install_adumb_rxncor_distance_walls",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery.restore_charmm_state_from_restart",
    ) as restore:
        retry = prepare_adumb_rc_before_overlap_chunk(
            guard,
            overlap_context="HEAT",
            chunk_index=3,
            n_chunks=400,
            final_restart=stage,
            force_rewind=True,
        )
    assert retry is True
    assert restore.call_args_list[0].args[0] in (stage, baseline)


def test_parse_adumb_umbrella_bounds_negative_min() -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        parse_adumb_umbrella_bounds,
    )

    script = """
    rxncor define rdif combination rcl 1.0 rcn -1.0
    umbrella rxncor nresol 40 trig 0 poly 6 min -6.0 max 6.0 name rdif
    """
    assert parse_adumb_umbrella_bounds(script) == (-6.0, 6.0)


def test_parse_adumb_umbrella_bounds_skips_distance_only_rcl() -> None:
    """Distance umbrella on rcl must not arm the xi=r(ClC)-r(CN) soft window."""
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        adumb_umbrella_is_bond_difference,
        parse_adumb_umbrella_bounds,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.restraints import (
        adumb_rc_wall_pairs_for_name,
    )

    script = """
    rxncor define rcl distance pcl pc
    rxncor define rcn distance pc pn
    rxncor define rdif combination rcl 1.0 rcn -1.0
    rxncor set nrxn 1 rcl
    umbrella rxncor nresol 40 trig 0 poly 6 min 0.0 max 6.0 name rcl
    """
    assert adumb_umbrella_is_bond_difference(script) is False
    assert parse_adumb_umbrella_bounds(script) == (None, None)
    assert adumb_rc_wall_pairs_for_name("rcl") == (("CL1", "C1"),)
    assert adumb_rc_wall_pairs_for_name("rcn") == (("C1", "N1"),)
    assert adumb_rc_wall_pairs_for_name("rdif") == (("CL1", "C1"), ("C1", "N1"))
    from mmml.interfaces.pycharmmInterface.mlpot.restraints import (
        adumb_rc_wall_pairs_for_names,
    )

    assert adumb_rc_wall_pairs_for_names(["rcl", "rcn"]) == (
        ("CL1", "C1"),
        ("C1", "N1"),
    )


def test_charmm_output_indicates_failure_detects_unrecognized() -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.restraints import (
        _charmm_output_indicates_failure,
    )

    log = "***** Unrecognized command: noe *****"
    assert _charmm_output_indicates_failure(log) == "CHARMM reported unrecognized command(s)"
