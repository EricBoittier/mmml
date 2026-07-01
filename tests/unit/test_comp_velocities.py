"""Unit tests for COMP velocity helpers (mocked PyCHARMM)."""

from __future__ import annotations

import types
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest

from mmml.interfaces.pycharmmInterface.mlpot.comp_velocities import (
    DEFAULT_COMP_FORCE_SCALE,
    apply_selective_force_damp_recipe,
    build_high_force_selection,
    clear_comp_for_production,
    clear_comparison_coordinates,
    coor_get_comparison_capi,
    coor_set_comparison_capi,
    get_comparison_array,
    mirror_comparison_velocities_for_dynamics,
    prepare_comp_for_heat,
    prepare_comp_for_iasvel0,
    set_comparison_array,
    sync_comparison_velocities_akma,
    sync_comparison_velocities_from_comparison,
    sync_comparison_velocities_from_restart,
    zero_comparison_scalars,
)


def _mock_pycharmm_module(n_atoms: int = 4):
    mod = MagicMock()
    mod.psf.get_natom.return_value = n_atoms
    mod.coor.get_comparison.return_value = pd.DataFrame(
        {
            "x": np.zeros(n_atoms),
            "y": np.zeros(n_atoms),
            "z": np.zeros(n_atoms),
            "w": np.zeros(n_atoms),
        }
    )
    mod.coor.get_forces.return_value = pd.DataFrame(
        {
            "dx": np.array([0.5, 2.0, 0.1, 5.0]),
            "dy": np.zeros(n_atoms),
            "dz": np.zeros(n_atoms),
        }
    )
    mod.select.find.return_value = 0
    mod.select.none_selection.return_value = (False,) * n_atoms
    sel = MagicMock()
    sel.get_n_selected.return_value = 2
    sel.store.return_value = "highf"
    mod.SelectAtoms.return_value = sel
    return mod


@patch("mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.coor_set_comparison_capi")
def test_set_comparison_array_roundtrip(mock_set_capi):
    mock_set_capi.return_value = 2
    values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    set_comparison_array(values)
    mock_set_capi.assert_called_once_with(values)


@patch("mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.coor_get_comparison_capi")
def test_get_comparison_array_uses_capi(mock_get_capi):
    mock_get_capi.return_value = np.array([[1.0, 2.0, 3.0, 0.0]])
    out = get_comparison_array()
    mock_get_capi.assert_called_once()
    assert out.shape == (1, 4)
    assert out[0, 0] == pytest.approx(1.0)


def test_coor_set_comparison_capi():
    fake_charmm = MagicMock()
    fake_charmm.coor_set_comparison.return_value = 2
    fake_lib = types.SimpleNamespace(charmm=fake_charmm)
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.comp_velocities._charmm_lib",
        return_value=fake_lib,
    ):
        out = coor_set_comparison_capi(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    assert out == 2
    fake_charmm.coor_set_comparison.assert_called_once()


@patch("mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.coor_set_comparison_capi")
def test_sync_comparison_velocities_akma(mock_set_capi):
    sync_comparison_velocities_akma(np.array([[100.0, 0.0, 0.0], [0.0, 200.0, 0.0]]))
    mock_set_capi.assert_called_once()
    passed = mock_set_capi.call_args[0][0]
    assert passed.shape == (2, 4)
    assert passed[0, 0] == pytest.approx(100.0)
    assert passed[1, 1] == pytest.approx(200.0)


@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.sync_comparison_velocities_from_main",
    return_value=True,
)
def test_mirror_comparison_velocities_for_dynamics_syncs_when_iasvel_zero(mock_sync):
    kw = {"iasvel": 0, "start": False}
    mirror_comparison_velocities_for_dynamics(kw)
    mock_sync.assert_called_once()


@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.comparison_matches_main_positions",
    return_value=True,
)
@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.sync_comparison_velocities_from_restart",
    return_value=False,
)
@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.sync_comparison_velocities_from_comparison",
    return_value=False,
)
@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.sync_comparison_velocities_from_main",
    return_value=False,
)
def test_mirror_comparison_velocities_for_dynamics_raises_when_comp_is_positions(
    mock_main, mock_comp, mock_restart, mock_match
):
    kw = {"iasvel": 0, "start": False}
    with pytest.raises(RuntimeError, match="COMP holds main coordinates"):
        mirror_comparison_velocities_for_dynamics(kw, restart_read_path="/tmp/fake.res")


@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.comparison_matches_main_positions",
    return_value=False,
)
@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.sync_comparison_velocities_from_main",
    return_value=False,
)
@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.sync_comparison_velocities_from_comparison",
    return_value=True,
)
def test_mirror_comparison_velocities_for_dynamics_uses_warm_comp(
    mock_comp, mock_main, mock_match
):
    mirror_comparison_velocities_for_dynamics({"iasvel": 0, "start": False})
    mock_main.assert_called_once()
    mock_comp.assert_called_once()


@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.sync_comparison_velocities_from_main",
    return_value=False,
)
@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.sync_comparison_velocities_from_comparison",
    return_value=False,
)
@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.sync_comparison_velocities_from_restart",
    return_value=True,
)
def test_mirror_comparison_velocities_for_dynamics_uses_restart(mock_restart, mock_comp, mock_main):
    mirror_comparison_velocities_for_dynamics(
        {"iasvel": 0, "start": False},
        restart_read_path="/tmp/handoff.res",
    )
    mock_restart.assert_called_once_with("/tmp/handoff.res")


@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.sync_comparison_velocities_from_main",
)
def test_mirror_comparison_velocities_for_dynamics_skips_when_iasvel_one(mock_sync):
    mirror_comparison_velocities_for_dynamics({"iasvel": 1, "start": False})
    mock_sync.assert_not_called()


@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.sync_comparison_velocities_from_main",
)
def test_mirror_comparison_velocities_for_dynamics_skips_when_start_true(mock_sync):
    mirror_comparison_velocities_for_dynamics({"iasvel": 0, "start": True})
    mock_sync.assert_not_called()


@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.assert_comparison_holds_velocities_not_positions",
)
@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.sync_charmm_velocities_akma",
)
def test_refresh_bussi_comp_velocity_handoff_syncs_and_validates(
    mock_sync_main, mock_assert
):
    from mmml.interfaces.pycharmmInterface.mlpot.comp_velocities import (
        refresh_bussi_comp_velocity_handoff,
    )

    vel = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=float)
    refresh_bussi_comp_velocity_handoff(vel, context="test")
    mock_sync_main.assert_called_once()
    mock_assert.assert_called_once_with(context="test")


def test_assert_comparison_holds_velocities_not_positions_raises_when_comp_is_main(
    monkeypatch,
):
    from mmml.interfaces.pycharmmInterface.mlpot.comp_velocities import (
        assert_comparison_holds_velocities_not_positions,
    )

    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.comparison_matches_main_positions",
        lambda **_: True,
    )
    with pytest.raises(RuntimeError, match="COMP holds main coordinates"):
        assert_comparison_holds_velocities_not_positions(context="chunk 2")


def test_coor_get_comparison_capi():
    fake_charmm = MagicMock()

    def _fill(x, y, z, w):
        for i in range(2):
            x[i] = float(i + 1)
            y[i] = float(i + 2)
            z[i] = float(i + 3)
            w[i] = 0.0

    fake_charmm.coor_get_comparison.side_effect = _fill
    fake_lib = types.SimpleNamespace(charmm=fake_charmm)
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.comp_velocities._charmm_lib",
        return_value=fake_lib,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.comp_velocities._import_pycharmm",
    ) as mock_import:
        mock_import.return_value.psf.get_natom.return_value = 2
        out = coor_get_comparison_capi()
    assert out.shape == (2, 4)
    assert out[0, 0] == pytest.approx(1.0)
    assert out[1, 2] == pytest.approx(4.0)


@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.sync_comparison_velocities_akma",
)
@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.charmm_velocities_akma_for_thermostat",
)
@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.velocities_are_cold",
    return_value=False,
)
def test_sync_comparison_velocities_from_main_warm(mock_cold, mock_vel, mock_sync):
    from mmml.interfaces.pycharmmInterface.mlpot.comp_velocities import (
        sync_comparison_velocities_from_main,
    )

    vel = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=float)
    mock_vel.return_value = vel
    assert sync_comparison_velocities_from_main() is True
    mock_sync.assert_called_once()
    np.testing.assert_allclose(mock_sync.call_args[0][0], vel)


@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.charmm_velocities_akma_for_thermostat",
    return_value=None,
)
def test_sync_comparison_velocities_from_main_missing(mock_vel):
    from mmml.interfaces.pycharmmInterface.mlpot.comp_velocities import (
        sync_comparison_velocities_from_main,
    )

    assert sync_comparison_velocities_from_main() is False


@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.velocities_are_cold",
    return_value=False,
)
@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.comparison_matches_main_positions",
    return_value=True,
)
@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.coor_get_comparison_capi",
)
def test_sync_comparison_velocities_from_comparison_rejects_positions(
    mock_get, mock_pos, mock_cold
):
    mock_get.return_value = np.array([[1.0, 2.0, 3.0, 0.0]])
    assert sync_comparison_velocities_from_comparison() is False


@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.comparison_velocities_akma",
)
@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.comp_velocities._import_pycharmm",
)
def test_comparison_matches_main_positions_true(mock_import, mock_comp):
    mock_import.return_value.coor.get_positions.return_value = pd.DataFrame(
        {"x": [1.0], "y": [2.0], "z": [3.0]}
    )
    mock_comp.return_value = np.array([[1.0, 2.0, 3.0]])
    from mmml.interfaces.pycharmmInterface.mlpot.comp_velocities import (
        comparison_matches_main_positions,
    )

    assert comparison_matches_main_positions() is True


@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.comparison_matches_main_positions",
    return_value=False,
)
@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.velocities_are_cold",
    return_value=False,
)
@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.comparison_velocities_akma",
)
def test_sync_comparison_velocities_from_comparison_warm(mock_comp, mock_cold, mock_pos):
    mock_comp.return_value = np.array([[10.0, 0.0, 0.0], [0.0, 20.0, 0.0]])
    assert sync_comparison_velocities_from_comparison() is True
    mock_comp.assert_called_once()


@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.comparison_matches_main_positions",
    return_value=False,
)
@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.velocities_are_cold",
    return_value=False,
)
@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.velocities_are_pathological",
    return_value=True,
)
@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.comparison_velocities_akma",
)
def test_sync_comparison_velocities_from_comparison_rejects_pathological(
    mock_comp, mock_path, mock_cold, mock_pos
):
    mock_comp.return_value = np.array([[1.0e6, 0.0, 0.0], [0.0, 2.0e6, 0.0]])
    assert sync_comparison_velocities_from_comparison() is False


@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.comparison_matches_main_positions",
    return_value=False,
)
@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.velocities_are_cold",
    return_value=False,
)
@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.velocities_are_pathological",
    return_value=False,
)
@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.comparison_velocities_akma",
)
def test_sync_comparison_velocities_from_comparison_rejects_spatial_coords(
    mock_comp, mock_path, mock_cold, mock_pos
):
    mock_comp.return_value = np.array([[9999.0, 0.0, 0.0], [0.0, 9999.0, 0.0]])
    assert sync_comparison_velocities_from_comparison() is False


@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.comparison_matches_main_positions",
    return_value=False,
)
@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.velocities_are_cold",
    return_value=True,
)
@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.comparison_velocities_akma",
)
def test_sync_comparison_velocities_from_comparison_cold(mock_comp, mock_cold, mock_pos):
    mock_comp.return_value = np.zeros((2, 3))
    assert sync_comparison_velocities_from_comparison() is False


@patch("mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.sync_comparison_velocities_akma")
@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation.read_restart_velocities",
)
@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.velocities_are_cold",
    return_value=False,
)
def test_sync_comparison_velocities_from_restart_warm(mock_cold, mock_read, mock_sync, tmp_path):
    res = tmp_path / "handoff.res"
    res.write_text("REST\n", encoding="ascii")
    vel = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    mock_read.return_value = vel
    assert sync_comparison_velocities_from_restart(res) is True
    mock_sync.assert_called_once()
    np.testing.assert_allclose(mock_sync.call_args[0][0], vel)


@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation.read_restart_velocities",
    return_value=None,
)
def test_sync_comparison_velocities_from_restart_missing(mock_read):
    assert sync_comparison_velocities_from_restart(None) is False
    assert sync_comparison_velocities_from_restart("/no/such/file.res") is False


@patch("mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.coor_set_comparison_capi")
@patch("mmml.interfaces.pycharmmInterface.mlpot.comp_velocities._import_pycharmm")
def test_clear_comparison_coordinates_zeros_comp(mock_import, mock_set_capi):
    mock_import.return_value.psf.get_natom.return_value = 3
    clear_comparison_coordinates()
    mock_set_capi.assert_called_once()
    zeros = mock_set_capi.call_args[0][0]
    assert zeros.shape == (3, 4)
    assert np.all(zeros == 0.0)


@patch("mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.run_charmm_script")
def test_zero_comparison_scalars(mock_script):
    zero_comparison_scalars()
    assert mock_script.call_count == 4
    scripts = [c.args[0] for c in mock_script.call_args_list]
    assert scripts == [
        "scalar xcomp set 0 select all end",
        "scalar ycomp set 0 select all end",
        "scalar zcomp set 0 select all end",
        "scalar wcomp set 0 select all end",
    ]


@patch("mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.run_charmm_script")
@patch("mmml.interfaces.pycharmmInterface.mlpot.comp_velocities._import_pycharmm")
def test_build_high_force_selection(mock_import, mock_script):
    pycharmm = _mock_pycharmm_module()
    mock_import.return_value = pycharmm
    highf_sel = MagicMock()
    highf_sel.get_selection.return_value = (False, True, False, True)
    highf_sel.store.return_value = "highf"
    pycharmm.SelectAtoms.return_value = highf_sel
    name, n_sel = build_high_force_selection(1.0, store_name="highf")
    assert name == "highf"
    assert n_sel == 2
    pycharmm.SelectAtoms.assert_called_once()
    _, kwargs = pycharmm.SelectAtoms.call_args
    assert kwargs["atom_nums"] == [1, 3]
    highf_sel.store.assert_called_once_with(name="highf")
    mock_script.assert_not_called()


@patch("mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.run_charmm_script")
@patch("mmml.interfaces.pycharmmInterface.mlpot.comp_velocities._import_pycharmm")
def test_build_high_force_selection_excludes_hydrogen(mock_import, mock_script):
    pycharmm = _mock_pycharmm_module()
    mock_import.return_value = pycharmm
    highf_sel = MagicMock()
    heavy_only = MagicMock()
    heavy_only.get_selection.return_value = (False, True, False, False)
    heavy_only.store.return_value = "highf"
    highf_sel.__and__ = MagicMock(return_value=heavy_only)
    pycharmm.SelectAtoms.side_effect = [highf_sel, MagicMock()]
    name, n_sel = build_high_force_selection(1.0, exclude_hydrogen=True)
    assert name == "highf"
    assert n_sel == 1
    highf_sel.__and__.assert_called_once()
    heavy_only.store.assert_called_once_with(name="highf")


@patch("mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.run_charmm_script")
@patch("mmml.interfaces.pycharmmInterface.mlpot.comp_velocities._import_pycharmm")
def test_build_high_force_selection_hydrogen_only(mock_import, mock_script):
    pycharmm = _mock_pycharmm_module()
    mock_import.return_value = pycharmm
    highf_sel = MagicMock()
    h_only = MagicMock()
    h_only.get_selection.return_value = (True, False, False, False)
    h_only.store.return_value = "highf"
    highf_sel.__and__ = MagicMock(return_value=h_only)
    pycharmm.SelectAtoms.side_effect = [highf_sel, MagicMock()]
    name, n_sel = build_high_force_selection(1.0, hydrogen_only=True)
    assert name == "highf"
    assert n_sel == 1
    h_only.store.assert_called_once_with(name="highf")


@patch("mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.unstore_selection")
@patch("mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.build_high_force_selection")
@patch("mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.zero_comparison_scalars")
@patch("mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.run_charmm_script")
def test_apply_selective_force_damp_recipe_highf_only(
    mock_script,
    mock_zero,
    mock_build,
    mock_unstore,
):
    mock_build.return_value = ("highf", 2)
    n = apply_selective_force_damp_recipe(
        min_force_kcalmol_A=1.0,
        force_scale=0.01,
    )
    assert n == 2
    mock_zero.assert_called_once_with("all")
    scripts = [c.args[0] for c in mock_script.call_args_list]
    assert all(" select highf end" in s for s in scripts if "copy" in s or " mult " in s)
    assert not any(" select all end" in s and "copy" in s for s in scripts)
    assert scripts[-1] == "scalar wcomp set 0 select all end"
    mock_unstore.assert_called_once_with("highf")


@patch("mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.unstore_selection")
@patch("mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.build_high_force_selection")
@patch("mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.zero_comparison_scalars")
@patch("mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.run_charmm_script")
def test_apply_selective_force_damp_recipe_empty_highf(
    mock_script,
    mock_zero,
    mock_build,
    mock_unstore,
):
    mock_build.return_value = ("highf", 0)
    n = apply_selective_force_damp_recipe(min_force_kcalmol_A=10.0)
    assert n == 0
    mock_zero.assert_called_once_with("all")
    assert mock_script.call_args_list == [call("scalar wcomp set 0 select all end")]
    mock_unstore.assert_called_once_with("highf")


@patch("mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.apply_selective_force_damp_recipe")
@patch("mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.zero_comparison_scalars")
@patch("mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.run_charmm_script")
def test_prepare_comp_for_iasvel0_default_recipe(mock_script, mock_zero, mock_recipe):
    mock_recipe.return_value = 3
    n = prepare_comp_for_iasvel0(min_force_kcalmol_A=1.0, force_scale=DEFAULT_COMP_FORCE_SCALE)
    assert n == 3
    mock_script.assert_called_once_with("ENER")
    mock_recipe.assert_called_once()
    mock_zero.assert_not_called()


@patch("mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.apply_selective_force_damp_recipe")
@patch("mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.zero_comparison_scalars")
@patch("mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.run_charmm_script")
def test_prepare_comp_for_iasvel0_zero_only(mock_script, mock_zero, mock_recipe):
    n = prepare_comp_for_iasvel0(zero_only=True)
    assert n == 0
    mock_script.assert_has_calls(
        [
            call("ENER"),
            call("scalar wcomp set 0 select all end"),
        ]
    )
    mock_zero.assert_called_once_with("all")
    mock_recipe.assert_not_called()


@patch("mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.prepare_comp_for_iasvel0")
def test_prepare_comp_for_heat_targets_hydrogen(mock_prepare):
    mock_prepare.return_value = 4
    n = prepare_comp_for_heat(min_force_kcalmol_A=2.0, force_scale=0.02)
    assert n == 4
    mock_prepare.assert_called_once_with(
        min_force_kcalmol_A=2.0,
        force_scale=0.02,
        hydrogen_only=True,
        exclude_hydrogen=False,
    )


@patch("mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.clear_comparison_coordinates")
def test_clear_comp_for_production(mock_clear_coords):
    clear_comp_for_production()
    mock_clear_coords.assert_called_once()


@patch("mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.clear_comparison_coordinates")
def test_clear_comp_for_production_honors_quiet(mock_clear_coords):
    clear_comp_for_production(quiet=True)
    mock_clear_coords.assert_called_once()
