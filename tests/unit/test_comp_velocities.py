"""Unit tests for COMP velocity helpers (mocked PyCHARMM)."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest

from mmml.interfaces.pycharmmInterface.mlpot.comp_velocities import (
    DEFAULT_COMP_FORCE_SCALE,
    apply_selective_force_damp_recipe,
    build_high_force_selection,
    get_comparison_array,
    prepare_comp_for_iasvel0,
    set_comparison_array,
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
    sel = MagicMock()
    sel.get_n_selected.return_value = 2
    sel.store.return_value = "highf"
    mod.SelectAtoms.return_value = sel
    return mod


@patch("mmml.interfaces.pycharmmInterface.mlpot.comp_velocities._import_pycharmm")
def test_set_comparison_array_roundtrip(mock_import):
    pycharmm = _mock_pycharmm_module()
    mock_import.return_value = pycharmm
    values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    set_comparison_array(values)
    df = pycharmm.coor.set_comparison.call_args[0][0]
    assert list(df.columns) == ["x", "y", "z", "w"]
    assert df["x"].tolist() == [1.0, 4.0]
    assert df["w"].tolist() == [0.0, 0.0]

    pycharmm.coor.get_comparison.return_value = df
    out = get_comparison_array()
    assert out.shape == (2, 4)
    assert out[0, 0] == pytest.approx(1.0)
    assert out[0, 3] == pytest.approx(0.0)


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
    name, n_sel = build_high_force_selection(1.0, store_name="highf")
    assert name == "highf"
    assert n_sel == 2
    pycharmm.SelectAtoms.assert_called_once()
    _, kwargs = pycharmm.SelectAtoms.call_args
    assert kwargs["atom_nums"] == [1, 3]
    pycharmm.SelectAtoms.return_value.store.assert_called_once_with(name="highf")
    mock_script.assert_not_called()


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
