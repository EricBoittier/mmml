"""PBC helpers must use KEY_LIBRARY C API, not ``open``/``nbonds``/``crystal`` scripts."""

from __future__ import annotations

from unittest.mock import patch

import pytest


def test_prepare_charmm_pbc_uses_crystal_api_not_open_script() -> None:
    from mmml.interfaces.pycharmmInterface.mlpot import pbc_env

    with (
        patch(
            "mmml.interfaces.pycharmmInterface.charmm_mpi.mpi_charmm_script"
        ) as mock_script,
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.pbc_env._image_setup_byres_all"
        ) as mock_image,
        patch("pycharmm.crystal.define_cubic", return_value=1) as mock_def,
        patch("pycharmm.crystal.build", return_value=1) as mock_build,
    ):
        pbc_env.prepare_charmm_pbc(32.0)

    mock_def.assert_called_once_with(32.0)
    mock_build.assert_called_once()
    mock_image.assert_called_once_with(0.0, 0.0, 0.0)
    joined = "\n".join(str(c.args[0]) for c in mock_script.call_args_list)
    assert "open read" not in joined.lower()
    assert "crystal defi" not in joined.lower()


def test_apply_pbc_nbonds_uses_nbonds_api() -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import apply_pbc_nbonds

    with patch(
        "mmml.interfaces.pycharmmInterface.nbonds_config.apply_nbonds_kwargs"
    ) as mock_apply:
        cuts = apply_pbc_nbonds(nbxmod=5, cubic_box_side_A=32.0)
    mock_apply.assert_called_once()
    assert cuts.cubic_box_side_A == pytest.approx(32.0)


def test_apply_vacuum_nbonds_uses_nbonds_api(monkeypatch: pytest.MonkeyPatch) -> None:
    from mmml.interfaces.pycharmmInterface.nbonds_config import apply_vacuum_nbonds

    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.prepare_charmm_vacuum",
        lambda: None,
    )
    with patch(
        "mmml.interfaces.pycharmmInterface.nbonds_config.apply_nbonds_kwargs"
    ) as mock_apply:
        apply_vacuum_nbonds(nbxmod=5)
    mock_apply.assert_called_once()
