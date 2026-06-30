"""PBC MLpot registration ordering (no JAX — fast pytest collection)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def test_register_mlpot_pbc_rebuilds_after_param_swap():
    from mmml.interfaces.pycharmmInterface.mlpot import setup as mlpot_setup

    call_order: list[str] = []
    fake_pycharmm = MagicMock()
    fake_pycharmm.coor.get_natom.return_value = 4
    fake_sel = MagicMock()
    fake_sel.get_atom_indexes.return_value = [0, 1, 2, 3]

    def _finalize(_sel, *, cubic_box_side_A, verbose=False):
        call_order.append("finalize_pbc_exclusions")

    def _suspend(*, verbose=False):
        call_order.append("crystal_free")

    def _block(*args, **kwargs):
        call_order.append("block")
        return "all"

    class _FakeMLpot:
        def __init__(self, *, skip_iblo_inb_update=False, **kwargs):
            call_order.append("mlpot")
            if skip_iblo_inb_update:
                call_order.append("skip_iblo")

    with patch.object(mlpot_setup, "_import_pycharmm", return_value=fake_pycharmm), patch.object(
        mlpot_setup,
        "_finalize_pbc_mlpot_exclusions_after_param_read",
        side_effect=_finalize,
    ), patch.object(
        mlpot_setup,
        "_suspend_pbc_for_cgenff_param_read",
        side_effect=_suspend,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.block_terms.apply_mlpot_registration_mm_off",
        side_effect=_block,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.mlpot_limits.validate_mlpot_system_size",
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_levels.charmm_relaxed_bomlev",
        return_value=MagicMock(__enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False)),
    ):
        fake_pycharmm.MLpot = _FakeMLpot
        mlpot_setup.register_mlpot(
            MagicMock(),
            [1, 1, 1, 1],
            fake_sel,
            use_pbc=True,
            cubic_box_side_A=50.0,
        )

    assert call_order == [
        "crystal_free",
        "block",
        "finalize_pbc_exclusions",
        "mlpot",
        "skip_iblo",
    ]


def test_register_mlpot_pbc_block_skips_crystal_free_before_prm():
    from mmml.interfaces.pycharmmInterface.mlpot import setup as mlpot_setup

    call_order: list[str] = []
    fake_pycharmm = MagicMock()
    fake_sel = MagicMock()
    fake_sel.get_atom_indexes.return_value = [0, 1]

    def _suspend(*, verbose=False):
        call_order.append("crystal_free")

    def _block(*args, **kwargs):
        call_order.append("block")
        return "all"

    def _install(_sel, *, update=True):
        call_order.append("install_exclusions")

    class _FakeMLpot:
        def __init__(self, *, skip_iblo_inb_update=False, **kwargs):
            call_order.append("mlpot")

    with patch.object(mlpot_setup, "_import_pycharmm", return_value=fake_pycharmm), patch.object(
        mlpot_setup,
        "_suspend_pbc_for_cgenff_param_read",
        side_effect=_suspend,
    ), patch.object(mlpot_setup, "_install_ml_exclusions", side_effect=_install), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.block_terms.apply_mlpot_registration_mm_off",
        side_effect=_block,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.block_terms.mlpot_use_block_registration",
        return_value=True,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.mlpot_limits.validate_mlpot_system_size",
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_levels.charmm_relaxed_bomlev",
        return_value=MagicMock(__enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False)),
    ):
        fake_pycharmm.MLpot = _FakeMLpot
        mlpot_setup.register_mlpot(
            MagicMock(),
            [1, 1],
            fake_sel,
            use_pbc=True,
            cubic_box_side_A=50.0,
            use_block_registration=True,
        )

    assert "crystal_free" not in call_order
    assert call_order == ["block", "install_exclusions", "mlpot"]


def test_register_mlpot_vacuum_skips_pre_block_exclusions():
    from mmml.interfaces.pycharmmInterface.mlpot import setup as mlpot_setup

    call_order: list[str] = []
    fake_pycharmm = MagicMock()
    fake_sel = MagicMock()
    fake_sel.get_atom_indexes.return_value = [0, 1]

    def _block(*args, **kwargs):
        call_order.append("block")
        return "all"

    def _mlpot(**kwargs):
        call_order.append("mlpot")
        if kwargs.get("skip_iblo_inb_update"):
            call_order.append("skip_iblo")
        return MagicMock()

    with patch.object(mlpot_setup, "_import_pycharmm", return_value=fake_pycharmm), patch.object(
        mlpot_setup, "_install_ml_exclusions"
    ) as mock_install, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.block_terms.apply_mlpot_registration_mm_off",
        side_effect=_block,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.mlpot_limits.validate_mlpot_system_size",
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_levels.charmm_relaxed_bomlev",
        return_value=MagicMock(__enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False)),
    ), patch.object(fake_pycharmm, "MLpot", side_effect=_mlpot), patch.object(
        fake_pycharmm, "UpdateNonBondedScript"
    ) as mock_nb:
        mock_nb.return_value.run = MagicMock()
        mlpot_setup.register_mlpot(
            MagicMock(),
            [1, 1],
            fake_sel,
            use_pbc=False,
        )

    mock_install.assert_not_called()
    assert call_order == ["block", "mlpot"]


def test_register_mlpot_pbc_requires_skip_iblo_parameter():
    from mmml.interfaces.pycharmmInterface.mlpot import setup as mlpot_setup

    fake_sel = MagicMock()
    fake_sel.get_atom_indexes.return_value = [0, 1]

    class _OldMLpot:
        def __init__(self, **kwargs):
            pass

    class _FakePycharmm:
        MLpot = _OldMLpot

    with patch.object(mlpot_setup, "_import_pycharmm", return_value=_FakePycharmm()), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.block_terms.apply_mlpot_registration_mm_off",
        return_value="all",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.mlpot_limits.validate_mlpot_system_size",
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_levels.charmm_relaxed_bomlev",
        return_value=MagicMock(__enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False)),
    ), pytest.raises(RuntimeError, match="skip_iblo_inb_update"):
        mlpot_setup.register_mlpot(
            MagicMock(),
            [1, 1],
            fake_sel,
            use_pbc=True,
        )
