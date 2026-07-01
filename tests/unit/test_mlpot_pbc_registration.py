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
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.assert_charmm_pbc_lattice_ready_for_mlpot",
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

    def _finalize(_sel, *, cubic_box_side_A, verbose=False):
        call_order.append("finalize_pbc_exclusions")

    def _install(_sel, *, update=True):
        call_order.append("install_exclusions")

    class _FakeMLpot:
        def __init__(self, *, skip_iblo_inb_update=False, **kwargs):
            call_order.append("mlpot")

    with patch.object(mlpot_setup, "_import_pycharmm", return_value=fake_pycharmm), patch.object(
        mlpot_setup,
        "_suspend_pbc_for_cgenff_param_read",
        side_effect=_suspend,
    ), patch.object(
        mlpot_setup,
        "_finalize_pbc_mlpot_exclusions_after_param_read",
        side_effect=_finalize,
    ), patch(
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
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.assert_charmm_pbc_lattice_ready_for_mlpot",
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
    assert call_order == ["block", "finalize_pbc_exclusions", "mlpot"]


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


def test_register_mlpot_pbc_requires_mpirun_for_mpi_lib():
    from mmml.interfaces.pycharmmInterface.mlpot import setup as mlpot_setup

    fake_pycharmm = MagicMock()
    fake_sel = MagicMock()
    fake_sel.get_atom_indexes.return_value = [0, 1]

    with patch.object(mlpot_setup, "_import_pycharmm", return_value=fake_pycharmm), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.mlpot_limits.validate_mlpot_system_size",
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.charmm_lib_links_mpi",
        return_value=True,
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._under_mpirun",
        return_value=False,
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_levels.charmm_relaxed_bomlev",
        return_value=MagicMock(__enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False)),
    ), pytest.raises(RuntimeError, match="MLpot PBC registration"):
        mlpot_setup.register_mlpot(
            MagicMock(),
            [1, 1],
            fake_sel,
            use_pbc=True,
            cubic_box_side_A=40.0,
        )


def test_finalize_pbc_exclusions_uses_prepare_charmm_pbc():
    from mmml.interfaces.pycharmmInterface.mlpot import setup as mlpot_setup

    fake_sel = MagicMock()
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.prepare_charmm_pbc",
    ) as prepare, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.apply_pbc_nbonds",
    ) as apply_nb, patch.object(
        mlpot_setup,
        "_install_ml_exclusions",
    ) as install, patch.object(
        mlpot_setup,
        "_import_pycharmm",
    ) as import_py:
        fake_pycharmm = MagicMock()
        import_py.return_value = fake_pycharmm
        mlpot_setup._finalize_pbc_mlpot_exclusions_after_param_read(
            fake_sel,
            cubic_box_side_A=32.0,
            verbose=False,
        )
    prepare.assert_called_once_with(32.0)
    install.assert_called_once_with(fake_sel, update=False)
    apply_nb.assert_called_once_with(nbxmod=5, cubic_box_side_A=32.0)
    fake_pycharmm.nbonds.update_bnbnd.assert_not_called()
    fake_pycharmm.image.update_bimag.assert_called_once()


def test_register_mlpot_context_skips_user_check_when_jax_deferred():
    from mmml.interfaces.pycharmmInterface.mlpot import run_workflow

    z = __import__("numpy").zeros(4, dtype=int)
    r = __import__("numpy").zeros((4, 3))
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.run_workflow.load_physnet_mlpot_bundle",
        return_value=(None, None, MagicMock()),
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.run_workflow.register_mlpot",
        return_value=MagicMock(),
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.run_workflow.select_all_atoms",
        return_value=MagicMock(),
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.defer_jax_warmup_until_after_mlpot_sd",
        return_value=True,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.run_workflow.refresh_nbonds_after_mlpot_pbc",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.run_workflow.sync_charmm_positions",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.run_workflow.get_charmm_positions_array",
        return_value=r,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.rebind_mlpot_calculator_from_pycmodel",
        return_value=True,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.assert_mlpot_user_active",
    ) as assert_user:
        run_workflow._register_mlpot_context(
            z,
            r,
            __import__("pathlib").Path("ckpt"),
            len(z),
            1,
            defer_jax_warmup=True,
            verbose=True,
        )
    assert_user.assert_not_called()
