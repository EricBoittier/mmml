"""CGENFF pre-minimize must not clear CHARMM crystal when use_pbc is set."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
    CharmmMmMinimizeConfig,
    _prepare_charmm_mm_minimize_list_frequencies,
)


def test_charmm_mm_minimize_config_accepts_use_pbc() -> None:
    cfg = CharmmMmMinimizeConfig(nstep_sd=10, use_pbc=True)
    assert cfg.use_pbc is True


def test_prepare_charmm_mm_minimize_list_frequencies_aligns_pbc_imgfrq() -> None:
    pycharmm = MagicMock()
    inb = _prepare_charmm_mm_minimize_list_frequencies(
        pycharmm,
        use_pbc=True,
        nstep=1000,
        inbfrq=50,
    )
    assert inb == 50
    pycharmm.nbonds.set_inbfrq.assert_called_once_with(50)
    pycharmm.nbonds.set_imgfrq.assert_called_once_with(50)


def test_prepare_charmm_mm_minimize_list_frequencies_clears_imgfrq_vacuum() -> None:
    pycharmm = MagicMock()
    inb = _prepare_charmm_mm_minimize_list_frequencies(
        pycharmm,
        use_pbc=False,
        nstep=100,
        inbfrq=50,
    )
    assert inb == 50
    pycharmm.nbonds.set_inbfrq.assert_called_once_with(50)
    pycharmm.nbonds.set_imgfrq.assert_called_once_with(0)


def test_prepare_charmm_mm_minimize_list_frequencies_harmonizes_to_nstep() -> None:
    pycharmm = MagicMock()
    inb = _prepare_charmm_mm_minimize_list_frequencies(
        pycharmm,
        use_pbc=True,
        nstep=25,
        inbfrq=50,
    )
    assert inb == 25
    pycharmm.nbonds.set_imgfrq.assert_called_once_with(25)


def test_minimize_charmm_mm_only_prepares_pbc_list_frequencies_before_sd() -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import minimize_charmm_mm_only

    pycharmm = MagicMock()
    minimize = MagicMock()
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._import_pycharmm_modules",
        return_value=(pycharmm, MagicMock(), MagicMock(), minimize, MagicMock()),
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.block_terms.apply_charmm_mm_block",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        return_value=__import__("numpy").zeros((10, 3)),
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.probe_charmm_cubic_box_side_A",
        return_value=(32.0, "pbound"),
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.restore_charmm_cubic_crystal_lattice",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._prepare_charmm_mm_minimize_list_frequencies",
        return_value=50,
    ) as prepare_freq:
        minimize_charmm_mm_only(
            CharmmMmMinimizeConfig(nstep_sd=100, nstep_abnr=0, use_pbc=True, verbose=False)
        )
    prepare_freq.assert_called_once_with(
        pycharmm,
        use_pbc=True,
        nstep=100,
        inbfrq=50,
    )
    minimize.run_sd.assert_called_once()
    assert minimize.run_sd.call_args.kwargs["inbfrq"] == 50


def test_minimize_charmm_mm_only_restores_pbc_after_cgenff_param_suspend() -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import minimize_charmm_mm_only

    pycharmm = MagicMock()
    minimize = MagicMock()
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._import_pycharmm_modules",
        return_value=(pycharmm, MagicMock(), MagicMock(), minimize, MagicMock()),
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.block_terms.apply_charmm_mm_block",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        return_value=__import__("numpy").zeros((10, 3)),
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._prepare_charmm_mm_minimize_list_frequencies",
        return_value=50,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.probe_charmm_cubic_box_side_A",
        return_value=(32.0, "pbound"),
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.restore_charmm_cubic_crystal_lattice",
    ) as restore_lattice:
        minimize_charmm_mm_only(
            CharmmMmMinimizeConfig(nstep_sd=10, nstep_abnr=0, use_pbc=True, verbose=False)
        )
    restore_lattice.assert_called_once_with(32.0, quiet=True)
