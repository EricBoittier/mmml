"""Unit tests for MLpot CGENFF parameter swap."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from mmml.interfaces.pycharmmInterface.mlpot import cgenff_prm_swap


def test_zeroed_cgenff_prm_paths():
    full = cgenff_prm_swap.cgenff_prm_path()
    assert full.name == "par_all36_cgenff.prm"
    assert "mmml/data/charmm" in str(full).replace("\\", "/")
    assert cgenff_prm_swap.zeroed_cgenff_prm_path().name == "zeroed_par_all36_cgenff.prm"
    assert (
        cgenff_prm_swap.zeroed_cgenff_prm_path(bonded_only=True).name
        == "zeroed_bonded_par_all36_cgenff.prm"
    )
    repo = Path(__file__).resolve().parents[2]
    assert (repo / "mmml/data/charmm/par_all36_cgenff.prm").is_file()
    assert full.is_file()


def test_read_cgenff_prm_uses_flex_for_append_swap():
    import inspect

    from mmml.interfaces.pycharmmInterface.nbonds_config import (
        CGENFF_PRM_BOMLEV,
        read_cgenff_prm,
    )

    src = inspect.getsource(read_cgenff_prm)
    assert "flex=True" in src
    assert f"CGENFF_PRM_BOMLEV" in src
    assert CGENFF_PRM_BOMLEV == -5
    assert "read_cgenff_prm(path, append=True)" in inspect.getsource(
        cgenff_prm_swap._read_cgenff_prm
    )


def test_apply_full_cgenff_params_reads_bonded_restore_and_checks_bonds():
    bonded = cgenff_prm_swap.bonded_cgenff_prm_path()
    with mock.patch.object(cgenff_prm_swap, "_read_cgenff_prm") as read_fn, mock.patch.object(
        cgenff_prm_swap, "assert_psf_bonds_present", return_value=400
    ) as bond_fn:
        cgenff_prm_swap.apply_full_cgenff_params(verbose=False)

    read_fn.assert_called_once_with(bonded)
    bond_fn.assert_called_once()
    assert cgenff_prm_swap.active_cgenff_prm_mode() == "full"


def test_apply_zeroed_cgenff_params_bonded_only(tmp_path: Path, monkeypatch):
    bonded = tmp_path / "zeroed_bonded_par_all36_cgenff.prm"
    bonded.write_text("END\n", encoding="utf-8")
    monkeypatch.setattr(
        cgenff_prm_swap,
        "zeroed_cgenff_prm_path",
        lambda *, bonded_only=False: bonded if bonded_only else tmp_path / "zeroed.prm",
    )
    with mock.patch.object(cgenff_prm_swap, "_read_cgenff_prm") as read_fn:
        cgenff_prm_swap.apply_zeroed_cgenff_params(bonded_only=True)
    read_fn.assert_called_once_with(bonded)
    assert cgenff_prm_swap.active_cgenff_prm_mode() == "zeroed_bonded"


def test_read_cgenff_prm_append_suspends_pbc_before_read():
    import inspect

    from mmml.interfaces.pycharmmInterface.nbonds_config import read_cgenff_prm

    src = inspect.getsource(read_cgenff_prm)
    assert "if append:" in src
    assert "suspend_pbc_before_cgenff_param_append()" in src


def test_read_cgenff_prm_replace_skips_pbc_suspend():
    import inspect

    from mmml.interfaces.pycharmmInterface.nbonds_config import read_cgenff_prm

    src = inspect.getsource(read_cgenff_prm)
    assert src.index("if append:") < src.index("def _read()")


def test_assert_psf_bonds_present_raises():
    with mock.patch.object(cgenff_prm_swap, "psf_bond_count", return_value=0):
        with pytest.raises(RuntimeError, match="PSF has 0 bonds"):
            cgenff_prm_swap.assert_psf_bonds_present(min_bonds=1)
