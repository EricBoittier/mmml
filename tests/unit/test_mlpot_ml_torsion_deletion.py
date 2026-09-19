"""Unit tests: ML torsion deletion at zeroed-CGenFF MLpot registration (#217)."""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path
from unittest import mock

import pytest

from mmml.interfaces.pycharmmInterface.mlpot import cgenff_prm_swap

_block_terms_path = (
    Path(__file__).resolve().parents[2]
    / "mmml/interfaces/pycharmmInterface/mlpot/block_terms.py"
)
_spec = importlib.util.spec_from_file_location("block_terms", _block_terms_path)
block_terms = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(block_terms)
@pytest.fixture(autouse=True)
def _reset_cgenff_swap_state():
    mode = cgenff_prm_swap._active_mode
    cgenff_prm_swap.clear_ml_torsions_deleted()
    yield
    cgenff_prm_swap.clear_ml_torsions_deleted()
    cgenff_prm_swap._active_mode = mode


def _registration_pycharmm(n_total: int, charges: list[float]) -> mock.Mock:
    pycharmm = mock.Mock()
    pycharmm.coor.get_natom.return_value = n_total
    pycharmm.psf.get_charges.return_value = list(charges)
    return pycharmm


def _run_registration(sel, pycharmm, counts=None):
    counts = counts if counts is not None else [None, None]
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cgenff_prm_swap.apply_zeroed_cgenff_params"
    ) as zero_prm_fn, mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cgenff_prm_swap.assert_psf_bonds_present",
        return_value=400,
    ), mock.patch.object(block_terms, "_import_pycharmm", return_value=pycharmm), mock.patch.object(
        block_terms, "_psf_torsion_counts", side_effect=counts
    ):
        tag = block_terms.zero_mlpot_psf_mm_terms(sel)
    return tag, zero_prm_fn


def test_zero_mlpot_psf_mm_terms_deletes_ml_torsions_after_zeroed_append():
    sel = mock.Mock()
    sel.get_atom_indexes.return_value = [0, 1, 2]
    pycharmm = _registration_pycharmm(3, [0.5, -0.2, 0.1])
    calls: list[str] = []
    pycharmm.psf.delete_dihedrals.side_effect = lambda *a: calls.append("dihe")
    pycharmm.psf.delete_impropers.side_effect = lambda *a: calls.append("impr")
    pycharmm.psf.delete_cmaps.side_effect = lambda *a: calls.append("cmap")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tag, zero_prm_fn = _run_registration(sel, pycharmm)

    assert tag == "all"
    zero_prm_fn.assert_called_once_with(bonded_only=True, verbose=False)
    assert calls == ["dihe", "impr", "cmap"]
    # The PSF deletion does not replace the all-ML bonded SKIPE.
    pycharmm.lingo.charmm_script.assert_called_once_with(
        "SKIPE " + " ".join(block_terms.ALL_ML_SKIPE_BONDED)
    )
    for fn in (
        pycharmm.psf.delete_dihedrals,
        pycharmm.psf.delete_impropers,
        pycharmm.psf.delete_cmaps,
    ):
        fn.assert_called_once_with(sel, sel)
    # Bonds/angles stay in the PSF: nonbond exclusions come from the bond list.
    pycharmm.psf.delete_bonds.assert_not_called()
    pycharmm.psf.delete_angles.assert_not_called()
    pycharmm.psf.delete_connectivity.assert_not_called()
    assert cgenff_prm_swap.ml_torsions_deleted()


def test_zero_mlpot_psf_mm_terms_hybrid_deletes_ml_only_and_warns():
    sel = mock.Mock()
    sel.get_atom_indexes.return_value = [0, 1, 2]
    sel.store.return_value = "mmml_ml"
    pycharmm = _registration_pycharmm(6, [0.1] * 6)
    counts = [
        {"dihedrals": 24, "impropers": 0, "cmaps": 0},
        {"dihedrals": 12, "impropers": 0, "cmaps": 0},
    ]
    with pytest.warns(UserWarning, match="hybrid PSF registration"):
        tag, _ = _run_registration(sel, pycharmm, counts)

    assert tag == "mmml_ml"
    pycharmm.psf.delete_dihedrals.assert_called_once_with(sel, sel)
    # SKIPE would drop the MM molecules' bonded terms as well.
    pycharmm.lingo.charmm_script.assert_not_called()
    charges = pycharmm.psf.set_charge.call_args.args[0]
    assert charges == [0.0, 0.0, 0.0, 0.1, 0.1, 0.1]


def test_delete_ml_torsion_terms_reports_removed_counts():
    sel = mock.Mock()
    pycharmm = mock.Mock()
    counts = [
        {"dihedrals": 24, "impropers": 2, "cmaps": 1},
        {"dihedrals": 12, "impropers": 1, "cmaps": 0},
    ]
    with mock.patch.object(block_terms, "_psf_torsion_counts", side_effect=counts):
        removed = block_terms.delete_ml_torsion_terms(sel, pycharmm=pycharmm)
    assert removed == {"dihedrals": 12, "impropers": 1, "cmaps": 1}


def test_delete_ml_torsion_terms_all_ml_raises_on_leftover_terms():
    sel = mock.Mock()
    pycharmm = mock.Mock()
    counts = [
        {"dihedrals": 24, "impropers": 0, "cmaps": 0},
        {"dihedrals": 24, "impropers": 0, "cmaps": 0},
    ]
    with mock.patch.object(
        block_terms, "_psf_torsion_counts", side_effect=counts
    ), pytest.raises(RuntimeError, match="double-count"):
        block_terms.delete_ml_torsion_terms(sel, all_ml=True, pycharmm=pycharmm)


def test_psf_torsion_counts_unreadable_returns_none():
    assert block_terms._psf_torsion_counts(mock.Mock()) is None


def test_full_cgenff_restore_warns_when_ml_torsions_were_deleted():
    cgenff_prm_swap.mark_ml_torsions_deleted()
    with mock.patch.object(cgenff_prm_swap, "_read_cgenff_prm"), mock.patch.object(
        cgenff_prm_swap, "assert_psf_bonds_present", return_value=400
    ), pytest.warns(UserWarning, match="does not bring them back"):
        cgenff_prm_swap.apply_full_cgenff_params(force=True)

    cgenff_prm_swap.clear_ml_torsions_deleted()
    with mock.patch.object(cgenff_prm_swap, "_read_cgenff_prm"), mock.patch.object(
        cgenff_prm_swap, "assert_psf_bonds_present", return_value=400
    ), warnings.catch_warnings():
        warnings.simplefilter("error")
        cgenff_prm_swap.apply_full_cgenff_params(force=True)
