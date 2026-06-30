"""Tests for MLpot CHARMM BLOCK coefficient strings."""

from __future__ import annotations

from unittest import mock

import importlib.util
from pathlib import Path
from unittest import mock

_block_terms_path = (
    Path(__file__).resolve().parents[2]
    / "mmml/interfaces/pycharmmInterface/mlpot/block_terms.py"
)
_spec = importlib.util.spec_from_file_location("block_terms", _block_terms_path)
block_terms = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(block_terms)


def test_mlpot_block_all_atoms_zeros_elec_vdw_no_impr():
    sel = mock.Mock()
    sel.get_atom_indexes.return_value = list(range(10))
    sel.store = mock.Mock()
    scripts: list[str] = []

    with mock.patch.object(block_terms, "_import_pycharmm") as imp, mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_levels.run_charmm_script_quiet",
        side_effect=scripts.append,
    ):
        imp.return_value.coor.get_natom.return_value = 10
        block_terms.apply_mlpot_energy_block(sel)

    script = scripts[0]
    assert "IMPr" not in script and "IMPR" not in script
    assert "ELEC 0.0" in script
    assert "VDW 0.0" in script
    assert "COEFF 1 1 0.0" in script


def test_mlpot_internal_block_coeff_scaled():
    assert "BOND 0.1" in block_terms._mlpot_internal_block_coeff_line(0.1)
    assert block_terms._mlpot_internal_block_coeff_line(0.0) == block_terms._ML_SELF_ZERO


def test_mlpot_use_block_registration_defaults_false(monkeypatch):
    monkeypatch.delenv("MMML_MLPOT_USE_BLOCK", raising=False)
    assert block_terms.mlpot_use_block_registration() is False
    monkeypatch.setenv("MMML_MLPOT_USE_BLOCK", "1")
    assert block_terms.mlpot_use_block_registration() is True


def test_apply_mlpot_registration_mm_off_uses_psf_by_default(monkeypatch):
    sel = mock.Mock()
    sel.get_atom_indexes.return_value = list(range(10))
    monkeypatch.delenv("MMML_MLPOT_USE_BLOCK", raising=False)
    with mock.patch.object(
        block_terms, "zero_mlpot_psf_mm_terms", return_value="all"
    ) as zero_fn, mock.patch.object(block_terms, "apply_mlpot_energy_block") as block_fn:
        tag = block_terms.apply_mlpot_registration_mm_off(sel)
    assert tag == "all"
    zero_fn.assert_called_once()
    block_fn.assert_not_called()


def test_apply_mlpot_registration_mm_off_honors_use_block(monkeypatch):
    sel = mock.Mock()
    monkeypatch.delenv("MMML_MLPOT_USE_BLOCK", raising=False)
    with mock.patch.object(
        block_terms, "zero_mlpot_psf_mm_terms"
    ) as zero_fn, mock.patch.object(
        block_terms, "apply_mlpot_energy_block", return_value="all"
    ) as block_fn:
        tag = block_terms.apply_mlpot_registration_mm_off(sel, use_block=True)
    assert tag == "all"
    block_fn.assert_called_once()
    zero_fn.assert_not_called()


def test_zero_mlpot_psf_mm_terms_zeros_via_block_and_charges():
    sel = mock.Mock()
    sel.get_atom_indexes.return_value = [0, 1, 2]
    fake_psf = mock.Mock()
    fake_psf.get_charges.return_value = [0.5, -0.2, 0.1]

    with mock.patch.object(
        block_terms, "apply_mlpot_energy_block", return_value="all"
    ) as block_fn, mock.patch.object(block_terms, "_import_pycharmm") as imp:
        pycharmm = imp.return_value
        pycharmm.psf = fake_psf
        tag = block_terms.zero_mlpot_psf_mm_terms(sel)

    assert tag == "all"
    block_fn.assert_called_once()
    fake_psf.delete_connectivity = getattr(fake_psf, "delete_connectivity", None)
    if hasattr(fake_psf, "delete_connectivity"):
        assert not fake_psf.delete_connectivity.called
    fake_psf.set_charge.assert_called_once()
    assert fake_psf.set_charge.call_args.args[0] == [0.0, 0.0, 0.0]


def test_zero_mlpot_psf_mm_terms_periodic_external_uses_periodic_block():
    sel = mock.Mock()
    with mock.patch.object(
        block_terms, "apply_mlpot_periodic_external_block", return_value="all"
    ) as periodic_fn, mock.patch.object(
        block_terms, "apply_mlpot_energy_block"
    ) as block_fn, mock.patch.object(block_terms, "_import_pycharmm") as imp:
        imp.return_value.psf.get_charges.return_value = [0.0, 0.0]
        imp.return_value.psf.set_charge = mock.Mock()
        tag = block_terms.zero_mlpot_psf_mm_terms(sel, periodic_external=True)

    assert tag == "all"
    periodic_fn.assert_called_once()
    block_fn.assert_not_called()


def test_apply_mlpot_registration_mm_off_periodic_external_uses_psf(monkeypatch):
    sel = mock.Mock()
    sel.get_atom_indexes.return_value = list(range(10))
    monkeypatch.delenv("MMML_MLPOT_USE_BLOCK", raising=False)
    with mock.patch.object(
        block_terms, "zero_mlpot_psf_mm_terms", return_value="all"
    ) as zero_fn, mock.patch.object(block_terms, "apply_mlpot_periodic_external_block") as block_fn:
        tag = block_terms.apply_mlpot_registration_mm_off(sel, periodic_external=True)
    assert tag == "all"
    zero_fn.assert_called_once()
    assert zero_fn.call_args.kwargs.get("periodic_external") is True
    block_fn.assert_not_called()


def test_mlpot_block_partial_ml_zeros_ml_block_elec_vdw():
    sel = mock.Mock()
    sel.get_atom_indexes.return_value = [0, 1, 2]
    sel.store.return_value = "mmml_ml"

    scripts: list[str] = []

    with mock.patch.object(block_terms, "_import_pycharmm") as imp, mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_levels.run_charmm_script_quiet",
        side_effect=scripts.append,
    ):
        imp.return_value.coor.get_natom.return_value = 10
        tag = block_terms.apply_mlpot_energy_block(sel)

    assert tag == "mmml_ml"
    script = scripts[0]
    assert "COEFF 2 2 0.0 BOND 0.0 ANGL 0.0 DIHEdral 0.0 ELEC 0.0 VDW 0.0" in script
    assert "COEFF 1 2 0.0" in script
