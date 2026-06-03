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

    with mock.patch.object(block_terms, "_import_pycharmm") as imp:
        imp.return_value.coor.get_natom.return_value = 10
        imp.return_value.lingo.charmm_script = mock.Mock()
        block_terms.apply_mlpot_energy_block(sel)

    script = imp.return_value.lingo.charmm_script.call_args[0][0]
    assert "IMPr" not in script and "IMPR" not in script
    assert "ELEC 0.0" in script
    assert "VDW 0.0" in script
    assert "COEFF 1 1 0.0" in script


def test_mlpot_internal_block_coeff_scaled():
    assert "BOND 0.1" in block_terms._mlpot_internal_block_coeff_line(0.1)
    assert block_terms._mlpot_internal_block_coeff_line(0.0) == block_terms._ML_SELF_ZERO


def test_mlpot_block_partial_ml_zeros_ml_block_elec_vdw():
    sel = mock.Mock()
    sel.get_atom_indexes.return_value = [0, 1, 2]
    sel.store.return_value = "mmml_ml"

    with mock.patch.object(block_terms, "_import_pycharmm") as imp:
        imp.return_value.coor.get_natom.return_value = 10
        imp.return_value.lingo.charmm_script = mock.Mock()
        tag = block_terms.apply_mlpot_energy_block(sel)

    assert tag == "mmml_ml"
    script = imp.return_value.lingo.charmm_script.call_args[0][0]
    assert "COEFF 2 2 0.0 BOND 0.0 ANGL 0.0 DIHEdral 0.0 ELEC 0.0 VDW 0.0" in script
    assert "COEFF 1 2 0.0" in script
