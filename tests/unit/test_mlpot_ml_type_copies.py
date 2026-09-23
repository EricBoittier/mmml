"""Unit tests: ML atom type copies for hybrid MLpot registration (#225)."""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest import mock

import pytest

from mmml.interfaces.pycharmmInterface.mlpot import cgenff_prm_swap, ml_type_copies


@pytest.fixture(autouse=True)
def _clear_copies():
    mode = cgenff_prm_swap._active_mode
    ml_type_copies.clear_ml_type_copies()
    yield
    ml_type_copies.clear_ml_type_copies()
    cgenff_prm_swap._active_mode = mode


def test_copy_type_name_prefixes_and_avoids_collisions():
    assert ml_type_copies.copy_type_name("CG331", set()) == "MLCG331"
    assert ml_type_copies.copy_type_name("CG331", {"MLCG331"}) == "MLX00000"
    # CHARMM type names are at most 8 characters.
    assert ml_type_copies.copy_type_name("ABCDEFG", set()) == "MLX00000"


def test_flex_lookup_last_match_either_direction():
    cols = [[1, 2, 1], [2, 1, 3]]
    assert ml_type_copies.flex_lookup((2, 1), cols, 3) == 1
    assert ml_type_copies.flex_lookup((3, 1), cols, 3) == 2
    assert ml_type_copies.flex_lookup((4, 4), cols, 3) is None
    with pytest.raises(RuntimeError, match="equivalence"):
        ml_type_copies.flex_lookup((1, 2), [[1, -1], [2, 5]], 2)


def test_ml_bonded_copy_prm_text_zero_force_constants():
    text = ml_type_copies.ml_bonded_copy_prm_text(
        {"MLCG331": 12.011},
        {("MLCG331", "MLHGA3"): 1.111},
        {("MLHGA3", "MLCG331", "MLHGA3"): 108.4},
    )
    lines = text.splitlines()
    assert "MASS -1 MLCG331" in text
    assert lines.index("ATOMS") < lines.index("BONDS") < lines.index("ANGLES")
    bond = next(ln for ln in lines if ln.startswith("MLCG331 "))
    assert bond.split()[2:] == ["0.0", "1.1110"]
    angle = next(ln for ln in lines if ln.startswith("MLHGA3 "))
    assert angle.split()[3:] == ["0.0", "108.4000"]
    assert lines[-1] == "END"
    # Re-registration reuses existing copies: no ATOMS section.
    assert "ATOMS" not in ml_type_copies.ml_bonded_copy_prm_text({}, {}, {})


def _fake_charmm(iac_after):
    """Two-molecule system: atoms 0-1 (ML) and 2-3 (MM), types A=0 and B=1."""
    pycharmm = mock.Mock()
    pycharmm.psf.get_iac.side_effect = [[0, 1, 0, 1], iac_after]
    pycharmm.psf.get_amass.return_value = [12.0, 1.0, 12.0, 1.0]
    pycharmm.psf.get_nbond.return_value = 2
    pycharmm.psf.get_ib_jb.return_value = ([1, 3], [2, 4])
    atc = ["A", "B"]
    pycharmm.param.get_atc.side_effect = [atc, atc + ["MLA", "MLB"]]
    pycharmm.param.get_vdwr.return_value = [2.0, 1.0, 2.0, 1.0]
    pycharmm.param.get_epsilon.return_value = [-0.1, -0.02, -0.1, -0.02]
    itc = [7, 8, 0, 0]
    tables = {
        "param.cbai": [1],
        "param.cbaj": [2],
        "param.cbb": [1.1],
        "param.ctai": [2],
        "param.ctaj": [1],
        "param.ctak": [1],
        "param.ctb": [math.radians(109.5)],
        "param.itc": itc,
    }
    counts = {"param.ncb": 1, "param.nct": 1, "psf.ntheta": 1}
    patches = [
        mock.patch.object(
            ml_type_copies,
            "_fortran_symbol",
            side_effect=lambda lib, var, ctype: SimpleNamespace(value=counts[var]),
        ),
        mock.patch.object(
            ml_type_copies, "_static_array", side_effect=lambda lib, var, ctype, n: tables[var]
        ),
        mock.patch.object(
            ml_type_copies,
            "_allocatable_array",
            side_effect=lambda lib, var, ctype, n: {"psf.it": [2], "psf.jt": [1], "psf.kt": [3]}[var],
        ),
        mock.patch("mmml.interfaces.pycharmmInterface.nbonds_config.read_cgenff_prm"),
    ]
    return pycharmm, itc, patches


def _apply(pycharmm, patches):
    mocks = [p.start() for p in patches]
    try:
        prm_texts: list[str] = []
        read_fn = mocks[-1]
        read_fn.side_effect = lambda path, append: prm_texts.append(path.read_text())
        counts = ml_type_copies.apply_ml_type_copies([0, 1], "mmml_ml", pycharmm=pycharmm)
        return counts, prm_texts
    finally:
        for p in patches:
            p.stop()


def test_apply_ml_type_copies_moves_only_ml_atoms():
    pycharmm, itc, patches = _fake_charmm([2, 3, 0, 1])
    counts, prm_texts = _apply(pycharmm, patches)

    assert counts == {"types": 2, "bonds": 1, "angles": 1}
    text = prm_texts[0]
    assert "MASS -1 MLA" in text and "MASS -1 MLB" in text
    # ML bond A-B: copy names, zero force, CHARMM b0.
    assert any(ln.split() == ["MLA", "MLB", "0.0", "1.1000"] for ln in text.splitlines())
    # Angle 2-1-3 crosses into the MM molecule: only ML atoms get copy names.
    assert any(ln.split()[:3] == ["MLB", "MLA", "A"] for ln in text.splitlines())
    # Copies share the originals' VDW groups.
    assert itc[2:] == [7, 8]
    scripts = [c.args[0] for c in pycharmm.lingo.charmm_script.call_args_list]
    assert scripts == [
        "SCALAR TYPE SET 3 SELE MMML_ML .AND. CHEM A END",
        "SCALAR TYPE SET 4 SELE MMML_ML .AND. CHEM B END",
    ]
    assert ml_type_copies.ml_type_copies_active()

    pycharmm.lingo.charmm_script.reset_mock()
    assert ml_type_copies.restore_ml_atom_types(pycharmm=pycharmm)
    scripts = [c.args[0] for c in pycharmm.lingo.charmm_script.call_args_list]
    assert scripts == ["SCALAR TYPE SET 1 SELE CHEM MLA END", "SCALAR TYPE SET 2 SELE CHEM MLB END"]
    assert not ml_type_copies.restore_ml_atom_types(pycharmm=pycharmm)


def test_apply_ml_type_copies_raises_when_atoms_not_moved():
    pycharmm, _, patches = _fake_charmm([0, 1, 0, 1])
    with pytest.raises(RuntimeError, match="0 of 2 ML atoms moved"):
        _apply(pycharmm, patches)


def test_full_cgenff_restore_moves_ml_atoms_back():
    with mock.patch.object(
        ml_type_copies, "restore_ml_atom_types"
    ) as restore_fn, mock.patch.object(cgenff_prm_swap, "_read_cgenff_prm"), mock.patch.object(
        cgenff_prm_swap, "assert_psf_bonds_present", return_value=4
    ):
        cgenff_prm_swap.apply_full_cgenff_params(force=True)
    restore_fn.assert_called_once_with()
