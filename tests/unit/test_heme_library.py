"""CHARMM heme stream is a library residue, not a CGenFF name."""

from __future__ import annotations

import pytest

from karml.interfaces.pycharmmInterface.cgenff_residues import (
    is_cgenff_residue_name,
    require_cgenff_residue_name,
)
from karml.interfaces.pycharmmInterface.heme_library import (
    heme_library_residue_names,
    heme_reference_coordinate_table,
    heme_reference_positions,
    heme_stream_cards,
    heme_toppar_paths,
    is_heme_library_residue,
    segment_terminal_patches,
    topology_family,
    topology_residue_context,
)
from karml.interfaces.pycharmmInterface.nbonds_config import _rtf_path_for_append


def test_heme_stream_is_the_protein_library() -> None:
    rtf, prm, stream = heme_toppar_paths()
    assert rtf.name == "top_all36_prot.rtf"
    assert prm.name == "par_all36m_prot.prm"
    assert stream.name == "toppar_all36_prot_heme.str"
    names = heme_library_residue_names()
    assert "HEME" in names
    assert "PHEM" not in names  # PRES patch, not a RESI
    assert is_heme_library_residue("heme")
    assert not is_cgenff_residue_name("HEME")
    assert require_cgenff_residue_name("heme") == "HEME"


def test_heme_only_selects_the_protein_topology() -> None:
    assert topology_family(None) == "cgenff"
    assert topology_family(("ACO",)) == "cgenff"
    assert topology_family(("HEME",)) == "heme"
    assert topology_family(("HEME", "TIP3")) == "heme"
    assert topology_family(("HEME", "ALA", "TIP3", "CO")) == "heme"
    with pytest.raises(ValueError, match="toppar_all36_prot_heme"):
        topology_family(("HEME", "MEOH"))


def test_heme_stream_cards_drop_the_script_commands() -> None:
    _rtf, _prm, stream = heme_toppar_paths()
    rtf_card, prm_card = heme_stream_cards(stream)
    assert "RESI HEME" in rtf_card
    assert not rtf_card.lstrip().lower().startswith("read")
    assert "\nend" in rtf_card.lower() or rtf_card.lower().startswith("end")
    assert "return" not in rtf_card.lower()
    assert "NONBONDED" in prm_card
    assert "FE " in prm_card
    assert not prm_card.lstrip().lower().startswith("read")
    assert "return" not in prm_card.lower()

    import os

    raw_path = _write_for_append(rtf_card)
    append_path = _rtf_path_for_append(raw_path)
    try:
        append_text = open(append_path, encoding="utf-8").read()
    finally:
        for path in {raw_path, append_path}:
            try:
                os.remove(path)
            except OSError:
                pass
    assert "RESI HEME" in append_text
    assert not any(
        len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit()
        for line in append_text.splitlines()
        if (parts := line.split())
    )


def _write_for_append(text: str) -> str:
    import os
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".rtf")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


def test_heme_reference_coordinates_match_the_residue() -> None:
    import re

    import numpy as np

    _rtf, _prm, stream = heme_toppar_paths()
    body = stream.read_text(encoding="utf-8").split("RESI HEME", 1)[1].split("RESI O2", 1)[0]
    names = re.findall(r"^ATOM\s+(\S+)", body, re.M)
    table = heme_reference_coordinate_table()
    assert list(table) == names
    assert table["FE"] == pytest.approx((15.09006, 27.69400, 3.29659))
    placed = heme_reference_positions(names)
    assert placed is not None
    assert placed.shape == (73, 3)
    assert np.linalg.norm(placed.mean(axis=0)) < 1e-8
    span = placed.max(axis=0) - placed.min(axis=0)
    assert float(span.max()) > 8.0
    assert heme_reference_positions(["FE", "NO_SUCH_ATOM"]) is None


def test_heme_segment_drops_peptide_terminal_patches() -> None:
    assert segment_terminal_patches() == {}
    with topology_residue_context(("HEME",)):
        assert segment_terminal_patches() == {
            "first_patch": "NONE",
            "last_patch": "NONE",
        }
    assert segment_terminal_patches() == {}
