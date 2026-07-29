"""make-box / setupBox must use KEY_LIBRARY PBC APIs (no lingo nbonds/crystal)."""

from __future__ import annotations

from pathlib import Path


def test_setup_box_generic_uses_prepare_charmm_pbc() -> None:
    src = Path("mmml/interfaces/pycharmmInterface/setupBox.py").read_text(
        encoding="utf-8"
    )
    body = src.split("def setup_box_generic")[1].split("\ndef ")[0]
    assert "prepare_charmm_pbc" in body
    assert "charmm_script(pbcs)" not in body
    assert "nbonds atom cutnb" not in body
    # 5-char CGenFF names (CH3CL): do not use lingo READ SEQU PDB UNIT.
    assert "READ SEQU PDB UNIT" not in body
    assert "sequence_string" in body
    assert "sync_charmm_positions" in body


def test_minimize_box_does_not_use_lingo_nbonds() -> None:
    src = Path("mmml/interfaces/pycharmmInterface/setupBox.py").read_text(
        encoding="utf-8"
    )
    body = src.split("def minimize_box")[1].split("\ndef ")[0]
    assert "nbonds atom" not in body
    assert "charmm_script" not in body
    assert "run_abnr" in body


def test_make_box_cli_uses_density_for_neat_liquid() -> None:
    src = Path("mmml/cli/make/make_box.py").read_text(encoding="utf-8")
    assert "determine_n_molecules_from_density" in src
    # Neat path (solvent is None) must size N from density when provided.
    neat = src.split("if args.solvent is None:")[1].split("else:")[0]
    assert "args.density is not None" in neat
    assert "determine_n_molecules_from_density" in neat


def test_make_box_cli_stages_pdb_before_packmol() -> None:
    """``--pdb`` must stage to pdb/initial.pdb and still run Packmol solvation."""
    src = Path("mmml/cli/make/make_box.py").read_text(encoding="utf-8")
    assert "_stage_solute_pdb" in src
    assert "run_packmol_solvation" in src
    # Staging must not short-circuit packing (old bug: --pdb skipped Packmol).
    assert "pdb_path = args.pdb" not in src


def test_setup_box_generic_uses_read_cgenff_toppar() -> None:
    src = Path("mmml/interfaces/pycharmmInterface/setupBox.py").read_text(
        encoding="utf-8"
    )
    body = src.split("def setup_box_generic")[1].split("\ndef ")[0]
    assert "read_cgenff_toppar" in body
    assert "read.rtf(rtf)" not in body


def test_run_packmol_solvation_restores_resnames() -> None:
    src = Path("mmml/interfaces/pycharmmInterface/setupBox.py").read_text(
        encoding="utf-8"
    )
    body = src.split("def run_packmol_solvation")[1].split("\ndef ")[0]
    assert "rewrite_packmol_pdb_resnames" in body
    assert "chain A" not in body