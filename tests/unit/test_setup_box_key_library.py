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
