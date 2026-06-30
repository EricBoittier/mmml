"""Guards for import-time / vacuum CHARMM scripts under MPI mpirun."""

from __future__ import annotations

from pathlib import Path


def test_crystal_free_skips_under_mpi_linked_mpirun():
    source = Path("mmml/interfaces/pycharmmInterface/import_pycharmm.py").read_text(
        encoding="utf-8"
    )
    assert "def should_skip_vacuum_charmm_init()" in source
    crystal = source.split("def crystal_free_charmm")[1].split("\ndef ")[0]
    assert "should_skip_vacuum_charmm_init()" in crystal


def test_reset_block_skips_under_mpi_linked_mpirun():
    source = Path("mmml/interfaces/pycharmmInterface/import_pycharmm.py").read_text(
        encoding="utf-8"
    )
    assert "def should_skip_charmm_reset_block()" in source
    reset = source.split("def reset_block")[1].split("\ndef ")[0]
    assert "should_skip_charmm_reset_block()" in reset
