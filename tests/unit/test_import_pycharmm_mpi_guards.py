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


def test_crystal_free_for_param_append_bypasses_vacuum_skip():
    source = Path("mmml/interfaces/pycharmmInterface/import_pycharmm.py").read_text(
        encoding="utf-8"
    )
    fn = source.split("def crystal_free_charmm_for_param_append")[1].split("\ndef ")[0]
    assert "if should_skip_vacuum_charmm_init()" not in fn
    assert "_run_crystal_free" in fn
    run_fn = source.split("def _run_crystal_free")[1].split("\ndef ")[0]
    assert "mpi_charmm_script" in run_fn


def test_reset_block_skips_under_mpi_linked_mpirun():
    source = Path("mmml/interfaces/pycharmmInterface/import_pycharmm.py").read_text(
        encoding="utf-8"
    )
    assert "def should_skip_charmm_reset_block()" in source
    skip_pos = source.index("def should_skip_charmm_reset_block()")
    reset_pos = source.index("def reset_block()")
    assert skip_pos < reset_pos
    reset = source.split("def reset_block")[1].split("\ndef ")[0]
    assert "should_skip_charmm_reset_block()" in reset
