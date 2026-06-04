"""Unit tests for MLpot force NPZ checkpoint writer."""

from __future__ import annotations

import numpy as np

from mmml.interfaces.pycharmmInterface.mlpot.force_checkpoint import (
    ForceCheckpointConfig,
    ForceCheckpointWriter,
    configure_force_checkpoint,
    flush_force_checkpoint,
    get_force_checkpoint_writer,
)


def test_writer_records_and_flushes(tmp_path) -> None:
    cfg = ForceCheckpointConfig(
        enabled=True,
        interval=1,
        output_path=tmp_path / "forces.npz",
        n_monomers=2,
        atoms_per_monomer=3,
    )
    w = ForceCheckpointWriter(config=cfg)
    n = 6
    for step in (1, 2, 3):
        f = np.arange(n * 3, dtype=float).reshape(n, 3) + step
        pos = np.zeros((n, 3)) + step
        w.record(step, total_forces=f, positions=pos, ml_forces=f * 0.5)
    out = w.flush()
    assert out is not None and out.is_file()
    data = np.load(out)
    assert data["step"].tolist() == [1, 2, 3]
    assert data["forces"].shape == (3, n, 3)
    assert "max_force_per_monomer" in data.files


def test_configure_and_flush_global(tmp_path) -> None:
    configure_force_checkpoint(
        ForceCheckpointConfig(
            enabled=True,
            interval=2,
            output_path=tmp_path / "f.npz",
            n_monomers=1,
            atoms_per_monomer=5,
        )
    )
    writer = get_force_checkpoint_writer()
    assert writer is not None
    writer.record(2, total_forces=np.ones((5, 3)))
    writer.record(4, total_forces=np.ones((5, 3)) * 2)
    path = flush_force_checkpoint()
    assert path is not None
    assert get_force_checkpoint_writer() is None
    data = np.load(path)
    assert data["step"].tolist() == [2, 4]
