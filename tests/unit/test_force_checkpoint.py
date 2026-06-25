"""Unit tests for force checkpoint physical-force sign."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np


def test_maybe_record_forces_uses_physical_forces() -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.force_checkpoint import (
        ForceCheckpointConfig,
        ForceCheckpointWriter,
        configure_force_checkpoint,
        maybe_record_forces,
    )

    writer = ForceCheckpointWriter(
        config=ForceCheckpointConfig(enabled=True, interval=1, output_path=None)
    )
    configure_force_checkpoint(writer.config)
    # Re-bind active writer (configure creates new instance)
    from mmml.interfaces.pycharmmInterface.mlpot import force_checkpoint as fc

    fc._active = writer

    physical = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float64)
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    with (
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.cli_common.charmm_total_forces_kcalmol_A",
            return_value=physical,
        ),
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.cli_common.charmm_positions_angstrom",
            return_value=pos,
        ),
    ):
        maybe_record_forces(0)

    assert len(writer.forces) == 1
    np.testing.assert_allclose(writer.forces[0], physical)
    np.testing.assert_allclose(writer.positions[0], pos)
