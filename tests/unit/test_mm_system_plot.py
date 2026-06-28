"""Unit tests for :mod:`mmml.interfaces.pycharmmInterface.mm_system_plot`."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np

from mmml.interfaces.pycharmmInterface.mm_system_plot import plot_mm_system_diagnostics


def test_plot_mm_system_diagnostics_smoke(tmp_path) -> None:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [-0.24, 0.93, 0.0],
        ],
        dtype=float,
    )
    out = tmp_path / "diag.png"
    fig = plot_mm_system_diagnostics(
        positions,
        symbols=["O", "H", "H"],
        charges=[-0.8, 0.4, 0.4],
        subsystems={"all": np.ones(3, dtype=bool)},
        excluded_pairs=frozenset({(0, 1), (0, 2)}),
        save_path=out,
        show=False,
    )
    assert out.is_file()
    assert fig is not None
    import matplotlib.pyplot as plt

    plt.close(fig)
