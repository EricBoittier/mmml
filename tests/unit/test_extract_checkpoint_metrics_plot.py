from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mmml.cli.misc.extract_checkpoint_metrics import (
    plot_training_comparison,
    plot_training_metrics,
)


def _synthetic_metrics(n: int = 40) -> dict[str, np.ndarray]:
    epochs = np.arange(1, n + 1, dtype=float)
    valid_loss = 1e10 * np.exp(-0.25 * epochs) + 8.0 + 0.05 * np.random.default_rng(0).normal(size=n)
    train_loss = valid_loss * 1.1
    valid_energy = 0.2 * np.exp(-0.12 * epochs) + 0.15
    valid_forces = 0.08 * np.exp(-0.1 * epochs) + 0.05
    return {
        "epochs": epochs,
        "train_loss": train_loss,
        "valid_loss": valid_loss,
        "train_energy_mae": valid_energy * 1.05,
        "valid_energy_mae": valid_energy,
        "train_forces_mae": valid_forces * 1.05,
        "valid_forces_mae": valid_forces,
        "train_dipole_mae": np.full(n, np.nan),
        "valid_dipole_mae": np.full(n, np.nan),
        "lr_eff": np.full(n, 1e-3),
        "best_loss": np.minimum.accumulate(valid_loss),
    }


def test_plot_training_metrics_ef_only(tmp_path: Path) -> None:
    out = tmp_path / "curves.png"
    plot_training_metrics(
        _synthetic_metrics(),
        out,
        ckpt_name="test-run",
        log_loss=True,
        verbose=False,
        ef_only=True,
    )
    assert out.is_file()
    assert out.stat().st_size > 10_000


def test_plot_training_comparison(tmp_path: Path) -> None:
    runs = [
        ("n800/r1", _synthetic_metrics(30)),
        ("n800/r2", _synthetic_metrics(35)),
        ("n1600/r1", _synthetic_metrics(32)),
    ]
    out = tmp_path / "compare.png"
    plot_training_comparison(runs, out, ef_only=True, verbose=False)
    assert out.is_file()


def test_plot_training_comparison_with_table(tmp_path: Path) -> None:
    runs = [
        ("n800/r1", _synthetic_metrics(30)),
        ("n800/r2", _synthetic_metrics(35)),
    ]
    table = [
        ["n_train", "r1 E", "r2 E", "r1 F", "r2 F"],
        ["800", "0.50", "0.62", "0.10", "0.11"],
    ]
    out = tmp_path / "compare_table.png"
    plot_training_comparison(
        runs,
        out,
        ef_only=True,
        verbose=False,
        summary_table=table,
        plot_style="nature",
    )
    assert out.is_file()
    assert out.stat().st_size > 10_000


def test_warmup_trim_allows_plot_with_spikes(tmp_path: Path) -> None:
    m = _synthetic_metrics(20)
    m["valid_loss"][0] = 1e12
    m["train_loss"][0] = 1e12
    out = tmp_path / "spike.png"
    plot_training_metrics(m, out, verbose=False, ef_only=True)
    assert out.is_file()
