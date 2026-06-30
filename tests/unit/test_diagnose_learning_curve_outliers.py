from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mmml.cli.misc.diagnose_learning_curve_outliers import (
    RunRecord,
    detect_spikes,
    flag_run_outliers,
    load_runs,
    plot_outlier_summary,
    reproduce_train_indices,
    run_diagnosis,
)


def test_detect_spikes_flags_jump() -> None:
    epochs = list(range(1, 11))
    values = [1.0, 1.0, 1.0, 1.0, 1.0, 8.0, 1.0, 1.0, 1.0, 1.0]
    spikes = detect_spikes(epochs, values, metric="valid_loss", relative_factor=5.0, jump_factor=3.0)
    assert spikes
    assert any(s.epoch == 6 for s in spikes)


def test_flag_run_outliers_marks_high_test_energy() -> None:
    runs = [
        RunRecord("aco", 800, 1, 1042, Path("."), 0.5, None, None, None, None),
        RunRecord("aco", 800, 2, 2042, Path("."), 3.0, None, None, None, None),
        RunRecord("aco", 800, 3, 3042, Path("."), 0.6, None, None, None, None),
    ]
    flag_run_outliers(runs)
    assert runs[0].run_outlier is False
    assert runs[1].run_outlier is True
    assert runs[2].run_outlier is False


def test_reproduce_train_indices_matches_physnet_seed_prefix() -> None:
    idx_800 = reproduce_train_indices(3042, 800, 30, 16800)
    idx_1600 = reproduce_train_indices(3042, 1600, 60, 16800)
    assert idx_800.tolist() == idx_1600[:800].tolist()


def test_load_runs_from_fixture(tmp_path: Path) -> None:
    run_dir = tmp_path / "aco" / "n800" / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "seed": 1042,
                "test_eval": {"energy_mae_kcal_mol": 0.5, "forces_mae_kcal_mol": 0.1},
                "training_final": {"valid_loss": 1.0, "valid_energy_mae": 0.02},
            }
        )
    )
    runs = load_runs(tmp_path, "aco")
    assert len(runs) == 1
    assert runs[0].seed == 1042
    assert runs[0].test_energy_mae == pytest.approx(0.5)


def test_plot_outlier_summary(tmp_path: Path) -> None:
    runs = [
        RunRecord("aco", 800, 1, 1042, Path("."), 0.5, 0.1, 0.02, 1.0, None, [], False, ""),
        RunRecord("aco", 800, 2, 2042, Path("."), 3.0, 0.1, 0.04, 1.0, None, [], True, "bad"),
    ]
    seed_summary = {
        1042: {"outlier_runs": 0, "test_energy_mae_mean": 0.5, "test_energy_mae_max": 0.5, "spike_epochs": 0},
        2042: {"outlier_runs": 1, "test_energy_mae_mean": 3.0, "test_energy_mae_max": 3.0, "spike_epochs": 0},
    }
    npz_path = tmp_path / "train.npz"
    np.savez(npz_path, E=np.array([-81.0, -40.0, -92.0]), F=np.zeros((3, 1, 3)), N=np.array([20, 20, 20]))
    out = tmp_path / "summary.png"
    plot_outlier_summary(
        runs,
        seed_summary,
        [{"index": 1, "energy": -40.0, "fmax": 4.0, "exclusive_to_outlier_runs": 1, "outlier_seeds": [2042]}],
        [{"index": 2, "energy": -92.0, "fmax": 3.0, "composite_score": 10.0, "natoms": 20}],
        out,
        train_npz=npz_path,
        dataset="aco",
        plot_style="nature",
        verbose=False,
    )
    assert out.is_file()
