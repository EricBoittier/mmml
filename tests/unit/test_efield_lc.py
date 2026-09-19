"""Learning-curve train subsets (synthetic NPZ only)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mmml.data.efield_lc import (
    DEFAULT_LC_FRACS,
    format_lc_table,
    frac_tag,
    n_train_for_frac,
    prepare_lc_grid,
    prepare_lc_split,
    summarize_lc_runs,
)
from mmml.data.spice_alpha import write_physnet_npz

pytestmark = pytest.mark.data_loading


def _split_dir(tmp_path: Path, *, n_train: int = 40, n_valid: int = 8) -> Path:
    src = tmp_path / "full"
    src.mkdir()
    pad = 3

    def _payload(n: int, e0: float):
        return {
            "R": np.zeros((n, pad, 3), np.float32),
            "Z": np.ones((n, pad), np.int32),
            "N": np.full((n,), pad, np.int32),
            "E": np.linspace(e0, e0 - 1.0, n, dtype=np.float32),
            "F": np.zeros((n, pad, 3), np.float32),
            "D": np.zeros((n, 3), np.float32),
            "polar": np.eye(3, dtype=np.float32)[None].repeat(n, 0),
            "Ef": np.zeros((n, 3), np.float32),
            "_mmml_units": np.array(
                json.dumps({"E": "ev", "F": "ev_angstrom", "polar": "bohr3"})
            ),
        }

    write_physnet_npz(_payload(n_train, -10.0), src / "energies_forces_dipoles_train.npz")
    write_physnet_npz(_payload(n_valid, -20.0), src / "energies_forces_dipoles_valid.npz")
    return src


def test_n_train_for_frac_aligns_to_batch():
    assert n_train_for_frac(15975, 0.01, 4) == 156
    assert n_train_for_frac(15975, 0.10, 4) == 1596
    with pytest.raises(ValueError, match="BATCH_SIZE"):
        n_train_for_frac(10, 0.01, 4)


def test_frac_tag():
    assert frac_tag(0.01) == "p01"
    assert frac_tag(0.03) == "p03"
    assert frac_tag(0.10) == "p10"
    assert DEFAULT_LC_FRACS == (0.01, 0.03, 0.10, 0.30)


def test_prepare_lc_split_keeps_valid_and_shrinks_train(tmp_path: Path):
    src = _split_dir(tmp_path)
    dest = tmp_path / "splits_p10"
    meta = prepare_lc_split(src, dest, frac=0.10, batch_size=4, seed=0)
    assert meta["n_train"] == 4
    train = np.load(dest / "energies_forces_dipoles_train.npz")
    valid = np.load(dest / "energies_forces_dipoles_valid.npz")
    assert train["E"].shape[0] == 4
    assert valid["E"].shape[0] == 8
    np.testing.assert_allclose(valid["E"], np.load(src / "energies_forces_dipoles_valid.npz")["E"])
    assert "polar" in train.files
    man = json.loads((dest / "lc_manifest.json").read_text(encoding="utf-8"))
    assert man["frac"] == 0.10
    assert man["seed"] == 0


def test_prepare_lc_grid_and_summarize(tmp_path: Path):
    src = _split_dir(tmp_path, n_train=40)
    rows = prepare_lc_grid(src, tmp_path / "lc", fracs=(0.10, 0.30), batch_size=4, seed=0)
    assert [r["n_train"] for r in rows] == [4, 12]
    ckpt = tmp_path / "ckpt_p10"
    ckpt.mkdir()
    (ckpt / "history.jsonl").write_text(
        json.dumps(
            {
                "epoch": 3,
                "n_train": 4,
                "valid_polar_mae_bohr3": 1.5,
                "valid_loss": 0.2,
                "best_epoch": 2,
                "best_valid_loss": 0.1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    table = format_lc_table(summarize_lc_runs([ckpt]))
    assert "1.5" in table
    assert "ckpt_p10" in table


def test_submit_script_is_polar_only_and_local():
    text = Path("scripts/spice_alpha/submit_efield_lc.sh").read_text(encoding="utf-8")
    assert "ENERGY_WEIGHT=" in text
    assert "0.01,0.03,0.10,0.30" in text
    assert "zenodo" not in text.lower()
    assert "22868157" in text
    assert "sbatch" in text
