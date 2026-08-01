"""NpT density analysis: equilibration detection, block errors, and refusals.

Synthetic traces with a known answer, so these pin the statistics rather than
any particular MD run.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

_SPEC = importlib.util.spec_from_file_location(
    "analyze_npt_density",
    Path(__file__).resolve().parents[2] / "scripts" / "analyze_npt_density.py",
)
den = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(den)


def _write(path: Path, rho, *, time_ps=None, temperature=298.0):
    rho = np.asarray(rho, dtype=float)
    with h5py.File(path, "w") as fh:
        fh.create_dataset("density_g_cm3", data=rho)
        fh.create_dataset(
            "time_ps",
            data=np.arange(rho.size, dtype=float) * 0.01 if time_ps is None else time_ps,
        )
        fh.attrs["temperature_target"] = float(temperature)


def test_recovers_planted_density_and_calls_it_equilibrated(tmp_path, capsys):
    rng = np.random.default_rng(0)
    rho = 0.99700 + rng.normal(scale=2e-4, size=3000)  # flat, noisy
    p = tmp_path / "npt.h5"
    _write(p, rho)

    assert den.main(["--traj", str(p), "--species", "TIP3", "--discard-frac", "0.4"]) == 0
    out = capsys.readouterr().out
    assert "0.997" in out
    assert "EQUILIBRATED" in out and "NOT EQUILIBRATED" not in out


def test_flags_a_drifting_trace_as_not_equilibrated(tmp_path, capsys):
    n = 3000
    t = np.arange(n) * 0.01
    rho = 0.95 + 0.002 * t  # steady expansion, never settles
    p = tmp_path / "drift.h5"
    _write(p, rho, time_ps=t)

    assert den.main(["--traj", str(p), "--discard-frac", "0.4"]) == 0
    out = capsys.readouterr().out
    assert "NOT EQUILIBRATED" in out
    assert "Do not quote this density" in out


def test_block_sem_exceeds_naive_for_correlated_density():
    rng = np.random.default_rng(1)
    x = np.convolve(rng.normal(size=4000), np.ones(200) / 200.0, mode="same")
    naive = float(np.std(x, ddof=1) / np.sqrt(x.size))
    assert den.block_average_sem(x, 10) > 5 * naive


def test_refuses_a_trajectory_without_density(tmp_path):
    """An NVT/NVE run cannot measure density; say so instead of inventing one."""
    p = tmp_path / "nvt.h5"
    with h5py.File(p, "w") as fh:
        fh.create_dataset("potential_energy", data=np.zeros(100))
    with pytest.raises(SystemExit, match="NpT"):
        den.main(["--traj", str(p)])


def test_refuses_nonfinite_density(tmp_path):
    """A diverged barostat must not be averaged away."""
    rho = np.full(500, 0.997)
    rho[300] = np.nan
    p = tmp_path / "nan.h5"
    _write(p, rho)
    with pytest.raises(SystemExit, match="non-finite"):
        den.main(["--traj", str(p)])


def test_warns_when_reference_temperature_mismatches(tmp_path, capsys):
    rng = np.random.default_rng(2)
    rho = 0.6819 + rng.normal(scale=1e-4, size=2000)
    p = tmp_path / "amm.h5"
    _write(p, rho, temperature=298.0)  # ammonia reference is 239.8 K

    den.main(["--traj", str(p), "--species", "AMM1", "--discard-frac", "0.4"])
    out = capsys.readouterr().out
    assert "WARNING" in out and "temperature" in out


def test_ammonia_reference_is_at_its_boiling_point():
    ref, ref_T = den.REFERENCES["AMM1"]
    assert ref_T < 250.0
    assert 0.6 < ref < 0.75
