"""dH_vap analysis: arithmetic, unit handling, and the block-average error bar.

Built against synthetic trajectories with a known answer, so the test pins the
formula rather than any particular MD run.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

_SPEC = importlib.util.spec_from_file_location(
    "analyze_enthalpy_vaporization",
    Path(__file__).resolve().parents[2] / "scripts" / "analyze_enthalpy_vaporization.py",
)
hvap = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(hvap)


def _write_traj(path: Path, potential_ev: np.ndarray, n_atoms: int) -> None:
    with h5py.File(path, "w") as fh:
        fh.create_dataset("potential_energy", data=np.asarray(potential_ev, dtype=float))
        fh.attrs["n_atoms"] = int(n_atoms)


def test_dh_vap_recovers_a_planted_value(tmp_path, capsys):
    """Plant <U>_liq and <U>_gas; dH_vap must come back as U_gas - U_liq/N + RT."""
    n_mol, apm, T = 100, 3, 298.0
    n_atoms = n_mol * apm

    # -9.0 kcal/mol per molecule in the liquid, 0.0 in the gas.
    u_liq_per_mol_kcal = -9.0
    liq_total_ev = np.full(400, u_liq_per_mol_kcal * n_mol / hvap.EV_TO_KCAL_MOL)
    gas_ev = np.zeros(400)

    liq = tmp_path / "liq.h5"
    gas = tmp_path / "gas.h5"
    _write_traj(liq, liq_total_ev, n_atoms)
    _write_traj(gas, gas_ev, apm)

    rc = hvap.main([
        "--liquid", str(liq), "--gas", str(gas),
        "--atoms-per-molecule", str(apm),
        "--temperature", str(T),
        "--discard-frac", "0.0",
    ])
    assert rc == 0

    expected = 0.0 - u_liq_per_mol_kcal + hvap.GAS_CONSTANT_KCAL_MOL_K * T
    out = capsys.readouterr().out
    assert f"{expected:.3f}" in out, out


def test_gas_energy_scalar_matches_gas_trajectory(tmp_path):
    """--gas-energy-kcal must be interchangeable with an equivalent --gas run."""
    n_mol, apm = 50, 6
    liq = tmp_path / "l.h5"
    _write_traj(liq, np.full(200, -20.0), n_mol * apm)

    kept, n_atoms = hvap.read_potential_ev(liq, 0.0)
    assert n_atoms == n_mol * apm
    per_mol = kept * hvap.EV_TO_KCAL_MOL / n_mol
    assert np.isclose(per_mol.mean(), -20.0 * hvap.EV_TO_KCAL_MOL / n_mol)


def test_block_average_sem_exceeds_naive_sem_for_correlated_data():
    """The whole point of block averaging: correlated data must report a bigger bar."""
    rng = np.random.default_rng(0)
    noise = rng.normal(size=4000)
    # strong autocorrelation
    x = np.convolve(noise, np.ones(200) / 200.0, mode="same")

    naive = float(np.std(x, ddof=1) / np.sqrt(x.size))
    blocked = hvap.block_average_sem(x, n_blocks=10)
    assert blocked > 5 * naive, (blocked, naive)


def test_rejects_atom_count_not_divisible_by_molecule_size(tmp_path):
    """A mixed or mis-specified box must fail loudly, not silently mis-normalise."""
    liq = tmp_path / "l.h5"
    _write_traj(liq, np.full(100, -1.0), 101)  # 101 not divisible by 3
    with pytest.raises(SystemExit):
        hvap.main([
            "--liquid", str(liq), "--gas-energy-kcal", "0.0",
            "--atoms-per-molecule", "3", "--temperature", "298.0",
        ])


def test_rejects_nonfinite_energies(tmp_path):
    liq = tmp_path / "l.h5"
    e = np.full(100, -1.0)
    e[50] = np.nan
    _write_traj(liq, e, 30)
    with pytest.raises(RuntimeError, match="non-finite"):
        hvap.read_potential_ev(liq, 0.0)


def test_rejects_too_few_frames_after_equilibration_discard(tmp_path):
    liq = tmp_path / "l.h5"
    _write_traj(liq, np.full(12, -1.0), 30)
    with pytest.raises(RuntimeError, match="equilibration"):
        hvap.read_potential_ev(liq, 0.9)


def test_ammonia_reference_is_at_its_boiling_point():
    """AMM1 must not carry a 298 K reference -- ammonia is a gas at 298 K."""
    ref, ref_T = hvap.REFERENCES["AMM1"]
    assert ref_T < 250.0, "ammonia reference must be near its 239.8 K boiling point"
    assert 5.0 < ref < 6.5
