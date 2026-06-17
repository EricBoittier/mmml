"""Unit tests for cross-backend MD handoff I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mmml.cli.run.md_handoff import (
    MdHandoffState,
    detect_handoff_format,
    handoff_to_npz_dict,
    load_handoff,
    load_handoff_from_res,
    save_handoff_npz,
    save_handoff_to_res,
)


@pytest.fixture
def nve_stub() -> Path:
    stub = (
        Path(__file__).resolve().parents[1]
        / "functionality/mlpot/output/dynamics/nve_stub.res"
    )
    assert stub.is_file(), f"missing fixture: {stub}"
    return stub


def test_detect_handoff_format_res(nve_stub: Path) -> None:
    assert detect_handoff_format(nve_stub) == "res"


def test_cluster_geometry_from_handoff_skips_packmol_layout() -> None:
    from mmml.cli.run.md_handoff import MdHandoffState, cluster_geometry_from_handoff

    handoff = MdHandoffState(
        positions=np.zeros((100, 3), dtype=float),
        atomic_numbers=np.array([6, 1, 1, 1, 1] * 20, dtype=int),
        cell=np.diag([38.0, 38.0, 38.0]),
        pbc=True,
    )
    z, r0, per, labels, summary = cluster_geometry_from_handoff(
        handoff,
        composition="DCM:20",
    )
    assert len(z) == 100
    assert per == [5] * 20
    assert labels == ["DCM"] * 20
    assert summary == {"DCM": 20}
    assert r0.shape == (100, 3)


def test_pycharmm_stage_dcd_frames_counts_overlap_chunks(tmp_path: Path) -> None:
    from mmml.cli.run.md_stage_summary import pycharmm_stage_dcd_frames

    out = tmp_path
    (out / "equi_dcm_20.chunk.0000.dcd").write_bytes(b"")
    (out / "equi_dcm_20.chunk.0001.dcd").write_bytes(b"")
    assert pycharmm_stage_dcd_frames(out, "equi", "dcm_20") == 0


def test_load_handoff_from_res_round_trip_coords(nve_stub: Path) -> None:
    handoff = load_handoff_from_res(nve_stub)
    assert handoff.positions.shape == (20, 3)
    assert handoff.step == 2000
    assert handoff.pbc is False or handoff.cell is not None
    assert np.all(np.isfinite(handoff.positions))


def test_npz_round_trip(tmp_path: Path) -> None:
    z = np.array([6, 1, 1, 1, 1], dtype=np.int32)
    pos = np.random.default_rng(0).normal(size=(5, 3))
    vel = np.random.default_rng(1).normal(size=(5, 3))
    state = MdHandoffState(
        positions=pos,
        atomic_numbers=z,
        velocities=vel,
        cell=np.diag([20.0, 20.0, 20.0]),
        pbc=True,
        temperature_K=300.0,
        pressure_atm=1.0,
        step=42,
        metadata={"source": "test"},
    )
    path = save_handoff_npz(state, tmp_path / "state.npz")
    loaded = load_handoff(path)
    np.testing.assert_allclose(loaded.positions, pos)
    np.testing.assert_allclose(loaded.velocities, vel)
    assert loaded.step == 42
    assert loaded.temperature_K == pytest.approx(300.0)


def test_save_handoff_to_res_with_template(nve_stub: Path, tmp_path: Path) -> None:
    pos = load_handoff_from_res(nve_stub).positions
    shift = pos + 0.01
    state = MdHandoffState(
        positions=shift,
        atomic_numbers=np.zeros(len(shift), dtype=np.int32),
        velocities=None,
        step=2000,
        metadata={"source": "test"},
    )
    out = tmp_path / "patched.res"
    save_handoff_to_res(state, out, template_res=nve_stub)
    reloaded = load_handoff_from_res(out)
    np.testing.assert_allclose(reloaded.positions, shift, rtol=0, atol=1e-10)


def test_handoff_to_npz_dict_serializes_metadata() -> None:
    state = MdHandoffState(
        positions=np.zeros((2, 3)),
        atomic_numbers=np.array([1, 1], dtype=np.int32),
        metadata={"backend": "jaxmd"},
    )
    d = handoff_to_npz_dict(state)
    assert "metadata" in d
    assert "positions" in d
