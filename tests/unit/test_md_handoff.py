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


def test_find_latest_charmm_restart_prefers_equi_segment(tmp_path: Path) -> None:
    from mmml.cli.run.md_handoff import find_latest_charmm_restart_in_dir

    (tmp_path / "heat_DCM20.res").write_text("heat", encoding="utf-8")
    equi0 = tmp_path / "equi_DCM20.0.res"
    equi1 = tmp_path / "equi_DCM20.1.res"
    equi0.write_text("equi0", encoding="utf-8")
    equi1.write_text("equi1", encoding="utf-8")
    chosen = find_latest_charmm_restart_in_dir(tmp_path)
    assert chosen == equi1


def test_load_dependency_handoff_falls_back_to_staged_res(
    tmp_path: Path, nve_stub: Path
) -> None:
    import shutil

    from mmml.cli.run.md_handoff import load_dependency_handoff

    dep = tmp_path / "equil"
    dep.mkdir()
    shutil.copy(nve_stub, dep / "equi_DCM20.res")
    handoff = load_dependency_handoff(dep)
    assert handoff is not None
    assert handoff.positions.shape == (20, 3)


def test_apply_handoff_geometry_wraps_pbc_monomers() -> None:
    from ase import Atoms

    from mmml.cli.run.md_handoff import MdHandoffState, apply_handoff_geometry_to_atoms

    L = 32.0
    cell = np.diag([L, L, L])
    pos = np.zeros((10, 3), dtype=float)
    pos[0:5] = [1.0, 16.0, 16.0]
    pos[5:10] = [33.0, 16.0, 16.0]
    z = np.array([6, 1, 1, 1, 1] * 2, dtype=np.int32)
    handoff = MdHandoffState(
        positions=pos.copy(),
        atomic_numbers=z,
        cell=cell,
        pbc=True,
    )
    atoms = Atoms(numbers=z, positions=pos)
    atoms.set_cell(cell)
    atoms.set_pbc(True)
    offsets = np.array([0, 5, 10], dtype=int)
    apply_handoff_geometry_to_atoms(
        atoms, handoff, monomer_offsets=offsets, sync_charmm=False
    )
    wrapped = atoms.get_positions()
    assert not np.allclose(wrapped, pos)
    assert np.all(wrapped[:, 0] >= -1.0e-6)
    assert np.all(wrapped[:, 0] < L + 1.0e-6)


def test_resolve_handoff_restart_template_uses_continue_from_final_res(
    nve_stub: Path, tmp_path: Path
) -> None:
    import argparse

    from mmml.cli.run.md_handoff import resolve_handoff_restart_template

    handoff_dir = tmp_path / "jaxmd_nve" / "handoff"
    handoff_dir.mkdir(parents=True)
    final_res = handoff_dir / "final.res"
    final_res.write_text(nve_stub.read_text())
    npz = handoff_dir / "state.npz"
    npz.write_bytes(b"placeholder")

    handoff = MdHandoffState(
        positions=np.zeros((3, 3)),
        atomic_numbers=np.array([1, 1, 1], dtype=int),
    )
    args = argparse.Namespace(
        handoff_template_res=None,
        continue_from=str(npz),
    )
    got = resolve_handoff_restart_template(handoff, args, {})
    assert got == final_res.resolve()


def test_prepare_pycharmm_handoff_continuation_writes_seed(
    nve_stub: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import argparse

    from mmml.cli.run.md_handoff import prepare_pycharmm_handoff_continuation

    handoff_dir = tmp_path / "dep" / "handoff"
    handoff_dir.mkdir(parents=True)
    (handoff_dir / "final.res").write_text(nve_stub.read_text())

    pos = np.random.default_rng(0).random((3, 3))
    vel = np.random.default_rng(1).random((3, 3))
    handoff = MdHandoffState(
        positions=pos,
        atomic_numbers=np.array([1, 1, 1], dtype=int),
        velocities=vel,
        cell=np.diag([32.0, 32.0, 32.0]),
        pbc=True,
    )
    args = argparse.Namespace(
        handoff_template_res=None,
        continue_from=str(handoff_dir / "state.npz"),
        continue_velocities=True,
        restart_from=None,
    )
    restored: list[Path] = []

    def _fake_restore(path: Path, *, read_unit: int = 93) -> None:
        restored.append(Path(path))

    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery.restore_charmm_state_from_restart",
        _fake_restore,
    )
    seed = prepare_pycharmm_handoff_continuation(
        handoff, args, tmp_path / "prod", {}, quiet=True
    )
    assert seed is not None
    assert seed.name == "continue_seed.res"
    assert seed.is_file()
    assert restored == [seed]
    assert args.restart_from == seed
