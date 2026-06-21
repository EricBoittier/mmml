"""Unit tests for cross-backend MD handoff I/O."""

from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import patch

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


def test_format_fortran_float_rejects_non_finite() -> None:
    from mmml.cli.run.md_handoff import _format_fortran_float

    with pytest.raises(ValueError, match="non-finite"):
        _format_fortran_float(float("nan"))


def test_is_usable_restart_template_accepts_valid_overlap(
    tmp_path: Path, nve_stub: Path
) -> None:
    from mmml.cli.run.md_handoff import _is_usable_restart_template

    overlap = tmp_path / "heat_tag.overlap_a.res"
    overlap.write_text(nve_stub.read_text(encoding="ascii"), encoding="ascii")
    assert _is_usable_restart_template(overlap)
    assert _is_usable_restart_template(nve_stub)
    assert not _is_usable_restart_template(tmp_path / "continue_seed.res")


def test_resolve_handoff_restart_template_prefers_heat_over_overlap(
    nve_stub: Path, tmp_path: Path
) -> None:
    import argparse

    from mmml.cli.run.md_handoff import (
        MdHandoffState,
        resolve_handoff_restart_template,
    )

    job = tmp_path / "init"
    job.mkdir(parents=True)
    heat = job / "heat_dcm_52.res"
    overlap = job / "heat_dcm_52.overlap_a.res"
    heat.write_text(nve_stub.read_text(encoding="ascii"), encoding="ascii")
    overlap.write_text(nve_stub.read_text(encoding="ascii"), encoding="ascii")

    handoff = MdHandoffState(
        positions=np.zeros((20, 3)),
        atomic_numbers=np.ones(20, dtype=int),
    )
    args = argparse.Namespace(handoff_template_res=None, continue_from=None)
    paths = {"heat_res": heat, "equi_res": None, "prod_res": None}
    got = resolve_handoff_restart_template(handoff, args, paths)
    assert got == heat.resolve()


def test_resolve_handoff_restart_template_falls_back_to_overlap_scratch(
    nve_stub: Path, tmp_path: Path
) -> None:
    import argparse

    from mmml.cli.run.md_handoff import (
        MdHandoffState,
        resolve_handoff_restart_template,
    )

    job = tmp_path / "init"
    job.mkdir(parents=True)
    overlap = job / "equi_dcm_52.overlap_b.res"
    overlap.write_text(nve_stub.read_text(encoding="ascii"), encoding="ascii")

    handoff = MdHandoffState(
        positions=np.zeros((20, 3)),
        atomic_numbers=np.ones(20, dtype=int),
    )
    args = argparse.Namespace(handoff_template_res=None, continue_from=None)
    paths = {"heat_res": job / "missing.res", "equi_res": None, "prod_res": None}
    got = resolve_handoff_restart_template(handoff, args, paths)
    assert got == overlap.resolve()


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
        positions=np.zeros((20, 3)),
        atomic_numbers=np.array([1] * 20, dtype=int),
    )
    args = argparse.Namespace(
        handoff_template_res=None,
        continue_from=str(npz),
    )
    got = resolve_handoff_restart_template(handoff, args, {})
    assert got == final_res.resolve()


def test_resolve_handoff_restart_template_rejects_mismatched_atom_count(
    nve_stub: Path, tmp_path: Path
) -> None:
    import argparse
    from mmml.cli.run.md_handoff import resolve_handoff_restart_template

    handoff_dir = tmp_path / "mismatch" / "handoff"
    handoff_dir.mkdir(parents=True)
    final_res = handoff_dir / "final.res"
    # nve_stub has 20 atoms
    final_res.write_text(nve_stub.read_text())
    npz = handoff_dir / "state.npz"
    npz.write_bytes(b"placeholder")

    # handoff has 3 atoms, so it shouldn't match final_res with 20 atoms
    handoff = MdHandoffState(
        positions=np.zeros((3, 3)),
        atomic_numbers=np.array([1, 1, 1], dtype=int),
    )
    args = argparse.Namespace(
        handoff_template_res=None,
        continue_from=str(npz),
    )
    got = resolve_handoff_restart_template(handoff, args, {})
    # Since final_res has 20 atoms, and handoff has 3 atoms, it should reject it and return None
    assert got is None


def test_prepare_pycharmm_handoff_continuation_writes_seed(
    nve_stub: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import argparse

    from mmml.cli.run.md_handoff import (
        prepare_pycharmm_handoff_continuation,
        resolve_handoff_restart_template,
    )

    handoff_dir = tmp_path / "dep" / "handoff"
    handoff_dir.mkdir(parents=True)
    (handoff_dir / "final.res").write_text(nve_stub.read_text())

    pos = np.random.default_rng(0).random((20, 3))
    vel = np.random.default_rng(1).random((20, 3))
    handoff = MdHandoffState(
        positions=pos,
        atomic_numbers=np.ones(20, dtype=int),
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
    (handoff_dir / "state.npz").write_bytes(b"npz")
    template = resolve_handoff_restart_template(handoff, args, {})
    assert template is not None

    seed_path = tmp_path / "prod" / "handoff" / "continue_seed.res"
    restored: list[Path] = []

    def _fake_save(handoff_obj, path, *, template_res=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("restart stub")
        return path

    def _fake_restore(path: Path, *, read_unit: int = 93) -> None:
        restored.append(Path(path))

    bonded_mm_recovery = importlib.import_module(
        "mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery"
    )
    monkeypatch.setattr(bonded_mm_recovery, "restore_charmm_state_from_restart", _fake_restore)
    with patch("mmml.cli.run.md_handoff.save_handoff_to_res", _fake_save):
        seed = prepare_pycharmm_handoff_continuation(
            handoff, args, tmp_path / "prod", {}, quiet=True
        )
    assert seed == seed_path.resolve()
    assert restored == [seed_path.resolve()]
    assert args.restart_from == seed_path.resolve()


def test_resolve_existing_file_path(tmp_path: Path) -> None:
    from mmml.cli.run.md_handoff import _resolve_existing_file_path
    import mmml.cli.run.md_handoff as md_handoff

    # 1. Test None or empty path
    assert _resolve_existing_file_path(None) is None
    assert _resolve_existing_file_path("") is None

    # 2. Test an existing file at direct absolute path
    test_file = tmp_path / "test_file.res"
    test_file.write_text("dummy content")
    assert _resolve_existing_file_path(test_file) == test_file.resolve()

    # Get the repo root used by the helper
    repo_root = Path(md_handoff.__file__).resolve().parents[3]

    # 3. Test matching triggers (e.g. artifacts)
    dummy_artifact = repo_root / "artifacts" / "test_dummy_artifact.res"
    dummy_artifact.parent.mkdir(parents=True, exist_ok=True)
    dummy_artifact.write_text("dummy artifact")
    try:
        # Resolve path using trigger
        fake_mount_path = "/invalid_mount/root/artifacts/test_dummy_artifact.res"
        resolved = _resolve_existing_file_path(fake_mount_path)
        assert resolved == dummy_artifact.resolve()
    finally:
        if dummy_artifact.is_file():
            dummy_artifact.unlink()

    # 4. Test grandparent/parent/filename fallback
    dummy_campaign_file = repo_root / "artifacts" / "pbc_solvent_burst" / "test_camp" / "handoff" / "final.res"
    dummy_campaign_file.parent.mkdir(parents=True, exist_ok=True)
    dummy_campaign_file.write_text("dummy campaign restart")
    try:
        fake_camp_path = "/invalid_mount/root/test_camp/handoff/final.res"
        resolved = _resolve_existing_file_path(fake_camp_path)
        assert resolved == dummy_campaign_file.resolve()
    finally:
        if dummy_campaign_file.is_file():
            dummy_campaign_file.unlink()
            # Clean up the directory structure we created if empty
            try:
                dummy_campaign_file.parent.rmdir()
                dummy_campaign_file.parent.parent.rmdir()
            except OSError:
                pass


def test_resolve_handoff_restart_template_resolves_varying_mount(
    nve_stub: Path, tmp_path: Path
) -> None:
    import argparse
    from mmml.cli.run.md_handoff import resolve_handoff_restart_template
    import mmml.cli.run.md_handoff as md_handoff

    repo_root = Path(md_handoff.__file__).resolve().parents[3]

    dummy_camp_res = repo_root / "artifacts" / "pbc_solvent_burst" / "test_camp_mount" / "handoff" / "final.res"
    dummy_camp_res.parent.mkdir(parents=True, exist_ok=True)
    dummy_camp_res.write_text(nve_stub.read_text())

    npz_path = repo_root / "artifacts" / "pbc_solvent_burst" / "test_camp_mount" / "handoff" / "state.npz"
    npz_path.write_bytes(b"placeholder")

    try:
        handoff = MdHandoffState(
            positions=np.zeros((20, 3)),
            atomic_numbers=np.ones(20, dtype=int),
        )
        foreign_npz_path = "/foreign_mount_point/artifacts/pbc_solvent_burst/test_camp_mount/handoff/state.npz"
        args = argparse.Namespace(
            handoff_template_res=None,
            continue_from=foreign_npz_path,
        )
        got = resolve_handoff_restart_template(handoff, args, {})
        assert got == dummy_camp_res.resolve()
    finally:
        if dummy_camp_res.is_file():
            dummy_camp_res.unlink()
        if npz_path.is_file():
            npz_path.unlink()
        try:
            dummy_camp_res.parent.rmdir()
            dummy_camp_res.parent.parent.rmdir()
            dummy_camp_res.parent.parent.parent.rmdir()
        except OSError:
            pass


def test_resolve_handoff_restart_template_fallback_any_res(
    nve_stub: Path, tmp_path: Path
) -> None:
    import argparse
    from mmml.cli.run.md_handoff import resolve_handoff_restart_template
    import mmml.cli.run.md_handoff as md_handoff

    repo_root = Path(md_handoff.__file__).resolve().parents[3]

    camp_dir = repo_root / "artifacts" / "pbc_solvent_burst" / "test_camp_fallback"
    camp_dir.mkdir(parents=True, exist_ok=True)
    handoff_dir = camp_dir / "handoff"
    handoff_dir.mkdir(parents=True, exist_ok=True)

    other_res = camp_dir / "some_other_step.res"
    other_res.write_text(nve_stub.read_text())

    foreign_npz_path = str(handoff_dir / "state.npz")

    try:
        handoff = MdHandoffState(
            positions=np.zeros((20, 3)),
            atomic_numbers=np.ones(20, dtype=int),
        )
        args = argparse.Namespace(
            handoff_template_res=None,
            continue_from=foreign_npz_path,
        )
        got = resolve_handoff_restart_template(handoff, args, {})
        assert got == other_res.resolve()
    finally:
        if other_res.is_file():
            other_res.unlink()
        try:
            handoff_dir.rmdir()
            camp_dir.rmdir()
            camp_dir.parent.rmdir()
        except OSError:
            pass




