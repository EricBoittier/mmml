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


def test_save_handoff_to_res_crystal_parameters_patching(tmp_path: Path) -> None:
    # 1. Create a dummy template containing crystal parameters
    template_content = """REST  SYNTHETIC-HANDOFF      0
 !NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,SEED,FIRSTT,FINALT,TBATH,TOL,IHTFRQ,IUNSAV
         2         0         0         1         0         0         0  0.000000000000000D+00  3.000000000000000D+02  3.000000000000000D+02  1.000000000000000D-10         0        -1
 !CRYSTAL PARAMETERS
 1.225025245184120D+02 0.000000000000000D+00 0.000000000000000D+00
 0.000000000000000D+00 1.225025245184120D+02 0.000000000000000D+00
 0.000000000000000D+00 0.000000000000000D+00 1.225025245184120D+02
 !X, Y, Z
  1.0D+00 2.0D+00 3.0D+00
  4.0D+00 5.0D+00 6.0D+00
"""
    template_res = tmp_path / "template.res"
    template_res.write_text(template_content, encoding="ascii")

    # 2. Handoff state with a different box size (25.0) and positions
    positions = np.array([[1.1, 2.1, 3.1], [4.1, 5.1, 6.1]], dtype=float)
    state = MdHandoffState(
        positions=positions,
        atomic_numbers=np.array([1, 1], dtype=np.int32),
        cell=np.diag([25.0, 25.0, 25.0]),
        pbc=True,
    )

    out = tmp_path / "patched.res"
    save_handoff_to_res(state, out, template_res=template_res)
    
    # Reload and check crystal parameters are updated
    content = out.read_text(encoding="ascii")
    assert " !CRYSTAL PARAMETERS" in content
    assert "2.500000000000000D+01" in content
    assert "1.225025245184120D+02" not in content

    # 3. Test that if pbc is False/cell is None, the !CRYSTAL PARAMETERS section is removed
    state_vac = MdHandoffState(
        positions=positions,
        atomic_numbers=np.array([1, 1], dtype=np.int32),
        cell=None,
        pbc=False,
    )
    out_vac = tmp_path / "patched_vac.res"
    save_handoff_to_res(state_vac, out_vac, template_res=template_res)
    content_vac = out_vac.read_text(encoding="ascii")
    assert " !CRYSTAL PARAMETERS" not in content_vac


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


def test_align_handoff_positions_for_charmm_pbc_shifts_jaxmd_wrap():
    from mmml.cli.run.md_handoff import (
        MdHandoffState,
        align_handoff_positions_for_charmm_pbc,
        monomer_offsets_uniform,
    )

    L = 40.0
    pos = np.zeros((10, 3), dtype=float)
    pos[0:5] = [1.0, 20.0, 20.0]
    pos[5:10] = [39.0, 20.0, 20.0]
    handoff = MdHandoffState(
        positions=pos,
        atomic_numbers=np.ones(10, dtype=np.int32),
        metadata={"backend": "jaxmd"},
    )
    offsets = monomer_offsets_uniform(10, 2)
    aligned = align_handoff_positions_for_charmm_pbc(
        pos,
        monomer_offsets=offsets,
        box_side_A=L,
        handoff=handoff,
        quiet=True,
    )
    assert np.all(aligned[:, 0] >= -0.5 * L - 1.0e-6)
    assert np.all(aligned[:, 0] <= 0.5 * L + 1.0e-6)
    assert not np.allclose(aligned, pos)


def test_align_handoff_positions_skips_pycharmm_handoff():
    from mmml.cli.run.md_handoff import (
        MdHandoffState,
        align_handoff_positions_for_charmm_pbc,
        monomer_offsets_uniform,
    )

    pos = np.array([[1.0, 2.0, 3.0]], dtype=float)
    handoff = MdHandoffState(
        positions=pos,
        atomic_numbers=np.array([6], dtype=np.int32),
        metadata={"backend": "pycharmm"},
    )
    aligned = align_handoff_positions_for_charmm_pbc(
        pos,
        monomer_offsets=monomer_offsets_uniform(1, 1),
        box_side_A=32.0,
        handoff=handoff,
        quiet=True,
    )
    assert np.allclose(aligned, pos)


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


def test_save_handoff_to_res_unusable_template_fallback_to_memory(
    nve_stub: Path, tmp_path: Path
) -> None:
    import sys
    from unittest.mock import MagicMock
    from mmml.cli.run.md_handoff import save_handoff_to_res

    state = MdHandoffState(
        positions=np.zeros((3, 3)),
        atomic_numbers=np.array([1, 1, 1], dtype=np.int32),
    )
    out = tmp_path / "fallback.res"

    orig_modules = sys.modules.copy()
    try:
        sys.modules["mmml.interfaces.pycharmmInterface.import_pycharmm"] = MagicMock()
        mock_recovery = MagicMock()
        mock_recovery.rewrite_dynamics_restart_validated.return_value = True
        sys.modules["mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery"] = mock_recovery

        mock_setup = MagicMock()
        sys.modules["mmml.interfaces.pycharmmInterface.mlpot.setup"] = mock_setup

        res_path = save_handoff_to_res(state, out, template_res=nve_stub)
        assert res_path == out
        mock_setup.sync_charmm_positions.assert_called_once()
        mock_recovery.rewrite_dynamics_restart_validated.assert_called_once_with(out)
    finally:
        sys.modules.clear()
        sys.modules.update(orig_modules)


def test_save_handoff_to_res_unusable_template_resolves_fallback(
    nve_stub: Path, tmp_path: Path
) -> None:
    from mmml.cli.run.md_handoff import save_handoff_to_res, load_handoff_from_res

    # Create directories for failed template and a fallback
    bad_dir = tmp_path / "init" / "handoff"
    bad_dir.mkdir(parents=True)
    bad_template = bad_dir / "final.res"
    # An unusable template: startswith continue_seed or has NaN or wrong natom (nve_stub has 20 atoms)
    bad_template.write_text("continue_seed_style_invalid")

    # A usable fallback template in same folder or parent folder
    good_template = tmp_path / "init" / "fallback_good.res"
    good_template.write_text(nve_stub.read_text(encoding="ascii"), encoding="ascii")

    # State we want to save
    pos = np.random.default_rng(2).random((20, 3))
    state = MdHandoffState(
        positions=pos,
        atomic_numbers=np.ones(20, dtype=int),
    )
    out = tmp_path / "patched.res"

    # Save handoff using bad template. It should find the good fallback and succeed!
    res_path = save_handoff_to_res(state, out, template_res=bad_template)
    assert res_path == out
    assert out.is_file()

    # Load and verify coordinates are patched from our state
    reloaded = load_handoff_from_res(out)
    np.testing.assert_allclose(reloaded.positions, pos, rtol=1e-5, atol=1e-5)


def test_find_usable_fallback_template_prefers_newest_restart(
    nve_stub: Path, tmp_path: Path
) -> None:
    import os
    import time
    from mmml.cli.run.md_handoff import _find_usable_fallback_template

    bad_template = tmp_path / "handoff" / "final.res"
    bad_template.parent.mkdir(parents=True)
    bad_template.write_text("continue_seed_style_invalid")

    older = tmp_path / "heat_old.res"
    newer = tmp_path / "heat_new.res"
    older.write_text(nve_stub.read_text(encoding="ascii"), encoding="ascii")
    newer.write_text(nve_stub.read_text(encoding="ascii"), encoding="ascii")

    old_time = time.time() - 3600
    os.utime(older, (old_time, old_time))
    os.utime(newer, (time.time(), time.time()))

    found = _find_usable_fallback_template(bad_template, expected_natom=20)
    assert found is not None
    assert found.name == "heat_new.res"


def test_prepare_pycharmm_handoff_continuation_no_template_invalid_restart(
    tmp_path: Path,
) -> None:
    import sys
    import argparse
    from unittest.mock import MagicMock, patch
    from mmml.cli.run.md_handoff import prepare_pycharmm_handoff_continuation

    pos = np.random.default_rng(0).random((20, 3))
    handoff = MdHandoffState(
        positions=pos,
        atomic_numbers=np.ones(20, dtype=int),
    )
    args = argparse.Namespace(
        handoff_template_res=None,
        continue_from=None,
        continue_velocities=True,
        restart_from=None,
    )

    orig_modules = sys.modules.copy()
    try:
        sys.modules["mmml.interfaces.pycharmmInterface.import_pycharmm"] = MagicMock()

        mock_recovery = MagicMock()
        # Return False to simulate validation failure (e.g. uninitialized coords)
        mock_recovery.rewrite_dynamics_restart_validated.return_value = False
        sys.modules["mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery"] = mock_recovery

        mock_setup = MagicMock()
        sys.modules["mmml.interfaces.pycharmmInterface.mlpot.setup"] = mock_setup

        with patch("mmml.cli.run.md_handoff.resolve_handoff_restart_template", return_value=None):
            result = prepare_pycharmm_handoff_continuation(
                handoff, args, tmp_path / "prod", {}, quiet=False
            )

        seed = tmp_path / "prod" / "handoff" / "continue_seed.res"
        assert result == seed.resolve()
        assert seed.is_file()
        mock_setup.sync_charmm_positions.assert_called()
        mock_recovery.rewrite_dynamics_restart_validated.assert_called_once_with(seed)
    finally:
        sys.modules.clear()
        sys.modules.update(orig_modules)


def test_handoff_from_charmm_raises_on_all_zero_positions() -> None:
    import sys
    from unittest.mock import MagicMock
    from mmml.cli.run.md_handoff import handoff_from_charmm

    orig_modules = sys.modules.copy()
    try:
        # Mock setup to return zeros
        mock_setup = MagicMock()
        mock_setup.get_charmm_positions_array.return_value = np.zeros((10, 3))
        sys.modules["mmml.interfaces.pycharmmInterface.mlpot.setup"] = mock_setup

        # Mock run_state_checkpoint to return velocities or None
        mock_checkpoint = MagicMock()
        mock_checkpoint._charmm_velocities_array.return_value = None
        sys.modules["mmml.interfaces.pycharmmInterface.mlpot.run_state_checkpoint"] = mock_checkpoint

        # Mock dynamics_validation
        mock_validation = MagicMock()
        sys.modules["mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation"] = mock_validation

        # Call and expect RuntimeError
        with pytest.raises(RuntimeError, match="handoff_from_charmm: CHARMM returned all-zero positions"):
            handoff_from_charmm(
                atomic_numbers=np.ones(10, dtype=int),
                restart_path=None,
            )
    finally:
        sys.modules.clear()
        sys.modules.update(orig_modules)


def test_handoff_from_charmm_recovers_from_restart(tmp_path: Path) -> None:
    import sys
    from unittest.mock import MagicMock
    from mmml.cli.run.md_handoff import handoff_from_charmm

    orig_modules = sys.modules.copy()
    try:
        # Mock setup to return zeros
        mock_setup = MagicMock()
        mock_setup.get_charmm_positions_array.return_value = np.zeros((10, 3))
        sys.modules["mmml.interfaces.pycharmmInterface.mlpot.setup"] = mock_setup

        # Mock run_state_checkpoint to return velocities or None
        mock_checkpoint = MagicMock()
        mock_checkpoint._charmm_velocities_array.return_value = None
        sys.modules["mmml.interfaces.pycharmmInterface.mlpot.run_state_checkpoint"] = mock_checkpoint

        # Mock dynamics_validation to return valid coordinates from read_restart_coordinates
        mock_validation = MagicMock()
        valid_coords = np.ones((10, 3))
        mock_validation.read_restart_coordinates.return_value = valid_coords
        sys.modules["mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation"] = mock_validation

        # Mock import_pycharmm and pbc_env so the rest of handoff_from_charmm works or falls back
        sys.modules["mmml.interfaces.pycharmmInterface.import_pycharmm"] = MagicMock()
        mock_pbc = MagicMock()
        mock_pbc.resolve_charmm_cubic_box_side_A.return_value = (None, None)
        mock_pbc.parse_cubic_box_side_from_charmm_restart.return_value = None
        sys.modules["mmml.interfaces.pycharmmInterface.mlpot.pbc_env"] = mock_pbc

        # Write a dummy restart file so cand.is_file() is True
        dummy_restart = tmp_path / "dummy.res"
        dummy_restart.write_text("dummy restart")

        with pytest.warns(UserWarning, match="handoff_from_charmm: CHARMM in-memory positions are all zero"):
            state = handoff_from_charmm(
                atomic_numbers=np.ones(10, dtype=int),
                restart_path=dummy_restart,
            )
        np.testing.assert_allclose(state.positions, valid_coords)

    finally:
        sys.modules.clear()
        sys.modules.update(orig_modules)


def test_cluster_geometry_from_handoff_warns_on_all_zeros() -> None:
    from mmml.cli.run.md_handoff import MdHandoffState, cluster_geometry_from_handoff

    handoff = MdHandoffState(
        positions=np.zeros((10, 3)),
        atomic_numbers=np.ones(10, dtype=int),
    )
    with pytest.warns(UserWarning, match="cluster_geometry_from_handoff: handoff positions are all zero"):
        cluster_geometry_from_handoff(
            handoff,
            n_molecules=2,
        )


def test_synthetic_restart_usability_and_round_trip(tmp_path: Path) -> None:
    from mmml.cli.run.md_handoff import (
        _write_synthetic_charmm_restart,
        _is_usable_restart_template,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        read_restart_coordinates,
        read_restart_velocities,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
        parse_cubic_box_side_from_charmm_restart,
    )

    positions = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ], dtype=float)
    velocities = np.array([
        [-0.1, -0.2, -0.3],
        [-0.4, -0.5, -0.6],
        [-0.7, -0.8, -0.9],
    ], dtype=float)
    cell = np.diag([30.0, 30.0, 30.0])
    handoff = MdHandoffState(
        positions=positions,
        atomic_numbers=np.array([6, 1, 1], dtype=np.int32),
        velocities=velocities,
        cell=cell,
        pbc=True,
    )

    out_file = tmp_path / "synthetic.res"
    _write_synthetic_charmm_restart(handoff, out_file)

    assert out_file.is_file()
    
    # 1. Verify coordinate parsing
    coords = read_restart_coordinates(out_file)
    assert coords is not None
    np.testing.assert_allclose(coords, positions, rtol=1e-10, atol=1e-10)

    # 2. Verify velocity parsing
    vels = read_restart_velocities(out_file)
    assert vels is not None
    np.testing.assert_allclose(vels, velocities, rtol=1e-10, atol=1e-10)

    # 3. Verify box side parsing
    side = parse_cubic_box_side_from_charmm_restart(out_file)
    assert side is not None
    assert side == pytest.approx(30.0)

    # 4. Verify template validation check passes
    assert _is_usable_restart_template(out_file, expected_natom=3)

    # 5. Verify load_handoff_from_res parses it properly
    reloaded = load_handoff_from_res(out_file, atomic_numbers=handoff.atomic_numbers)
    np.testing.assert_allclose(reloaded.positions, positions, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(reloaded.velocities, velocities, rtol=1e-10, atol=1e-10)
    assert reloaded.cell is not None
    np.testing.assert_allclose(reloaded.cell, cell, rtol=1e-10, atol=1e-10)
    assert reloaded.pbc is True


def test_save_handoff_automatically_resolves_template(nve_stub: Path, tmp_path: Path) -> None:
    from mmml.cli.run.md_handoff import save_handoff, load_handoff_from_res

    # Create a valid stage restart file in the output directory
    (tmp_path / "heat_DCM.res").write_text(nve_stub.read_text(encoding="ascii"), encoding="ascii")

    # Call save_handoff with template_res=None
    pos = np.random.default_rng(3).random((20, 3))
    state = MdHandoffState(
        positions=pos,
        atomic_numbers=np.ones(20, dtype=int),
    )
    
    paths = save_handoff(state, tmp_path, template_res=None)
    
    assert "res" in paths
    final_res = paths["res"]
    assert final_res.name == "final.res"
    assert final_res.is_file()

    # Load and verify coordinates are patched from our state
    reloaded = load_handoff_from_res(final_res)
    np.testing.assert_allclose(reloaded.positions, pos, rtol=1e-5, atol=1e-5)


def test_save_handoff_to_res_charmm_write_fails_falls_back_to_patching(
    nve_stub: Path, tmp_path: Path
) -> None:
    from unittest.mock import patch
    from mmml.cli.run.md_handoff import save_handoff_to_res, load_handoff_from_res

    # State we want to save (20 atoms to match nve_stub)
    pos = np.random.default_rng(42).random((20, 3))
    state = MdHandoffState(
        positions=pos,
        atomic_numbers=np.ones(20, dtype=int),
    )
    out = tmp_path / "patched_fallback.res"

    # Mock _write_handoff_restart_via_charmm to raise ValueError
    with patch(
        "mmml.cli.run.md_handoff._write_handoff_restart_via_charmm",
        side_effect=ValueError("Simulated validation failure"),
    ), patch(
        "mmml.cli.run.md_handoff.pycharmm",
        create=True,
    ):
        res_path = save_handoff_to_res(state, out, template_res=nve_stub)
        assert res_path == out
        assert out.is_file()

        # It should have successfully patched using the fallback offline path!
        reloaded = load_handoff_from_res(out)
        np.testing.assert_allclose(reloaded.positions, pos, rtol=1e-5, atol=1e-5)








