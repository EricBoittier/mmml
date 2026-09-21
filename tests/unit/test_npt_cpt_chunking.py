"""NPT CPT stability chunking and post-dynamics finite-state checks."""

from __future__ import annotations

from contextlib import nullcontext
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
    DEFAULT_CPT_DYNAMICS_CHUNK_NSTEP,
    _cpt_stability_chunk_nstep,
    _cpt_subchunk_use_in_memory_handoff,
    _dynamics_chunk_state_corrupt,
)
from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
    charmm_coordinates_are_finite,
    charmm_dynamics_energy_is_finite,
    charmm_dynamics_state_is_finite,
    validate_charmm_dynamics_state_after_chunk,
)
from tests.unit.pycharmm_stubs import fake_pycharmm_modules


def test_default_cpt_chunk_nstep_is_250():
    assert DEFAULT_CPT_DYNAMICS_CHUNK_NSTEP == 250


def test_cpt_stability_chunk_nstep_skips_short_and_non_cpt():
    assert _cpt_stability_chunk_nstep({"cpt": True}, 200) is None
    assert _cpt_stability_chunk_nstep({"cpt": False}, 5000) is None


def test_cpt_stability_chunk_nstep_for_long_npt():
    assert _cpt_stability_chunk_nstep({"cpt": True}, 1000) == 250


def test_cpt_stability_chunk_nstep_env_override(monkeypatch):
    monkeypatch.setenv("MMML_CPT_DYNAMICS_CHUNK_NSTEP", "100")
    assert _cpt_stability_chunk_nstep({"cpt": True}, 500) == 100


def test_cpt_subchunk_defaults_to_in_memory_handoff(monkeypatch):
    monkeypatch.delenv("MMML_CPT_READYN_SUBCHUNK", raising=False)
    assert _cpt_subchunk_use_in_memory_handoff() is True


def test_cpt_subchunk_readyn_handoff_opt_in(monkeypatch):
    monkeypatch.setenv("MMML_CPT_READYN_SUBCHUNK", "1")
    assert _cpt_subchunk_use_in_memory_handoff() is False


def test_charmm_coordinates_are_finite_detects_nan():
    pos = pd.DataFrame({"x": [0.0, np.nan], "y": [0.0, 0.0], "z": [0.0, 0.0]})
    fake_coor = mock.MagicMock()
    fake_coor.get_positions.return_value = pos
    with fake_pycharmm_modules(coor=fake_coor):
        assert not charmm_coordinates_are_finite()


def test_charmm_dynamics_energy_is_finite_detects_nan():
    row = pd.DataFrame([{"USER": -1.0, "TOTE": np.nan}])
    fake_energy = mock.MagicMock()
    fake_energy.get_energy.return_value = row
    with fake_pycharmm_modules(energy=fake_energy):
        assert not charmm_dynamics_energy_is_finite()


def test_charmm_dynamics_state_is_finite_requires_both():
    pos = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 0.0], "z": [0.0, 0.0]})
    row = pd.DataFrame([{"USER": -1.0, "GRMS": 0.1}])
    fake_coor = mock.MagicMock()
    fake_coor.get_positions.return_value = pos
    fake_energy = mock.MagicMock()
    fake_energy.get_energy.return_value = row
    with fake_pycharmm_modules(coor=fake_coor, energy=fake_energy):
        assert charmm_dynamics_state_is_finite()


def test_sync_charmm_lists_after_mini_skips_stale_grms_gate():
    from mmml.interfaces.pycharmmInterface.mlpot import dynamics

    fake_lingo = mock.MagicMock()
    fake_pycharmm = mock.MagicMock(lingo=fake_lingo)
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation.assert_charmm_dynamics_chunk_safe"
    ) as assert_safe, mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._import_pycharmm_modules",
        return_value=(fake_pycharmm, None, None, None, None, None),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_levels.charmm_silent_command",
        return_value=nullcontext(),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_nbond_diagnostics.maybe_snapshot_nbond_state"
    ):
        dynamics.sync_charmm_lists_after_mini(quiet=True)

    assert_safe.assert_called_once_with(
        context="CHARMM UPDATE after mini (sync NB/MLpot lists)",
        check_grms=False,
    )
    fake_lingo.charmm_script.assert_has_calls([mock.call("UPDATE"), mock.call("ENER")])


def test_charmm_dynamics_energy_is_plausible_rejects_blowup_totke():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        charmm_dynamics_energy_is_plausible,
        charmm_dynamics_state_is_finite,
    )

    pos = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 0.0], "z": [0.0, 0.0]})
    row = pd.DataFrame([{"USER": -87539.0, "TOTKe": 9.37e8}])
    fake_coor = mock.MagicMock()
    fake_coor.get_positions.return_value = pos
    fake_energy = mock.MagicMock()
    fake_energy.get_energy.return_value = row
    with fake_pycharmm_modules(coor=fake_coor, energy=fake_energy):
        assert not charmm_dynamics_energy_is_plausible()
        assert not charmm_dynamics_state_is_finite()


def test_validate_charmm_dynamics_state_raises_on_energy_blowup():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        validate_charmm_dynamics_state_after_chunk,
    )

    pos = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 0.0], "z": [0.0, 0.0]})
    row = pd.DataFrame([{"USER": -87539.0, "TOTKe": 9.37e8}])
    fake_coor = mock.MagicMock()
    fake_coor.get_positions.return_value = pos
    fake_energy = mock.MagicMock()
    fake_energy.get_energy.return_value = row
    with fake_pycharmm_modules(coor=fake_coor, energy=fake_energy):
        with pytest.raises(RuntimeError, match="energies exceed"):
            validate_charmm_dynamics_state_after_chunk(context="HEAT")


def test_validate_charmm_dynamics_state_raises_on_corruption():
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation.charmm_coordinates_are_finite",
        return_value=False,
    ):
        with pytest.raises(RuntimeError, match="non-finite"):
            validate_charmm_dynamics_state_after_chunk(context="EQUI")


def test_dynamics_chunk_state_corrupt_checks_memory_and_restart(tmp_path):
    bad_restart = tmp_path / "bad.res"
    bad_restart.write_text("REST\n!X, Y, Z\nNAN\n", encoding="utf-8")
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation.charmm_dynamics_state_is_finite",
        return_value=True,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation.restart_has_nonfinite_coordinates",
        return_value=True,
    ):
        assert _dynamics_chunk_state_corrupt(
            overlap_context="EQUI",
            restart_path=bad_restart,
        )


def test_materialize_cpt_subchunk_skips_nstep0_velocity_assign(tmp_path):
    """CPT sub-chunk handoff must snapshot in-memory barostat state, not nstep=0 assign."""
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        _materialize_cpt_subchunk_restart_handoff,
    )

    write_path = tmp_path / "heat.cptsc_a.res"
    chunk_kw = {"cpt": True, "hoover reft": 300.0, "timestep": 0.00025}

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics.assign_velocities_at_temperature",
    ) as assign, mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery.rewrite_dynamics_restart_validated",
        return_value=True,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._valid_restart_file",
        return_value=write_path,
    ):
        _materialize_cpt_subchunk_restart_handoff(
            write_path,
            global_step=250,
            overlap_context="HEAT",
            mlpot_ctx=mock.Mock(),
            chunk_kw=chunk_kw,
        )

    assign.assert_not_called()


def test_cpt_stability_chunking_skips_constant_volume():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import _cpt_stability_chunk_nstep

    assert _cpt_stability_chunk_nstep({"cpt": True, "pmass": 500}, 100000) is not None
    assert _cpt_stability_chunk_nstep({"cpt": True, "pmass": 0}, 100000) is None
    assert _cpt_stability_chunk_nstep({"cpt": False}, 100000) is None


# --- CPT sub-chunk DCD merge (each sub-chunk used to overwrite the chunk DCD) ---

_DYN = "mmml.interfaces.pycharmmInterface.mlpot.dynamics"
_N_ATOMS = 4


class _FakeDynaDcd:
    """Stand-in for ``_run_dynamics_chunk`` writing a real DCD like CHARMM does.

    CHARMM truncates the trajectory on every ``dyna`` open and writes
    ``nstep // nsavc`` frames for a continuation segment, at local steps
    ``nsavc, 2*nsavc, ...``. Every coordinate of a frame holds its global step
    (``start_step`` + steps run so far + local step) so tests can check which
    steps survive the merge.
    """

    def __init__(self, *, segment_local_restart: bool = False, start_step: int = 0) -> None:
        self.start_step = int(start_step)
        self.calls: list[dict] = []
        self.traj_paths: list = []
        # CHARMM with the in-memory CPT handoff restarts its step counter at each
        # overlap chunk: the restart JHSTRT is segment-local (250, 500), not global.
        self.segment_local_restart = segment_local_restart
        self.steps = 0

    def __call__(self, kw, io, *, extra_iokw=None, **_kwargs):
        from pathlib import Path

        from mmml.utils.dcd_writer import save_trajectory_dcd

        traj = io.trajectory if io is not None else None
        self.calls.append(dict(kw))
        self.traj_paths.append(traj)
        if traj is not None and "nsavc" in kw:
            nstep = int(kw["nstep"])
            nsavc = int(kw["nsavc"])
            n_frames = nstep // nsavc
            base = self.start_step + self.steps
            steps = [base + k * nsavc for k in range(1, n_frames + 1)]
            pos = np.array(steps, dtype=float)[:, None, None] * np.ones((1, _N_ATOMS, 3))
            boxes = [np.array([32.0, 32.0, 32.0])] * max(1, n_frames)
            save_trajectory_dcd(
                Path(traj),
                pos,
                [None] * _N_ATOMS,
                boxes=boxes,
                steps_per_frame=nsavc,
            )
        self.steps += int(kw["nstep"])
        if io is not None and io.restart_write is not None:
            from pathlib import Path

            text = "REST\n"
            if self.segment_local_restart:
                text = (
                    f"REST     1    {self.steps:5d}\n"
                    " !NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NSAVL\n"
                    f"   10     0     250       1      10    {self.steps:5d}     297       0       0\n"
                )
            Path(io.restart_write).write_text(text, encoding="utf-8")
        return mock.Mock()


def _run_overlap_cpt_chunk(
    tmp_path,
    *,
    nsavc: int,
    chunk_index: int,
    chunk_nstep: int = 500,
    pmass: float = 400.0,
    corrupt_after: int | None = None,
    restart_write=None,
    segment_local_restart: bool = False,
):
    """Drive the real outer harmonize + CPT sub-chunk runner for one overlap chunk."""
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        CharmmTrajectoryFiles,
        _harmonize_overlap_chunk_frequencies,
        _overlap_should_drop_chunk_trajectory,
        _run_cpt_stability_subchunked,
        _drop_trajectory_io,
    )

    kw = {
        "nstep": chunk_nstep,
        "nsavc": nsavc,
        "_target_dcd_nsavc": nsavc,
        "timestep": 0.0005,
        "cpt": True,
        "pmass": pmass,
    }
    start = chunk_index * chunk_nstep
    _harmonize_overlap_chunk_frequencies(kw, chunk_nstep, global_step_start=start, split_trajectory=True)
    cpt_sub = _cpt_stability_chunk_nstep(kw, chunk_nstep)
    chunk_traj = tmp_path / f"prod.{chunk_index:04d}.dcd"
    io = CharmmTrajectoryFiles(trajectory=chunk_traj, restart_write=restart_write)
    if _overlap_should_drop_chunk_trajectory(
        suppress_trajectory=bool(kw.get("_suppress_trajectory", False)),
        has_nsavc="nsavc" in kw,
        split_trajectory=True,
        bussi_sub_chunk=None,
        cpt_sub_chunk=cpt_sub,
    ):
        io = _drop_trajectory_io(io)
    fake = _FakeDynaDcd(segment_local_restart=segment_local_restart, start_step=start)
    n_state = {"n": 0}

    def corrupt(**_kw):
        n_state["n"] += 1
        return corrupt_after is not None and n_state["n"] > corrupt_after

    with (
        mock.patch(f"{_DYN}._run_dynamics_chunk", side_effect=fake),
        mock.patch(f"{_DYN}._dynamics_chunk_state_corrupt", side_effect=corrupt),
        mock.patch(
            f"{_DYN}._materialize_cpt_subchunk_restart_handoff",
            side_effect=lambda path, **_k: path,
        ),
    ):
        if cpt_sub is None:
            from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
                _run_dynamics_chunk,
            )

            _run_dynamics_chunk(kw, io)
        else:
            _run_cpt_stability_subchunked(
                kw,
                io,
                overlap_context="PROD",
                rng_base=1,
                chunk_nstep=cpt_sub,
                total_nstep=chunk_nstep,
                extra_iokw={},
                log_banner=False,
                global_step_offset=start,
            )
    return fake, chunk_traj


def _frames(path):
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        count_dcd_frames,
        count_readable_dcd_frames,
    )

    return count_dcd_frames(path), count_readable_dcd_frames(path)


def _frame_steps(path):
    """Global step stored in each frame's coordinates by ``_FakeDynaDcd``."""
    from mmml.utils.dcd_writer import _dcd_frame_byte_size, _dcd_header_byte_size

    data = path.read_bytes()
    hdr, n_frames, natoms, uc = _dcd_header_byte_size(data)
    fb = _dcd_frame_byte_size(natoms, has_unitcell=uc)
    x_off = (4 + 48 + 4 if uc else 0) + 4
    out = []
    for i in range(n_frames):
        o = hdr + i * fb + x_off
        out.append(int(round(float(np.frombuffer(data[o : o + 4], dtype="<f4")[0]))))
    return out


def _dcd_header(path):
    """``(NSET, ISTART, NSAVC, NSTEP)`` from a DCD header."""
    import struct

    return struct.unpack("<4i", path.read_bytes()[8:24])


def test_cpt_subchunks_merge_all_frames_nsavc125(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        overlap_chunk_dcd_paths,
    )

    fake, chunk_traj = _run_overlap_cpt_chunk(tmp_path, nsavc=125, chunk_index=0)

    written = [p for p in fake.traj_paths if p is not None]
    assert [c["nstep"] for c in fake.calls] == [250, 250]
    assert len(written) == 2 and len(set(written)) == 2
    assert all(".cptsub" in p.name for p in written)
    assert _frames(chunk_traj) == (4, 4)
    assert list(tmp_path.glob("*.cptsub*")) == []
    assert overlap_chunk_dcd_paths(tmp_path / "prod.dcd") == [chunk_traj]


def test_cpt_subchunks_nsavc500_single_writer(tmp_path):
    fake, chunk_traj = _run_overlap_cpt_chunk(tmp_path, nsavc=500, chunk_index=1)

    # Only the sub-chunk holding global step 1000 writes; it cannot use
    # nsavc == nstep, so it writes 875 and 1000 and the merge keeps 1000.
    assert [p is not None for p in fake.traj_paths] == [False, True]
    assert _frames(chunk_traj) == (1, 1)
    assert _frame_steps(chunk_traj) == [1000]
    assert _dcd_header(chunk_traj)[1:3] == (1000, 500)
    assert list(tmp_path.glob("*.cptsub*")) == []


def test_cpt_subchunks_nsavc4000_nonsave_and_save_chunk(tmp_path):
    _fake0, traj0 = _run_overlap_cpt_chunk(tmp_path, nsavc=4000, chunk_index=0)
    assert not traj0.exists()
    assert list(tmp_path.glob("*.dcd")) == []

    _fake7, traj7 = _run_overlap_cpt_chunk(tmp_path, nsavc=4000, chunk_index=7)
    assert _frames(traj7) == (1, 1)
    assert list(tmp_path.glob("*.cptsub*")) == []


def test_cpt_subchunks_early_break_keeps_frames(tmp_path):
    fake, chunk_traj = _run_overlap_cpt_chunk(tmp_path, nsavc=125, chunk_index=0, chunk_nstep=1000, corrupt_after=1)

    # Sub-chunk 0 ok, sub-chunk 1 reports corrupt state -> loop breaks.
    assert [c["nstep"] for c in fake.calls] == [250, 250]
    assert _frames(chunk_traj) == (4, 4)
    assert list(tmp_path.glob("*.cptsub*")) == []


def test_cpt_subchunks_readyn_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("MMML_CPT_READYN_SUBCHUNK", "1")
    fake, chunk_traj = _run_overlap_cpt_chunk(
        tmp_path,
        nsavc=125,
        chunk_index=0,
        restart_write=tmp_path / "prod.res",
    )

    written = [p for p in fake.traj_paths if p is not None]
    assert len(written) == 2 and all(".cptsub" in p.name for p in written)
    assert [bool(c.get("restart")) for c in fake.calls] == [False, True]
    assert _frames(chunk_traj) == (4, 4)
    assert list(tmp_path.glob("*.cptsub*")) == []


@pytest.mark.parametrize(("nsavc", "frames"), [(125, 4), (500, 1)])
def test_cpt_subchunks_segment_local_restart_step_runs_all_subchunks(tmp_path, nsavc, frames):
    # Live DCM:308 CPT prod: in overlap chunks >= 1 the in-memory handoff restart
    # held a segment-local JHSTRT (250), read as global "250 - offset 500 < 250"
    # -> "short restart" break after sub-chunk 0. Half of every later chunk's
    # steps (and its DCD saves, e.g. the nsavc=500 save at step 1000) were lost.
    fake, chunk_traj = _run_overlap_cpt_chunk(
        tmp_path,
        nsavc=nsavc,
        chunk_index=1,
        restart_write=tmp_path / "prod.res",
        segment_local_restart=True,
    )

    assert [c["nstep"] for c in fake.calls] == [250, 250]
    assert _frames(chunk_traj) == (frames, frames)
    assert list(tmp_path.glob("*.cptsub*")) == []


def test_cpt_subchunks_short_global_restart_still_breaks(tmp_path):
    from pathlib import Path

    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        CharmmTrajectoryFiles,
        _run_cpt_stability_subchunked,
    )

    res = tmp_path / "prod.res"
    calls: list[int] = []

    def fake(kw, io, **_k):
        calls.append(int(kw["nstep"]))
        # Global counter that stopped 100 steps into this 250-step sub-chunk.
        Path(io.restart_write).write_text(
            "REST     1     600\n"
            " !NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NSAVL\n"
            "   10     0     250       1      10      600     297       0       0\n",
            encoding="utf-8",
        )
        return mock.Mock()

    with (
        mock.patch(f"{_DYN}._run_dynamics_chunk", side_effect=fake),
        mock.patch(f"{_DYN}._dynamics_chunk_state_corrupt", return_value=False),
    ):
        _run_cpt_stability_subchunked(
            {"nstep": 500, "cpt": True, "pmass": 400.0, "timestep": 0.0005},
            CharmmTrajectoryFiles(restart_write=res),
            overlap_context="PROD",
            rng_base=1,
            chunk_nstep=250,
            total_nstep=500,
            log_banner=False,
            global_step_offset=500,
        )

    assert calls == [250]


def test_cpt_constant_volume_no_subfiles(tmp_path):
    fake, chunk_traj = _run_overlap_cpt_chunk(tmp_path, nsavc=125, chunk_index=2, pmass=0.0)

    assert [c["nstep"] for c in fake.calls] == [500]
    assert fake.traj_paths == [chunk_traj]
    assert _frames(chunk_traj) == (4, 4)


def test_run_dynamics_with_io_cpt_overlap_merges(tmp_path):
    from pathlib import Path

    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        CharmmTrajectoryFiles,
        run_dynamics_with_io,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        count_overlap_chunk_dcd_frames,
        overlap_chunk_dcd_paths,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import (
        DynamicsOverlapConfig,
    )

    cfg = DynamicsOverlapConfig(
        action="error",
        min_distance_A=0.5,
        check_interval=500,
        n_monomers=2,
        use_pbc=False,
    )
    pos_ok = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
        dtype=float,
    )
    dcd = tmp_path / "prod.dcd"
    io = CharmmTrajectoryFiles(restart_write=tmp_path / "prod.res", trajectory=dcd)
    fake = _FakeDynaDcd()
    with (
        mock.patch(f"{_DYN}._run_dynamics_chunk", side_effect=fake),
        mock.patch(
            f"{_DYN}._materialize_cpt_subchunk_restart_handoff",
            side_effect=lambda path, **_k: Path(path),
        ),
        mock.patch(f"{_DYN}._prepare_overlap_chunk_after_restart"),
        mock.patch(f"{_DYN}._dynamics_chunk_state_corrupt", return_value=False),
        mock.patch(
            "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
            return_value=pos_ok,
        ),
        mock.patch(f"{_DYN}._refresh_restart_write_after_chunk"),
        mock.patch(f"{_DYN}._write_overlap_chunk_numbered_restart", return_value=None),
    ):
        run_dynamics_with_io(
            {
                "nstep": 2000,
                "nsavc": 125,
                "timestep": 0.0005,
                "cpt": True,
                "pmass": 400.0,
            },
            io,
            overlap=cfg,
            overlap_context="PROD",
        )

    assert [c["nstep"] for c in fake.calls] == [250] * 8
    written = [p for p in fake.traj_paths if p is not None]
    assert len(written) == 8 and all(".cptsub" in Path(p).name for p in written)
    assert len(overlap_chunk_dcd_paths(dcd)) == 4
    assert count_overlap_chunk_dcd_frames(dcd) == (16, 16)
    assert list(tmp_path.glob("*.cptsub*")) == []


# --- CPT sub-chunk DCD cadence: frames land on the exact global save steps ---

_CADENCE_TARGETS = [20, 100, 125, 150, 250, 270, 300, 333, 500, 600, 700, 4000]


def _overlap_cpt_frame_steps(tmp_path, *, nsavc, total, chunk_nstep=500, pmass=400.0):
    """Global steps of all frames written across ``total // chunk_nstep`` overlap chunks."""
    steps: list[int] = []
    n_calls = 0
    for ci in range(total // chunk_nstep):
        fake, traj = _run_overlap_cpt_chunk(
            tmp_path, nsavc=nsavc, chunk_index=ci, chunk_nstep=chunk_nstep, pmass=pmass
        )
        n_calls += len(fake.calls)
        assert all(1 <= int(c["nstep"]) <= 250 for c in fake.calls)
        assert all(
            int(c["nsavc"]) < int(c["nstep"])
            for c, p in zip(fake.calls, fake.traj_paths)
            if p is not None
        )
        if traj.is_file():
            chunk_steps = _frame_steps(traj)
            nset, istart, hdr_nsavc, _nstep = _dcd_header(traj)
            assert nset == len(chunk_steps)
            if pmass > 0:
                assert hdr_nsavc == nsavc
                assert istart == chunk_steps[0]
            steps += chunk_steps
    assert list(tmp_path.glob("*.cptsub*")) == []
    return steps, n_calls


@pytest.mark.parametrize("total", [2000, 20000])
@pytest.mark.parametrize("nsavc", _CADENCE_TARGETS)
def test_cpt_subchunk_frames_are_exact_global_saves(tmp_path, nsavc, total):
    steps, n_calls = _overlap_cpt_frame_steps(tmp_path, nsavc=nsavc, total=total)

    assert steps == list(range(nsavc, total + 1, nsavc))
    # The DCD cadence never adds dyna calls (each one redraws velocities).
    assert n_calls == total // 250


def test_cpt_subchunk_boundaries_do_not_depend_on_nsavc(tmp_path):
    def call_lengths(nsavc):
        fake, _traj = _run_overlap_cpt_chunk(
            tmp_path, nsavc=nsavc, chunk_index=1, chunk_nstep=501, pmass=400.0
        )
        return [int(c["nstep"]) for c in fake.calls]

    lengths = {nsavc: call_lengths(nsavc) for nsavc in (1, 7, 150, 251, 270, 501)}
    assert all(v == [250, 249, 2] for v in lengths.values()), lengths


@pytest.mark.parametrize("nsavc", [20, 125, 150, 270, 333, 4000])
def test_cpt_subchunk_frames_exact_without_overlap(tmp_path, nsavc):
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        CharmmTrajectoryFiles,
        _run_cpt_stability_subchunked,
    )

    total = 20000
    dcd = tmp_path / "prod.dcd"
    fake = _FakeDynaDcd()
    with (
        mock.patch(f"{_DYN}._run_dynamics_chunk", side_effect=fake),
        mock.patch(f"{_DYN}._dynamics_chunk_state_corrupt", return_value=False),
    ):
        _run_cpt_stability_subchunked(
            {"nstep": total, "nsavc": nsavc, "_target_dcd_nsavc": nsavc, "timestep": 0.0005, "cpt": True, "pmass": 400.0},
            CharmmTrajectoryFiles(trajectory=dcd),
            overlap_context="PROD",
            rng_base=1,
            chunk_nstep=250,
            total_nstep=total,
            log_banner=False,
        )

    assert sum(int(c["nstep"]) for c in fake.calls) == total
    assert _frame_steps(dcd) == list(range(nsavc, total + 1, nsavc))
    assert _dcd_header(dcd)[2] == nsavc
    assert list(tmp_path.glob("*.cptsub*")) == []


def test_cpt_subchunk_nstep_fixed_boundaries():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        cpt_subchunk_nstep,
    )

    def split(total, S):
        done, out = 0, []
        while done < total:
            out.append(cpt_subchunk_nstep(done, total, S))
            done += out[-1]
        return out

    assert split(500, 250) == [250, 250]
    assert split(501, 250) == [250, 249, 2]
    assert split(620, 250) == [250, 250, 120]
    assert split(250, 250) == [250]
    assert split(5, 2) == [2, 2, 1]
    assert split(3, 1) == [1, 1, 1]


def test_cpt_dcd_segment_brute_force_on_fixed_boundaries():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        cpt_dcd_segment,
        cpt_subchunk_nstep,
    )

    for S in (250, 100, 7, 3):
        for outer in (333, 500, 501, 750):
            for t in [*range(1, 130), 149, 150, 199, 250, 251, 270, 333, 499, 500, 600, 4000]:
                for cs in range(0, 3 * outer, outer):
                    done, got = 0, []
                    while done < outer:
                        n = cpt_subchunk_nstep(done, outer, S)
                        seg = cpt_dcd_segment(cs + done, n, t)
                        assert seg.start == cs + done and seg.nstep == n
                        assert not seg.dropped
                        if seg.nsavc is not None:
                            assert 1 <= seg.nsavc < seg.nstep
                        got += seg.kept_steps()
                        done += n
                    exp = list(range((cs // t + 1) * t, cs + outer + 1, t))
                    assert got == exp, (S, outer, t, cs)


def test_cpt_dcd_segment_one_step_save_is_dropped_not_raised():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import cpt_dcd_segment

    seg = cpt_dcd_segment(9, 1, 5)
    assert seg.nsavc is None and seg.dropped == (10,)
    assert cpt_dcd_segment(10, 1, 5).dropped == ()


@pytest.mark.parametrize("stability", [1, 2])
def test_cpt_tiny_stability_size_does_not_abort(tmp_path, stability):
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        CharmmTrajectoryFiles,
        _run_cpt_stability_subchunked,
    )

    total, nsavc = 7, 1
    dcd = tmp_path / "tiny.dcd"
    fake = _FakeDynaDcd()
    with (
        mock.patch(f"{_DYN}._run_dynamics_chunk", side_effect=fake),
        mock.patch(f"{_DYN}._dynamics_chunk_state_corrupt", return_value=False),
    ):
        _run_cpt_stability_subchunked(
            {"nstep": total, "nsavc": nsavc, "_target_dcd_nsavc": nsavc, "timestep": 0.0005, "cpt": True, "pmass": 400.0},
            CharmmTrajectoryFiles(trajectory=dcd),
            overlap_context="PROD",
            rng_base=1,
            chunk_nstep=stability,
            total_nstep=total,
            log_banner=False,
        )
    assert sum(int(c["nstep"]) for c in fake.calls) == total
    steps = _frame_steps(dcd) if dcd.is_file() else []
    # 2-step calls save exactly (nsavc=1 < 2); the 1-step calls' saves are dropped.
    assert steps == ([1, 2, 3, 4, 5, 6] if stability == 2 else [])
    assert list(tmp_path.glob("*.cptsub*")) == []


def test_merge_error_keeps_subfiles(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import _merge_cpt_subchunk_dcds
    from mmml.utils.dcd_writer import save_trajectory_dcd

    a, b = tmp_path / "c.cptsub000.dcd", tmp_path / "c.cptsub001.dcd"
    save_trajectory_dcd(a, np.zeros((2, 4, 3)), [None] * 4, boxes=[np.ones(3)] * 2)
    save_trajectory_dcd(b, np.zeros((2, 4, 3)), [None] * 4, boxes=None)
    with pytest.raises(ValueError, match="unit-cell"):
        _merge_cpt_subchunk_dcds([a, b], tmp_path / "c.dcd")
    assert a.is_file() and b.is_file()


def test_merge_failure_does_not_mask_dynamics_error(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        CharmmTrajectoryFiles,
        _run_cpt_stability_subchunked,
    )

    fake = _FakeDynaDcd()
    n = {"calls": 0}

    def dyna(kw, io, **k):
        n["calls"] += 1
        if n["calls"] == 2:
            raise RuntimeError("dyna blew up")
        return fake(kw, io, **k)

    with (
        mock.patch(f"{_DYN}._run_dynamics_chunk", side_effect=dyna),
        mock.patch(f"{_DYN}._dynamics_chunk_state_corrupt", return_value=False),
        mock.patch(f"{_DYN}._merge_cpt_subchunk_dcds", side_effect=ValueError("merge broke")),
        pytest.raises(RuntimeError, match="dyna blew up"),
    ):
        _run_cpt_stability_subchunked(
            {"nstep": 500, "nsavc": 125, "timestep": 0.0005, "cpt": True, "pmass": 400.0},
            CharmmTrajectoryFiles(trajectory=tmp_path / "prod.dcd"),
            overlap_context="PROD",
            rng_base=1,
            chunk_nstep=250,
            total_nstep=500,
            log_banner=False,
        )


def test_failed_dynamics_still_salvages_frames(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        CharmmTrajectoryFiles,
        _run_cpt_stability_subchunked,
    )

    fake = _FakeDynaDcd()

    def dyna(kw, io, **k):
        if len(fake.calls) == 1:
            raise RuntimeError("dyna blew up")
        return fake(kw, io, **k)

    dcd = tmp_path / "prod.dcd"
    with (
        mock.patch(f"{_DYN}._run_dynamics_chunk", side_effect=dyna),
        mock.patch(f"{_DYN}._dynamics_chunk_state_corrupt", return_value=False),
        pytest.raises(RuntimeError),
    ):
        _run_cpt_stability_subchunked(
            {"nstep": 500, "nsavc": 125, "timestep": 0.0005, "cpt": True, "pmass": 400.0},
            CharmmTrajectoryFiles(trajectory=dcd),
            overlap_context="PROD",
            rng_base=1,
            chunk_nstep=250,
            total_nstep=500,
            log_banner=False,
        )
    assert _frame_steps(dcd) == [125, 250]
    assert list(tmp_path.glob("*.cptsub*")) == []


@pytest.mark.parametrize(
    ("step", "offset", "steps_done", "n", "handoff", "short"),
    [
        # in-memory handoff: chunk-local counter must hit steps_done + n exactly
        (250, 500, 0, 250, False, False),
        (500, 500, 250, 250, False, False),
        (250, 500, 250, 250, False, True),  # 0-step sub-chunk (counter unchanged)
        (100, 500, 0, 250, False, True),
        (1000, 500, 250, 250, False, False),  # global counter reaching the end
        (600, 500, 0, 250, False, True),  # global counter short of 750
        # READYN restart handoff: global counter
        (750, 500, 0, 250, True, False),
        (1000, 500, 250, 250, True, False),
        (250, 500, 0, 250, True, True),  # a chunk-local value is short here
        (750, 500, 250, 250, True, True),  # 0-step sub-chunk
        (-1, 0, 0, 250, True, True),
        (None, 0, 0, 250, False, False),
    ],
)
def test_cpt_subchunk_restart_is_short_by_mode(step, offset, steps_done, n, handoff, short):
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import _cpt_subchunk_restart_is_short

    assert (
        _cpt_subchunk_restart_is_short(
            step,
            global_step_offset=offset,
            steps_done=steps_done,
            n=n,
            restart_handoff=handoff,
        )
        is short
    )
