"""Unit tests for ASE Bussi heat thermostat integration."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities import (
    apply_bussi_velocity_rescale,
    assign_bussi_fallback_velocities,
    calculate_bussi_rescale_alpha,
    target_kinetic_energy_kcalmol,
)
from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
    _requested_heat_thermostat,
    resolve_heat_thermostat,
)
from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
    _apply_bussi_in_memory_continuation_kw,
    _bussi_heat_chunk_nstep,
    _ensure_bussi_heat_continuation_iasvel,
    bussi_heat_ramp_spec_from_kw,
    heat_ramp_spec_from_kw,
    prepare_bussi_heat_dynamics_kw,
)


def test_requested_heat_thermostat_defaults_to_bussi():
    args = mock.Mock(heat_thermostat="bussi")
    assert _requested_heat_thermostat(args) == "bussi"


def test_calculate_bussi_rescale_alpha_is_stochastic():
    target_ke = target_kinetic_energy_kcalmol(300.0, ndof=30)
    alpha_a = calculate_bussi_rescale_alpha(
        target_ke * 0.8,
        target_kinetic_energy=target_ke,
        ndof=30,
        coupling_time_ps=0.1,
        elapsed_time_ps=0.1,
        rng=np.random.default_rng(1),
    )
    alpha_b = calculate_bussi_rescale_alpha(
        target_ke * 0.8,
        target_kinetic_energy=target_ke,
        ndof=30,
        coupling_time_ps=0.1,
        elapsed_time_ps=0.1,
        rng=np.random.default_rng(2),
    )
    assert alpha_a > 0.0
    assert alpha_b > 0.0
    assert alpha_a != pytest.approx(alpha_b)


def test_prepare_bussi_heat_dynamics_kw_disables_charmm_ihtfrq():
    kw = {
        "firstt": 0.0,
        "finalt": 240.0,
        "timestep": 0.00025,
        "nstep": 1000,
    }
    prepare_bussi_heat_dynamics_kw(
        kw,
        nstep=1000,
        ihtfrq=100,
        timestep_ps=0.00025,
    )
    assert kw["ihtfrq"] == 0
    assert "TEMINC" not in kw
    assert kw["_heat_thermostat"] == "bussi"
    spec = bussi_heat_ramp_spec_from_kw(kw)
    assert spec is not None
    assert spec["thermostat"] == "bussi"
    assert int(spec["ihtfrq"]) == 100
    assert _bussi_heat_chunk_nstep(kw, 1000) == 100
    ramp = heat_ramp_spec_from_kw(kw)
    assert ramp is not None
    assert ramp["thermostat"] == "bussi"


def test_capture_charmm_velocities_for_bussi_prefers_live_memory_over_restart(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities import (
        capture_charmm_velocities_for_bussi,
    )

    warm = np.array([[100.0, 0.0, 0.0]], dtype=float)
    restart = tmp_path / "dyn.res"
    restart.write_text(
        "REST     0     1\n"
        "       1 !NTITLE followed by title\n"
        "* t\n"
        "\n"
        " !NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NSAVL\n"
        "         1           0           0           0           0           0           0\n"
        " !X, Y, Z\n"
        " 0.000000000000000D+00 0.000000000000000D+00 0.000000000000000D+00\n"
        " !VELOCITIES\n"
        " 1.000000000000000D+02 0.000000000000000D+00 0.000000000000000D+00\n",
        encoding="ascii",
    )
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.charmm_velocities_akma",
        return_value=warm,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.sync_charmm_velocities_akma",
    ) as sync, mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.velocities_are_cold",
        return_value=False,
    ):
        out = capture_charmm_velocities_for_bussi(restart_path=restart)
    assert out is not None
    np.testing.assert_allclose(out, warm)
    sync.assert_called_once()


def test_capture_charmm_velocities_for_bussi_skips_cold_restart(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities import (
        capture_charmm_velocities_for_bussi,
    )

    warm = np.array([[50.0, 0.0, 0.0]], dtype=float)
    restart = tmp_path / "dyn.res"
    restart.write_text(
        "REST     0     1\n"
        "       1 !NTITLE followed by title\n"
        "* t\n"
        "\n"
        " !NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NSAVL\n"
        "         1           0           0           0           0           0           0\n"
        " !X, Y, Z\n"
        " 0.000000000000000D+00 0.000000000000000D+00 0.000000000000000D+00\n"
        " !VELOCITIES\n"
        " 1.000000000000000D-08 0.000000000000000D+00 0.000000000000000D+00\n",
        encoding="ascii",
    )

    def _cold(vel, **_kwargs):
        arr = np.asarray(vel, dtype=float).reshape(-1, 3)
        return float(np.max(np.abs(arr))) < 1.0

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.charmm_velocities_akma",
        return_value=warm,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.sync_charmm_velocities_akma",
    ) as sync, mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.velocities_are_cold",
        side_effect=_cold,
    ):
        out = capture_charmm_velocities_for_bussi(restart_path=restart)
    assert out is not None
    np.testing.assert_allclose(out, warm)
    sync.assert_called_once()


def test_capture_charmm_velocities_for_bussi_from_restart(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities import (
        capture_charmm_velocities_for_bussi,
    )

    restart = tmp_path / "dyn.res"
    restart.write_text(
        "REST     0     1\n"
        "       1 !NTITLE followed by title\n"
        "* t\n"
        "\n"
        " !NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NSAVL\n"
        "         1           0           0           0           0           0           0\n"
        " !X, Y, Z\n"
        " 0.000000000000000D+00 0.000000000000000D+00 0.000000000000000D+00\n"
        " !VELOCITIES\n"
        " 1.000000000000000D+02 0.000000000000000D+00 0.000000000000000D+00\n",
        encoding="ascii",
    )
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.charmm_velocities_akma",
        return_value=None,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.sync_charmm_velocities_akma",
    ) as sync, mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.velocities_are_cold",
        return_value=False,
    ):
        out = capture_charmm_velocities_for_bussi(restart_path=restart)
    assert out is not None
    assert out.shape == (1, 3)
    sync.assert_called_once()


def test_read_restart_velocities_charmm_vx_vy_vz(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        read_restart_velocities,
    )

    restart = tmp_path / "charmm.res"
    restart.write_text(
        "REST     0    50\n"
        "       1 !NTITLE followed by title\n"
        "* t\n"
        "\n"
        " !NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NSAVL\n"
        "         1           0          50          49          50           0           3\n"
        " !X, Y, Z\n"
        " 0.000000000000000D+00 0.000000000000000D+00 0.000000000000000D+00\n"
        " !VX, VY, VZ\n"
        " 1.000000000000000D+02 0.000000000000000D+00 0.000000000000000D+00\n",
        encoding="ascii",
    )
    vel = read_restart_velocities(restart)
    assert vel is not None
    assert vel.shape == (1, 3)
    assert vel[0, 0] == pytest.approx(100.0)


def test_resolve_dynamics_init_velocities_falls_back_to_iasvel_one_when_cold():
    from unittest.mock import patch

    import numpy as np

    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        _resolve_dynamics_init_velocities,
    )

    kw = {"start": False, "iasvel": 0, "firstt": 12.0}
    cold = np.zeros((4, 3), dtype=float)
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities._resolve_bussi_rescale_velocities",
        return_value=cold,
    ):
        out = _resolve_dynamics_init_velocities(kw, restart_read_path="/tmp/x.res")
    assert out is None
    assert kw["iasvel"] == 1
    assert kw["firstt"] == pytest.approx(12.0)


def test_resolve_dynamics_init_velocities_bussi_uses_mb_fallback_not_iasvel_one():
    from unittest.mock import patch

    import numpy as np

    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        _resolve_dynamics_init_velocities,
    )

    cold = np.zeros((3, 3), dtype=float)
    warm = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    kw = {
        "start": False,
        "iasvel": 0,
        "firstt": 10.0,
        "_heat_thermostat": "bussi",
        "_bussi_ramp": {"firstt": 10.0, "finalt": 50.0, "teminc": 0.8, "ihtfrq": 50},
        "_skip_ase_cold_velocity_assign": True,
    }
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities._resolve_bussi_rescale_velocities",
        return_value=cold,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.assign_bussi_fallback_velocities",
        return_value=(10.0, warm),
    ) as assign:
        out = _resolve_dynamics_init_velocities(kw, restart_read_path=None)
    assign.assert_called_once()
    assert out is not None
    assert kw["iasvel"] == 0
    assert out["vx"].shape == (3,)


def test_run_dynamics_captures_bussi_velocities_before_velos_del():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import run_dynamics

    call_order: list[str] = []

    def _capture(**_kwargs):
        call_order.append("capture")

    def _release():
        call_order.append("release")

    kw = {
        "nstep": 50,
        "start": True,
        "iasvel": 1,
        "firstt": 10.0,
        "finalt": 50.0,
        "timestep": 0.0001,
        "_heat_thermostat": "bussi",
        "_bussi_ramp": {"firstt": 10.0, "finalt": 50.0, "teminc": 0.8, "ihtfrq": 50},
        "_bussi_rescale_interval": 50,
        "_post_dyna_restart_write": "/tmp/fake.res",
    }
    fake_dyn = mock.MagicMock()
    fake_pycharmm = mock.MagicMock()
    fake_pycharmm.DynamicsScript = mock.MagicMock(return_value=fake_dyn)
    with mock.patch.dict("sys.modules", {"pycharmm": fake_pycharmm}), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._run_dynamics_via_c_api",
        return_value=fake_dyn,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._dynamics_c_api_available",
        return_value=True,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._release_charmm_dynamics_api_buffers",
        side_effect=_release,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._apply_dynamics_io_setters",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.capture_charmm_velocities_for_bussi",
        side_effect=_capture,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.mirror_comparison_velocities_for_dynamics",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.maybe_assign_velocities_via_ase_if_cold",
    ):
        run_dynamics(kw)
    assert call_order == ["release", "capture", "release"]


def test_post_dyna_restart_write_path_prefers_staging_alias(tmp_path):
    from mmml.interfaces.pycharmmInterface.charmm_paths import CharmmIoAlias
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        _post_dyna_restart_write_path,
    )

    original = tmp_path / "Heat.res"
    staging = tmp_path / "heat.res"
    staging.write_text("REST\n", encoding="ascii")
    alias = CharmmIoAlias(original=original, alias=staging, for_write=True)
    out = _post_dyna_restart_write_path(original, [alias])
    assert out == staging


def test_post_dyna_restart_write_path_returns_staging_before_file_exists(tmp_path):
    from mmml.interfaces.pycharmmInterface.charmm_paths import CharmmIoAlias
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        _post_dyna_restart_write_path,
    )

    original = tmp_path / "Heat.res"
    staging = tmp_path / "heat.res"
    alias = CharmmIoAlias(original=original, alias=staging, for_write=True)
    out = _post_dyna_restart_write_path(original, [alias])
    assert out == staging


def test_resolve_restart_velocities_read_paths_includes_overlap_slots(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities import (
        resolve_restart_velocities_read_paths,
    )

    final = tmp_path / "heat.res"
    paths = resolve_restart_velocities_read_paths(final)
    assert final.resolve() in paths
    assert (tmp_path / "heat.a.res").resolve() in paths
    assert (tmp_path / "heat.b.res").resolve() in paths


def _write_restart_with_velocities(path: Path, vx: float) -> None:
    path.write_text(
        "REST     0     1\n"
        "       1 !NTITLE followed by title\n"
        "* t\n"
        "\n"
        " !NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NSAVL\n"
        "         1           0           0           0           0           0           0\n"
        " !X, Y, Z\n"
        " 0.000000000000000D+00 0.000000000000000D+00 0.000000000000000D+00\n"
        " !VELOCITIES\n"
        f" {vx:.15E} 0.000000000000000D+00 0.000000000000000D+00\n",
        encoding="ascii",
    )


def test_resolve_restart_velocities_read_paths_from_scratch_slot(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities import (
        resolve_restart_velocities_read_paths,
    )

    scratch = tmp_path / "heat.a.res"
    paths = resolve_restart_velocities_read_paths(scratch)
    assert scratch.resolve() in paths
    assert (tmp_path / "heat.b.res").resolve() in paths
    assert (tmp_path / "heat.res").resolve() in paths


def test_read_restart_velocities_akma_falls_back_to_alternate_slot(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities import (
        _read_restart_velocities_akma,
    )

    cold = tmp_path / "heat.a.res"
    warm = tmp_path / "heat.b.res"
    _write_restart_with_velocities(cold, 1.0e-8)
    _write_restart_with_velocities(warm, 100.0)

    def _cold(vel, **_kwargs):
        arr = np.asarray(vel, dtype=float).reshape(-1, 3)
        return float(np.max(np.abs(arr))) < 50.0

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.velocities_are_cold",
        side_effect=_cold,
    ):
        vel = _read_restart_velocities_akma(cold, quiet=True)
    assert vel is not None
    assert vel[0, 0] == pytest.approx(100.0)


def test_read_restart_velocities_akma_falls_back_to_prior_restart(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities import (
        _read_restart_velocities_akma,
    )

    current = tmp_path / "heat.a.res"
    prior = tmp_path / "pretreat.res"
    _write_restart_with_velocities(current, 1.0e-8)
    _write_restart_with_velocities(prior, 80.0)

    def _cold(vel, **_kwargs):
        arr = np.asarray(vel, dtype=float).reshape(-1, 3)
        return float(np.max(np.abs(arr))) < 50.0

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.velocities_are_cold",
        side_effect=_cold,
    ):
        vel = _read_restart_velocities_akma(
            current,
            fallback_paths=[prior],
            quiet=True,
        )
    assert vel is not None
    assert vel[0, 0] == pytest.approx(80.0)


def test_read_restart_velocities_akma_skips_natom_mismatch(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities import (
        _read_restart_velocities_akma,
    )

    wrong = tmp_path / "pretreat.res"
    _write_restart_with_velocities(wrong, 100.0)
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.charmm_psf_natom",
        return_value=990,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities._restart_file_matches_psf",
        return_value=False,
    ):
        assert _read_restart_velocities_akma(
            None,
            fallback_paths=[wrong],
            quiet=True,
        ) is None


def test_read_restart_velocities_akma_rejects_pathological(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities import (
        _read_restart_velocities_akma,
    )

    hot = tmp_path / "heat.b.res"
    _write_restart_with_velocities(hot, 1.0e12)

    assert _read_restart_velocities_akma(hot, quiet=True) is None


def test_read_restart_velocities_akma_rejects_coords_as_velocities(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities import (
        _read_restart_velocities_akma,
    )

    restart = tmp_path / "coords.res"
    restart.write_text(
        "REST     0     1\n"
        "       1 !NTITLE followed by title\n"
        "* t\n"
        "\n"
        " !NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NSAVL\n"
        "         1           0           0           0           0           0           0\n"
        " !X, Y, Z\n"
        " 1.500000000000000D+00 2.500000000000000D+00 3.500000000000000D+00\n"
        " !VX, VY, VZ\n"
        " 1.500000000000000D+00 2.500000000000000D+00 3.500000000000000D+00\n",
        encoding="ascii",
    )

    assert _read_restart_velocities_akma(restart, quiet=True) is None


def test_apply_bussi_velocity_rescale_rejects_pathological():
    from mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities import (
        apply_bussi_velocity_rescale,
    )

    masses = np.array([12.0], dtype=float)
    bad = np.array([[1.0e12, 0.0, 0.0]], dtype=float)
    good = np.array([[100.0, 0.0, 0.0]], dtype=float)
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.charmm_masses_amu",
        return_value=masses,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities._resolve_bussi_rescale_velocities",
        return_value=bad,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.assign_bussi_fallback_velocities",
        return_value=(40.0, good),
    ) as assign, mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.calculate_bussi_rescale_alpha",
        return_value=1.0,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.sync_charmm_velocities_akma",
    ):
        apply_bussi_velocity_rescale(40.0, timestep_ps=0.0005, quiet=True)
    assign.assert_called_once()


def test_apply_bussi_velocity_rescale_rejects_pathological_after_alpha():
    from mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities import (
        apply_bussi_velocity_rescale,
    )

    masses = np.array([12.0], dtype=float)
    warm = np.array([[1.0e8, 0.0, 0.0]], dtype=float)
    good = np.array([[100.0, 0.0, 0.0]], dtype=float)
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.charmm_masses_amu",
        return_value=masses,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities._resolve_bussi_rescale_velocities",
        return_value=warm,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.assign_bussi_fallback_velocities",
        return_value=(28.0, good),
    ) as assign, mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.velocities_are_pathological",
        side_effect=[False, True],
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.calculate_bussi_rescale_alpha",
        return_value=0.6065,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.sync_charmm_velocities_akma",
    ):
        apply_bussi_velocity_rescale(28.0, timestep_ps=0.0005, quiet=True)
    assign.assert_called_once()


def test_bussi_overlap_skip_scratch_restart_write():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        _bussi_overlap_skip_scratch_restart_write,
        prepare_bussi_heat_dynamics_kw,
    )

    kw = {"firstt": 10.0, "finalt": 300.0, "timestep": 0.0001, "nstep": 50}
    prepare_bussi_heat_dynamics_kw(kw, nstep=50, ihtfrq=50, timestep_ps=0.0001)
    assert _bussi_overlap_skip_scratch_restart_write(
        mem_handoff=True,
        chunk_kw=kw,
        chunk_index=0,
        n_chunks=8,
    )
    assert _bussi_overlap_skip_scratch_restart_write(
        mem_handoff=True,
        chunk_kw=kw,
        chunk_index=6,
        n_chunks=8,
    )
    assert not _bussi_overlap_skip_scratch_restart_write(
        mem_handoff=True,
        chunk_kw=kw,
        chunk_index=7,
        n_chunks=8,
    )


def test_apply_bussi_velocity_rescale_syncs_charmm():
    masses = np.array([12.0, 1.0, 1.0], dtype=float)
    v_akma = np.ones((3, 3), dtype=float) * 100.0
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.charmm_masses_amu",
        return_value=masses,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.capture_charmm_velocities_for_bussi",
        side_effect=[v_akma, v_akma * 1.1],
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.sync_charmm_velocities_akma",
    ) as sync, mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.calculate_bussi_rescale_alpha",
        return_value=1.1,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.estimate_kinetic_temperature_k",
        return_value=295.0,
    ):
        measured, alpha = apply_bussi_velocity_rescale(
            300.0,
            timestep_ps=0.00025,
            rescale_interval_steps=100,
            quiet=True,
        )
    sync.assert_called_once()
    assert alpha == pytest.approx(1.1)
    assert measured == pytest.approx(295.0)


def test_apply_bussi_velocity_rescale_assigns_when_velocities_missing():
    masses = np.array([12.0, 1.0, 1.0], dtype=float)
    v_akma = np.ones((3, 3), dtype=float) * 50.0
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.charmm_masses_amu",
        return_value=masses,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities._resolve_bussi_rescale_velocities",
        return_value=v_akma,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.sync_charmm_velocities_akma",
    ) as sync, mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.calculate_bussi_rescale_alpha",
        return_value=1.0,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.estimate_kinetic_temperature_k",
        return_value=10.0,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.estimate_kinetic_energy_kcalmol",
        return_value=1.0,
    ):
        measured, alpha = apply_bussi_velocity_rescale(
            10.0,
            timestep_ps=0.00025,
            rescale_interval_steps=50,
            quiet=True,
        )
    sync.assert_called_once()
    assert alpha == pytest.approx(1.0)
    assert measured == pytest.approx(10.0)


def test_assign_bussi_fallback_velocities_uses_numpy_when_ase_fails():
    masses = np.array([12.0, 1.0, 1.0], dtype=float)
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.assign_maxwell_boltzmann_velocities_via_ase",
        side_effect=RuntimeError("ase unavailable"),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.charmm_masses_amu",
        return_value=masses,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.sync_charmm_velocities_akma",
    ) as sync:
        measured, vel = assign_bussi_fallback_velocities(300.0, quiet=True, seed=0)
    sync.assert_called_once()
    assert vel.shape == (3, 3)
    assert float(np.max(np.abs(vel))) > 0.0
    assert measured > 0.0


def test_apply_bussi_velocity_rescale_never_raises_when_fallback_assigns():
    masses = np.array([12.0, 1.0, 1.0], dtype=float)
    v_akma = np.ones((3, 3), dtype=float) * 80.0
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.charmm_masses_amu",
        return_value=masses,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities._resolve_bussi_rescale_velocities",
        return_value=v_akma,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.sync_charmm_velocities_akma",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.calculate_bussi_rescale_alpha",
        return_value=1.0,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.estimate_kinetic_temperature_k",
        return_value=12.0,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.estimate_kinetic_energy_kcalmol",
        return_value=2.0,
    ):
        measured, alpha = apply_bussi_velocity_rescale(
            14.0,
            timestep_ps=0.0001,
            rescale_interval_steps=50,
            quiet=True,
        )
    assert alpha == pytest.approx(1.0)
    assert measured == pytest.approx(12.0)


def test_resolve_heat_thermostat_keeps_bussi_after_pretreat(monkeypatch):
    import argparse

    args = argparse.Namespace(
        heat_thermostat="bussi",
        charmm_mm_pretreat=True,
        setup="pbc_liquid",
        quiet=True,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.resolve_charmm_mm_pretreat_for_staged",
        lambda *_a, **_k: True,
    )
    assert resolve_heat_thermostat(args) == "bussi"


def test_apply_bussi_in_memory_continuation_keeps_iasvel_zero():
    kw = {
        "firstt": 10.0,
        "finalt": 50.0,
        "tstruct": 10.0,
        "tbath": 10.0,
        "timestep": 0.0001,
        "nstep": 50,
    }
    prepare_bussi_heat_dynamics_kw(kw, nstep=50, ihtfrq=50, timestep_ps=0.0001)
    _apply_bussi_in_memory_continuation_kw(kw)
    assert kw["iasvel"] == 0
    assert kw["start"] is False
    assert kw["iunrea"] == -1
    assert kw["_skip_ase_cold_velocity_assign"] is True
    assert kw["_bussi_comp_only_handoff"] is True
    assert "firstt" not in kw
    assert "finalt" not in kw
    assert "tstruct" not in kw


def test_resolve_dynamics_init_velocities_bussi_comp_only_skips_c_api():
    from unittest.mock import patch

    import numpy as np

    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        _resolve_dynamics_init_velocities,
    )

    warm = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    kw = {
        "start": False,
        "iasvel": 0,
        "_bussi_comp_only_handoff": True,
        "_heat_thermostat": "bussi",
    }
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.last_synced_velocities_akma_raw",
        return_value=warm,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities._resolve_bussi_rescale_velocities",
    ) as resolve:
        out = _resolve_dynamics_init_velocities(kw, restart_read_path=None)
    assert out is None
    resolve.assert_not_called()


def test_ensure_bussi_heat_continuation_iasvel_for_overlap_chunk():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        _apply_overlap_chunk_dynamics_kw,
    )

    kw = {
        "firstt": 10.0,
        "finalt": 50.0,
        "tstruct": 10.0,
        "timestep": 0.0001,
        "nstep": 50,
        "start": False,
        "iasvel": 1,
    }
    prepare_bussi_heat_dynamics_kw(kw, nstep=50, ihtfrq=50, timestep_ps=0.0001)
    _apply_overlap_chunk_dynamics_kw(kw, chunk_index=1, has_restart_read=False)
    assert kw["iasvel"] == 0
    assert kw["start"] is False
    assert kw["_bussi_comp_only_handoff"] is True
    assert "firstt" not in kw
    assert "tstruct" not in kw


def test_overlap_chunk_uses_memory_handoff_for_bussi(monkeypatch):
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        _overlap_chunk_uses_memory_handoff,
        prepare_bussi_heat_dynamics_kw,
    )

    kw = {"firstt": 10.0, "finalt": 50.0, "timestep": 0.0001, "nstep": 50}
    prepare_bussi_heat_dynamics_kw(kw, nstep=50, ihtfrq=50, timestep_ps=0.0001)
    monkeypatch.delenv("MMML_BUSSI_READYN_OVERLAP", raising=False)
    assert _overlap_chunk_uses_memory_handoff(
        mock.Mock(),
        chunk_index=1,
        n_chunks=4,
        bussi_heat=True,
    )
    monkeypatch.setenv("MMML_BUSSI_READYN_OVERLAP", "1")
    assert not _overlap_chunk_uses_memory_handoff(
        mock.Mock(),
        chunk_index=1,
        n_chunks=4,
        bussi_heat=True,
    )


def test_prepare_post_rescue_overlap_handoff_bussi_uses_in_memory_kw():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        _prepare_post_rescue_overlap_handoff,
        prepare_bussi_heat_dynamics_kw,
    )

    chunk_kw = {
        "firstt": 10.0,
        "finalt": 50.0,
        "timestep": 0.0001,
        "nstep": 50,
        "restart": True,
        "iunrea": 3,
    }
    prepare_bussi_heat_dynamics_kw(
        chunk_kw, nstep=50, ihtfrq=50, timestep_ps=0.0001
    )
    ctx = mock.Mock(
        use_pbc=True,
        charmm_cubic_box_side_A=180.0,
        _overlap_post_rescue_cold_start=False,
    )
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.ensure_charmm_crystal_for_cpt",
    ):
        _prepare_post_rescue_overlap_handoff(chunk_kw, mlpot_ctx=ctx)

    assert chunk_kw["restart"] is False
    assert chunk_kw["iasvel"] == 0
    assert chunk_kw["start"] is False
    assert chunk_kw["iunrea"] == -1
    assert chunk_kw["_skip_ase_cold_velocity_assign"] is True


def test_apply_post_rescue_overlap_handoff_bussi_returns_in_memory():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        CharmmTrajectoryFiles,
        _apply_post_rescue_overlap_handoff,
        prepare_bussi_heat_dynamics_kw,
    )

    chunk_kw = {"firstt": 10.0, "finalt": 50.0, "timestep": 0.0001, "nstep": 50}
    prepare_bussi_heat_dynamics_kw(
        chunk_kw, nstep=50, ihtfrq=50, timestep_ps=0.0001
    )
    chunk_io = CharmmTrajectoryFiles(
        restart_read=Path("/tmp/heat.a.res"),
        restart_write=Path("/tmp/heat.b.res"),
    )
    out_io, in_memory = _apply_post_rescue_overlap_handoff(
        chunk_io,
        chunk_kw,
        steps_done=500,
        mlpot_ctx=mock.Mock(),
        overlap=None,
        overlap_context="HEAT",
    )
    assert in_memory is True
    assert out_io is chunk_io


def test_harmonize_overlap_chunk_preserves_nsavv_when_suppressing_dcd():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        _harmonize_overlap_chunk_frequencies,
    )

    kw = {"nsavc": 499, "nsavv": 499, "nprint": 499}
    _harmonize_overlap_chunk_frequencies(kw, 50, global_step_start=0)
    assert kw["_suppress_trajectory"] is True
    assert kw["nsavc"] == 49
    assert kw["nsavv"] == 50
    assert "nprint" not in kw

    kw_cadence = {
        "nsavc": 499,
        "nsavv": 499,
        "_dyn_freq_cadence": 50,
    }
    _harmonize_overlap_chunk_frequencies(kw_cadence, 50, global_step_start=0)
    assert kw_cadence["_suppress_trajectory"] is True
    assert kw_cadence["nsavv"] == 50


def test_estimate_akma_velocities_from_position_delta():
    from mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities import (
        estimate_akma_velocities_from_position_delta,
        estimate_kinetic_temperature_k,
    )

    masses = np.array([12.0, 1.0, 1.0], dtype=float)
    p0 = np.zeros((3, 3), dtype=float)
    p1 = p0.copy()
    p1[0, 0] = 0.025  # 0.025 Å in 0.025 ps => 1 Å/ps on atom 0
    vel = estimate_akma_velocities_from_position_delta(
        p0, p1, dt_ps=0.025, masses_amu=masses
    )
    assert vel is not None
    assert vel.shape == (3, 3)
    temp = estimate_kinetic_temperature_k(vel, masses)
    assert temp is not None
    assert float(temp) > 0.5


def test_read_restart_coordinate_frames_two_snapshots(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        read_restart_coordinate_frames,
    )

    restart = tmp_path / "hist.res"
    restart.write_text(
        "REST     0     1\n"
        "       1 !NTITLE followed by title\n"
        "* t\n"
        "\n"
        " !NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NSAVL\n"
        "         1           0          50          49          50           0           0\n"
        " !X, Y, Z\n"
        " 0.000000000000000D+00 0.000000000000000D+00 0.000000000000000D+00\n"
        " !X, Y, Z\n"
        " 2.500000000000000D-02 0.000000000000000D+00 0.000000000000000D+00\n",
        encoding="ascii",
    )
    frames = read_restart_coordinate_frames(restart)
    assert len(frames) == 2
    assert frames[1][0, 0] == pytest.approx(0.025)


def test_try_bussi_finite_difference_from_memory(monkeypatch):
    from mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities import (
        _try_bussi_finite_difference_velocities,
    )

    masses = np.array([12.0], dtype=float)
    p0 = np.zeros((1, 3), dtype=float)
    p1 = np.array([[0.025, 0.0, 0.0]], dtype=float)

    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        lambda: p1,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.charmm_masses_amu",
        lambda: masses,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities._charmm_cubic_cell_matrix",
        lambda: None,
    )

    vel = _try_bussi_finite_difference_velocities(
        pos_before=p0,
        dt_ps=0.025,
        quiet=True,
    )
    assert vel is not None
    assert vel.shape == (1, 3)
