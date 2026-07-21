"""JAX-MD PBC neighbor-list defaults (NVE stability + throughput)."""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from mmml.cli.run.jaxmd_runner import (
    JAXMD_FIRE_DT_HIGH_F_PS,
    JAXMD_FIRE_DT_VERY_HIGH_F_PS,
    JAXMD_FIRE_HARD_START_MAX_STEPS_PER_STAGE,
    _nl_update_positions,
    fire_stage_blew_up,
    jaxmd_fire_dt_backoff_schedule,
    resolve_jaxmd_fire_dt_start_ps,
    resolve_jaxmd_fire_stage_steps,
    resolve_jaxmd_steps_per_loop_call,
    resolve_mm_pair_list_capacity,
    resolve_pre_md_fire_start_positions,
    run_jaxmd_fire_with_dt_backoff,
    should_attempt_fire_template_rebuild,
    should_skip_first_fire_when_pbc_fire_follows,
    should_skip_jaxmd_fire,
)
from mmml.interfaces.pycharmmInterface.mm_energy_forces import (
    DEFAULT_JAX_MD_SKIN_DISTANCE_A,
    format_mm_pair_update_stats_summary,
    neighbor_pair_cache_should_reuse,
)

PBC_RECORDING_BLOCK_STEPS = 800
PBC_BOX_A = np.array([40.0, 40.0, 40.0])


def resolve_pbc_loop_steps(jax_md_update_interval: int | None) -> int:
    """Resolve the production PBC case: dynamic MM pairs inside a fixed recording block."""
    return resolve_jaxmd_steps_per_loop_call(
        steps_per_recording=PBC_RECORDING_BLOCK_STEPS,
        use_pbc=True,
        has_update_fn=True,
        jax_md_update_interval=jax_md_update_interval,
    )


def test_default_skin_is_quarter_angstrom():
    assert DEFAULT_JAX_MD_SKIN_DISTANCE_A == 0.25


def test_nl_update_positions_preserves_jax_arrays(monkeypatch):
    monkeypatch.delenv("MMML_MM_NL_FORCE_HOST", raising=False)
    positions = jnp.zeros((2, 3))
    assert _nl_update_positions(positions) is positions


def test_nl_update_positions_force_host_escape_hatch(monkeypatch):
    monkeypatch.setenv("MMML_MM_NL_FORCE_HOST", "1")
    positions = jnp.zeros((2, 3))
    out = _nl_update_positions(positions)
    assert isinstance(out, np.ndarray)


def test_resolve_mm_pair_list_capacity_uses_axis0_not_last():
    """Regression: shape[-1]==2 produced fill fractions like 74400%."""
    pair_idx = np.zeros((20000, 2), dtype=np.int32)
    assert resolve_mm_pair_list_capacity(pair_idx=pair_idx) == 20000


def test_resolve_mm_pair_list_capacity_prefers_get_stats():
    update_fn = SimpleNamespace(get_stats=lambda: {"pair_capacity": 12345})
    pair_idx = np.zeros((20000, 2), dtype=np.int32)
    assert resolve_mm_pair_list_capacity(update_fn=update_fn, pair_idx=pair_idx) == 12345


def test_pre_md_fire_start_keeps_box_frame_under_pbc():
    """PBC FIRE must not COM-center (that plus per-atom wrap splits monomers)."""
    R = np.array([[10.0, 0.0, 0.0], [12.0, 0.0, 0.0]], dtype=np.float32)
    masses = np.array([12.0, 1.0], dtype=np.float32)
    out = resolve_pre_md_fire_start_positions(R, masses, use_pbc=True)
    np.testing.assert_allclose(np.asarray(out), R, atol=1e-6)


def test_pre_md_fire_start_com_centers_in_free_space():
    R = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
    masses = np.array([1.0, 1.0], dtype=np.float32)
    out = np.asarray(resolve_pre_md_fire_start_positions(R, masses, use_pbc=False))
    np.testing.assert_allclose(out.mean(axis=0), [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(out[1, 0] - out[0, 0], 2.0, atol=1e-6)


def test_skin_zero_interval_one_never_reuses():
    R = np.zeros((4, 3), dtype=np.float64)
    assert not neighbor_pair_cache_should_reuse(
        calls=1,
        interval=1,
        skin=0.0,
        R=R,
        last_R=R.copy(),
        box=PBC_BOX_A,
        last_box=PBC_BOX_A.copy(),
        have_cache=True,
    )


def test_default_skin_interval_one_reuses_small_step():
    R0 = np.zeros((4, 3), dtype=np.float64)
    R1 = R0.copy()
    R1[0, 0] = 0.05  # < skin/2 for default skin=0.25
    assert neighbor_pair_cache_should_reuse(
        calls=1,
        interval=1,
        skin=DEFAULT_JAX_MD_SKIN_DISTANCE_A,
        R=R1,
        last_R=R0,
        box=PBC_BOX_A,
        last_box=PBC_BOX_A.copy(),
        have_cache=True,
    )


def test_default_skin_rejects_disp_beyond_half_skin():
    R0 = np.zeros((4, 3), dtype=np.float64)
    R1 = R0.copy()
    R1[0, 0] = 0.13  # > 0.125 = skin/2
    assert not neighbor_pair_cache_should_reuse(
        calls=1,
        interval=1,
        skin=DEFAULT_JAX_MD_SKIN_DISTANCE_A,
        R=R1,
        last_R=R0,
        box=PBC_BOX_A,
        last_box=PBC_BOX_A.copy(),
        have_cache=True,
    )


def test_resolve_steps_per_loop_call_defaults_to_one_for_pbc_with_update_fn():
    assert resolve_pbc_loop_steps(jax_md_update_interval=None) == 1


def test_resolve_steps_per_loop_call_honors_pbc_update_interval():
    assert resolve_pbc_loop_steps(jax_md_update_interval=10) == 10


def test_resolve_steps_per_loop_call_uses_divisor_for_recording_blocks():
    assert resolve_pbc_loop_steps(jax_md_update_interval=30) == 25


def test_format_mm_pair_update_stats_summary():
    line = format_mm_pair_update_stats_summary(
        {"calls": 1000, "reused": 950, "updates": 50, "reallocs": 0, "fallbacks": 0}
    )
    assert "950/1000 reused (95.0%)" in line
    assert "reallocs=0" in line


def test_jaxmd_and_ase_cli_defaults_use_interval_one_conservative_skin():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    jaxmd_src = (root / "mmml/cli/run/md_pbc_suite/jaxmd.py").read_text(encoding="utf-8")
    ase_src = (root / "mmml/cli/run/md_pbc_suite/ase.py").read_text(encoding="utf-8")
    assert "DEFAULT_JAX_MD_SKIN_DISTANCE_A" in jaxmd_src
    assert "default=1" in jaxmd_src.split("jax-md-update-interval")[1][:120]
    assert "DEFAULT_JAX_MD_SKIN_DISTANCE_A" in ase_src
    assert "default=1," in ase_src.split('"--jax-md-update-interval"')[1][:200]
    assert "default=1.75" in ase_src.split('"--jax-md-capacity-multiplier"')[1][:200]


def test_resolve_jaxmd_fire_dt_start_shrinks_for_soft_geometry():
    assert resolve_jaxmd_fire_dt_start_ps(0.08) == pytest.approx(1.0e-4)
    assert resolve_jaxmd_fire_dt_start_ps(0.3) == pytest.approx(3.0e-4)
    assert resolve_jaxmd_fire_dt_start_ps(0.8) == pytest.approx(1.0e-3)


def test_resolve_jaxmd_fire_dt_start_colder_for_hard_geometry():
    """max|F| ≳ 1 must not inherit the historical 1e-3 ps inertial step."""
    assert resolve_jaxmd_fire_dt_start_ps(1.0) == pytest.approx(JAXMD_FIRE_DT_HIGH_F_PS)
    assert resolve_jaxmd_fire_dt_start_ps(7.0) == pytest.approx(JAXMD_FIRE_DT_VERY_HIGH_F_PS)
    assert resolve_jaxmd_fire_dt_start_ps(7.0) < 1.0e-3


def test_jaxmd_fire_dt_backoff_schedule_descends():
    sched = jaxmd_fire_dt_backoff_schedule(1.0e-4)
    assert sched[0] == pytest.approx(1.0e-4)
    assert len(sched) >= 2
    assert sched[1] < sched[0]


def test_fire_stage_blew_up_factor_and_abs_rise():
    assert fire_stage_blew_up(85.0, best_max_f_eVA=6.5, stage_start_max_f_eVA=7.7)
    # Absolute rise vs stage start (5 eV/Å) catches mid-stage spikes sooner.
    assert fire_stage_blew_up(13.0, best_max_f_eVA=6.5, stage_start_max_f_eVA=7.7)
    assert not fire_stage_blew_up(8.0, best_max_f_eVA=6.5, stage_start_max_f_eVA=7.7)


def test_should_skip_first_fire_when_pbc_fire_follows():
    assert should_skip_first_fire_when_pbc_fire_follows(
        use_pbc=True, first_fire_steps=1000, pbc_fire_steps=500
    )
    assert not should_skip_first_fire_when_pbc_fire_follows(
        use_pbc=True, first_fire_steps=1000, pbc_fire_steps=0
    )
    assert not should_skip_first_fire_when_pbc_fire_follows(
        use_pbc=False, first_fire_steps=1000, pbc_fire_steps=500
    )


def test_resolve_jaxmd_fire_stage_steps_caps_hard_starts():
    assert resolve_jaxmd_fire_stage_steps(1000, 7.0) == JAXMD_FIRE_HARD_START_MAX_STEPS_PER_STAGE
    assert resolve_jaxmd_fire_stage_steps(1000, 0.2) == 1000


def test_should_attempt_fire_template_rebuild_on_blowup_or_stall():
    assert should_attempt_fire_template_rebuild(
        {"stages": [{"blew_up": True}], "start_max_f": 7.0},
        6.5,
    )
    assert should_attempt_fire_template_rebuild(
        {"stages": [{"blew_up": False}], "start_max_f": 7.0},
        6.8,
    )
    assert not should_attempt_fire_template_rebuild(
        {"stages": [{"blew_up": False}], "start_max_f": 7.0},
        0.5,
    )


def test_run_jaxmd_fire_aborts_stage_on_blowup_and_tries_colder_dt():
    """A force spike must not freeze the backoff after a tiny early improvement."""
    masses = jnp.ones(2)
    pos0 = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    call = {"n": 0}

    def force_fn(pos, **_kwargs):
        call["n"] += 1
        # First evaluations soft; after a few steps explode (stage blow-up).
        if call["n"] < 4:
            return jnp.array([[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]])
        return jnp.array([[20.0, 0.0, 0.0], [-20.0, 0.0, 0.0]])

    def shift_fn(R, dR, **_kwargs):
        return R + dR

    _pos, best_f, info = run_jaxmd_fire_with_dt_backoff(
        force_fn=force_fn,
        shift_fn=shift_fn,
        positions=pos0,
        masses=masses,
        n_steps=20,
        dt_schedule=(1.0e-3, 3.0e-4),
        worsen_limit=100,
        blowup_factor=3.0,
    )
    assert info["blew_up"] is True
    assert any(s.get("blew_up") for s in info["stages"])
    # Must advance past the first (hot) stage rather than declaring success.
    assert len(info["stages"]) >= 2
    assert best_f < 20.0


def test_should_skip_jaxmd_fire_when_already_soft():
    assert should_skip_jaxmd_fire(0.086)
    assert should_skip_jaxmd_fire(0.10)
    assert not should_skip_jaxmd_fire(0.11)
    assert not should_skip_jaxmd_fire(0.05, skip_below_eVA=0.0)
