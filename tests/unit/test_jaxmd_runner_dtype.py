"""JAX-MD integrator carry follows the configured ML compute dtype."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from karml.cli.run.jaxmd_runner import (
    _JAXMD_DTYPE,
    as_jaxmd_dtype,
    directional_force_energy_error,
    normalize_jaxmd_state,
    nve_force_energy_ablation_verdict,
    nve_force_energy_should_attempt_rescue,
)


def test_as_jaxmd_dtype_uses_configured_dtype():
    f64 = jnp.ones((10, 3), dtype=jnp.float64)
    out = as_jaxmd_dtype(f64)
    assert out.dtype == _JAXMD_DTYPE


def test_as_jaxmd_dtype_casts_float32_to_configured_dtype():
    f32 = jnp.ones((10, 3), dtype=jnp.float32)
    out = as_jaxmd_dtype(f32)
    assert out.dtype == _JAXMD_DTYPE


def test_normalize_jaxmd_state_casts_carry_fields():
    class _State:
        def __init__(self):
            self.position = jnp.ones((2, 3), dtype=jnp.float64)
            self.momentum = jnp.ones((2, 3), dtype=jnp.float64)
            self.mass = jnp.ones((2,), dtype=jnp.float64)

        def set(self, **kwargs):
            out = _State()
            out.position = kwargs.get("position", self.position)
            out.momentum = kwargs.get("momentum", self.momentum)
            out.mass = kwargs.get("mass", self.mass)
            return out

    normed = normalize_jaxmd_state(_State())
    assert normed.position.dtype == _JAXMD_DTYPE
    assert normed.momentum.dtype == _JAXMD_DTYPE
    assert normed.mass.dtype == _JAXMD_DTYPE


def test_directional_force_energy_error_accepts_conservative_force():
    slope, relerr = directional_force_energy_error(
        energy_plus=1.98,
        energy_minus=2.02,
        epsilon_A=0.01,
        projected_force_eV_A=2.0,
    )
    assert np.isclose(slope, -2.0)
    assert relerr < 1.0e-12


def test_directional_force_energy_error_detects_wrong_force():
    _, relerr = directional_force_energy_error(
        energy_plus=1.98,
        energy_minus=2.02,
        epsilon_A=0.01,
        projected_force_eV_A=1.0,
    )
    assert np.isclose(relerr, 0.5)


def test_nve_force_energy_ablation_verdict_ml_path():
    text = nve_force_energy_ablation_verdict(0.28, 0.30, 0.20)
    assert "PBC ML-dimer" in text
    assert "not MM pairs" in text


def test_nve_force_energy_ablation_verdict_mm_path():
    text = nve_force_energy_ablation_verdict(0.28, 0.05, 0.20)
    assert "suspect MM" in text


def test_nve_force_energy_ablation_verdict_q0_hellmann_feynman():
    text = nve_force_energy_ablation_verdict(
        1.2, 0.001, 0.20, mm_charge_mode="q0", used_frozen_mm_charges=False
    )
    assert "Hellmann–Feynman" in text
    assert "q0" in text
    # Once the preflight freezes q, fall back to the generic MM verdict.
    text_frozen = nve_force_energy_ablation_verdict(
        0.28, 0.05, 0.20, mm_charge_mode="q0", used_frozen_mm_charges=True
    )
    assert "suspect MM" in text_frozen


def test_nve_force_energy_ablation_verdict_both_hybrid_worse():
    text = nve_force_energy_ablation_verdict(0.50, 0.25, 0.20)
    assert "MM/hybrid assembly adds" in text


def test_nve_force_energy_ablation_verdict_hybrid_pass_ml_fail():
    text = nve_force_energy_ablation_verdict(0.15, 0.23, 0.20)
    assert "hybrid gate passed" in text
    assert "continuing" in text


def test_nve_force_energy_should_attempt_rescue():
    assert nve_force_energy_should_attempt_rescue(
        0.25, 0.20, rescue_enabled=True, rescue_already_attempted=False
    )
    assert not nve_force_energy_should_attempt_rescue(
        0.15, 0.20, rescue_enabled=True, rescue_already_attempted=False
    )
    assert not nve_force_energy_should_attempt_rescue(
        0.25, 0.20, rescue_enabled=True, rescue_already_attempted=True
    )
    assert not nve_force_energy_should_attempt_rescue(
        0.25, 0.20, rescue_enabled=False, rescue_already_attempted=False
    )


def test_nve_etot_drift_rescue_helpers():
    from karml.cli.run.jaxmd_runner import (
        nve_etot_drift_grace_threshold_eV,
        nve_etot_drift_halved_dt_ps,
        nve_etot_drift_rescue_tricks,
        nve_etot_drift_should_attempt_rescue,
    )

    assert nve_etot_drift_should_attempt_rescue(
        rescue_enabled=True, attempts_used=0, max_attempts=5
    )
    assert nve_etot_drift_should_attempt_rescue(
        rescue_enabled=True, attempts_used=4, max_attempts=5
    )
    assert not nve_etot_drift_should_attempt_rescue(
        rescue_enabled=True, attempts_used=5, max_attempts=5
    )
    assert not nve_etot_drift_should_attempt_rescue(
        rescue_enabled=False, attempts_used=0, max_attempts=5
    )
    assert "grace" in nve_etot_drift_rescue_tricks(0)
    assert "dt_halve" in nve_etot_drift_rescue_tricks(1)
    assert "charmm_rescue" in nve_etot_drift_rescue_tricks(2)
    assert nve_etot_drift_grace_threshold_eV(
        current_threshold_eV=0.5, grace_eV=2.5, attempt_1_based=1
    ) == pytest.approx(2.5)
    assert nve_etot_drift_grace_threshold_eV(
        current_threshold_eV=2.5, grace_eV=2.5, attempt_1_based=3
    ) == pytest.approx(5.0)
    assert nve_etot_drift_halved_dt_ps(0.00025) == pytest.approx(0.000125)
    assert nve_etot_drift_halved_dt_ps(0.00008, min_dt_fs=0.05) == pytest.approx(
        0.00005
    )


def test_jaxmd_suite_nve_preflight_cli_defaults():
    """NVE gates must be wired into jargs (not only suite argparse)."""
    from karml.cli.run.md_pbc_suite import jaxmd as jaxmd_suite

    src = Path(jaxmd_suite.__file__).read_text()
    assert "--nve-etot-drift-abort-eV" in src
    assert "--nve-etot-drift-rescue" in src
    assert "--nve-etot-drift-rescue-attempts" in src
    assert "--nve-max-f-start-eVA" in src
    assert "--nve-force-energy-relative-tolerance" in src
    assert "--nve-force-energy-ml-only-diagnose" in src
    assert "--nve-force-energy-rescue" in src
    assert "--nve-force-energy-rescue-fire-steps" in src
    assert "nve_max_f_start_eVA=" in src
    assert "nve_etot_drift_abort_eV=" in src
    assert "nve_etot_drift_rescue=" in src
    assert "nve_force_energy_ml_only_diagnose=" in src
    assert "nve_force_energy_rescue=" in src
    assert "nve_force_energy_rescue_fire_steps=" in src
    assert "default=1000" in src or "default: 1000" in src
    # Early NVE abort must not crash on missing HDF5 path.
    assert '_hdf5 if _hdf5 else' in src or "last_hdf5_path" in src


def test_resolve_nve_max_f_start_gate_scales_with_system_size():
    from karml.cli.run.jaxmd_runner import (
        NVE_MAX_F_START_BASE_EVA,
        resolve_nve_max_f_start_gate_eVA,
    )

    g_small, s_small = resolve_nve_max_f_start_gate_eVA(1.5, n_atoms=50)
    assert s_small == pytest.approx(1.0)
    assert g_small == pytest.approx(1.5)

    g_ref, s_ref = resolve_nve_max_f_start_gate_eVA(1.5, n_atoms=100)
    assert s_ref == pytest.approx(1.0)
    assert g_ref == pytest.approx(NVE_MAX_F_START_BASE_EVA)

    g_liq, s_liq = resolve_nve_max_f_start_gate_eVA(1.5, n_atoms=2709)
    assert s_liq == pytest.approx((2709 / 100) ** 0.5)
    # TIP3:903: size-scaled default clears a ~6.7 eV/Å post-FIRE start.
    assert g_liq == pytest.approx(1.5 * s_liq)
    assert g_liq > 6.7

    g_off, s_off = resolve_nve_max_f_start_gate_eVA(0.0, n_atoms=2709)
    assert g_off == 0.0
    assert s_off == pytest.approx(1.0)

    g_cap, _ = resolve_nve_max_f_start_gate_eVA(1.5, n_atoms=1_000_000)
    assert g_cap == pytest.approx(15.0)


def test_nve_requires_float64_message_in_runner():
    from karml.cli.run import jaxmd_runner as jr

    src = Path(jr.__file__).read_text()
    assert "NVE requires JAX float64" in src
    assert "jax_enable_x64" in src
    assert "NVE force–energy ML-only ablation" in src
    assert "nve_force_energy_ablation_verdict" in src
    assert "NVE preflight rescue" in src
    assert "force_rebuild=True" in src
    assert "NVE E_tot drift → repair & restart" in src
    assert "nve_etot_drift_rescue_tricks" in src


def test_nve_pbc_does_not_write_molecular_wrap_into_integrator_state():
    """Whole-monomer ±L wraps in state caused ~0.1 eV E_tot jumps at image crossings.

    NL binning may use a wrapped copy; energy/forces must see continuous unwrapped R.
    """
    from karml.cli.run import jaxmd_runner as jr

    src = Path(jr.__file__).read_text(encoding="utf-8")
    assert "wrapped_for_nl" in src
    assert "Do NOT write" in src or "do NOT write" in src
    # The old hot-path pattern must stay gone.
    assert "Wrap coordinates first so neighbor list binning" not in src
    # Still wrap for NL / export, not as the NVE state update before sim().
    step_block = src.split("elif use_pbc and update_fn is not None:")[1].split(
        "else:\n                        state = sim"
    )[0]
    assert "wrapped_for_nl" in step_block
    assert "state.set(position=as_jaxmd_dtype(wrapped" not in step_block


def test_configure_jaxmd_dtype_honours_explicit_dtype(monkeypatch):
    import jax

    from karml.cli.run import jaxmd_runner

    if not jax.config.read("jax_enable_x64"):
        pytest.skip("float64 needs jax_enable_x64")
    monkeypatch.setattr(jaxmd_runner, "_JAXMD_DTYPE", jaxmd_runner._JAXMD_DTYPE)
    assert jaxmd_runner.configure_jaxmd_dtype("float64") == jnp.float64
    assert jaxmd_runner.as_jaxmd_dtype(np.ones(3, dtype=np.float32)).dtype == jnp.float64


def test_epot_blow_up_ignores_arbitrary_zero_crossing():
    from karml.cli.run.jaxmd_runner import epot_blew_up

    # 26 Sep ACO:266 NVT: -87.4 -> +1.1 eV in 0.8 ps is thermalisation (0.03 eV/atom).
    assert not epot_blew_up(1.118, -87.43, 2660)
    assert epot_blew_up(-87.43 + 0.6 * 2660, -87.43, 2660)


def test_nve_float32_runs_with_warning_unless_strict(monkeypatch):
    from karml.cli.run.jaxmd_runner import NVE_REQUIRE_FLOAT64_ENV, nve_float64_policy

    monkeypatch.delenv(NVE_REQUIRE_FLOAT64_ENV, raising=False)
    assert nve_float64_policy(True, jnp.float64) == ("ok", "")
    action, msg = nve_float64_policy(True, jnp.float32)
    assert action == "warn" and "float32" in msg
    assert nve_float64_policy(False, jnp.float32)[0] == "warn"
    action, msg = nve_float64_policy(True, jnp.float32, require_float64=True)
    assert action == "refuse" and msg.startswith("NVE requires JAX float64")
    monkeypatch.setenv(NVE_REQUIRE_FLOAT64_ENV, "1")
    assert nve_float64_policy(True, jnp.float32)[0] == "refuse"
    assert nve_float64_policy(True, jnp.float64)[0] == "ok"


def test_nve_float32_skips_fd_preflight_only_for_float32():
    """The FD gate runs only on float64 (float32 noise > the 0.01 A FD signal)."""
    from karml.cli.run import jaxmd_runner as jr

    src = Path(jr.__file__).read_text(encoding="utf-8")
    assert "if fd_tol > 0.0 and is_f64:" in src
    assert 'run_sim.recoveries["nve_float32_fd_preflight_skipped"] = True' in src


def test_nve_require_float64_flag_reaches_runner():
    import inspect

    from karml.cli.run.md_pbc_suite import jaxmd

    src = inspect.getsource(jaxmd)
    block = src[src.index("jargs = SimpleNamespace(") :]
    block = block[: block.index("set_up_nhc_sim_routine(")]
    assert "nve_require_float64=" in block
    assert jaxmd.build_parser().parse_args(["--nve-require-float64"]).nve_require_float64 is True
    assert jaxmd.build_parser().parse_args([]).nve_require_float64 is False


def test_cast_carry_like_keeps_input_dtypes():
    import jax

    from karml.cli.run.jaxmd_runner import cast_carry_like

    if not jax.config.read("jax_enable_x64"):
        pytest.skip("needs x64 to produce float64 leaves")
    old = {"force": jnp.zeros((2, 3), jnp.float32), "box": jnp.ones((), jnp.float32), "n": jnp.int32(1)}
    new = {"force": jnp.ones((2, 3), jnp.float64), "box": jnp.full((), 2.0, jnp.float64), "n": jnp.int32(2)}
    out = cast_carry_like(new, old)
    assert {k: v.dtype for k, v in out.items()} == {k: v.dtype for k, v in old.items()}
    same = cast_carry_like(old, old)
    assert all(same[k] is old[k] or np.array_equal(same[k], old[k]) for k in old)


def test_summarize_jaxmd_recoveries():
    from karml.cli.run.jaxmd_runner import summarize_jaxmd_recoveries

    out = summarize_jaxmd_recoveries(None, None)
    assert out == {
        "mm_pair_list_refits": 0,
        "mm_pair_capacity_grows": 0,
        "mm_pair_reallocs": 0,
        "mm_pair_fallbacks": 0,
        "nve_float32_fd_preflight_skipped": False,
    }
    out = summarize_jaxmd_recoveries(
        {"nve_float32_fd_preflight_skipped": True}, {"list_refits": 2, "capacity_grows": 1}
    )
    assert out["mm_pair_list_refits"] == 2 and out["mm_pair_capacity_grows"] == 1
    assert out["nve_float32_fd_preflight_skipped"] is True
