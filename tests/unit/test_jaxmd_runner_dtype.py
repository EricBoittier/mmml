"""JAX-MD integrator carry follows the configured ML compute dtype."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from mmml.cli.run.jaxmd_runner import (
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
    from mmml.cli.run.jaxmd_runner import (
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
    from mmml.cli.run.md_pbc_suite import jaxmd as jaxmd_suite

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


def test_nve_requires_float64_message_in_runner():
    from mmml.cli.run import jaxmd_runner as jr

    src = Path(jr.__file__).read_text()
    assert "NVE requires JAX float64" in src
    assert "jax_enable_x64" in src
    assert "NVE force–energy ML-only ablation" in src
    assert "nve_force_energy_ablation_verdict" in src
    assert "NVE preflight rescue" in src
    assert "force_rebuild=True" in src
    assert "NVE E_tot drift → repair & restart" in src
    assert "nve_etot_drift_rescue_tricks" in src
