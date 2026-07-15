"""JAX-MD integrator carry follows the configured ML compute dtype."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from mmml.cli.run.jaxmd_runner import (
    _JAXMD_DTYPE,
    as_jaxmd_dtype,
    directional_force_energy_error,
    normalize_jaxmd_state,
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


def test_jaxmd_suite_nve_preflight_cli_defaults():
    """NVE gates must be wired into jargs (not only suite argparse)."""
    from mmml.cli.run.md_pbc_suite import jaxmd as jaxmd_suite

    src = Path(jaxmd_suite.__file__).read_text()
    assert "--nve-etot-drift-abort-eV" in src
    assert "--nve-max-f-start-eVA" in src
    assert "--nve-force-energy-relative-tolerance" in src
    assert "nve_max_f_start_eVA=" in src
    assert "nve_etot_drift_abort_eV=" in src
