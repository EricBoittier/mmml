"""jax-md metal units measure time in 1/unit['time'] ps (~10.18 fs): integrators must get dt_ps * unit['time']."""
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax_md = pytest.importorskip("jax_md")
import jax.numpy as jnp  # noqa: E402
from jax_md import simulate, space, units  # noqa: E402


def _period_fs(dt_fs: float, convert: bool) -> float:
    """Period of a 1-amu particle on a k = 1 eV/A^2 spring, measured from nve trajectories."""
    unit = units.metal_unit_system()
    dt_ps = dt_fs * 1e-3
    dt = dt_ps * float(unit["time"]) if convert else dt_ps
    _, shift = space.free()
    energy = lambda R: 0.5 * jnp.sum(R**2)  # noqa: E731
    init, apply = simulate.nve(energy, shift, dt)
    state = init(jax.random.PRNGKey(0), jnp.array([[1.0]]), kT=0.0, mass=1.0)
    xs = []
    for _ in range(4000):
        state = apply(state)
        xs.append(float(state.position[0, 0]))
    xs = np.array(xs)
    up = np.where((xs[:-1] < 0) & (xs[1:] >= 0))[0]
    assert len(up) >= 2, "no full oscillation in 4000 steps"
    return float(np.mean(np.diff(up))) * dt_fs


def test_metal_time_unit_conversion_gives_physical_period():
    # analytic: T = 2 pi sqrt(m/k) = 2 pi * 10.1805 fs for m = 1 amu, k = 1 eV/A^2
    expected = 2 * np.pi * 1e3 / float(units.metal_unit_system()["time"])
    assert _period_fs(0.5, convert=True) == pytest.approx(expected, rel=0.01)


def test_runner_passes_converted_dt_to_integrators():
    src = (Path(__file__).resolve().parents[2] / "karml/cli/run/jaxmd_runner.py").read_text()
    assert "dt_md = dt * _time_unit" in src
    assert "simulate.nve(wrapped_force_fn, shift, dt_md)" in src
    assert "nhc_tau * dt_md" in src
    assert 'metal_unit_system()["time"]' in src  # FIRE minimiser
