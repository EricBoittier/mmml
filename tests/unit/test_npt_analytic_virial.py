"""Analytic NpT virial (``make_npt_energy_fn(virial="analytic")``).

The barostat's dE/d(perturbation) is the chain rule ``(-F) . dreal/dp + (dE/dbox) . dbox/dp``
instead of a central difference of the energy along the strain. In float64 both agree;
in float32 the difference of two large, rounded energies is noise (the hybrid ACO:266
32 A box gave 0.1-2 katm per call at h = 1e-3), while the analytic form is not.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("jax_md")

from jax_md import quantity  # noqa: E402

import mmml.cli.run.jaxmd_runner as runner  # noqa: E402

SIGMA = 3.4
EPS = 0.0104
L_BOX = 12.0


@pytest.fixture(autouse=True)
def _x64():
    prev = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _frac(seed=0, n_side=3, jitter=0.03, dtype=jnp.float64):
    g = (np.arange(n_side) + 0.5) / n_side
    f = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    f = f + np.random.default_rng(seed).uniform(-jitter, jitter, f.shape)
    return jnp.asarray(np.mod(f, 1.0), dtype=dtype)


def _make_lj(offset=0.0):
    """Minimum-image LJ on real positions with an explicit box (lattice shift under stop_gradient)."""

    def energy(real_pos, box, neighbor=None):
        box = jnp.asarray(box)
        lengths = jnp.diagonal(box) if box.ndim == 2 else box
        d = real_pos[:, None, :] - real_pos[None, :, :]
        d = d - lengths * jax.lax.stop_gradient(jnp.round(d / lengths))
        n = real_pos.shape[0]
        r2 = jnp.sum(d * d, -1) + jnp.eye(n, dtype=d.dtype)
        sr6 = (SIGMA**2 / r2) ** 3
        e = 4.0 * EPS * (sr6 * sr6 - sr6)
        return 0.5 * jnp.sum(jnp.where(jnp.eye(n, dtype=bool), 0.0, e)) + offset

    def force(real_pos, box, neighbor=None):
        return -jax.grad(energy)(real_pos, box)

    return energy, force


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_analytic_matches_finite_difference_in_float64(seed):
    e, f = _make_lj()
    _, fn_fd = runner.make_npt_energy_fn(e, f, dtype=jnp.float64, virial="fd")
    _, fn_an = runner.make_npt_energy_fn(e, f, dtype=jnp.float64, virial="analytic")
    frac = _frac(seed)
    box = jnp.eye(3, dtype=jnp.float64) * L_BOX
    p_fd = float(quantity.pressure(fn_fd, frac, box, kinetic_energy=0.3))
    p_an = float(quantity.pressure(fn_an, frac, box, kinetic_energy=0.3))
    assert p_an == pytest.approx(p_fd, rel=1e-7)
    s_fd = np.asarray(quantity.stress(fn_fd, frac, box))
    s_an = np.asarray(quantity.stress(fn_an, frac, box))
    np.testing.assert_allclose(s_an, s_fd, rtol=1e-6, atol=1e-12)


def test_analytic_box_term_is_material():
    """The lattice (dE/dbox) channel carries real weight: dropping it changes P."""
    e, f = _make_lj()
    _, fn_an = runner.make_npt_energy_fn(e, f, dtype=jnp.float64, virial="analytic")
    frac = _frac(0, jitter=0.08)
    box = jnp.eye(3, dtype=jnp.float64) * L_BOX
    real = frac @ box.T
    central = float(jnp.sum(f(real, box) * real)) / (3 * L_BOX**3)
    p_vir = float(quantity.pressure(fn_an, frac, box, kinetic_energy=0.0))
    assert abs(p_vir - central) > 1e-3 * abs(p_vir)


def test_analytic_survives_float32_energy_rounding():
    """A large energy offset (absolute PhysNet/MM sums) wrecks the float32 difference only."""
    e64, f64 = _make_lj()
    _, ref = runner.make_npt_energy_fn(e64, f64, dtype=jnp.float64, virial="fd")
    frac64 = _frac(1)
    box64 = jnp.eye(3, dtype=jnp.float64) * L_BOX
    p_ref = float(quantity.pressure(ref, frac64, box64, kinetic_energy=0.0))

    e32, f32 = _make_lj(offset=jnp.float32(5.0e4))
    frac = jnp.asarray(frac64, jnp.float32)
    box = jnp.asarray(box64, jnp.float32)
    _, fd32 = runner.make_npt_energy_fn(e32, f32, dtype=jnp.float32, virial="fd")
    _, an32 = runner.make_npt_energy_fn(e32, f32, dtype=jnp.float32, virial="analytic")
    p_fd = float(quantity.pressure(fd32, frac, box, kinetic_energy=0.0))
    p_an = float(quantity.pressure(an32, frac, box, kinetic_energy=0.0))
    assert abs(p_an - p_ref) <= 1e-4 * abs(p_ref)
    assert abs(p_fd - p_ref) > 10 * abs(p_an - p_ref)


def test_env_selects_virial(monkeypatch):
    e, f = _make_lj()
    monkeypatch.setenv(runner.NPT_VIRIAL_ENV, "bogus")
    with pytest.raises(ValueError):
        runner.make_npt_energy_fn(e, f, dtype=jnp.float64)
