"""jaxmd_runner's NpT energy must use jax-md's LINEAR-strain perturbation.

jax-md (``quantity.pressure`` and the ``npt_nose_hoover`` box force) calls the
energy with ``perturbation = 1 + eps`` where ``eps`` is a linear strain: every
displacement is scaled by ``1 + eps`` so ``V -> V (1 + eps)**3``, and

    P = (2 K - dU/deps) / (3 V)          dU/deps = 3 V dU/dV.

``jaxmd_runner`` used to apply ``perturbation**(1/3)`` to the box (a volume
factor), so dU/deps came out 3x too small and the barostat saw
P_kin + P_vir / 3. Its self-check compared the custom VJP with a finite
difference of the *same* mis-scaled forward and reported 0.000 %.

These tests compare against references that never go through the runner's
``perturbation`` handling:

* an explicit isotropic volume change at fixed fractional coordinates;
* jax-md's own native Lennard-Jones energy, whose perturbation handling defines
  the convention.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jax_md = pytest.importorskip("jax_md")

from jax_md import energy as jmd_energy  # noqa: E402
from jax_md import quantity, simulate, space  # noqa: E402

import mmml.cli.run.jaxmd_runner as runner  # noqa: E402

SIGMA = 3.4  # Angstrom (argon-like)
EPS = 0.0104  # eV
L_BOX = 12.0


@pytest.fixture(autouse=True)
def _x64():
    prev = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _lattice_frac(seed=0, n_side=3, jitter=0.03):
    g = (np.arange(n_side) + 0.5) / n_side
    frac = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    rng = np.random.default_rng(seed)
    frac = frac + rng.uniform(-jitter, jitter, frac.shape)
    return jnp.asarray(np.mod(frac, 1.0), dtype=jnp.float64)


def _mic_lj_energy_of_real(real_pos, box, neighbor=None):
    """Full minimum-image LJ; depends on the box explicitly through the MIC."""
    box = jnp.asarray(box)
    lengths = jnp.diagonal(box) if box.ndim == 2 else box
    d = real_pos[:, None, :] - real_pos[None, :, :]
    d = d - lengths * jnp.round(d / lengths)
    n = real_pos.shape[0]
    r2 = jnp.sum(d * d, -1) + jnp.eye(n)  # keep diagonal finite
    sr6 = (SIGMA**2 / r2) ** 3
    e = 4.0 * EPS * (sr6 * sr6 - sr6)
    return 0.5 * jnp.sum(jnp.where(jnp.eye(n, dtype=bool), 0.0, e))


def _mic_lj_force_of_real(real_pos, box, neighbor=None):
    return -jax.grad(_mic_lj_energy_of_real)(real_pos, box)


def _legacy_apply(box, perturbation, dtype=None):
    """The pre-fix forward: perturbation treated as a VOLUME factor."""
    box = jnp.asarray(box, dtype=dtype)
    if perturbation is None:
        return box
    return box * jnp.power(jnp.asarray(perturbation, dtype=dtype), 1.0 / 3.0)


def _legacy_volume_factor_npt(energy_of_real, force_of_real):
    """Pre-fix behaviour: volume-factor forward, VJP = FD of that forward."""
    _, fn = runner.make_npt_energy_fn(
        energy_of_real, force_of_real, dtype=jnp.float64,
        apply_perturbation=_legacy_apply,
    )
    return fn


def _reference_dE_dV(frac, box, rel_dv=1e-6):
    """Explicit V -> V(1 +/- dv), L -> L (V'/V)^(1/3), fixed fractional coords."""
    V = float(jnp.linalg.det(box))
    es = []
    for s in (1.0, -1.0):
        box_s = box * (1.0 + s * rel_dv) ** (1.0 / 3.0)
        es.append(float(_mic_lj_energy_of_real(space.transform(box_s, frac), box_s)))
    return (es[0] - es[1]) / (2.0 * rel_dv * V)


def _make():
    return runner.make_npt_energy_fn(
        _mic_lj_energy_of_real, _mic_lj_force_of_real, dtype=jnp.float64
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_jaxmd_pressure_matches_independent_finite_difference(seed):
    """(1) quantity.pressure through the runner's energy == 2K/(3V) - dE/dV."""
    _, npt_energy_fn = _make()
    frac = _lattice_frac(seed)
    box = jnp.eye(3, dtype=jnp.float64) * L_BOX
    V = L_BOX**3
    K = 0.37  # eV, arbitrary kinetic energy
    p_jaxmd = float(
        quantity.pressure(npt_energy_fn, frac, box, kinetic_energy=K, neighbor=None)
    )
    dE_dV = _reference_dE_dV(frac, box)
    p_ref = 2.0 * K / (3.0 * V) - dE_dV
    assert abs(dE_dV) > 1e-6, "virial must be material for the test to mean anything"
    assert p_jaxmd == pytest.approx(p_ref, rel=1e-7, abs=0.0)


def test_legacy_volume_factor_convention_gives_one_third_virial():
    """The pre-fix forward reproduces the P_vir/3 bug; guards the test's power."""
    legacy = _legacy_volume_factor_npt(_mic_lj_energy_of_real, _mic_lj_force_of_real)
    _, fixed = _make()
    frac = _lattice_frac(0)
    box = jnp.eye(3, dtype=jnp.float64) * L_BOX
    V = L_BOX**3
    p_ref_vir = -_reference_dE_dV(frac, box)
    p_legacy_vir = -float(runner.barostat_dU_deps(legacy, frac, box, None)) / (3 * V)
    p_fixed_vir = -float(runner.barostat_dU_deps(fixed, frac, box, None)) / (3 * V)
    assert p_legacy_vir / p_ref_vir == pytest.approx(1.0 / 3.0, rel=1e-6)
    assert p_fixed_vir / p_ref_vir == pytest.approx(1.0, rel=1e-7)


def test_pressure_matches_jax_md_native_lennard_jones():
    """Independent convention reference: jax-md's own pair energy."""
    box = jnp.eye(3, dtype=jnp.float64) * L_BOX
    displacement, _ = space.periodic_general(box, fractional_coordinates=True)
    native = jmd_energy.lennard_jones_pair(
        displacement, sigma=SIGMA, epsilon=EPS, r_onset=4.0, r_cutoff=5.5
    )

    def energy_of_real(real_pos, box_eff, neighbor=None):
        frac = space.transform(jnp.linalg.inv(box_eff), real_pos)
        return native(frac, box=box_eff)

    def force_of_real(real_pos, box_eff, neighbor=None):
        return -jax.grad(energy_of_real)(real_pos, box_eff)

    _, npt_energy_fn = runner.make_npt_energy_fn(
        energy_of_real, force_of_real, dtype=jnp.float64
    )
    frac = _lattice_frac(4)
    K = 0.2
    p_native = float(quantity.pressure(native, frac, box, kinetic_energy=K))
    p_runner = float(quantity.pressure(npt_energy_fn, frac, box, kinetic_energy=K))
    assert p_runner == pytest.approx(p_native, rel=1e-7)
    # Anisotropic (matrix perturbation) path agrees with jax-md's stress too.
    s_native = np.asarray(quantity.stress(native, frac, box))
    s_runner = np.asarray(quantity.stress(npt_energy_fn, frac, box))
    np.testing.assert_allclose(s_runner, s_native, rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("p0", [1.0, 1.02])
def test_custom_vjp_perturbation_cotangent_matches_autodiff_and_fd(p0):
    """(2) The custom-VJP cotangent == jax.grad of the forward == FD."""
    raw_fn, npt_energy_fn = _make()
    frac = _lattice_frac(5)
    box = jnp.eye(3, dtype=jnp.float64) * L_BOX
    got = float(jax.grad(lambda p: npt_energy_fn(frac, box, None, p, None, None))(p0))
    auto = float(jax.grad(lambda p: raw_fn(frac, box=box, perturbation=p))(p0))
    h = 1e-6
    fd = (
        float(raw_fn(frac, box=box, perturbation=p0 + h))
        - float(raw_fn(frac, box=box, perturbation=p0 - h))
    ) / (2 * h)
    assert got == pytest.approx(auto, rel=1e-7)
    assert got == pytest.approx(fd, rel=1e-6)
    # At p = 1 the linear-strain derivative is the pair virial -sum_ij f_ij.d_ij,
    # which here (a pure MIC pair potential) is -3 V dE/dV.
    if p0 == 1.0:
        assert got == pytest.approx(3 * L_BOX**3 * _reference_dE_dV(frac, box), rel=1e-7)


def test_selfcheck_uses_independent_reference_and_flags_legacy():
    _, fixed = _make()
    legacy = _legacy_volume_factor_npt(_mic_lj_energy_of_real, _mic_lj_force_of_real)
    frac = _lattice_frac(7)
    box = jnp.eye(3, dtype=jnp.float64) * L_BOX
    ok = runner.npt_virial_selfcheck(
        fixed, _mic_lj_energy_of_real, frac, box, None, dtype=jnp.float64
    )
    bad = runner.npt_virial_selfcheck(
        legacy, _mic_lj_energy_of_real, frac, box, None, dtype=jnp.float64
    )
    assert ok["ratio"] == pytest.approx(1.0, rel=1e-6)
    assert ok["rel_err"] < 1e-6
    assert bad["ratio"] == pytest.approx(1.0 / 3.0, rel=1e-5)
    assert bad["rel_err"] > 0.5


def test_short_npt_run_barostat_pressure_matches_recomputed_true_pressure():
    """(3) Along a real NpT trajectory the barostat's P equals the true P."""
    _, npt_energy_fn = _make()
    frac = _lattice_frac(8, jitter=0.02)
    box = jnp.eye(3, dtype=jnp.float64) * L_BOX
    kT = 8.617333262e-5 * 120.0
    dt = 0.002  # ps (metal units)
    init_fn, apply_fn = simulate.npt_nose_hoover(
        npt_energy_fn,
        space.periodic_general(box, fractional_coordinates=True)[1],
        dt=dt,
        pressure=jnp.asarray(1e-5),
        kT=kT,
        barostat_kwargs=runner.default_nhc_kwargs(100 * dt),
        thermostat_kwargs=runner.default_nhc_kwargs(20 * dt),
    )
    mass = 39.95
    state = init_fn(jax.random.PRNGKey(0), frac, box=box, neighbor=None, mass=mass)
    step = jax.jit(lambda s: apply_fn(s, neighbor=None, pressure=jnp.asarray(1e-5)))
    ratios = []
    for _ in range(4):
        for _ in range(25):
            state = step(state)
        b = simulate.npt_box(state)
        V = float(jnp.linalg.det(b))
        K = float(quantity.kinetic_energy(momentum=state.momentum, mass=state.mass))
        p_baro = float(
            quantity.pressure(npt_energy_fn, state.position, b, kinetic_energy=K, neighbor=None)
        )
        p_kin = 2 * K / (3 * V)
        p_true_vir = -_reference_dE_dV(state.position, b)
        ratios.append((p_baro - p_kin) / p_true_vir)
    np.testing.assert_allclose(ratios, 1.0, rtol=1e-6)
