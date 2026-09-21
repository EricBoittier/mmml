"""Tests for mmml.fit.gas_phase (no CHARMM / GPU)."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import mmml.fit.gas_phase as gp
from mmml.fit.gas_phase import (
    LangevinConfig,
    make_langevin_runner,
    block_average,
    bonded_pairs,
    blocking_analysis,
    combined_sem,
    dhvap_from_cohesive_kcal_mol,
    dhvap_kcal_mol,
    flat_bottom_bond_restraint,
    physnet_monomer_energy_fn,
    replica_mean_sem,
    run_gas_phase,
)
from mmml.fit.reweight import KB_KCAL_MOL_K

REPO = Path(__file__).resolve().parents[2]
CKPT = REPO / "examples" / "ckpts_json" / "DESdimers_params.json"


def _ar1(n: int, phi: float, seed: int) -> np.ndarray:
    """Unit-variance AR(1); SEM of the mean = sqrt((1+phi)/(1-phi)/n)."""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n) * np.sqrt(1.0 - phi**2)
    x = np.empty(n)
    x[0] = rng.standard_normal()
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    return x


def test_dhvap_formula_scalar_and_array():
    assert dhvap_kcal_mol(-1000.0, -1010.0, 298.15) == pytest.approx(10.0 + KB_KCAL_MOL_K * 298.15)
    out = dhvap_kcal_mol(np.array([-5.0, -5.0]), np.array([-12.0, -11.0]), np.array([200.0, 298.0]))
    np.testing.assert_allclose(out, [7.0 + 200 * KB_KCAL_MOL_K, 6.0 + 298 * KB_KCAL_MOL_K])
    # RT at 298.15 K is 0.5925 kcal/mol
    assert dhvap_kcal_mol(0.0, 0.0, 298.15) == pytest.approx(0.5925, abs=1e-4)


def test_dhvap_from_cohesive_matches_full_formula():
    e_intra_gas, e_intra_liq, u_inter, T = -1100.0, -1100.3, -9.0, 200.0
    full = dhvap_kcal_mol(e_intra_gas, e_intra_liq + u_inter, T)
    coh = dhvap_from_cohesive_kcal_mol(u_inter, T, delta_e_intra=e_intra_gas - e_intra_liq)
    assert coh == pytest.approx(full)
    assert dhvap_from_cohesive_kcal_mol(-9.0, 0.0, 0.0) == pytest.approx(9.0)
    with pytest.raises(TypeError):
        dhvap_from_cohesive_kcal_mol(-9.0, 0.0)  # dE_intra must be stated


def test_blocking_constant_series_is_finite():
    res = blocking_analysis(np.full(256, 3.0))
    assert res.mean == 3.0 and res.sem == 0.0


def test_bonded_pairs_and_flat_bottom_guard():
    z = [6, 1, 1]
    x = np.array([[0.0, 0.0, 0.0], [1.09, 0.0, 0.0], [5.0, 0.0, 0.0]])
    pairs = bonded_pairs(x, z)
    np.testing.assert_array_equal(pairs, [[0, 1]])
    guard = flat_bottom_bond_restraint(x, pairs, half_width_A=0.2, k_kcal_A2=100.0)
    assert float(guard(jnp.asarray(x))) == 0.0
    y = x.copy()
    y[1, 0] = 1.09 + 0.15  # inside the flat bottom
    assert float(guard(jnp.asarray(y))) == 0.0
    y[1, 0] = 1.09 - 0.3  # 0.1 A beyond the wall
    assert float(guard(jnp.asarray(y))) == pytest.approx(0.5 * 100.0 * 0.1**2, rel=1e-4)


def test_langevin_restraint_energy_reported_separately():
    """A guard acting on a free 'bond' is recorded but kept out of E_pot."""
    x0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    guard = flat_bottom_bond_restraint(x0, np.array([[0, 1]]), half_width_A=0.0, k_kcal_A2=50.0)

    def energy(x):
        return 0.0 * jnp.sum(x)

    cfg = LangevinConfig(dt_fs=1.0, n_equil_steps=500, n_steps=8000, record_every=10, n_replicas=4)
    (res,) = run_gas_phase(energy, x0, np.full(2, 12.0), [300.0], cfg, restraint_fn=guard)
    assert res.e_pot_mean == 0.0
    # 1D harmonic bond: <E_rst> = kT/2
    assert res.restraint_energy_mean == pytest.approx(0.5 * KB_KCAL_MOL_K * 300.0, rel=0.25)
    assert res.restraint_active_fraction > 0.9


def test_block_average_iid_matches_naive_sem():
    x = np.random.default_rng(0).standard_normal(4096)
    mean, sem = block_average(x, 64)
    assert mean == pytest.approx(x.mean())
    assert sem == pytest.approx(1.0 / np.sqrt(x.size), rel=0.3)


def test_block_average_drops_leading_remainder():
    x = np.arange(10.0)
    mean, _ = block_average(x, 3)  # keeps x[1:]
    assert mean == pytest.approx(np.arange(1.0, 10.0).mean())
    with pytest.raises(ValueError):
        block_average(x, 1)
    with pytest.raises(ValueError):
        block_average(x[:2], 3)


@pytest.mark.parametrize("seed", [1, 2])
def test_blocking_recovers_correlated_sem(seed):
    n, phi = 2**16, 0.9
    x = _ar1(n, phi, seed)
    true_sem = np.sqrt((1 + phi) / (1 - phi) / n)
    naive = x.std() / np.sqrt(n)
    res = blocking_analysis(x)
    assert res.sems[0] == pytest.approx(naive, rel=1e-3)
    assert res.sem == pytest.approx(true_sem, rel=0.3)
    assert res.statistical_inefficiency == pytest.approx((1 + phi) / (1 - phi), rel=0.6)
    # Fixed-count blocks much longer than tau also recover it.
    _, sem_blocks = block_average(x, 32)
    assert sem_blocks == pytest.approx(true_sem, rel=0.4)
    # Block sizes double each level.
    np.testing.assert_array_equal(res.block_sizes, 2 ** np.arange(res.block_sizes.size))


def test_blocking_iid_has_plateau_at_naive():
    x = np.random.default_rng(3).standard_normal(2**14)
    res = blocking_analysis(x)
    assert res.plateau
    assert res.sem == pytest.approx(1.0 / np.sqrt(x.size), rel=0.15)


def test_replica_mean_sem_combines_independent_replicas():
    series = np.stack([_ar1(2**13, 0.8, s) + 3.0 for s in range(6)])
    mean, sem, between = replica_mean_sem(series)
    true_sem = np.sqrt(9.0 / 2**13) / np.sqrt(6)
    assert mean == pytest.approx(3.0, abs=4 * true_sem)
    assert sem == pytest.approx(true_sem, rel=0.3)
    assert np.isfinite(between) and between > 0


def test_langevin_harmonic_equipartition():
    """<E_pot> = <E_kin> = 3N/2 kT for an isotropic harmonic 'molecule'."""
    k_spring = 100.0  # kcal/mol/A^2

    n_atoms, T = 3, 300.0
    x0 = 2.0 * np.eye(n_atoms)  # well separated: no pair below the distance floor

    def energy(x):
        return 0.5 * k_spring * jnp.sum((x - x0) ** 2)

    cfg = LangevinConfig(
        dt_fs=1.0,
        friction_per_fs=0.05,
        n_equil_steps=1000,
        n_steps=20000,
        record_every=10,
        n_replicas=8,
        seed=7,
    )
    (res,) = run_gas_phase(energy, x0, np.full(n_atoms, 12.0), [T], cfg)
    expected = 1.5 * n_atoms * KB_KCAL_MOL_K * T
    assert res.e_kin_expected == pytest.approx(expected)
    assert res.e_pot_mean == pytest.approx(expected, rel=0.05)
    assert abs(res.e_pot_mean - expected) < 5 * max(res.e_pot_sem, 1e-6)
    assert res.temperature_kin_K == pytest.approx(T, rel=0.05)
    assert res.n_samples == 2000
    assert res.sampled_ps == pytest.approx(20.0)
    assert res.e_pot_sem == max(res.e_pot_sem_blocking, res.e_pot_sem_between)
    assert res.e_pot_sem_between == pytest.approx(res.e_pot_sem_blocking, rel=0.8)
    assert res.topology_preserved and res.topology_changed_replicas == []
    assert not res.bond_graph_checked
    assert res.n_replicas_used == 8 and res.unstable_replicas == []
    assert res.dropped_replicas == []
    assert len(res.min_pair_distance_A) == 8
    assert all(np.isnan(res.min_distance_ratio))  # no atomic numbers
    assert all(np.isfinite(res.min_pair_distance_A))
    assert "e_pot_series" not in res.as_dict()


def test_langevin_unstable_integration_raises():
    """omega * dt >> 2: every replica blows up and is rejected."""

    def energy(x):
        return 0.5 * 1e6 * jnp.sum(x**2)

    cfg = LangevinConfig(dt_fs=1.0, n_equil_steps=10, n_steps=400, record_every=10, n_replicas=2)
    with pytest.raises(FloatingPointError):
        run_gas_phase(energy, np.zeros((2, 3)), np.ones(2), [300.0], cfg, min_blocks=4)


@pytest.mark.skipif(not CKPT.is_file(), reason="bundled PhysNet checkpoint missing")
def test_physnet_monomer_energy_fn_invariances():
    z = [6, 6, 6, 8, 1, 1, 1, 1, 1, 1]  # acetone
    x = np.array(
        [
            [0.000, 0.000, 0.000],
            [1.290, 0.760, 0.000],
            [-1.290, 0.760, 0.000],
            [0.000, -1.215, 0.000],
            [2.140, 0.080, 0.000],
            [1.340, 1.410, 0.880],
            [1.340, 1.410, -0.880],
            [-2.140, 0.080, 0.000],
            [-1.340, 1.410, 0.880],
            [-1.340, 1.410, -0.880],
        ],
        dtype=np.float32,
    )
    efn = physnet_monomer_energy_fn(CKPT, z)
    e0 = float(efn(jnp.asarray(x)))
    assert np.isfinite(e0)
    e_shift = float(efn(jnp.asarray(x + np.float32(5.0))))
    assert e_shift == pytest.approx(e0, abs=1e-2)
    g = np.asarray(jax.grad(efn)(jnp.asarray(x)))
    assert np.all(np.isfinite(g))
    assert np.abs(g.sum(axis=0)).max() < 1e-2  # no net force


def test_combined_sem_uses_between_replica_spread():
    """Non-ergodic replicas: blocking sees no error in the offsets, between does."""
    offsets = np.array([0.0, 5.0, -3.0, 8.0])
    series = np.stack([_ar1(2**12, 0.5, s) + o for s, o in enumerate(offsets)])
    mean, sem_blk, between = replica_mean_sem(series)
    assert between > 10 * sem_blk
    assert combined_sem(sem_blk, between) == between
    assert combined_sem(0.3, 0.1) == 0.3
    assert combined_sem(0.3, float("nan")) == 0.3  # one replica


def test_blocking_short_series_is_biased_low_and_replicas_catch_it():
    """At ~4000 samples and tau ~ 100, blocking underestimates; the replica spread does not."""
    n, phi, n_rep = 4000, 0.99, 16
    series = np.stack([_ar1(n, phi, 100 + s) for s in range(n_rep)])
    true_sem = np.sqrt((1 + phi) / (1 - phi) / n) / np.sqrt(n_rep)
    _, sem_blk, between = replica_mean_sem(series)
    assert sem_blk < 0.9 * true_sem
    assert combined_sem(sem_blk, between) >= sem_blk
    assert between == pytest.approx(true_sem, rel=0.5)


# C (0), H (1) bonded; H (2) far away. Covalent C-H cutoff 1.2 * 1.07 = 1.28 A.
_Z_CHH = [6, 1, 1]
_X_CHH = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
_CFG_SMALL = LangevinConfig(
    dt_fs=1.0,
    friction_per_fs=0.05,
    n_equil_steps=500,
    n_steps=4000,
    record_every=10,
    n_replicas=4,
    seed=3,
)


def _tethered(pull_to_0_A: float | None, k: float = 400.0):
    """Atoms tethered to _X_CHH; optionally atom 2 pulled to distance ``pull_to_0_A`` of atom 0."""
    x0 = jnp.asarray(_X_CHH)

    def energy(x):
        e = 0.5 * k * jnp.sum((x[:2] - x0[:2]) ** 2)
        if pull_to_0_A is None:
            return e + 0.5 * k * jnp.sum((x[2] - x0[2]) ** 2)
        d = jnp.linalg.norm(x[2] - x[0])
        return e + 0.5 * k * (d - pull_to_0_A) ** 2

    return energy


def test_topology_preserved_for_intact_molecule():
    (res,) = run_gas_phase(_tethered(None), _X_CHH, np.full(3, 12.0), [300.0], _CFG_SMALL, atomic_numbers=_Z_CHH)
    assert res.bond_graph_checked and res.topology_preserved
    assert res.topology_changed_replicas == [] and res.n_replicas_used == 4
    assert "topology_preserved" in res.as_dict()


def test_new_bond_is_flagged_and_dropped():
    """Atom 2 falls into a C-H 'hole': the thermostat keeps T normal, the graph check fires."""
    energy = _tethered(1.0)
    with pytest.warns(RuntimeWarning, match="topology"), pytest.raises(RuntimeError):
        run_gas_phase(energy, _X_CHH, np.full(3, 12.0), [300.0], _CFG_SMALL, atomic_numbers=_Z_CHH)
    keep = LangevinConfig(**{**_CFG_SMALL.__dict__, "drop_topology_changes": False})
    with pytest.warns(RuntimeWarning, match="topology"):
        (res,) = run_gas_phase(energy, _X_CHH, np.full(3, 12.0), [300.0], keep, atomic_numbers=_Z_CHH)
    assert res.temperature_kin_K == pytest.approx(300.0, rel=0.2)  # passes the T test
    assert res.unstable_replicas == [] and res.dropped_replicas == []
    assert not res.topology_preserved
    assert res.topology_changed_replicas == [0, 1, 2, 3]
    assert res.n_replicas_used == 4  # kept, but flagged


def test_distance_floor_flags_fused_atoms_without_atomic_numbers():
    keep = LangevinConfig(**{**_CFG_SMALL.__dict__, "drop_topology_changes": False})
    with pytest.warns(RuntimeWarning, match="closer than 0.7"):
        (res,) = run_gas_phase(_tethered(0.3), _X_CHH, np.full(3, 12.0), [300.0], keep)
    assert not res.topology_preserved and len(res.topology_changed_replicas) == 4
    assert max(res.min_pair_distance_A) < 0.7
    off = LangevinConfig(**{**keep.__dict__, "min_distance_floor_A": None})
    (res,) = run_gas_phase(_tethered(0.3), _X_CHH, np.full(3, 12.0), [300.0], off)
    assert res.topology_preserved


def test_atomic_numbers_length_checked():
    with pytest.raises(ValueError, match="atomic numbers"):
        run_gas_phase(_tethered(None), _X_CHH, np.full(3, 12.0), [300.0], _CFG_SMALL, atomic_numbers=[6, 1])


# C (0) - H (1) at a realistic 1.09 A; H (2) far away.
_X_CH109 = np.array([[0.0, 0.0, 0.0], [1.09, 0.0, 0.0], [0.0, 3.0, 0.0]])


def _ch_well(r_min_A: float, k: float = 600.0):
    """Harmonic C-H bond with its minimum at ``r_min_A``; atoms 0, 2 tethered."""
    x0 = jnp.asarray(_X_CH109)

    def energy(x):
        d = jnp.linalg.norm(x[1] - x[0])
        tether = jnp.sum((x[0] - x0[0]) ** 2) + jnp.sum((x[2] - x0[2]) ** 2)
        return 0.5 * k * (d - r_min_A) ** 2 + 0.5 * k * tether

    return energy


def test_collapsed_bond_in_graph_is_flagged():
    """DESdimers-like C-H hole at 0.86 A: graph unchanged (< 1.28 A), above 0.7 A, yet flagged."""
    keep = LangevinConfig(**{**_CFG_SMALL.__dict__, "drop_topology_changes": False})
    with pytest.warns(RuntimeWarning, match="compressed"):
        (res,) = run_gas_phase(_ch_well(0.86), _X_CH109, np.full(3, 12.0), [300.0], keep, atomic_numbers=_Z_CHH)
    assert min(res.min_pair_distance_A) > 0.7  # the old global floor would not fire
    assert res.topology_changed_replicas == [0, 1, 2, 3] and not res.topology_preserved
    assert max(res.min_distance_ratio) < 0.8
    assert res.unstable_replicas == []
    with pytest.warns(RuntimeWarning), pytest.raises(RuntimeError):
        run_gas_phase(_ch_well(0.86), _X_CH109, np.full(3, 12.0), [300.0], _CFG_SMALL, atomic_numbers=_Z_CHH)


def test_intact_bond_at_equilibrium_not_flagged_at_high_T():
    (res,) = run_gas_phase(_ch_well(1.09), _X_CH109, np.full(3, 12.0), [500.0], _CFG_SMALL, atomic_numbers=_Z_CHH)
    assert res.topology_preserved and res.n_replicas_used == 4
    assert min(res.min_distance_ratio) > 0.8


def test_geometry_extrema_cover_every_step():
    """d_min recorded every 10 steps equals the min over the per-step series of the same trajectory."""
    x0 = np.stack([_X_CH109] * 2).astype(np.float32)
    ref = np.full((3, 3), 1.0)
    np.fill_diagonal(ref, np.inf)
    out = {}
    for every in (1, 10):
        cfg = LangevinConfig(dt_fs=1.0, friction_per_fs=0.05, record_every=every, n_replicas=2, seed=0)
        run = make_langevin_runner(_ch_well(1.0), np.full(3, 12.0), cfg, reference_distance_A=ref)
        v = jnp.asarray(np.random.default_rng(0).standard_normal(x0.shape) * 0.02, jnp.float32)
        _, _, _, d_min, _, n_bond, ratio = run(jnp.asarray(x0), v, jax.random.PRNGKey(1), 0.6, 200 // every)
        out[every] = (np.asarray(d_min), np.asarray(ratio))
    d1, r1 = out[1]
    d10, r10 = out[10]
    np.testing.assert_allclose(d10, d1.reshape(20, 10, 2).min(axis=1), rtol=1e-4)
    np.testing.assert_allclose(r10, r1.reshape(20, 10, 2).min(axis=1), rtol=1e-4)
    # The recorded-frame-only minimum would miss dips between frames.
    assert np.all(d10 <= d1[9::10] + 1e-6) and np.any(d10 < d1[9::10] - 1e-4)


def test_unstable_and_topology_drops_reported_separately(monkeypatch):
    """Replica 0 blows up, replica 1 changes topology: only 0 is 'unstable', both are dropped."""
    n_rec, n_rep, n_atoms, T = 64, 4, 3, 300.0
    kT = KB_KCAL_MOL_K * T
    rng = np.random.default_rng(0)

    def fake_runner(*args, **kwargs):
        def run(x, v, key, kT_, n_records):
            e_pot = rng.standard_normal((n_records, n_rep))
            e_pot[:, 0] = np.nan
            e_kin = np.full((n_records, n_rep), 1.5 * n_atoms * kT)
            d_min = np.full((n_records, n_rep), 1.0)
            n_bond = np.zeros((n_records, n_rep), np.int32)
            n_bond[-1, 1] = 1
            ratio = np.full((n_records, n_rep), np.inf)
            return (x, v, key), e_pot, e_kin, d_min, np.zeros_like(e_pot), n_bond, ratio

        return run

    monkeypatch.setattr(gp, "make_langevin_runner", fake_runner)
    cfg = LangevinConfig(n_equil_steps=10, n_steps=n_rec * 10, record_every=10, n_replicas=n_rep)
    with pytest.warns(RuntimeWarning, match="replicas \\[1\\]"):
        (res,) = run_gas_phase(_tethered(None), _X_CHH, np.full(3, 12.0), [T], cfg, min_blocks=4)
    assert res.unstable_replicas == [0]
    assert res.topology_changed_replicas == [1]
    assert res.dropped_replicas == [0, 1]
    assert res.n_replicas_used == 2
