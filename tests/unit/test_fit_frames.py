"""Frame store: loaders, decomposition cache and differentiable re-evaluation."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mmml.data.units import EV_TO_KCAL_MOL
from mmml.fit.frames import (
    FrameCache,
    FrameSet,
    check_decomposition,
    cubic_box_from_density,
    decompose,
    delta_u,
    density_g_cm3,
    load_frames,
    make_whole,
    mm_energies,
    u_theta,
)
from mmml.fit.lj_theta import init_theta


@pytest.fixture(autouse=True)
def _x64():
    """x64 for this module only (no global flag leaking into other tests)."""
    with jax.enable_x64(True):
        yield


APM = 2  # atoms per molecule in the synthetic system
Z = np.array([6, 8, 6, 8], dtype=np.int32)
MASS = 2 * (12.011 + 15.999)


def _frames(n_frames: int = 3, seed: int = 0) -> FrameSet:
    rng = np.random.default_rng(seed)
    box = np.tile([10.0, 11.0, 12.0], (n_frames, 1)) * rng.uniform(0.95, 1.05, (n_frames, 1))
    pos = rng.uniform(0.0, 9.0, (n_frames, 4, 3))
    pos[:, 1] = pos[:, 0] + [1.2, 0.0, 0.0]
    pos[:, 3] = pos[:, 2] + [0.0, 1.1, 0.0]
    masses = np.array([12.011, 15.999, 12.011, 15.999])
    return FrameSet(
        positions=pos,
        box=box,
        atomic_numbers=Z,
        masses=masses,
        temperature_K=250.0,
        n_molecules=2,
        density_g_cm3=density_g_cm3(masses.sum(), box),
        u_ref=np.full(n_frames, np.nan),  # no sampler energy recorded
    )


def _fake_mm(positions, pair_idx, pair_mask, cell, charges=None, lj_rmins=None, lj_epsilons=None):
    """Toy switch-free LJ, MIC: sum |eps_ij| [(Rmin_ij/r)^12 - 2 (Rmin_ij/r)^6], CHARMM eps <= 0.

    Like the real ``energy_with_lj``, padded / masked / i >= j rows get r = 1e6
    so (Rmin/r)^12 never overflows (inf * 0 = NaN in float32).
    """
    i, j = pair_idx[:, 0], pair_idx[:, 1]
    valid = (pair_mask > 0) & (i < j)
    d = positions[j] - positions[i]
    L = jnp.diag(cell)
    d = d - L * jnp.round(d / L)
    d = jnp.where(valid[:, None], d, 1e6)
    r = jnp.sqrt(jnp.sum(d * d, axis=-1))
    rm = lj_rmins[i] + lj_rmins[j]
    ep = jnp.sqrt(lj_epsilons[i] * lj_epsilons[j])  # |eps_ij|
    s6 = (rm / r) ** 6
    return jnp.sum(jnp.where(valid, pair_mask * ep * (s6 * s6 - 2.0 * s6), 0.0))


def _fake_update_fn():
    fn = SimpleNamespace()
    fn.energy_with_lj = _fake_mm
    fn.lj_rmins = np.array([1.9, 1.7, 1.9, 1.7])
    fn.lj_epsilons = np.array([-0.07, -0.12, -0.07, -0.12])
    fn.at_codes = np.array([0, 1, 0, 1])
    fn.atc_names = ["CG2O5", "OG2D3"]
    return fn


def _pairs_fn(n_pad: int):
    """Intermolecular pairs (atoms 0,1 x 2,3), padded with n_pad dummy rows."""

    def pairs(x, box):
        idx = np.array([[0, 2], [0, 3], [1, 2], [1, 3]] + [[0, 0]] * n_pad, dtype=np.int32)
        mask = np.array([1.0] * 4 + [0.0] * n_pad, dtype=np.float32)
        return idx, mask

    return pairs


def _terms_fn(x, box, mm_shift_eV: float = 0.0):
    # Deterministic "ML" terms in eV depending on geometry; mm_E is the
    # calculator's MM term, i.e. the same LJ at the base parameters.
    upd = _fake_update_fn()
    idx, msk = _pairs_fn(0)(x, box)
    e_mm = _fake_mm(
        jnp.asarray(x),
        jnp.asarray(idx),
        jnp.asarray(msk),
        jnp.diag(jnp.asarray(box)),
        lj_rmins=jnp.asarray(upd.lj_rmins),
        lj_epsilons=jnp.asarray(upd.lj_epsilons),
    )
    return {
        "internal_E": -100.0 - 0.01 * float(np.sum(x[:, 0])),
        "ml_2b_E": -0.2 - 0.001 * float(box[0]),
        "mm_E": float(e_mm) / EV_TO_KCAL_MOL + mm_shift_eV,
    }


N_PAD = 3
CAPACITY = 4 + N_PAD


def _fake_mm_fixed_capacity(positions, pair_idx, pair_mask, cell, **kw):
    # The real energy_with_lj closes over arrays sized to the live pair capacity.
    assert pair_idx.shape == (CAPACITY, 2) and pair_mask.shape == (CAPACITY,)
    return _fake_mm(positions, pair_idx, pair_mask, cell, **kw)


def _upd(**overrides):
    upd = _fake_update_fn()
    upd.energy_with_lj = _fake_mm_fixed_capacity
    for k, v in overrides.items():
        setattr(upd, k, v)
    return upd


def _cache(tmp_path=None, frames=None, **kw) -> FrameCache:
    path = None if tmp_path is None else tmp_path / "cache.npz"
    fs = _frames() if frames is None else frames
    return decompose(fs, _terms_fn, _pairs_fn(N_PAD), _upd(), cache_path=path, check="raise", **kw)


def test_density_and_cubic_box_round_trip():
    rho = np.array([0.78, 0.81])
    box = cubic_box_from_density(MASS * 100, rho)
    assert box.shape == (2, 3)
    np.testing.assert_allclose(density_g_cm3(MASS * 100, box), rho, rtol=1e-12)


def test_make_whole_rejoins_split_molecule_and_wraps_centroid():
    box = np.array([10.0, 10.0, 10.0])
    x = np.array([[9.8, 5.0, 5.0], [0.6, 5.0, 5.0], [3.0, 3.0, 3.0], [13.5, 3.0, 3.0]])
    w = make_whole(x, 2, box)
    np.testing.assert_allclose(np.linalg.norm(w[1] - w[0]), 0.8, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(w[3] - w[2]), 0.5, atol=1e-12)
    cog = w.reshape(2, 2, 3).mean(1)
    assert np.all((cog >= 0.0) & (cog < box))


def test_load_jaxmd_h5_recovers_box_from_density(tmp_path):
    fs = _frames(n_frames=5)
    L = np.array([20.0, 20.5, 21.0, 21.5, 22.0])
    rho = density_g_cm3(MASS, np.repeat(L[:, None], 3, 1))
    p = tmp_path / "run_npt.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("positions", data=fs.positions.astype(np.float32))
        f.create_dataset("potential_energy", data=np.linspace(-1.0, -1.1, 5))
        f.create_dataset("density_g_cm3", data=rho)
        f.attrs["atomic_numbers"] = Z
        f.attrs["temperature_target"] = 200.0
    out = load_frames(p, APM, start=1, stride=2)
    assert out.n_frames == 2 and out.n_molecules == 2 and out.temperature_K == 200.0
    np.testing.assert_allclose(out.box[:, 0], L[1::2], rtol=1e-4)
    np.testing.assert_allclose(out.u_ref, np.linspace(-1.0, -1.1, 5)[1::2] * EV_TO_KCAL_MOL)
    np.testing.assert_allclose(out.density_g_cm3, rho[1::2])
    assert out.positions.dtype == np.float64


def test_load_jaxmd_h5_nvt_needs_box(tmp_path):
    p = tmp_path / "run_nvt.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("positions", data=np.zeros((2, 4, 3), np.float32))
        f.attrs["atomic_numbers"] = Z
        f.attrs["temperature_target"] = 300.0
    with pytest.raises(ValueError, match="box_A"):
        load_frames(p, APM)
    out = load_frames(p, APM, box_A=np.array([15.0, 15.0, 15.0]))
    np.testing.assert_allclose(out.box, 15.0)
    assert np.all(np.isnan(out.u_ref))


def test_load_extxyz_with_cell(tmp_path):
    from ase import Atoms
    from ase.io import write

    fs = _frames(n_frames=2)
    images = []
    for f in range(2):
        a = Atoms(numbers=Z, positions=fs.positions[f], cell=fs.box[f], pbc=True)
        a.info["T_target_K"] = 200.0
        images.append(a)
    p = tmp_path / "frames.extxyz"
    write(p, images)
    out = load_frames(p, APM)
    np.testing.assert_allclose(out.box, fs.box, rtol=1e-10)
    np.testing.assert_allclose(out.density_g_cm3, fs.density_g_cm3, rtol=1e-10)
    assert out.temperature_K == 200.0 and np.all(np.isnan(out.u_ref))
    with pytest.raises(ValueError, match="multiple"):
        load_frames(p, 3)


def test_decompose_converts_units_and_caches(tmp_path):
    cache = _cache(tmp_path)
    fs = _frames()
    # Masked padding is dropped in storage; the live capacity is recorded.
    assert cache.pair_idx.shape == (3, 4, 2) and cache.pair_capacity == CAPACITY
    assert cache.pair_mask.sum(axis=1).tolist() == [4.0, 4.0, 4.0]
    for f in range(3):
        x = make_whole(fs.positions[f], APM, fs.box[f])
        e = _terms_fn(x, fs.box[f])
        assert cache.E_ml_mono[f] == pytest.approx(e["internal_E"] * EV_TO_KCAL_MOL)
        assert cache.E_ml_dimer[f] == pytest.approx(e["ml_2b_E"] * EV_TO_KCAL_MOL)
        assert cache.E_mm_calc[f] == pytest.approx(e["mm_E"] * EV_TO_KCAL_MOL)
    np.testing.assert_allclose(cache.E_mm0, cache.E_mm_calc, rtol=1e-10)
    np.testing.assert_allclose(cache.u_ref, cache.E_ml_mono + cache.E_ml_dimer + cache.E_mm0)
    loaded = FrameCache.load(tmp_path / "cache.npz")
    for name in ("E_ml_mono", "E_ml_dimer", "E_mm0", "pair_idx", "pair_mask", "base_rmins"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(cache, name))
    np.testing.assert_array_equal(loaded.frames.positions, cache.frames.positions)
    assert loaded.atc_names == ("CG2O5", "OG2D3") and loaded.frames.pressure_atm is None
    assert loaded.pair_capacity == CAPACITY
    assert loaded.frames.temperature_K == 250.0 and loaded.frames.n_molecules == 2
    assert loaded.fp_frames == cache.fp_frames and len(loaded.fp_lj) == 64

    # Matching cache is reused; the terms function only re-checks frame 0.
    calls = []

    def counting(x, box):
        calls.append(1)
        return _terms_fn(x, box)

    again = decompose(_frames(), counting, _pairs_fn(N_PAD), _upd(), cache_path=tmp_path / "cache.npz")
    np.testing.assert_array_equal(again.E_mm0, cache.E_mm0)
    assert len(calls) == 1

    def boom(x, box):
        raise AssertionError("terms_fn must not be called with verify_on_load=False")

    decompose(_frames(), boom, _pairs_fn(N_PAD), _upd(), cache_path=tmp_path / "cache.npz", verify_on_load=False)


def test_u_theta_at_theta0_reproduces_reference_and_delta_is_zero():
    cache = _cache()
    tm = cache.type_map()
    th0 = init_theta(tm)
    np.testing.assert_allclose(mm_energies(None, cache, _fake_mm_fixed_capacity), cache.E_mm0, rtol=1e-12)
    np.testing.assert_allclose(u_theta(th0, 1.0, cache, _fake_mm_fixed_capacity, tm), cache.u_ref, rtol=1e-12)
    np.testing.assert_allclose(delta_u(th0, 1.0, cache, _fake_mm_fixed_capacity, tm), 0.0, atol=1e-10)
    du = delta_u(th0, 0.5, cache, _fake_mm_fixed_capacity, tm)
    np.testing.assert_allclose(du, -0.5 * cache.E_ml_dimer, rtol=1e-12)


def test_u_theta_gradients_match_finite_differences():
    cache = _cache()
    tm = cache.type_map()
    th = {"log_eps": jnp.array([0.1, -0.05]), "log_sig": jnp.array([0.02, -0.01])}

    def total(th, lam):
        return jnp.sum(u_theta(th, lam, cache, _fake_mm_fixed_capacity, tm))

    g_th, g_lam = jax.grad(total, argnums=(0, 1))(th, 0.9)
    np.testing.assert_allclose(g_lam, cache.E_ml_dimer.sum(), rtol=1e-10)
    h = 1e-6
    for key in ("log_eps", "log_sig"):
        for k in range(2):
            e = jnp.zeros(2).at[k].set(h)
            up = total({**th, key: th[key] + e}, 0.9)
            dn = total({**th, key: th[key] - e}, 0.9)
            np.testing.assert_allclose(g_th[key][k], (up - dn) / (2 * h), rtol=1e-5, atol=1e-8)
    # Scaling epsilon of every type by s scales E_mm by s.
    s = 1.3
    th_s = {"log_eps": jnp.full(2, np.log(s)), "log_sig": jnp.zeros(2)}
    np.testing.assert_allclose(mm_energies(th_s, cache, _fake_mm_fixed_capacity, tm), s * cache.E_mm0, rtol=1e-10)


def test_mm_energies_requires_type_map_with_theta():
    cache = _cache()
    with pytest.raises(ValueError, match="type_map"):
        mm_energies(init_theta(cache.type_map()), cache, _fake_mm_fixed_capacity)


def test_frameset_select():
    fs = _frames(n_frames=4)
    sub = fs.select(np.array([0, 2]))
    assert sub.n_frames == 2 and sub.atoms_per_molecule == 2
    np.testing.assert_array_equal(sub.box, fs.box[[0, 2]])


def test_pad_pairs_drops_masked_rows_and_keeps_order():
    from mmml.fit.frames import _pad_pairs

    i1 = np.array([[0, 2], [0, 0], [1, 3], [0, 0]], dtype=np.int32)
    m1 = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    i2 = np.array([[1, 2]], dtype=np.int32)
    m2 = np.array([1.0], dtype=np.float32)
    idx, mask = _pad_pairs([i1, i2], [m1, m2])
    assert idx.shape == (2, 2, 2)
    np.testing.assert_array_equal(idx[0], [[0, 2], [1, 3]])
    np.testing.assert_array_equal(idx[1], [[1, 2], [0, 0]])
    np.testing.assert_array_equal(mask, [[1.0, 1.0], [1.0, 0.0]])


def test_decompose_rejects_changing_pair_capacity():
    calls = {"n": 0}

    def pairs(x, box):
        calls["n"] += 1
        return _pairs_fn(n_pad=calls["n"])(x, box)

    with pytest.raises(RuntimeError, match="capacity"):
        decompose(_frames(), _terms_fn, pairs, _fake_update_fn())


def test_mm_fake_has_attractive_well_and_masks_padding():
    # Sanity of the fake itself: E(r = Rmin_ij) = -|eps_ij|, padding contributes nothing.
    with jax.enable_x64(False):
        f32 = np.float32
        e = _fake_mm(
            jnp.asarray(np.array([[0.0, 0.0, 0.0], [3.6, 0.0, 0.0]], f32)),
            jnp.asarray(np.array([[0, 1], [0, 0], [1, 0]], np.int32)),  # padding and i > j rows
            jnp.asarray(np.array([1.0, 0.0, 1.0], f32)),
            jnp.asarray(np.eye(3, dtype=f32) * 50.0),
            lj_rmins=jnp.asarray(np.array([1.8, 1.8], f32)),
            lj_epsilons=jnp.asarray(np.array([-0.1, -0.1], f32)),
        )
        assert np.isfinite(float(e))
        np.testing.assert_allclose(float(e), -0.1, rtol=1e-5)


@pytest.mark.parametrize("build_x64", [False, True])
def test_delta_u_in_float32(build_x64):
    """delta_u is finite and ~0 at theta_0 when evaluated in float32 (cache built in either precision)."""
    with jax.enable_x64(build_x64):
        cache = _cache()
    with jax.enable_x64(False):
        tm = cache.type_map()
        th0 = init_theta(tm)
        e = mm_energies(None, cache, _fake_mm_fixed_capacity)
        assert e.dtype == jnp.float32 and np.all(np.isfinite(np.asarray(e)))
        du = np.asarray(delta_u(th0, 1.0, cache, _fake_mm_fixed_capacity, tm))
        assert du.dtype == np.float32 and np.all(np.isfinite(du))
        atol = 0.0 if not build_x64 else 1e-5 * np.abs(cache.E_mm0).max()
        np.testing.assert_allclose(du, 0.0, atol=atol)
        du_half = np.asarray(delta_u(th0, 0.5, cache, _fake_mm_fixed_capacity, tm))
        np.testing.assert_allclose(du_half, -0.5 * cache.E_ml_dimer, rtol=1e-5, atol=max(atol, 1e-4))
        g = jax.grad(lambda t: jnp.sum(delta_u(t, 1.0, cache, _fake_mm_fixed_capacity, tm)))(th0)
        assert all(np.all(np.isfinite(np.asarray(v))) for v in g.values())


def test_decompose_rejects_stale_cache(tmp_path):
    path = tmp_path / "cache.npz"
    c1 = _cache(tmp_path)
    # Different frames (count, positions).
    with pytest.raises(ValueError, match="frames differ"):
        decompose(_frames(5, seed=7), _terms_fn, _pairs_fn(N_PAD), _upd(), cache_path=path)
    # Same positions, different temperature.
    with pytest.raises(ValueError, match="frames differ"):
        decompose(replace(_frames(), temperature_K=300.0), _terms_fn, _pairs_fn(N_PAD), _upd(), cache_path=path)
    # Different base LJ (e.g. at_codes built with the wrong offset).
    upd2 = _upd(lj_epsilons=_fake_update_fn().lj_epsilons * 2)
    with pytest.raises(ValueError, match="base LJ"):
        decompose(_frames(), _terms_fn, _pairs_fn(N_PAD), upd2, cache_path=path)
    with pytest.raises(ValueError, match="base LJ"):
        decompose(_frames(), _terms_fn, _pairs_fn(N_PAD), _upd(at_codes=np.array([1, 0, 1, 0])), cache_path=path)
    # Different pair-list capacity.
    with pytest.raises(ValueError, match="capacity"):
        decompose(_frames(), _terms_fn, _pairs_fn(N_PAD + 1), _upd(), cache_path=path)
    # overwrite recomputes with the requested parameters.
    c2 = decompose(_frames(), _terms_fn, _pairs_fn(N_PAD), upd2, cache_path=path, overwrite=True, check="off")
    np.testing.assert_allclose(c2.base_epsilons, 2 * c1.base_epsilons)
    np.testing.assert_allclose(c2.E_mm0, 2 * c1.E_mm0, rtol=1e-12)
    np.testing.assert_allclose(FrameCache.load(path).base_epsilons, c2.base_epsilons)


def test_decompose_rejects_cache_without_fingerprint(tmp_path):
    cache = _cache()
    path = tmp_path / "old.npz"
    replace(cache, fp_frames="", fp_lj="").save(path)
    with pytest.raises(ValueError, match="fingerprint"):
        decompose(_frames(), _terms_fn, _pairs_fn(N_PAD), _upd(), cache_path=path)
    replace(cache, fp_model="").save(path)
    with pytest.raises(ValueError, match="fingerprint"):
        decompose(_frames(), _terms_fn, _pairs_fn(N_PAD), _upd(), cache_path=path)


def test_decompose_flags_mm_mismatch_with_calculator():
    def terms(x, box):
        return _terms_fn(x, box, mm_shift_eV=0.5)  # ~11.5 kcal/mol off

    with pytest.raises(ValueError, match="mm_E"):
        decompose(_frames(), terms, _pairs_fn(N_PAD), _upd(), check="raise")
    with pytest.warns(UserWarning, match="mm_E"):
        decompose(_frames(), terms, _pairs_fn(N_PAD), _upd())


def test_decompose_checks_sampler_potential():
    ref = _cache()
    # Sampler energy = decomposed U_0 + constant (e.g. a wall term): accepted.
    ok = _cache(frames=replace(_frames(), u_ref=ref.u_ref + 123.4))
    rep = check_decomposition(ok)
    assert rep["problems"] == [] and rep["u_ref_offset_kcal_mol"] == pytest.approx(123.4)
    # Sampler ran a different LJ assignment (at_codes swapped): U_0 differs non-uniformly.
    wrong_upd = _upd(
        at_codes=np.array([1, 0, 1, 0]),
        lj_rmins=np.array([1.7, 1.9, 1.7, 1.9]),
        lj_epsilons=np.array([-0.12, -0.07, -0.12, -0.07]),
    )
    e_wrong = np.asarray(
        mm_energies(
            None,
            replace(ref, base_rmins=wrong_upd.lj_rmins, base_epsilons=wrong_upd.lj_epsilons),
            _fake_mm_fixed_capacity,
        )
    )
    u_sampled = ref.E_ml_mono + ref.E_ml_dimer + e_wrong
    assert np.std(u_sampled - ref.u_ref) > 0.01 * 2
    with pytest.raises(ValueError, match="sampled Hamiltonian"):
        _cache(frames=replace(_frames(), u_ref=u_sampled))
    # NaN sampler energies (frames without energies) skip the check.
    assert "u_ref_offset_kcal_mol" not in check_decomposition(ref)


def test_load_jaxmd_h5_drops_nonfinite_and_blown_up_records(tmp_path):
    fs = _frames(n_frames=6)
    L = np.full(6, 20.0)
    rho = density_g_cm3(MASS, np.repeat(L[:, None], 3, 1))
    rho[2] = np.nan
    u = np.linspace(-1.0, -1.1, 6)
    u[4] = 8.4e7  # eV, blown-up integrator
    pos = fs.positions.copy()
    pos[1, 0, 0] = np.nan
    p = tmp_path / "run_npt.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("positions", data=pos)
        f.create_dataset("potential_energy", data=u)
        f.create_dataset("density_g_cm3", data=rho)
        f.attrs["atomic_numbers"] = Z
        f.attrs["temperature_target"] = 200.0
    with pytest.warns(UserWarning, match="dropping"):
        out = load_frames(p, APM)
    assert out.n_frames == 3
    np.testing.assert_allclose(out.u_ref, u[[0, 3, 5]] * EV_TO_KCAL_MOL)
    assert np.all(np.isfinite(out.box)) and np.all(np.isfinite(out.positions))
    with pytest.raises(ValueError, match="finite"):
        cubic_box_from_density(MASS, np.array([0.8, np.nan]))


def _write_staged_extxyz(path, stages, temps=None):
    from ase import Atoms
    from ase.io import write

    fs = _frames(n_frames=len(stages))
    images = []
    for f, st in enumerate(stages):
        a = Atoms(numbers=Z, positions=fs.positions[f], cell=fs.box[f], pbc=True)
        a.info["T_target_K"] = 200.0 if temps is None else temps[f]
        a.info["stage"] = st
        images.append(a)
    write(path, images)
    return fs


def test_load_extxyz_filters_stages(tmp_path):
    p = tmp_path / "staged.extxyz"
    fs = _write_staged_extxyz(p, ["heat", "heat", "equi", "equi", "prod"])
    with pytest.warns(UserWarning, match="heat"):
        out = load_frames(p, APM)
    assert out.n_frames == 3
    np.testing.assert_allclose(out.positions, fs.positions[2:])
    assert load_frames(p, APM, stages=None).n_frames == 5
    with pytest.warns(UserWarning, match="equi"):
        assert load_frames(p, APM, stages=("prod",)).n_frames == 1
    with pytest.raises(ValueError, match="no frames in stages"), pytest.warns(UserWarning):
        load_frames(p, APM, stages=("nope",))


def test_load_extxyz_rejects_mixed_temperatures(tmp_path):
    p = tmp_path / "mixed.extxyz"
    _write_staged_extxyz(p, ["equi", "equi"], temps=[200.0, 250.0])
    with pytest.raises(ValueError, match="different target temperatures"):
        load_frames(p, APM)


# ---------------------------------------------------------------------------
# Blow-up filter
# ---------------------------------------------------------------------------


def _write_h5(path, u_eV, n_frames=None):
    u_eV = np.atleast_1d(np.asarray(u_eV, dtype=np.float64))
    fs = _frames(n_frames=len(u_eV))
    rho = density_g_cm3(MASS, np.full((len(u_eV), 3), 20.0))
    with h5py.File(path, "w") as f:
        f.create_dataset("positions", data=fs.positions)
        f.create_dataset("potential_energy", data=u_eV)
        f.create_dataset("density_g_cm3", data=rho)
        f.attrs["atomic_numbers"] = Z
        f.attrs["temperature_target"] = 200.0
    return path


def test_blown_up_single_record_is_rejected(tmp_path):
    # The aco_T200 smoke failure: one record at 8.4e7 eV. A median of one is itself.
    p = _write_h5(tmp_path / "one.h5", [8.4e7])
    with pytest.raises(ValueError, match="no usable frames"), pytest.warns(UserWarning, match="unphysical"):
        load_frames(p, APM)
    # A single sane record loads, with a warning that it could not be cross-checked.
    p = _write_h5(tmp_path / "one_ok.h5", [-1.0])
    with pytest.warns(UserWarning, match="only 1 usable"):
        assert load_frames(p, APM).n_frames == 1


def test_blown_up_majority_does_not_evict_good_frame(tmp_path):
    p = _write_h5(tmp_path / "maj.h5", [-1.0, 8.4e7, 8.5e7])
    with pytest.warns(UserWarning, match="dropping 2/3"):
        out = load_frames(p, APM)
    np.testing.assert_allclose(out.u_ref, [-1.0 * EV_TO_KCAL_MOL])


def test_blow_up_filter_uses_largest_cluster_not_median(tmp_path):
    # Within the absolute bound; 4 atoms x 1 kcal/mol/atom = 4 kcal/mol (~0.17 eV) window.
    # Plain median (2.0 eV) would keep only frame 2; the cluster {0, 1} is the reference.
    u = [-1.0, -1.00001, 2.0, 3.5, 5.0]
    with pytest.warns(UserWarning, match="U_ref"):
        out = load_frames(_write_h5(tmp_path / "c.h5", u), APM)
    np.testing.assert_allclose(out.u_ref, np.array(u[:2]) * EV_TO_KCAL_MOL)
    # Ties (all clusters of size 1) go to the earliest record.
    with pytest.warns(UserWarning):
        out = load_frames(_write_h5(tmp_path / "t.h5", [3.0, -1.0, 6.0]), APM)
    np.testing.assert_allclose(out.u_ref, [3.0 * EV_TO_KCAL_MOL])
    # The absolute bound scales with the atoms: acetone-sized formation energies pass easily.
    from mmml.fit.frames import max_abs_potential_kcal_mol

    z_aco = np.array([6, 6, 6, 8, 1, 1, 1, 1, 1, 1] * 266)
    assert max_abs_potential_kcal_mol(z_aco) > 5e6 > 1e5  # total-energy scale (~2.2e6 kcal/mol) fits
    assert 8.4e7 * EV_TO_KCAL_MOL > max_abs_potential_kcal_mol(z_aco)


# ---------------------------------------------------------------------------
# Stale Hamiltonian
# ---------------------------------------------------------------------------


def test_cache_reuse_detects_changed_hamiltonian(tmp_path):
    path = tmp_path / "cache.npz"
    _cache(tmp_path)

    def half_mm(*a, **kw):  # stands in for another mm_switch_on
        return 0.5 * _fake_mm_fixed_capacity(*a, **kw)

    with pytest.raises(ValueError, match="E_mm"):
        decompose(_frames(), None, _pairs_fn(N_PAD), _upd(energy_with_lj=half_mm), cache_path=path)

    def scaled_terms(x, box):  # stands in for another checkpoint
        return {k: 0.9 * v for k, v in _terms_fn(x, box).items()}

    with pytest.raises(ValueError, match="E_ml_mono"):
        decompose(_frames(), scaled_terms, _pairs_fn(N_PAD), _upd(), cache_path=path)
    # Unchanged functions still reuse it.
    decompose(_frames(), _terms_fn, _pairs_fn(N_PAD), _upd(), cache_path=path)


def test_cache_fingerprint_covers_model_tag_and_whole(tmp_path):
    from mmml.fit.frames import hamiltonian_tag

    ck = tmp_path / "model.json"
    ck.write_text('{"w": 1}')
    cut = {"mm_switch_on": 6.0, "ml_switch_width": 1.5}
    tag = hamiltonian_tag(ck, cut)
    assert tag == hamiltonian_tag(ck, dict(reversed(cut.items())))
    assert tag != hamiltonian_tag(ck, {**cut, "mm_switch_on": 7.0})
    path = tmp_path / "cache.npz"
    _cache(tmp_path, model_tag=tag)
    decompose(_frames(), _terms_fn, _pairs_fn(N_PAD), _upd(), cache_path=path, model_tag=tag)
    ck.write_text('{"w": 2}')
    with pytest.raises(ValueError, match="model_tag"):
        decompose(_frames(), _terms_fn, _pairs_fn(N_PAD), _upd(), cache_path=path, model_tag=hamiltonian_tag(ck, cut))
    with pytest.raises(ValueError, match="whole"):
        decompose(_frames(), _terms_fn, _pairs_fn(N_PAD), _upd(), cache_path=path, model_tag=tag, whole=False)

    # terms_fn.model_tag (set by hybrid_term_fns) is picked up automatically.
    def tagged(x, box):
        return _terms_fn(x, box)

    tagged.model_tag = "other"
    with pytest.raises(ValueError, match="model_tag"):
        decompose(_frames(), tagged, _pairs_fn(N_PAD), _upd(), cache_path=path)


# ---------------------------------------------------------------------------
# Gradient memory
# ---------------------------------------------------------------------------


def _big_cache(n_frames: int, capacity: int, n_mol: int = 200) -> FrameCache:
    rng = np.random.default_rng(1)
    n_at = 2 * n_mol
    pos = rng.uniform(0.0, 30.0, (n_frames, n_at, 3))
    box = np.full((n_frames, 3), 30.0)
    n_pairs = 2000  # compact stored list; energy fn needs the full live capacity
    i = rng.integers(0, n_at - 1, (n_frames, n_pairs))
    j = np.minimum(i + 1 + rng.integers(0, 5, (n_frames, n_pairs)), n_at - 1)
    fs = FrameSet(
        positions=pos,
        box=box,
        atomic_numbers=np.tile(Z[:2], n_mol),
        masses=np.tile([12.011, 15.999], n_mol),
        temperature_K=300.0,
        n_molecules=n_mol,
        density_g_cm3=np.ones(n_frames),
        u_ref=np.full(n_frames, np.nan),
    )
    return FrameCache(
        frames=fs,
        E_ml_mono=np.zeros(n_frames),
        E_ml_dimer=np.zeros(n_frames),
        E_mm0=np.zeros(n_frames),
        E_mm_calc=np.zeros(n_frames),
        pair_idx=np.stack([i, j], -1).astype(np.int32),
        pair_mask=np.ones((n_frames, n_pairs), np.float32),
        base_rmins=np.tile([1.9, 1.7], n_mol),
        base_epsilons=np.tile([-0.07, -0.12], n_mol),
        at_codes=np.tile([0, 1], n_mol),
        atc_names=("CG2O5", "OG2D3"),
        pair_capacity=capacity,
    )


def test_delta_u_gradient_memory_does_not_scale_with_frames_times_capacity():
    capacity = 400_000

    def temp_bytes(n_frames: int) -> int:
        cache = _big_cache(n_frames, capacity)
        tm = cache.type_map()

        def loss(t):
            return jnp.sum(jnp.exp(-delta_u(t, 1.0, cache, _fake_mm, tm)))

        return jax.jit(jax.grad(loss)).lower(init_theta(tm)).compile().memory_analysis().temp_size_in_bytes

    small, large = temp_bytes(2), temp_bytes(16)
    per_frame_capacity_bytes = capacity * 8  # one float64 per padded pair row
    # Without remat every frame's padded intermediates are kept: >= 14 x capacity x 8 B more.
    assert large - small < 2 * per_frame_capacity_bytes, (small, large)
