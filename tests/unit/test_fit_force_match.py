"""Force-matching regularizer (synthetic data, fake MM energy; no CHARMM/GPU)."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mmml.fit import force_match as fm
from mmml.fit.lj_theta import LjTypeMap, init_theta

jax.config.update("jax_enable_x64", True)

N_MOL, N_PER, L = 4, 3, 12.0
N = N_MOL * N_PER


def fake_energy_with_lj(positions, pair_idx, pair_mask, cell, charges=None,
                        lj_rmins=None, lj_epsilons=None):
    """Quadratic 'LJ': sum_ij mask * sqrt(eps_i eps_j) * (r_ij - Rmin_ij)^2 with MIC."""
    i, j = pair_idx[:, 0], pair_idx[:, 1]
    box = jnp.diag(cell)
    d = positions[i] - positions[j]
    d = d - box * jnp.round(d / box)
    r = jnp.sqrt(jnp.sum(d**2, axis=-1))
    eps = jnp.sqrt(lj_epsilons[i] * lj_epsilons[j])  # CHARMM eps <= 0 -> product >= 0
    return jnp.sum(pair_mask * eps * (r - (lj_rmins[i] + lj_rmins[j])) ** 2)


def _type_map() -> LjTypeMap:
    at_codes = np.tile([0, 1, 2], N_MOL)
    return LjTypeMap.from_atc(at_codes, ["CG2O1", "OG2D3", "HGA3"], ["CG2O1", "OG2D3"])


def _pairs(rng):
    mol = np.repeat(np.arange(N_MOL), N_PER)
    ii, jj = np.triu_indices(N, 1)
    inter = mol[ii] != mol[jj]
    idx = np.stack([ii[inter], jj[inter]], axis=1)
    mask = (rng.random(len(idx)) > 0.2).astype(float)
    return idx, mask


def _cache(rng, n_frames=2, theta_true=None, lam_true=0.8, noise=0.0) -> fm.ForceMatchCache:
    tm = _type_map()
    base_r = rng.uniform(1.0, 2.0, N)
    base_e = -rng.uniform(0.05, 0.2, N)
    frames = []
    for k in range(n_frames):
        idx, mask = _pairs(rng)
        if k == 1:  # different capacity -> exercises padding
            idx, mask = idx[:-5], mask[:-5]
        frames.append(
            dict(
                positions=rng.uniform(0, L, (N, 3)),
                cells=np.diag([L, L, L]),
                f_pet=np.zeros((N, 3)),
                f_ml_mono=rng.normal(size=(N, 3)),
                f_ml_dimer=rng.normal(size=(N, 3)),
                f_mm_ref=np.zeros((N, 3)),
                pair_idx=idx,
                pair_mask=mask,
            )
        )
    cache = fm.ForceMatchCache.from_frames(
        frames, base_rmins=base_r, base_epsilons=base_e,
        molecule_index=fm.molecule_index_from_sizes([N_PER] * N_MOL),
    )
    # Teacher = hybrid at (theta_true, lam_true) (+ noise), so the optimum is known.
    theta_true = theta_true or {"log_eps": jnp.array([0.3, -0.2]), "log_sig": jnp.array([0.05, -0.1])}
    f_pet = np.stack([
        np.asarray(fm.hybrid_forces(theta_true, lam_true, cache, k, fake_energy_with_lj, tm))
        for k in range(n_frames)
    ]) + noise * rng.normal(size=(n_frames, N, 3))
    return fm.ForceMatchCache(**{**cache.__dict__, "f_pet": f_pet}), theta_true, tm


def test_mm_forces_are_negative_gradient_of_energy():
    rng = np.random.default_rng(0)
    cache, _, tm = _cache(rng)
    theta = {"log_eps": jnp.array([0.1, 0.2]), "log_sig": jnp.array([-0.05, 0.02])}
    f = fm.mm_forces(theta, cache, 0, fake_energy_with_lj, tm)
    rm, ep = fm.per_atom_lj(theta, tm, jnp.asarray(cache.base_rmins), jnp.asarray(cache.base_epsilons))
    x0 = jnp.asarray(cache.positions[0])
    d = jnp.asarray(rng.normal(size=(N, 3)))
    h = 1e-5

    def e(x):
        return fake_energy_with_lj(x, cache.pair_idx[0], cache.pair_mask[0], cache.cells[0],
                                   lj_rmins=rm, lj_epsilons=ep)

    fd = (e(x0 + h * d) - e(x0 - h * d)) / (2 * h)
    np.testing.assert_allclose(-jnp.sum(f * d), fd, rtol=1e-6)


def test_loss_is_zero_with_zero_gradient_at_teacher_parameters():
    rng = np.random.default_rng(1)
    cache, theta_true, tm = _cache(rng, lam_true=0.8)
    loss, (g_th, g_lam) = jax.value_and_grad(fm.fm_loss, argnums=(0, 1))(
        theta_true, 0.8, cache, fake_energy_with_lj, tm
    )
    assert float(loss) < 1e-20
    assert abs(float(g_lam)) < 1e-10
    for v in g_th.values():
        np.testing.assert_allclose(v, 0.0, atol=1e-10)
    assert float(fm.fm_loss(init_theta(tm), 1.0, cache, fake_energy_with_lj, tm)) > 1e-3


def test_loss_gradients_match_finite_differences_and_lambda_analytic():
    rng = np.random.default_rng(2)
    cache, _, tm = _cache(rng, noise=0.3)
    theta = {"log_eps": jnp.array([0.1, -0.3]), "log_sig": jnp.array([0.02, 0.04])}
    lam = 1.1
    g_th, g_lam = jax.grad(fm.fm_loss, argnums=(0, 1))(theta, lam, cache, fake_energy_with_lj, tm)

    # d/dlam = 2 mean_{frames,atoms} (F_hyb - F_pet) . F_dimer
    res = [np.asarray(fm.hybrid_forces(theta, lam, cache, k, fake_energy_with_lj, tm)) - cache.f_pet[k]
           for k in range(cache.n_frames)]
    g_lam_ref = np.mean([2 * np.mean(np.sum(r * cache.f_ml_dimer[k], -1)) for k, r in enumerate(res)])
    np.testing.assert_allclose(g_lam, g_lam_ref, rtol=1e-8)

    h = 1e-5
    for key in ("log_eps", "log_sig"):
        for t in range(2):
            e = jnp.zeros(2).at[t].set(h)
            lp = fm.fm_loss({**theta, key: theta[key] + e}, lam, cache, fake_energy_with_lj, tm)
            lm = fm.fm_loss({**theta, key: theta[key] - e}, lam, cache, fake_energy_with_lj, tm)
            np.testing.assert_allclose(g_th[key][t], (lp - lm) / (2 * h), rtol=1e-5, atol=1e-9)


def test_optimizer_recovers_teacher_lambda_and_theta():
    from jax.flatten_util import ravel_pytree
    from jax.scipy.optimize import minimize

    rng = np.random.default_rng(3)
    cache, theta_true, tm = _cache(rng, lam_true=0.7)
    p0 = {"theta": init_theta(tm), "lam": jnp.asarray(1.0)}
    flat0, unravel = ravel_pytree(p0)

    def loss(flat, c):
        p = unravel(flat)
        return fm.fm_loss(p["theta"], p["lam"], c, fake_energy_with_lj, tm)

    l0 = float(loss(flat0, cache))
    res = minimize(loss, flat0, args=(cache,), method="BFGS", options={"gtol": 1e-12, "maxiter": 500})
    p = unravel(res.x)
    assert float(loss(res.x, cache)) < 1e-10 * l0
    np.testing.assert_allclose(p["lam"], 0.7, atol=1e-4)
    for key in ("log_eps", "log_sig"):
        np.testing.assert_allclose(p["theta"][key], theta_true[key], atol=1e-4)


def _big_cache(n_frames: int, n_pairs: int) -> fm.ForceMatchCache:
    """Cache padded to ``n_pairs`` rows of non-uniform data (so constants cannot be splatted)."""
    rng = np.random.default_rng(12)
    cache, _, _ = _cache(rng, n_frames=n_frames)
    idx = np.stack([np.stack(np.triu_indices(N, 1), 1)[rng.integers(0, N * (N - 1) // 2, n_pairs)]
                    for _ in range(n_frames)]).astype(np.int32)
    mask = rng.random((n_frames, n_pairs))
    return fm.ForceMatchCache(**{**cache.__dict__, "pair_idx": idx, "pair_mask": mask})


def test_fm_loss_embeds_no_pair_constants_and_does_not_unroll_frames():
    tm = _type_map()
    th = init_theta(tm)
    sizes = {}
    for n_frames in (2, 5):
        cache = _big_cache(n_frames, 50_000)
        # cache passed as a jit argument (pytree): its arrays are parameters, not constants
        f = jax.jit(lambda t, lam, c: fm.fm_loss(t, lam, c, fake_energy_with_lj, tm))
        sizes[n_frames] = len(f.lower(th, 1.0, cache).as_text())
        # the eager path runs the same jitted body with the arrays as arguments
        lowered = fm._jitted_loss(fake_energy_with_lj, tm).lower(th, 1.0, cache).as_text()
        assert len(lowered) < 200_000
        np.testing.assert_allclose(
            fm.fm_loss(th, 1.0, cache, fake_energy_with_lj, tm), f(th, 1.0, cache), rtol=1e-12
        )
    # 50k pairs x (2 int32 + 1 float64) ~ 0.8 MB/frame if embedded; frames via lax.map
    assert sizes[2] < 200_000 and abs(sizes[5] - sizes[2]) < 2_000, sizes


def test_fm_loss_matches_per_frame_reference_and_device_cache():
    rng = np.random.default_rng(13)
    cache, _, tm = _cache(rng, n_frames=3, noise=0.5)
    theta = {"log_eps": jnp.array([0.2, -0.1]), "log_sig": jnp.array([0.03, 0.01])}
    ref = np.mean([
        np.mean(np.sum((np.asarray(fm.hybrid_forces(theta, 0.9, cache, k, fake_energy_with_lj, tm))
                        - cache.f_pet[k]) ** 2, -1))
        for k in range(cache.n_frames)
    ])
    np.testing.assert_allclose(fm.fm_loss(theta, 0.9, cache, fake_energy_with_lj, tm), ref, rtol=1e-12)
    dev = cache.to_device()
    assert isinstance(dev.pair_idx, jax.Array) and dev.fingerprint == cache.fingerprint
    np.testing.assert_allclose(fm.fm_loss(theta, 0.9, dev, fake_energy_with_lj, tm), ref, rtol=1e-12)


def test_pad_pairs_compacts_masked_rows_and_pads_with_real_pair():
    idx, mask = fm.pad_pairs(
        [np.array([[0, 1], [9, 9], [2, 3], [4, 5]]), np.array([[6, 7], [8, 9]])],
        [np.array([1.0, 0.0, 1.0, 0.0]), np.array([0.0, 1.0])],
    )
    assert idx.shape == (2, 2, 2) and mask.shape == (2, 2)
    np.testing.assert_array_equal(idx[0], [[0, 1], [2, 3]])
    np.testing.assert_array_equal(idx[1], [[8, 9], [8, 9]])  # filler = first valid pair
    np.testing.assert_array_equal(mask, [[1, 1], [1, 0]])
    idx, mask = fm.pad_pairs([np.array([[0, 1]]), np.array([[6, 7]])],
                             [np.array([1.0]), np.array([0.0])])
    np.testing.assert_array_equal(idx[1], [[0, 1]])  # no valid pair -> (0, 1), mask 0
    np.testing.assert_array_equal(mask[1], [0])
    assert np.all(idx[:, :, 0] != idx[:, :, 1])
    idx, mask = fm.pad_pairs([np.array([[0, 1], [2, 3]])], [np.array([0.0, 1.0])], capacity=3)
    np.testing.assert_array_equal(idx[0], [[2, 3], [2, 3], [2, 3]])
    np.testing.assert_array_equal(mask[0], [1, 0, 0])


def test_cache_npz_roundtrip(tmp_path):
    cache, _, _ = _cache(np.random.default_rng(4))
    path = tmp_path / "fm.npz"
    cache.save(path)
    back = fm.ForceMatchCache.load(path)
    for name, val in cache.__dict__.items():
        np.testing.assert_array_equal(getattr(back, name), val)


def test_make_molecules_whole_unwraps_and_wraps_centroid():
    rng = np.random.default_rng(5)
    box = np.array([10.0, 11.0, 12.0])
    centers = rng.uniform(0, 1, (N_MOL, 1, 3)) * box
    mol = centers + rng.normal(scale=0.7, size=(N_MOL, N_PER, 3))
    wrapped = (mol.reshape(-1, 3)) % box  # split molecules across faces
    whole = fm.make_molecules_whole(wrapped, [N_PER] * N_MOL, box).reshape(N_MOL, N_PER, 3)
    for m in range(N_MOL):
        d_in = mol[m] - mol[m, :1]
        np.testing.assert_allclose(whole[m] - whole[m, :1], d_in, atol=1e-12)
        c = whole[m].mean(0)
        assert np.all(c >= 0) and np.all(c < box)
        shift = (whole[m] - mol[m]) / box
        np.testing.assert_allclose(shift, np.round(shift), atol=1e-12)


def test_report_at_teacher_parameters_and_molecular_net_forces():
    cache, theta_true, tm = _cache(np.random.default_rng(6), lam_true=0.9)
    rep = fm.force_match_report(theta_true, 0.9, cache, fake_energy_with_lj, tm)
    assert rep["rmse_hybrid"] < 1e-10 and rep["mol_net_rmse_hybrid"] < 1e-10
    assert rep["rmse_ml_mono_only"] > 0.1
    np.testing.assert_allclose(rep["fm_loss"], 3 * rep["rmse_hybrid"] ** 2, atol=1e-18)
    net = fm.molecular_net_forces(np.ones((2, N, 3)), cache.molecule_index)
    np.testing.assert_allclose(net, np.full((2, N_MOL, 3), N_PER))


def _central_intramolecular_forces(rng, positions):
    """Pairwise central forces inside each molecule: zero net force and torque per molecule."""
    f = np.zeros_like(positions)
    for m in range(N_MOL):
        for a in range(N_PER):
            for b in range(a + 1, N_PER):
                i, j = m * N_PER + a, m * N_PER + b
                fij = rng.normal() * (positions[..., i, :] - positions[..., j, :])
                f[..., i, :] += fij
                f[..., j, :] -= fij
    return f


def test_molecular_torques_rigid_rotation_and_central_forces():
    rng = np.random.default_rng(8)
    x = rng.uniform(0, L, (2, N, 3))
    mol = fm.molecule_index_from_sizes([N_PER] * N_MOL)
    tq = fm.molecular_torques(_central_intramolecular_forces(rng, x), x, mol)
    np.testing.assert_allclose(tq, 0.0, atol=1e-10)
    # uniform force on a molecule gives zero torque about its centroid
    np.testing.assert_allclose(fm.molecular_torques(np.ones((2, N, 3)), x, mol), 0.0, atol=1e-10)
    # single force on atom 0 -> r0' x f0 with r0' measured from the centroid
    f = np.zeros((N, 3))
    f[0] = [0.0, 0.0, 1.0]
    arm = x[0, 0] - x[0, :N_PER].mean(0)
    np.testing.assert_allclose(fm.molecular_torques(f, x[0], mol)[0], np.cross(arm, f[0]))


def test_report_mol_keys_ignore_intramolecular_floor():
    rng = np.random.default_rng(9)
    cache, theta_true, tm = _cache(rng, lam_true=0.9)
    rep = fm.force_match_report(theta_true, 0.9, cache, fake_energy_with_lj, tm)
    assert "rmse_inter_hybrid" not in rep and "rms_inter_residual_pet" not in rep
    floor = _central_intramolecular_forces(rng, cache.positions)
    shifted = fm.ForceMatchCache(**{**cache.__dict__, "f_ml_mono": cache.f_ml_mono + floor})
    rep2 = fm.force_match_report(theta_true, 0.9, shifted, fake_energy_with_lj, tm)
    assert rep2["rmse_hybrid"] > 0.1  # per-atom metrics see the intramolecular error
    for key in ("mol_net_rmse_hybrid", "mol_torque_rmse_hybrid"):
        assert rep2[key] < 1e-10
    for key in ("mol_net_rmse_ml_only", "mol_torque_rmse_ml_only", "mol_torque_rmse_mm_only"):
        np.testing.assert_allclose(rep2[key], rep[key], rtol=1e-9, atol=1e-12)


class _Atoms:
    def __init__(self, rng, box):
        self.cell = SimpleNamespace(array=np.diag(box))
        self.positions = rng.uniform(0, L, (N, 3))

    def __len__(self):
        return N

    def get_atomic_numbers(self):
        return np.tile([6, 8, 1], N_MOL)


def _fake_calculators(rng, coulomb_extra=0.0):
    """Fake update fn / spherical calculator whose mm_F = -grad(energy_with_lj) (+ extra)."""
    idx, mask = _pairs(rng)
    calls = []

    def fake_update(positions, box=None):
        calls.append(np.asarray(box))
        return jnp.asarray(idx), jnp.asarray(mask)

    fake_update.lj_rmins = np.full(N, 1.5)
    fake_update.lj_epsilons = np.full(N, -0.1)
    fake_update.energy_with_lj = fake_energy_with_lj

    def fake_sph(**kw):
        x = jnp.asarray(kw["positions"], jnp.float64)
        cell = jnp.diag(jnp.asarray(kw["box"], jnp.float64))
        f_mm = -jax.grad(lambda r: fake_energy_with_lj(
            r, kw["mm_pair_idx"], kw["mm_pair_mask"], cell,
            lj_rmins=jnp.asarray(fake_update.lj_rmins), lj_epsilons=jnp.asarray(fake_update.lj_epsilons),
        ))(x)
        ones = np.ones((N, 3))
        return SimpleNamespace(internal_F=ones, ml_2b_F=2 * ones,
                               mm_F=(np.asarray(f_mm) + coulomb_extra) / fm.KCAL_PER_EV)

    return fake_update, fake_sph, idx, calls


def test_build_cache_with_fake_calculators(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    box = np.array([L, L, L])
    fake_update, fake_sph, idx, calls = _fake_calculators(rng)
    monkeypatch.setattr(fm, "pet_forces_kcal", lambda calc, z, x, cell: np.full((N, 3), 4.0))
    frames = [_Atoms(rng, box), _Atoms(rng, box)]
    kw = dict(atoms_per_monomer=[N_PER] * N_MOL, spherical_calculator=fake_sph,
              update_mm_pairs=fake_update, pet_calculator=object(), verbose=False)

    path = tmp_path / "sub" / "c.npz"
    cache = fm.build_force_match_cache(frames, cutoff_params=None, cache_path=path, fingerprint_extra="ckpt1", **kw)
    assert path.exists() and cache.n_frames == 2 and len(calls) == 2
    assert cache.pair_idx.shape[1] == len(idx)  # raw neighbour-list capacity kept
    np.testing.assert_allclose(calls[0], box)
    np.testing.assert_allclose(cache.f_ml_mono, fm.KCAL_PER_EV)
    np.testing.assert_allclose(cache.f_ml_dimer, 2 * fm.KCAL_PER_EV)
    np.testing.assert_allclose(cache.f_pet, 4.0)  # already kcal from pet_forces_kcal
    np.testing.assert_allclose(cache.base_epsilons, -0.1)
    assert fm.mm_parity_max_abs(cache, fake_energy_with_lj) < 1e-3

    # same inputs -> loaded from disk without touching the calculators
    again = fm.build_force_match_cache(frames, cutoff_params=None, cache_path=path, fingerprint_extra="ckpt1", **kw)
    assert len(calls) == 2
    np.testing.assert_array_equal(again.positions, cache.positions)
    assert again.fingerprint == cache.fingerprint != ""

    # different frame list / cutoffs / extra -> stale cache rebuilt, not returned
    more = frames + [_Atoms(rng, box)]
    bigger = fm.build_force_match_cache(more, cutoff_params=None, cache_path=path, fingerprint_extra="ckpt1", **kw)
    assert bigger.n_frames == 3 and len(calls) == 5
    moved = [_Atoms(rng, box), frames[1]]
    rebuilt = fm.build_force_match_cache(moved, cutoff_params=None, cache_path=path, fingerprint_extra="ckpt1", **kw)
    assert len(calls) == 7 and not np.allclose(rebuilt.positions[0], cache.positions[0])
    cut = SimpleNamespace(mm_switch_on=6.0, ml_switch_width=1.5)
    fm.build_force_match_cache(moved, cutoff_params=cut, cache_path=path, fingerprint_extra="ckpt1", **kw)
    assert len(calls) == 9
    fm.build_force_match_cache(moved, cutoff_params=SimpleNamespace(mm_switch_on=6.0, ml_switch_width=1.5),
                               cache_path=path, fingerprint_extra="ckpt1", **kw)
    assert len(calls) == 9  # equal cutoff values -> reused
    fm.build_force_match_cache(moved, cutoff_params=cut, cache_path=path, fingerprint_extra="ckpt2", **kw)
    assert len(calls) == 11
    with pytest.warns(UserWarning, match="checkpoint"):
        fm.build_force_match_cache(moved, cutoff_params=cut, cache_path=path, **kw)
    assert len(calls) == 13  # empty extra is a different fingerprint too


def test_build_cache_rebuilds_when_calculator_mm_parameters_change(tmp_path, monkeypatch):
    """Fixing atom types / LJ / charges in the calculator must not reuse stale f_mm_ref or base LJ."""
    rng = np.random.default_rng(14)
    upd, sph, _, calls = _fake_calculators(rng)
    monkeypatch.setattr(fm, "pet_forces_kcal", lambda calc, z, x, cell: np.zeros((N, 3)))
    frames = [_Atoms(rng, np.array([L, L, L]))]
    kw = dict(atoms_per_monomer=[N_PER] * N_MOL, spherical_calculator=sph, update_mm_pairs=upd,
              cutoff_params=None, pet_calculator=object(), cache_path=tmp_path / "c.npz",
              fingerprint_extra="ckpt", verbose=False)
    upd.at_codes = np.tile([0, 1, 2], N_MOL)
    upd.atc_names = ["CG2O1", "OG2D3", "HGA3"]
    fm.build_force_match_cache(frames, **kw)
    assert len(calls) == 1
    fm.build_force_match_cache(frames, **kw)
    assert len(calls) == 1  # unchanged -> reused

    upd.lj_rmins, upd.lj_epsilons = np.full(N, 1.0), np.full(N, -0.3)
    c = fm.build_force_match_cache(frames, **kw)
    assert len(calls) == 2
    np.testing.assert_allclose(c.base_rmins, 1.0)
    np.testing.assert_allclose(c.base_epsilons, -0.3)
    assert fm.mm_parity_max_abs(c, fake_energy_with_lj, 0, upd.lj_rmins, upd.lj_epsilons) < 1e-4

    upd.at_codes = np.tile([0, 2, 1], N_MOL)  # same LJ values, different types
    fm.build_force_match_cache(frames, **kw)
    assert len(calls) == 3
    upd.charges = np.full(N, 0.1)
    fm.build_force_match_cache(frames, **kw)
    assert len(calls) == 4
    fp = fm.cache_fingerprint(frames, kw["atoms_per_monomer"], None, "ckpt", update_mm_pairs=upd)
    assert fm.ForceMatchCache.load(kw["cache_path"]).fingerprint == fp

    # a file whose fingerprint matches but whose base LJ does not is rebuilt, not trusted
    stale = fm.ForceMatchCache.load(kw["cache_path"])
    fm.ForceMatchCache(**{**stale.__dict__, "base_rmins": stale.base_rmins + 0.5}).save(kw["cache_path"])
    c = fm.build_force_match_cache(frames, **kw)
    assert len(calls) == 5
    np.testing.assert_allclose(c.base_rmins, upd.lj_rmins)

    with pytest.raises(ValueError, match="pet_calculator"):
        fm.build_force_match_cache([_Atoms(rng, np.array([L, L, L]))], **{**kw, "pet_calculator": None})


def test_parity_guard_uses_calculator_base_lj_not_cached():
    rng = np.random.default_rng(15)
    upd, sph, _, _ = _fake_calculators(rng)
    frames = [_Atoms(rng, np.array([L, L, L]))]
    comps = fm.hybrid_force_components(sph, upd, frames[0].get_atomic_numbers(), frames[0].positions,
                                       frames[0].cell.array, N_MOL, None)
    comps.update(positions=frames[0].positions, cells=frames[0].cell.array, f_pet=np.zeros((N, 3)))
    cache = fm.ForceMatchCache.from_frames([comps], base_rmins=upd.lj_rmins, base_epsilons=upd.lj_epsilons,
                                           molecule_index=fm.molecule_index_from_sizes([N_PER] * N_MOL))
    assert fm.mm_parity_max_abs(cache, fake_energy_with_lj) < 1e-4
    assert fm.mm_parity_max_abs(cache, fake_energy_with_lj, 0, np.full(N, 1.0), np.full(N, -0.3)) > 1e-2


def test_build_cache_rebuilds_legacy_cache_without_fingerprint(tmp_path, monkeypatch):
    rng = np.random.default_rng(10)
    fake_update, fake_sph, _, calls = _fake_calculators(rng)
    monkeypatch.setattr(fm, "pet_forces_kcal", lambda calc, z, x, cell: np.zeros((N, 3)))
    legacy, _, _ = _cache(rng)
    path = tmp_path / "legacy.npz"
    np.savez_compressed(path, **{k: v for k, v in legacy.__dict__.items() if k != "fingerprint"})
    assert fm.ForceMatchCache.load(path).fingerprint == ""
    frames = [_Atoms(rng, np.array([L, L, L]))]
    out = fm.build_force_match_cache(
        frames, atoms_per_monomer=[N_PER] * N_MOL, spherical_calculator=fake_sph,
        update_mm_pairs=fake_update, cutoff_params=None, pet_calculator=object(),
        cache_path=path, fingerprint_extra="ckpt", verbose=False,
    )
    assert len(calls) == 1 and out.n_frames == 1 and out.fingerprint


def test_build_cache_rejects_mm_force_mismatch(monkeypatch):
    """E.g. a jax_pme hybrid: energy_with_lj is vdW only, calculator mm_F has Coulomb too."""
    rng = np.random.default_rng(11)
    fake_update, fake_sph, _, _ = _fake_calculators(rng, coulomb_extra=0.05)
    monkeypatch.setattr(fm, "pet_forces_kcal", lambda calc, z, x, cell: np.zeros((N, 3)))
    kw = dict(atoms_per_monomer=[N_PER] * N_MOL, spherical_calculator=fake_sph,
              update_mm_pairs=fake_update, cutoff_params=None, pet_calculator=object(), verbose=False)
    frames = [_Atoms(rng, np.array([L, L, L]))]
    with pytest.raises(ValueError, match="jax_pme"):
        fm.build_force_match_cache(frames, **kw)
    cache = fm.build_force_match_cache(frames, mm_parity_tol=None, **kw)
    np.testing.assert_allclose(fm.mm_parity_max_abs(cache, fake_energy_with_lj), 0.05, atol=1e-5)


def test_non_orthorhombic_cell_rejected():
    with pytest.raises(ValueError, match="orthorhombic"):
        fm._orthorhombic_lengths(np.array([[10.0, 1.0, 0], [0, 10, 0], [0, 0, 10]]))
