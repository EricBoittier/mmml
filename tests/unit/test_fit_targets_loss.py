"""Experimental targets and the reweighted liquid-observable loss (toy models)."""

from __future__ import annotations

import copy
import warnings
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mmml.fit.loss import (
    KCAL_TO_KJ,
    StatePointData,
    check_reference_consistency,
    ev_to_kcal_mol,
    liquid_loss,
    predict_observables,
    state_point_from_cache,
)
from mmml.data.units import EV_TO_KCAL_MOL
from mmml.fit.reweight import KB_KCAL_MOL_K
from mmml.fit.targets import (
    DEFAULT_REFERENCE_JSON,
    SIGMA_DHVAP_PRE1950_KJ_MOL,
    SIGMA_DHVAP_VAPOR_PRESSURE_KJ_MOL,
    DhvapPoint,
    StatePointTarget,
    boiling_point_K,
    build_state_point_targets,
    density_check_max_rel_error,
    density_g_cm3,
    dhvap_point_sigma,
    dhvap_points,
    load_reference,
)


@pytest.fixture(autouse=True)
def _x64():
    """float64 for the toy models, restored afterwards (other modules keep the default)."""
    with jax.enable_x64(True):
        yield


# Mirror of the cited reference JSON (subset) so tests do not need the file.
REF = {
    "ACO": {
        "T_boil_K": {"value": 329.3, "source": "NIST"},
        "dHvap_kJ_mol": [
            {"T_K": 298.15, "value": 31.27, "source": "Majer and Svoboda, 1985"},
            {"T_K": 228.0, "value": 32.9, "source": "Stephenson and Malanowski, 1987; from 178-243 K vapor pressure"},
            {"T_K": 293.0, "value": 32.1, "source": "Felsing and Durban, 1926 (via NIST WebBook)"},
            {"T_K": 329.3, "value": 29.1, "source": "Majer and Svoboda, 1985"},
        ],
        "density_DIPPR105": {
            "A": 57.6214,
            "B": 0.233955,
            "C": 507.803,
            "D": 0.254167,
            "range_K": [183, 507],
            "source": "DDBST DIPPR105",
            "check_kg_m3": {"195.96": 888.763, "202.44": 882.713, "293.16": 791.24, "299.64": 784.105},
        },
    },
    "DCM": {
        "dHvap_kJ_mol": [
            {"T_K": 298.15, "value": 29.03, "unc": 0.08, "source": "Manion, 2002"},
            {"T_K": 313.0, "value": 28.06, "source": "Majer and Svoboda, 1985"},
            {"T_K": [186, 312], "value": 29.4, "source": "Perry, 1926"},
        ],
    },
}


# ---------------------------------------------------------------- targets


def test_dippr105_reproduces_check_values():
    assert density_check_max_rel_error(REF, "ACO") < 1e-5
    assert float(density_g_cm3(REF, "aco", 293.16)) == pytest.approx(0.79124, rel=1e-5)
    with pytest.raises(ValueError):
        density_g_cm3(REF, "ACO", 150.0)
    with pytest.raises(KeyError):
        density_g_cm3(REF, "DCM", 298.15)


@pytest.mark.skipif(not DEFAULT_REFERENCE_JSON.exists(), reason="reference JSON not present")
def test_reference_file_density_checks():
    ref = load_reference(DEFAULT_REFERENCE_JSON)
    assert density_check_max_rel_error(ref, "ACO") < 1e-5
    aco = build_state_point_targets("ACO", ref=ref)  # 6 points minus T_b = 329.3 K
    assert len(aco) == 5 and all(t.rho_g_cm3 is not None for t in aco)
    assert max(t.T_K for t in aco) == 298.15
    by_T = {t.T_K: t.sigma_dhvap_kJ_mol for t in aco}
    assert by_T[298.15] == pytest.approx(0.5)  # Majer-Svoboda calorimetric
    assert by_T[293.0] == pytest.approx(SIGMA_DHVAP_PRE1950_KJ_MOL)  # Felsing 1926
    for T in (228.0, 254.0, 274.0):  # Stephenson-Malanowski, vapour-pressure derived
        assert by_T[T] == pytest.approx(SIGMA_DHVAP_VAPOR_PRESSURE_KJ_MOL)
    dcm = build_state_point_targets("DCM", ref=ref)  # 4 points minus T_b = 313 K
    assert len(dcm) == 3 and all(t.rho_g_cm3 is None for t in dcm)
    assert all(t.nominal_dhvap_kJ_mol for t in aco + dcm)


def test_dhvap_points_keep_range_entries():
    pts = dhvap_points(REF, "DCM")
    assert pts[-1].T_K is None and pts[-1].T_range_K == (186.0, 312.0)
    assert pts[0].unc_kJ_mol == pytest.approx(0.08)
    assert boiling_point_K(REF, "ACO") == pytest.approx(329.3) and boiling_point_K(REF, "DCM") is None


def test_dhvap_point_sigma_per_source():
    assert dhvap_point_sigma(DhvapPoint(298.15, 31.27, "Majer and Svoboda, 1985")) == pytest.approx(0.5)
    assert dhvap_point_sigma(DhvapPoint(298.15, 29.0, "Manion, 2002", 0.8)) == pytest.approx(0.8)
    assert dhvap_point_sigma(DhvapPoint(293.0, 32.1, "Felsing and Durban, 1926")) == pytest.approx(1.0)
    vp = DhvapPoint(248.0, 30.2, "Ganeff and Jungers, 2010; from 233-313 K (via NIST WebBook)")
    assert dhvap_point_sigma(vp) == pytest.approx(1.5)
    assert dhvap_point_sigma(DhvapPoint(250.0, 30.0, "x; from vapour pressure data"), 2.0) == pytest.approx(2.0)


def test_build_targets_default_temperatures():
    aco = build_state_point_targets("ACO", ref=REF)
    assert [t.T_K for t in aco] == [228.0, 293.0, 298.15]  # T_b point excluded
    assert [t.T_K for t in build_state_point_targets("ACO", ref=REF, exclude_boiling_point=False)] == [
        228.0,
        293.0,
        298.15,
        329.3,
    ]
    t298 = aco[2]
    assert t298.dhvap_kJ_mol == pytest.approx(31.27)
    assert t298.sigma_dhvap_kJ_mol == pytest.approx(0.5)
    assert t298.rho_g_cm3 == pytest.approx(float(density_g_cm3(REF, "ACO", 298.15)))
    assert t298.sigma_rho_g_cm3 == pytest.approx(0.005 * t298.rho_g_cm3)
    assert "Majer" in t298.dhvap_source and "DIPPR105" in t298.rho_source
    assert t298.nominal_dhvap_kJ_mol == pytest.approx(31.27)
    assert aco[0].sigma_dhvap_kJ_mol == pytest.approx(SIGMA_DHVAP_VAPOR_PRESSURE_KJ_MOL)
    assert aco[1].sigma_dhvap_kJ_mol == pytest.approx(SIGMA_DHVAP_PRE1950_KJ_MOL)
    flat = build_state_point_targets("ACO", ref=REF, per_source_sigma=False)
    assert all(t.sigma_dhvap_kJ_mol == pytest.approx(0.5) for t in flat)

    dcm = build_state_point_targets("DCM", ref=REF, sigma_dhvap_kJ_mol=0.05)
    assert [t.T_K for t in dcm] == [298.15, 313.0]  # range entry skipped
    assert all(t.rho_g_cm3 is None for t in dcm)
    assert dcm[0].sigma_dhvap_kJ_mol == pytest.approx(0.08)  # stated unc > default


def test_build_targets_requested_temperatures():
    tg = build_state_point_targets("ACO", [200.0, 298.0, 400.0], ref=REF, P_atm=2.0)
    assert [t.T_K for t in tg] == [200.0, 298.0, 400.0]
    assert tg[0].dhvap_kJ_mol is None and tg[0].rho_g_cm3 is not None
    assert tg[0].nominal_dhvap_kJ_mol == pytest.approx(32.9)  # closest point (228 K), unit checks only
    assert tg[1].dhvap_kJ_mol == pytest.approx(31.27)  # 298.15 within 1 K
    assert tg[2].P_atm == 2.0
    # DCM at a T without dHvap -> nothing to fit -> dropped
    assert build_state_point_targets("DCM", [250.0], ref=REF) == []
    ref2 = copy.deepcopy(REF)
    ref2["ACO"]["density_DIPPR105"]["range_K"] = [183, 250]
    assert build_state_point_targets("ACO", [298.15], ref=ref2)[0].rho_g_cm3 is None


# ---------------------------------------------------------------- loss (toy)
#
# N independent 1D "molecules": U(x) = sum_n [lam k x_n^2 / 2 + a x_n],
# sampled at (a, lam) = (0, 1). Exactly:
#   <x_n>        = -a / (lam k),   <U>/N = kT/2 - a^2 / (2 lam k)
#   rho_i        = rho0 + c mean_n x_n  ->  <rho> = rho0 - c a / (lam k)

T, K, N, RHO0, C_RHO, E_GAS = 300.0, 4.0, 2, 0.8, 0.05, -1.0
KT = KB_KCAL_MOL_K * T


def _toy_energy(theta, lam, sp):
    x = sp.frames
    return jnp.sum(0.5 * lam * K * x**2 + theta["a"] * x, axis=1)


def _toy_state_point(target, n_frames=400_000, seed=0):
    x = np.random.default_rng(seed).normal(0.0, np.sqrt(KT / K), size=(n_frames, N))
    u_ref = np.sum(0.5 * K * x**2, axis=1)
    rho = RHO0 + C_RHO * x.mean(axis=1)
    return StatePointData(target, N, u_ref, E_GAS, rho, x, sampled_T_K=T)


def _exact(a, lam):
    e_liq = 0.5 * KT - a**2 / (2 * lam * K)
    return RHO0 - C_RHO * a / (lam * K), KCAL_TO_KJ * (E_GAS - e_liq + KT)


def _target(rho=None, dh=None):
    return StatePointTarget("TOY", T, 1.0, rho, dh, 0.005, 0.1)


def test_theta0_gives_uniform_weights_and_sample_means():
    sp = _toy_state_point(_target(0.8, 10.0), n_frames=1000)
    pred = predict_observables({"a": jnp.asarray(0.0)}, 1.0, sp, _toy_energy)
    assert float(pred["ess_fraction"]) == pytest.approx(1.0)
    assert float(pred["rho_g_cm3"]) == pytest.approx(sp.rho_g_cm3.mean())
    assert float(pred["e_liq_per_mol_kcal"]) == pytest.approx(sp.u_ref_kcal_mol.mean() / N)


def test_reweighted_observables_match_analytic():
    sp = _toy_state_point(_target(0.8, 10.0))
    a, lam = 0.3, 1.1
    pred = predict_observables({"a": jnp.asarray(a)}, lam, sp, _toy_energy)
    rho_ex, dh_ex = _exact(a, lam)
    sd_x = np.sqrt(KT / K)
    assert float(pred["rho_g_cm3"]) == pytest.approx(rho_ex, abs=4 * C_RHO * sd_x / 400)
    assert float(pred["dhvap_kJ_mol"]) == pytest.approx(dh_ex, abs=5e-3)
    assert 0.3 < float(pred["ess_fraction"]) < 1.0


def test_loss_value_gradient_fd_and_diagnostics():
    a_true, lam_true = 0.25, 1.0
    rho_ex, dh_ex = _exact(a_true, lam_true)
    sps = [
        _toy_state_point(_target(rho_ex, dh_ex), n_frames=200_000, seed=1),
        _toy_state_point(_target(None, dh_ex), n_frames=200_000, seed=2),
    ]

    def fm(theta, lam):
        return (theta["a"] - 0.2) ** 2

    def f(a, lam):
        return liquid_loss({"a": a}, lam, sps, _toy_energy, prior_weight=0.3, fm_weight=2.0, fm_fn=fm)

    a0, lam0 = 0.1, 1.05
    (val, diag), (ga, gl) = jax.value_and_grad(f, argnums=(0, 1), has_aux=True)(a0, lam0)
    h = 1e-5
    fd_a = (f(a0 + h, lam0)[0] - f(a0 - h, lam0)[0]) / (2 * h)
    fd_l = (f(a0, lam0 + h)[0] - f(a0, lam0 - h)[0]) / (2 * h)
    assert float(ga) == pytest.approx(float(fd_a), rel=1e-6)
    assert float(gl) == pytest.approx(float(fd_l), rel=1e-6)
    assert float(ga) < 0.0  # a0 < a_true: loss decreases toward the truth

    prior = a0**2 + (lam0 - 1.0) ** 2
    fmv = (a0 - 0.2) ** 2
    assert float(diag["prior"]) == pytest.approx(prior)
    assert float(diag["fm"]) == pytest.approx(fmv)
    assert float(val) == pytest.approx(float(diag["chi2_total"]) + 0.3 * prior + 2.0 * fmv)
    assert float(diag["chi2_total"]) == pytest.approx(float(jnp.sum(diag["chi2"])))
    assert diag["T_K"].shape == (2,) and np.isnan(float(diag["rho_exp"][1]))
    assert np.all((diag["ess_fraction"] > 0) & (diag["ess_fraction"] <= 1))

    # Gradient descent on the data term alone recovers a_true (lam fixed at 1).
    def data_loss(a):
        return liquid_loss({"a": a}, 1.0, sps[:1], _toy_energy)[0]

    a = 0.0
    for _ in range(60):
        a = a - 0.02 * float(jax.grad(data_loss)(a))
    assert a == pytest.approx(a_true, abs=0.02)


def test_prior_centre_and_pytree_theta():
    sp = _toy_state_point(_target(None, 10.0), n_frames=100)

    def energy(theta, lam, sp):
        return _toy_energy({"a": theta["a"][0]}, lam, sp)

    theta = {"a": jnp.array([0.2, 0.0]), "b": jnp.array([1.0])}
    _, d = liquid_loss(theta, 1.0, [sp], energy, prior_weight=1.0)
    assert float(d["prior"]) == pytest.approx(0.04 + 1.0)
    _, d = liquid_loss(
        theta, 1.0, [sp], energy, prior_weight=1.0, theta_prior={"a": jnp.array([0.2, 0.0]), "b": jnp.array([0.5])}
    )
    assert float(d["prior"]) == pytest.approx(0.25)


def test_callable_gas_energy_enters_dhvap():
    sp = _toy_state_point(_target(None, 10.0), n_frames=100)
    sp_c = StatePointData(sp.target, N, sp.u_ref_kcal_mol, lambda th, lam: -2.0 * lam, None, sp.frames, sampled_T_K=T)
    p1 = predict_observables({"a": 0.0}, 1.0, sp, _toy_energy)
    p2 = predict_observables({"a": 0.0}, 1.0, sp_c, _toy_energy)
    assert float(p1["dhvap_kJ_mol"] - p2["dhvap_kJ_mol"]) == pytest.approx(KCAL_TO_KJ)
    assert np.isnan(float(p2["rho_g_cm3"]))


def test_density_target_without_frame_density_raises():
    sp = _toy_state_point(_target(0.8, None), n_frames=10)
    bad = StatePointData(sp.target, N, sp.u_ref_kcal_mol, E_GAS, None, sp.frames, sampled_T_K=T)
    with pytest.raises(ValueError, match="per-frame density"):
        liquid_loss({"a": 0.0}, 1.0, [bad], _toy_energy)


def test_reference_consistency_check():
    dh0 = _exact(0.0, 1.0)[1]
    sp = _toy_state_point(_target(0.8, dh0), n_frames=50)
    (rep,) = check_reference_consistency({"a": 0.0}, 1.0, [sp], _toy_energy)
    assert rep["max_abs_dU"] < 1e-12 and rep["ess_fraction"] == pytest.approx(1.0)
    assert rep["dhvap_kJ_mol"] == pytest.approx(KCAL_TO_KJ * (E_GAS - sp.u_ref_kcal_mol.mean() / N + KT))

    def shifted(theta, lam, sp):  # constant offset: weights intact, dHvap shifted
        return _toy_energy(theta, lam, sp) + 0.05

    with pytest.raises(ValueError, match="constant offset"):
        check_reference_consistency({"a": 0.0}, 1.0, [sp], shifted)
    (rep,) = check_reference_consistency({"a": 0.0}, 1.0, [sp], shifted, offset_tol_kcal_mol_per_molecule=0.1)
    assert rep["offset"] == pytest.approx(0.05) and rep["spread"] < 1e-12
    with pytest.raises(ValueError, match="shape"):
        check_reference_consistency({"a": 0.0}, 1.0, [sp], lambda t, lam, s: jnp.zeros(3))


# ------------------------------------------- realistic magnitudes (acetone box)
#
# hyb_aco.json: E_hybrid ~ -12800 eV ~ -2.95e5 kcal/mol for ACO:266, almost all
# of it the (theta-independent) PhysNet monomer sum.

N_ACO, T_ACO, F_ACO = 266, 200.0, 2000


def _aco_like(seed=0):
    rng = np.random.default_rng(seed)
    mono = (-12722.5 + 0.8 * rng.standard_normal(F_ACO)) * EV_TO_KCAL_MOL
    dimer = (-56.0 + 1.5 * rng.standard_normal(F_ACO)) * EV_TO_KCAL_MOL
    mm = (-33.0 + 3.0 * rng.standard_normal(F_ACO)) * EV_TO_KCAL_MOL
    u_ref = mono + dimer + mm
    e_gas = float(mono.mean() / N_ACO) + 0.5  # dHvap ~ 32 kJ/mol at theta_0
    return mono, dimer, mm, u_ref, e_gas


def _aco_sp(u_const=None, target_dh=32.0):
    mono, dimer, mm, u_ref, e_gas = _aco_like()
    tg = StatePointTarget("ACO", T_ACO, 1.0, None, target_dh, None, 0.5)
    frames = {"mono": mono, "dimer": dimer, "mm": mm}
    return StatePointData(tg, N_ACO, u_ref, e_gas, None, frames, sampled_T_K=T_ACO, u_const_kcal_mol=u_const)


def _aco_total(theta, lam, sp, dtype=jnp.float64):
    f = sp.frames
    return jnp.asarray(f["mono"], dtype) + lam * jnp.asarray(f["dimer"], dtype) + theta * jnp.asarray(f["mm"], dtype)


def _aco_dependent(theta, lam, sp, dtype=jnp.float64):
    f = sp.frames
    return lam * jnp.asarray(f["dimer"], dtype) + theta * jnp.asarray(f["mm"], dtype)


def test_consistency_catches_frame_noise_at_box_magnitudes():
    sp = _aco_sp()
    (rep,) = check_reference_consistency(1.0, 1.0, [sp], _aco_total)
    assert rep["spread"] < 1e-6 and 20.0 < rep["dhvap_kJ_mol"] < 45.0
    noise = 0.25 * np.sign(np.random.default_rng(1).standard_normal(F_ACO))

    def noisy(theta, lam, sp):  # passed the old rtol=1e-6 * |u_ref| check
        return _aco_total(theta, lam, sp) + noise

    assert float(predict_observables(1.0, 1.0, sp, noisy)["ess_fraction"]) < 0.9
    with pytest.raises(ValueError, match="varies across frames"):
        check_reference_consistency(1.0, 1.0, [sp], noisy)


def test_float32_needs_u_const_split():
    exact = KCAL_TO_KJ * (_aco_sp().e_gas_kcal_mol - _aco_like()[3].mean() / N_ACO + KB_KCAL_MOL_K * T_ACO)

    def total32(theta, lam, sp):
        return _aco_total(theta, lam, sp, jnp.float32)

    sp = _aco_sp()
    with pytest.raises(ValueError, match="u_const"):
        predict_observables(1.0, 1.0, sp, total32)
    with pytest.raises(ValueError, match="u_const"):
        check_reference_consistency(1.0, 1.0, [sp], total32)

    def dep32(theta, lam, sp):
        return _aco_dependent(theta, lam, sp, jnp.float32)

    sp = _aco_sp(u_const=_aco_like()[0])
    p = predict_observables(jnp.float32(1.0), jnp.float32(1.0), sp, dep32)
    assert p["ess_fraction"].dtype == jnp.float32
    assert float(p["ess_fraction"]) == pytest.approx(1.0, abs=1e-5)
    assert float(p["dhvap_kJ_mol"]) == pytest.approx(exact, abs=1e-3)
    (rep,) = check_reference_consistency(1.0, 1.0, [sp], dep32)
    assert rep["spread"] < 0.05 * KB_KCAL_MOL_K * T_ACO
    # float32 reweighting away from theta_0 matches float64
    p32 = predict_observables(jnp.float32(1.001), jnp.float32(0.999), sp, dep32)
    p64 = predict_observables(1.001, 0.999, sp, _aco_dependent)
    assert float(p32["dhvap_kJ_mol"]) == pytest.approx(float(p64["dhvap_kJ_mol"]), abs=2e-3)
    assert float(p32["ess_fraction"]) == pytest.approx(float(p64["ess_fraction"]), rel=1e-3)


def test_sampled_temperature_required_and_matched():
    tg = StatePointTarget("ACO", 200.0, 1.0, None, 32.0, None, 0.5)
    u = np.zeros(4)
    with pytest.raises(ValueError, match="sampled_T_K"):
        StatePointData(tg, 2, u, 0.0)
    with pytest.raises(ValueError, match="cannot be reweighted"):
        StatePointData(tg, 2, u, 0.0, sampled_T_K=298.0)
    StatePointData(tg, 2, u, 0.0, sampled_T_K=200.2)
    with pytest.raises(ValueError, match="u_const shape"):
        StatePointData(tg, 2, u, 0.0, sampled_T_K=200.0, u_const_kcal_mol=np.zeros(3))


def test_ev_units_mixup_is_caught():
    assert ev_to_kcal_mol([1.0, -2.0]) == pytest.approx([EV_TO_KCAL_MOL, -2 * EV_TO_KCAL_MOL])
    mono, dimer, mm, u_ref, e_gas = _aco_like()
    tg = StatePointTarget("ACO", T_ACO, 1.0, None, 32.0, None, 0.5)
    ev = {"mono": mono / EV_TO_KCAL_MOL, "dimer": dimer / EV_TO_KCAL_MOL, "mm": mm / EV_TO_KCAL_MOL}
    # Everything consistently in eV (JAX-MD potential_energy used verbatim).
    sp = StatePointData(tg, N_ACO, u_ref / EV_TO_KCAL_MOL, e_gas / EV_TO_KCAL_MOL, None, ev, sampled_T_K=T_ACO)
    with pytest.raises(ValueError, match="ev_to_kcal_mol"):
        check_reference_consistency(1.0, 1.0, [sp], _aco_total)
    # Converted: passes.
    sp_ok = StatePointData(
        tg,
        N_ACO,
        ev_to_kcal_mol(u_ref / EV_TO_KCAL_MOL),
        e_gas,
        None,
        {k: ev_to_kcal_mol(v) for k, v in ev.items()},
        sampled_T_K=T_ACO,
    )
    check_reference_consistency(1.0, 1.0, [sp_ok], _aco_total)


def test_state_point_from_cache():
    mono, dimer, mm, u_ref, e_gas = _aco_like()
    tg = StatePointTarget("ACO", T_ACO, 1.0, 0.9, 32.0, 0.0045, 0.5)

    def cache_with(u_sampled):
        fs = SimpleNamespace(
            n_molecules=N_ACO, density_g_cm3=np.full(F_ACO, 0.9), temperature_K=T_ACO, u_ref=u_sampled
        )
        # FrameCache.u_ref is rebuilt from the decomposition, not the sampler's record.
        return SimpleNamespace(frames=fs, u_ref=u_ref, E_ml_mono=mono, dimer=dimer, mm=mm)

    def dep(theta, lam, sp):
        return lam * jnp.asarray(sp.frames.dimer) + theta * jnp.asarray(sp.frames.mm)

    # Sampler recorded the same Hamiltonian (plus a constant wall term): passes.
    cache = cache_with(u_ref - 3.0)
    sp = state_point_from_cache(cache, tg, e_gas)
    assert sp.sampled_T_K == T_ACO and sp.frames is cache
    (rep,) = check_reference_consistency(1.0, 1.0, [sp], dep)
    assert rep["spread"] < 1e-8 and rep["sampled_checked"]
    assert rep["sampled_offset"] == pytest.approx(3.0) and rep["sampled_spread"] < 1e-6
    assert float(predict_observables(1.0, 1.0, sp, dep)["rho_g_cm3"]) == pytest.approx(0.9)

    # Sampler ran a different Hamiltonian: the u_ref-vs-energy_fn comparison
    # is a tautology here (spread 0), only the sampler record exposes it. A
    # 1 kcal/mol frame-to-frame mismatch (2.5 kT at 200 K) is 0.004
    # kcal/mol/molecule, which frames.check_decomposition's 0.01 would accept.
    other = u_ref + 1.0 * np.random.default_rng(3).standard_normal(F_ACO)
    sp_bad = state_point_from_cache(cache_with(other), tg, e_gas)
    with pytest.raises(ValueError, match="sampler-recorded"):
        check_reference_consistency(1.0, 1.0, [sp_bad], dep)

    # float32-recorded sampler potential (rounding only) is within tolerance.
    sp32 = state_point_from_cache(cache_with(u_ref.astype(np.float32).astype(np.float64)), tg, e_gas)
    assert check_reference_consistency(1.0, 1.0, [sp32], dep)[0]["sampled_checked"]

    # No sampler record (e.g. PET-MAD frames): warn, or raise on request.
    sp_nan = state_point_from_cache(cache_with(np.full(F_ACO, np.nan)), tg, e_gas)
    with pytest.warns(UserWarning, match="no sampler-recorded potential"):
        (rep,) = check_reference_consistency(1.0, 1.0, [sp_nan], dep)
    assert not rep["sampled_checked"] and np.isnan(rep["sampled_spread"])
    with pytest.raises(ValueError, match="no sampler-recorded potential"):
        check_reference_consistency(1.0, 1.0, [sp_nan], dep, require_sampled_potential=True)
    fs_no = SimpleNamespace(n_molecules=N_ACO, density_g_cm3=np.full(F_ACO, 0.9), temperature_K=T_ACO)
    sp_none = state_point_from_cache(SimpleNamespace(frames=fs_no, u_ref=u_ref, E_ml_mono=mono), tg, e_gas)
    assert np.all(np.isnan(sp_none.u_sampled_kcal_mol))

    # u_ref given directly (it is the sampler's record): no sampler comparison, no warning.
    sp_direct = StatePointData(tg, N_ACO, u_ref, e_gas, None, cache, sampled_T_K=T_ACO, u_const_kcal_mol=mono)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        (rep,) = check_reference_consistency(1.0, 1.0, [sp_direct], dep)
    assert not rep["sampled_checked"] and rep["unit_check"] == "target"
    with pytest.raises(ValueError, match="u_sampled shape"):
        StatePointData(tg, N_ACO, u_ref, e_gas, sampled_T_K=T_ACO, u_sampled_kcal_mol=u_ref[:3])


def test_ev_units_mixup_caught_without_dhvap_target():
    """200 K acetone has only a density target; the unit check must still run."""
    mono, dimer, mm, u_ref, e_gas = _aco_like()
    ev = {"mono": mono / EV_TO_KCAL_MOL, "dimer": dimer / EV_TO_KCAL_MOL, "mm": mm / EV_TO_KCAL_MOL}
    rho = np.full(F_ACO, 0.885)

    def sp_for(tg, frames, u, eg):
        return StatePointData(tg, N_ACO, u, eg, rho, frames, sampled_T_K=T_ACO)

    frames_kcal = {"mono": mono, "dimer": dimer, "mm": mm}
    for nominal, expect in ((None, "absolute"), (32.9, "nominal")):
        tg = StatePointTarget("ACO", T_ACO, 1.0, 0.885, None, 0.004, None, nominal_dhvap_kJ_mol=nominal)
        sp_ev = sp_for(tg, ev, u_ref / EV_TO_KCAL_MOL, e_gas / EV_TO_KCAL_MOL)
        with pytest.raises(ValueError, match="ev_to_kcal_mol"):
            check_reference_consistency(1.0, 1.0, [sp_ev], _aco_total)
        (rep,) = check_reference_consistency(1.0, 1.0, [sp_for(tg, frames_kcal, u_ref, e_gas)], _aco_total)
        assert rep["unit_check"] == expect and 20.0 < rep["dhvap_kJ_mol"] < 45.0
    (rep,) = check_reference_consistency(1.0, 1.0, [sp_ev], _aco_total, dhvap_ratio_bounds=None)
    assert rep["unit_check"] == "disabled"


def test_liquid_loss_float32_default_dtypes():
    """Production default (x64 off): value_and_grad runs in float32 and is finite."""
    mono, dimer, mm, u_ref, e_gas = _aco_like()
    tg = StatePointTarget("ACO", T_ACO, 1.0, 0.885, 32.0, 0.004, 0.5)
    rho = 0.885 + 0.002 * np.random.default_rng(4).standard_normal(F_ACO)
    frames = {"dimer": dimer, "mm": mm}
    sp = StatePointData(tg, N_ACO, u_ref, e_gas, rho, frames, sampled_T_K=T_ACO, u_const_kcal_mol=mono)

    def dep(theta, lam, sp):
        f = sp.frames
        return lam * jnp.asarray(f["dimer"]) + theta["s"] * jnp.asarray(f["mm"])

    with jax.enable_x64(False):
        theta = {"s": jnp.asarray(1.0)}
        assert theta["s"].dtype == jnp.float32
        (val, diag), (g_th, g_lam) = jax.value_and_grad(
            lambda th, lam: liquid_loss(th, lam, [sp], dep, prior_weight=0.1), argnums=(0, 1), has_aux=True
        )(theta, jnp.asarray(1.0))
        assert val.dtype == jnp.float32 and g_th["s"].dtype == jnp.float32
        assert np.isfinite(float(val)) and np.isfinite(float(g_th["s"])) and np.isfinite(float(g_lam))
        assert float(diag["ess_fraction"][0]) == pytest.approx(1.0, abs=1e-4)
