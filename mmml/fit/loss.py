"""Differentiable liquid-observable loss built on trajectory reweighting.

For each state point the frames were sampled at (theta_0, lambda_0) and
temperature ``sampled_T_K`` with total box potential ``u_ref`` (kcal/mol). The
total potential is split as ``U = u_const + V_theta`` where ``u_const`` is an
optional per-frame theta-independent part (the absolute PhysNet monomer
energies, ~3e5 kcal/mol for 266 acetones) and ``V_theta = energy_fn(theta,
lambda, sp)`` the theta-dependent rest (``lam * E_ml_dimer + E_mm(theta)``).
At (theta, lambda):

    w_i        = softmax(-beta [V_theta(x_i) - (u_ref_i - u_const_i)])
    <rho>      = sum_i w_i rho_i                  (rho_i from the box only)
    <E_liq>/N  = sum_i w_i (u_const_i + V_theta(x_i)) / N
    dHvap      = e_gas - <E_liq>/N + R T           (kcal/mol -> kJ/mol)

All large constants are removed in float64 on the host before anything is
cast to the JAX dtype, so the loss is float32-safe as long as ``energy_fn``
returns only the theta-dependent part; :func:`predict_observables` refuses
float32 energies whose magnitude cannot resolve ``0.01 kT``.

``e_gas`` is the mean gas-phase potential energy per molecule at T (kcal/mol;
a float or a callable of (theta, lambda)). Kinetic energies cancel classically;
the liquid P<V>/N term (~7e-3 kJ/mol for acetone at 1 atm: 122 A^3 per
molecule) is neglected.

    loss = sum_sp sum_obs ((pred - exp) / sigma)^2
           + prior_weight * (|theta - theta_prior|^2 + (lambda - 1)^2)
           + fm_weight * fm_fn(theta, lambda)

``u_const + V_theta0`` must be the Hamiltonian that was sampled, frame by
frame, in kcal/mol: check with :func:`check_reference_consistency` before
fitting. When ``u_ref`` is itself rebuilt from the same decomposition that
``energy_fn`` evaluates (:func:`state_point_from_cache`), comparing the two is
a tautology; the check is then made against ``u_sampled_kcal_mol``, the
potential the sampler recorded, and warns (or raises) when that is missing. JAX-MD HDF5 ``potential_energy`` is in eV (metal units); convert with
:func:`ev_to_kcal_mol` (or load via :mod:`mmml.fit.frames`).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from mmml.data.units import EV_TO_KCAL_MOL
from mmml.fit.reweight import (
    KB_KCAL_MOL_K,
    effective_sample_fraction,
    reweight_weights,
    reweighted_mean,
)
from mmml.fit.targets import StatePointTarget

KCAL_TO_KJ = 4.184

# Max |sampled_T_K - target.T_K| (K) accepted when pairing frames with a target.
SAMPLED_T_TOL_K = 0.5
# Resolution demanded of energy_fn's dtype, as a fraction of kT.
PRECISION_FRAC_KT = 0.01
# Liquid dHvap (kJ/mol) sanity range for the unit check when a state point has
# no dHvap reference: an eV-for-kcal/mol mixup divides dHvap by ~23.
DHVAP_SANITY_KJ_MOL = (8.0, 150.0)

# energy_fn(theta, lam, state_point) -> (n_frames,) theta-dependent box
# potential V_theta in kcal/mol (the total if ``u_const_kcal_mol`` is None).
EnergyFn = Callable[[Any, Any, "StatePointData"], jnp.ndarray]
GasEnergy = float | Callable[[Any, Any], jnp.ndarray]


def ev_to_kcal_mol(e_eV: Any) -> np.ndarray:
    """eV -> kcal/mol in float64 (JAX-MD metal-unit ``potential_energy``)."""
    return np.asarray(e_eV, dtype=np.float64) * EV_TO_KCAL_MOL


@dataclass(frozen=True)
class StatePointData:
    """Frames sampled at theta_0 for one state point, plus its target.

    ``frames`` is an opaque payload (positions, boxes, pair lists, ...) handed
    to ``energy_fn``; ``rho_g_cm3`` is the per-frame box density (None for NVT).
    ``u_ref_kcal_mol`` is the total sampled potential (kcal/mol, NOT eV) and
    ``u_const_kcal_mol`` its theta-independent part (None = 0), in which case
    ``energy_fn`` returns only the remainder. ``sampled_T_K`` is the thermostat
    temperature of the frames; it is required and must match ``target.T_K``
    (weights use ``beta`` at that T). ``u_sampled_kcal_mol`` is the potential
    the sampling run itself recorded (kcal/mol, NaN where absent); set it when
    ``u_ref`` was rebuilt rather than recorded, so that
    :func:`check_reference_consistency` can compare against the sampler. None
    means ``u_ref`` is the sampler's recorded potential.
    """

    target: StatePointTarget
    n_molecules: int
    u_ref_kcal_mol: np.ndarray  # (n_frames,) total potential at theta_0, kcal/mol
    e_gas_kcal_mol: GasEnergy
    rho_g_cm3: np.ndarray | None = None
    frames: Any = None
    sampled_T_K: float | None = None
    u_const_kcal_mol: np.ndarray | None = None  # (n_frames,) theta-independent part
    u_sampled_kcal_mol: np.ndarray | None = None  # (n_frames,) sampler-recorded potential

    def __post_init__(self) -> None:
        if self.sampled_T_K is None:
            raise ValueError("StatePointData needs sampled_T_K (thermostat T of the frames)")
        if abs(float(self.sampled_T_K) - float(self.target.T_K)) > SAMPLED_T_TOL_K:
            raise ValueError(
                f"frames sampled at {self.sampled_T_K} K cannot be reweighted to the target at {self.target.T_K} K"
            )
        n = np.shape(self.u_ref_kcal_mol)
        if self.u_const_kcal_mol is not None and np.shape(self.u_const_kcal_mol) != n:
            raise ValueError(f"u_const shape {np.shape(self.u_const_kcal_mol)} != u_ref {n}")
        if self.rho_g_cm3 is not None and np.shape(self.rho_g_cm3) != n:
            raise ValueError(f"rho shape {np.shape(self.rho_g_cm3)} != u_ref {n}")
        if self.u_sampled_kcal_mol is not None and np.shape(self.u_sampled_kcal_mol) != n:
            raise ValueError(f"u_sampled shape {np.shape(self.u_sampled_kcal_mol)} != u_ref {n}")

    def _host_terms(self) -> tuple[np.ndarray, float, np.ndarray, float]:
        """float64 ``(v_ref - v_shift, v_shift, u_const - c_mean, c_mean)``.

        ``v_ref = u_ref - u_const`` is what ``energy_fn`` must return at theta_0.
        """
        u_ref = np.asarray(self.u_ref_kcal_mol, dtype=np.float64)
        c = np.zeros_like(u_ref) if self.u_const_kcal_mol is None else np.asarray(self.u_const_kcal_mol, np.float64)
        v_ref = u_ref - c
        v_shift = float(v_ref.mean()) if v_ref.size else 0.0
        c_mean = float(c.mean()) if c.size else 0.0
        return v_ref - v_shift, v_shift, c - c_mean, c_mean


def state_point_from_cache(
    cache: Any, target: StatePointTarget, e_gas_kcal_mol: GasEnergy, *, frames: Any = None
) -> StatePointData:
    """:class:`StatePointData` from a :class:`mmml.fit.frames.FrameCache`.

    Uses the decomposed ``u_ref`` and ``E_ml_mono`` as ``u_const``, so
    ``energy_fn`` must return ``lam * E_ml_dimer + E_mm(theta)``. The
    sampler's recorded potential ``cache.frames.u_ref`` (kcal/mol, NaN if not
    recorded) becomes ``u_sampled_kcal_mol``: ``cache.u_ref`` is rebuilt from
    the same terms as ``energy_fn``, so only the recorded potential can show
    that the frames came from this Hamiltonian.
    """
    fs = cache.frames
    u_s = getattr(fs, "u_ref", None)
    n = np.shape(cache.u_ref)
    u_s = np.full(n, np.nan) if u_s is None else np.asarray(u_s, dtype=np.float64)
    return StatePointData(
        target,
        int(fs.n_molecules),
        np.asarray(cache.u_ref, dtype=np.float64),
        e_gas_kcal_mol,
        np.asarray(fs.density_g_cm3, dtype=np.float64),
        cache if frames is None else frames,
        sampled_T_K=float(fs.temperature_K),
        u_const_kcal_mol=np.asarray(cache.E_ml_mono, dtype=np.float64),
        u_sampled_kcal_mol=u_s,
    )


def _tree_sq_norm(tree: Any, center: Any | None = None) -> jnp.ndarray:
    if center is not None:
        tree = jax.tree_util.tree_map(lambda a, b: a - b, tree, center)
    leaves = jax.tree_util.tree_leaves(tree)
    return sum((jnp.sum(jnp.square(x)) for x in leaves), jnp.asarray(0.0))


def _e_gas(sp: StatePointData, theta: Any, lam: Any) -> jnp.ndarray:
    e = sp.e_gas_kcal_mol
    return jnp.asarray(e(theta, lam) if callable(e) else e)


def _check_precision(dtype: Any, v_ref: np.ndarray, T_K: float) -> None:
    """Raise if ``dtype`` cannot resolve ``PRECISION_FRAC_KT * kT`` at |v_ref|."""
    if not jnp.issubdtype(dtype, jnp.floating) or v_ref.size == 0:
        return
    ulp = float(jnp.finfo(dtype).eps) * float(np.max(np.abs(v_ref)))
    if ulp > PRECISION_FRAC_KT * KB_KCAL_MOL_K * T_K:
        raise ValueError(
            f"energy_fn returns {jnp.dtype(dtype).name} energies of magnitude "
            f"{np.max(np.abs(v_ref)):.4g} kcal/mol (resolution {ulp:.2g} kcal/mol > "
            f"{PRECISION_FRAC_KT} kT): pass the theta-independent part as u_const_kcal_mol "
            "and return only the theta-dependent energy, or enable jax_enable_x64"
        )


def predict_observables(theta: Any, lam: Any, sp: StatePointData, energy_fn: EnergyFn) -> dict[str, jnp.ndarray]:
    """Reweighted <rho> (g/cm^3), <E_liq>/N (kcal/mol), dHvap (kJ/mol), ESS fraction."""
    T = sp.target.T_K
    dv_ref, v_shift, dc, c_mean = sp._host_terms()
    v = jnp.asarray(energy_fn(theta, lam, sp))
    if v.shape != dv_ref.shape:
        raise ValueError(f"energy_fn returned shape {v.shape}, u_ref has {dv_ref.shape}")
    _check_precision(v.dtype, dv_ref + v_shift, T)
    dv = v - v_shift  # small: only frame-to-frame and theta-induced variation
    w = reweight_weights(dv, jnp.asarray(dv_ref, dtype=dv.dtype), T)
    n = sp.n_molecules
    e_liq = (c_mean + v_shift) / n + reweighted_mean((jnp.asarray(dc, dtype=dv.dtype) + dv) / n, w)
    dhvap = KCAL_TO_KJ * (_e_gas(sp, theta, lam) - e_liq + KB_KCAL_MOL_K * T)
    rho = reweighted_mean(jnp.asarray(sp.rho_g_cm3), w) if sp.rho_g_cm3 is not None else jnp.asarray(jnp.nan)
    return {
        "rho_g_cm3": rho,
        "e_liq_per_mol_kcal": e_liq,
        "dhvap_kJ_mol": dhvap,
        "ess_fraction": effective_sample_fraction(w),
    }


def liquid_loss(
    theta: Any,
    lam: Any,
    state_points: Sequence[StatePointData],
    energy_fn: EnergyFn,
    *,
    prior_weight: float = 0.0,
    theta_prior: Any | None = None,
    fm_weight: float = 0.0,
    fm_fn: Callable[[Any, Any], jnp.ndarray] | None = None,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Scalar loss and diagnostics; use with ``jax.value_and_grad(..., has_aux=True)``.

    Diagnostics are arrays over state points (NaN where a target is absent):
    ``T_K, ess_fraction, rho_pred, rho_exp, dhvap_pred, dhvap_exp,
    e_liq_per_mol_kcal, chi2`` plus scalars ``chi2_total, prior, fm, loss``.
    """
    nan = jnp.asarray(jnp.nan)
    rows: dict[str, list] = {
        k: []
        for k in (
            "T_K",
            "ess_fraction",
            "rho_pred",
            "rho_exp",
            "dhvap_pred",
            "dhvap_exp",
            "e_liq_per_mol_kcal",
            "chi2",
        )
    }
    chi2_total = jnp.asarray(0.0)
    for sp in state_points:
        tg = sp.target
        pred = predict_observables(theta, lam, sp, energy_fn)
        chi2 = jnp.asarray(0.0)
        if tg.rho_g_cm3 is not None:
            if sp.rho_g_cm3 is None:
                raise ValueError(f"density target at T={tg.T_K} K but no per-frame density")
            chi2 = chi2 + ((pred["rho_g_cm3"] - tg.rho_g_cm3) / tg.sigma_rho_g_cm3) ** 2
        if tg.dhvap_kJ_mol is not None:
            chi2 = chi2 + ((pred["dhvap_kJ_mol"] - tg.dhvap_kJ_mol) / tg.sigma_dhvap_kJ_mol) ** 2
        chi2_total = chi2_total + chi2
        rows["T_K"].append(jnp.asarray(tg.T_K))
        rows["ess_fraction"].append(pred["ess_fraction"])
        rows["rho_pred"].append(pred["rho_g_cm3"])
        rows["rho_exp"].append(nan if tg.rho_g_cm3 is None else jnp.asarray(tg.rho_g_cm3))
        rows["dhvap_pred"].append(pred["dhvap_kJ_mol"])
        rows["dhvap_exp"].append(nan if tg.dhvap_kJ_mol is None else jnp.asarray(tg.dhvap_kJ_mol))
        rows["e_liq_per_mol_kcal"].append(pred["e_liq_per_mol_kcal"])
        rows["chi2"].append(chi2)

    prior = _tree_sq_norm(theta, theta_prior) + jnp.sum(jnp.square(jnp.asarray(lam) - 1.0))
    fm = jnp.asarray(0.0) if fm_fn is None else jnp.asarray(fm_fn(theta, lam))
    total = chi2_total + prior_weight * prior + fm_weight * fm

    diag = {k: jnp.stack(v) if v else jnp.zeros(0) for k, v in rows.items()}
    diag.update(chi2_total=chi2_total, prior=prior, fm=fm, loss=total)
    return total, diag


def check_reference_consistency(
    theta0: Any,
    lam0: Any,
    state_points: Sequence[StatePointData],
    energy_fn: EnergyFn,
    *,
    spread_tol_kT: float = 0.05,
    offset_tol_kcal_mol_per_molecule: float = 1e-3,
    dhvap_ratio_bounds: tuple[float, float] | None = (0.5, 2.0),
    dhvap_abs_bounds_kJ_mol: tuple[float, float] = DHVAP_SANITY_KJ_MOL,
    sampled_spread_tol_kT: float = 0.1,
    require_sampled_potential: bool = False,
) -> list[dict[str, Any]]:
    """Raise unless ``u_const + energy_fn(theta0, lam0)`` reproduces the sampled potential.

    Reweighting is only valid when the refit Hamiltonian at theta_0 equals the
    sampled one (same terms, cutoffs, switching). With ``dU_i = U_theta0(x_i)
    - u_ref_i`` (float64 on the host):

    * spread ``max |dU_i - mean(dU)|`` must be ``<= spread_tol_kT * kT``: a
      frame-dependent mismatch distorts the weights (ESS < 1 at theta_0);
    * a constant offset ``|mean(dU)| / N`` must be ``<= offset_tol...``: it
      leaves the weights alone but shifts <E_liq>/N and dHvap;
    * sampler: when ``u_sampled_kcal_mol`` is set (u_ref rebuilt from the
      decomposition, where the two checks above are a tautology), the std over
      frames of ``U_theta0 - u_sampled`` must be ``<= max(sampled_spread_tol_kT
      * kT, 2 eps_float32 max|u_sampled|)`` (the floor is the resolution of a
      float32-recorded potential). A constant offset is allowed (terms outside
      the decomposition, e.g. a wall) and reported. If ``u_sampled`` is all NaN
      (e.g. PET-MAD frames, or a run that did not record its potential) the
      frames cannot be shown to come from this Hamiltonian: a warning, or an
      error with ``require_sampled_potential``;
    * units: the theta_0 dHvap divided by the experimental one (or, without a
      dHvap target, by ``target.nominal_dhvap_kJ_mol``) must lie in
      ``dhvap_ratio_bounds``; with neither it must lie in
      ``dhvap_abs_bounds_kJ_mol``. This catches eV-vs-kcal/mol mixups, which
      pass the other checks when u_ref and energy_fn share the wrong unit.
      ``dhvap_ratio_bounds=None`` disables it.

    Returns per state point ``{"T_K", "max_abs_dU", "spread", "offset",
    "ess_fraction", "dhvap_kJ_mol", "sampled_checked", "sampled_spread",
    "sampled_offset", "unit_check"}`` (energies in kcal/mol; the ``sampled_*``
    values are NaN when no sampler potential was compared).
    """
    out = []
    for sp in state_points:
        T = sp.target.T_K
        kT = KB_KCAL_MOL_K * T
        dv_ref, v_shift, dc, c_mean = sp._host_terms()
        v_raw = energy_fn(theta0, lam0, sp)
        v = np.asarray(v_raw, dtype=np.float64)
        if v.shape != dv_ref.shape:
            raise ValueError(f"T={T} K: energy shape {v.shape} != u_ref {dv_ref.shape}")
        _check_precision(jnp.asarray(v_raw).dtype, dv_ref + v_shift, T)
        du = (v - v_shift) - dv_ref
        offset = float(du.mean()) if du.size else 0.0
        spread = np.abs(du - offset)
        tol = spread_tol_kT * kT
        if not np.all(spread <= tol):
            i = int(np.argmax(spread))
            raise ValueError(
                f"T={T} K: U_theta0 - u_ref varies across frames (frame {i}: "
                f"|dU - <dU>|={spread[i]:.4g} kcal/mol > {tol:.4g} = {spread_tol_kT} kT); "
                "the reweighting Hamiltonian differs from the sampled one"
            )
        off_tol = offset_tol_kcal_mol_per_molecule * sp.n_molecules
        if abs(offset) > off_tol:
            raise ValueError(
                f"T={T} K: U_theta0 - u_ref has a constant offset {offset:.4g} kcal/mol "
                f"(> {off_tol:.4g}); the reweighting Hamiltonian differs from the sampled "
                "one (shifts <E_liq>/N and dHvap)"
            )
        u64 = v + c_mean + dc  # float64 total at theta_0
        sampled = _check_sampled(sp, u64, kT, sampled_spread_tol_kT, require_sampled_potential)
        w = np.exp(-(du - du.min()) / kT)
        w /= w.sum()
        e_liq = float(np.dot(w, u64)) / sp.n_molecules
        e_gas = float(np.asarray(_e_gas(sp, theta0, lam0)))
        dhvap = KCAL_TO_KJ * (e_gas - e_liq + kT)
        unit_check = _check_dhvap_units(sp.target, dhvap, dhvap_ratio_bounds, dhvap_abs_bounds_kJ_mol)
        out.append(
            {
                "T_K": float(T),
                "max_abs_dU": float(np.max(np.abs(du))) if du.size else 0.0,
                "spread": float(spread.max()) if du.size else 0.0,
                "offset": offset,
                "ess_fraction": float(1.0 / (np.sum(w**2) * w.size)) if du.size else 1.0,
                "dhvap_kJ_mol": dhvap,
                **sampled,
                "unit_check": unit_check,
            }
        )
    return out


def _check_sampled(
    sp: StatePointData, u_theta0: np.ndarray, kT: float, tol_kT: float, require: bool
) -> dict[str, Any]:
    """Compare ``U_theta0`` with the sampler-recorded potential (see caller)."""
    T = sp.target.T_K
    res: dict[str, Any] = {"sampled_checked": False, "sampled_spread": np.nan, "sampled_offset": np.nan}
    if sp.u_sampled_kcal_mol is None:
        return res  # u_ref is the sampler's own record
    u_s = np.asarray(sp.u_sampled_kcal_mol, dtype=np.float64)
    ok = np.isfinite(u_s)
    if ok.sum() < 2:
        msg = (
            f"T={T} K: no sampler-recorded potential (u_sampled is NaN); u_ref was rebuilt from "
            "the same decomposition as energy_fn, so nothing shows these frames were sampled "
            "from this Hamiltonian (e.g. PET-MAD frames): reweighting may be invalid"
        )
        if require:
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=3)
        return res
    if not ok.all():
        warnings.warn(f"T={T} K: sampler potential missing on {int((~ok).sum())} frames", stacklevel=3)
    d = u_theta0[ok] - u_s[ok]
    sd = float(np.std(d))
    tol = max(tol_kT * kT, 2.0 * float(np.finfo(np.float32).eps) * float(np.max(np.abs(u_s[ok]))))
    if not sd <= tol:
        raise ValueError(
            f"T={T} K: U_theta0 - (sampler-recorded potential) varies across frames: std "
            f"{sd:.4g} kcal/mol > {tol:.4g} ({sd / kT:.3g} kT); the reweighting Hamiltonian is "
            "not the one the frames were sampled from (LJ types? cutoffs? different model?)"
        )
    res.update(sampled_checked=True, sampled_spread=sd, sampled_offset=float(np.mean(d)))
    return res


def _check_dhvap_units(
    target: StatePointTarget,
    dhvap: float,
    ratio_bounds: tuple[float, float] | None,
    abs_bounds: tuple[float, float],
) -> str:
    """Unit sanity check on the theta_0 dHvap (kJ/mol); returns what was checked."""
    if ratio_bounds is None:
        return "disabled"
    hint = (
        "check that u_ref, energy_fn and e_gas are all in kcal/mol (JAX-MD potential_energy is eV: use ev_to_kcal_mol)"
    )
    ref = target.dhvap_kJ_mol or getattr(target, "nominal_dhvap_kJ_mol", None)
    if ref:
        lo, hi = ratio_bounds
        if not lo <= dhvap / ref <= hi:
            raise ValueError(
                f"T={target.T_K} K: dHvap at theta_0 = {dhvap:.4g} kJ/mol vs reference {ref:.4g} "
                f"(ratio outside [{lo}, {hi}]); {hint}"
            )
        return "target" if target.dhvap_kJ_mol else "nominal"
    lo, hi = abs_bounds
    if not lo <= dhvap <= hi:
        raise ValueError(
            f"T={target.T_K} K: dHvap at theta_0 = {dhvap:.4g} kJ/mol outside the sanity range "
            f"[{lo}, {hi}] kJ/mol (no dHvap reference at this state point); {hint}"
        )
    return "absolute"
