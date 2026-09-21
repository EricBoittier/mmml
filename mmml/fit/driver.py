"""Wire the reweighting modules into a fit of theta / lambda.

Used by ``mmml fit-liquid``. The toy MM path lets a JAX-MD NPT HDF5 be
decomposed and fitted without PhysNet or CHARMM (CI dry-run).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from mmml.data.units import EV_TO_KCAL_MOL
from mmml.fit.frames import FrameCache, FrameSet, HybridTermFns, decompose, mm_energies
from mmml.fit.lj_theta import LjTypeMap, Theta, init_theta, project_theta
from mmml.fit.loss import (
    KCAL_TO_KJ,
    StatePointData,
    check_reference_consistency,
    liquid_loss,
    state_point_from_cache,
)
from mmml.fit.reweight import KB_KCAL_MOL_K
from mmml.fit.targets import StatePointTarget, build_state_point_targets, load_reference
from mmml.models.mm_lj_scales import mm_lj_scales_metadata

EnergyFn = Callable[[Any, Any, StatePointData], jnp.ndarray]


def energy_fn_from_cache(mm_energy_with_lj: Callable, type_map: LjTypeMap) -> EnergyFn:
    """``lam * E_ml_dimer + E_mm(theta)``; matches :func:`state_point_from_cache`."""

    def energy_fn(theta: Any, lam: Any, sp: StatePointData) -> jnp.ndarray:
        cache = sp.frames
        return lam * jnp.asarray(cache.E_ml_dimer) + mm_energies(
            theta, cache, mm_energy_with_lj, type_map
        )

    return energy_fn


def e_gas_matching_dhvap(cache: FrameCache, target: StatePointTarget) -> float:
    """``<E_gas>`` (kcal/mol) that reproduces the target (or nominal) dHvap at theta_0."""
    n = cache.frames.n_molecules
    e_liq = float(np.mean(cache.u_ref)) / n
    dh = target.dhvap_kJ_mol or target.nominal_dhvap_kJ_mol or 30.0
    return e_liq - KB_KCAL_MOL_K * float(cache.frames.temperature_K) + float(dh) / KCAL_TO_KJ


def toy_hybrid_handles(frames: FrameSet) -> HybridTermFns:
    """Geometry-dependent fake ML terms + a CHARMM-like LJ ``energy_with_lj``.

    Intermolecular pairs are first-atom of molecule i with first-atom of j, so
    the pair capacity is independent of the box. Energies are in the hybrid
    calculator's eV (terms) / kcal/mol (MM) convention.
    """
    n_mol = int(frames.n_molecules)
    apm = int(frames.atoms_per_molecule)
    n_atoms = int(frames.n_atoms)
    idx_np = np.array(
        [[i * apm, j * apm] for i in range(n_mol) for j in range(i + 1, n_mol)] or [[0, 0]],
        dtype=np.int32,
    )
    mask_np = np.ones(len(idx_np), dtype=np.float32)
    if n_mol < 2:
        mask_np = np.zeros(1, dtype=np.float32)
    rmins = np.where(np.arange(n_atoms) % 2 == 0, 1.9, 1.7).astype(np.float64)
    eps = np.where(np.arange(n_atoms) % 2 == 0, -0.07, -0.12).astype(np.float64)
    codes = (np.arange(n_atoms) % 2).astype(np.int32)

    def energy_with_lj(positions, pair_idx, pair_mask, cell, charges=None, lj_rmins=None, lj_epsilons=None):
        i, j = pair_idx[:, 0], pair_idx[:, 1]
        valid = (pair_mask > 0) & (i < j)
        d = positions[j] - positions[i]
        L = jnp.diag(cell)
        d = d - L * jnp.round(d / L)
        d = jnp.where(valid[:, None], d, 1e6)
        r = jnp.sqrt(jnp.sum(d * d, axis=-1))
        rm = lj_rmins[i] + lj_rmins[j]
        ep = jnp.sqrt(lj_epsilons[i] * lj_epsilons[j])
        s6 = (rm / r) ** 6
        return jnp.sum(jnp.where(valid, pair_mask * ep * (s6 * s6 - 2.0 * s6), 0.0))

    def pairs(_x, _box):
        return idx_np, mask_np

    def terms(x, box):
        pi, pm = pairs(x, box)
        e_mm = energy_with_lj(
            jnp.asarray(x),
            jnp.asarray(pi),
            jnp.asarray(pm),
            jnp.diag(jnp.asarray(box)),
            lj_rmins=jnp.asarray(rmins),
            lj_epsilons=jnp.asarray(eps),
        )
        return {
            "internal_E": -100.0 - 0.01 * float(np.sum(x[:, 0])),
            "ml_2b_E": -0.2 - 0.001 * float(box[0]),
            "mm_E": float(e_mm) / EV_TO_KCAL_MOL,
        }

    terms.model_tag = "toy"  # type: ignore[attr-defined]
    upd = SimpleNamespace(
        energy_with_lj=energy_with_lj,
        lj_rmins=rmins,
        lj_epsilons=eps,
        at_codes=codes,
        atc_names=("CG2O5", "OG2D3"),
    )
    return HybridTermFns(terms=terms, pairs=pairs, update_fn=upd, model_tag="toy")


def state_point_for_cache(
    cache: FrameCache,
    molecule: str,
    *,
    e_gas: float | None = None,
    ref: Mapping[str, Any] | None = None,
) -> StatePointData:
    """Pair a cache with the experimental target at the frames' thermostat T."""
    T = float(cache.frames.temperature_K)
    targets = build_state_point_targets(molecule, [T], ref=ref if ref is not None else load_reference())
    if not targets:
        raise ValueError(
            f"no experimental rho/dHvap target for {molecule} at {T} K "
            "(acetone density covers 183-507 K; DCM dHvap is tabulated near 298 K)"
        )
    gas = float(e_gas) if e_gas is not None else e_gas_matching_dhvap(cache, targets[0])
    return state_point_from_cache(cache, targets[0], gas)


def jsonable(tree: Any) -> Any:
    """JSON-friendly conversion (NaN -> None)."""
    if isinstance(tree, dict):
        return {str(k): jsonable(v) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        return [jsonable(v) for v in tree]
    if isinstance(tree, (np.ndarray, jnp.ndarray)):
        return jsonable(np.asarray(tree).tolist())
    if isinstance(tree, (np.floating, float)):
        v = float(tree)
        return None if not np.isfinite(v) else v
    if isinstance(tree, (np.integer, int)):
        return int(tree)
    if isinstance(tree, (np.bool_, bool)):
        return bool(tree)
    if tree is None:
        return None
    return tree


def theta_sidecar_payload(theta: Theta, type_names: Sequence[str], lam: Any) -> dict[str, Any]:
    """``hybrid_mm.json``-style LJ scales plus the PhysNet dimer scale."""
    names = [str(n) for n in type_names]
    sig = np.exp(np.asarray(theta["log_sig"], dtype=np.float64))
    eps = np.exp(np.asarray(theta["log_eps"], dtype=np.float64))
    out = mm_lj_scales_metadata(
        learn_mm_lj_scales=True, type_names=names, sigma_scale=sig, epsilon_scale=eps
    )
    out["ml_dimer_scale"] = float(np.asarray(lam))
    return out


@dataclass
class FitResult:
    theta: Theta
    lam: jnp.ndarray
    loss: float
    diagnostics: dict[str, Any]
    history: list[dict[str, Any]]
    check: list[dict[str, Any]]
    type_names: tuple[str, ...]
    n_frames: int

    def as_json(self) -> dict[str, Any]:
        th = project_theta(self.theta)
        return jsonable(
            {
                "type_names": list(self.type_names),
                "eps_scale": np.exp(np.asarray(th["log_eps"])),
                "rmin_scale": np.exp(np.asarray(th["log_sig"])),
                "lambda": self.lam,
                "loss": self.loss,
                "n_frames": self.n_frames,
                "diagnostics": self.diagnostics,
                "history": self.history,
                "check": self.check,
            }
        )


def run_check(
    sp: StatePointData,
    energy_fn: EnergyFn,
    type_map: LjTypeMap,
    *,
    lam0: float = 1.0,
) -> list[dict[str, Any]]:
    theta0 = init_theta(type_map)
    return check_reference_consistency(theta0, lam0, [sp], energy_fn, require_sampled_potential=False)


def run_fit(
    sp: StatePointData,
    energy_fn: EnergyFn,
    type_map: LjTypeMap,
    *,
    n_steps: int = 0,
    lr: float = 0.02,
    prior_weight: float = 0.0,
    fit_lambda: bool = True,
    ess_min: float = 0.1,
) -> FitResult:
    """Adam steps on ``liquid_loss``; ``n_steps=0`` evaluates at theta_0 only."""
    theta = init_theta(type_map)
    lam = jnp.asarray(1.0)

    def packed_loss(params: dict[str, jnp.ndarray]) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        th = project_theta({"log_eps": params["log_eps"], "log_sig": params["log_sig"]})
        lm = params["lam"] if fit_lambda else jnp.asarray(1.0)
        return liquid_loss(th, lm, [sp], energy_fn, prior_weight=prior_weight)

    params = {"log_eps": theta["log_eps"], "log_sig": theta["log_sig"], "lam": lam}
    history: list[dict[str, Any]] = []
    opt = optax.adam(lr)
    opt_state = opt.init(params)
    val = diag = None
    for step in range(max(0, int(n_steps))):
        (val, diag), grads = jax.value_and_grad(packed_loss, has_aux=True)(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        params = {
            **project_theta({"log_eps": params["log_eps"], "log_sig": params["log_sig"]}),
            "lam": params["lam"] if fit_lambda else jnp.asarray(1.0),
        }
        ess = float(np.asarray(diag["ess_fraction"]).reshape(-1)[0])
        history.append({"step": step, "loss": float(val), "ess_fraction": ess})
        if ess < ess_min:
            history[-1]["ess_warn"] = True

    if val is None:
        val, diag = packed_loss(params)
    check = run_check(sp, energy_fn, type_map, lam0=1.0)
    theta_out = project_theta({"log_eps": params["log_eps"], "log_sig": params["log_sig"]})
    return FitResult(
        theta=theta_out,
        lam=params["lam"],
        loss=float(val),
        diagnostics=jsonable(diag),
        history=history,
        check=jsonable(check),
        type_names=type_map.type_names,
        n_frames=int(np.shape(sp.u_ref_kcal_mol)[0]),
    )


def decompose_frames(
    frames: FrameSet,
    handles: HybridTermFns,
    *,
    cache_path: str | Path | None = None,
    overwrite: bool = False,
    toy: bool = False,
) -> FrameCache:
    """Decompose hybrid terms. ``toy`` drops the sampler potential (not this Hamiltonian)."""
    if toy:
        frames = replace(frames, u_ref=np.full(frames.n_frames, np.nan))
    return decompose(
        frames,
        handles.terms,
        handles.pairs,
        handles.update_fn,
        cache_path=cache_path,
        overwrite=overwrite,
        check="raise",
        model_tag=handles.model_tag,
    )


def fit_from_cache(
    cache: FrameCache,
    molecule: str,
    *,
    e_gas: float | None = None,
    n_steps: int = 0,
    lr: float = 0.02,
    prior_weight: float = 0.0,
    fit_lambda: bool = True,
    mm_energy_with_lj: Callable | None = None,
    ref: Mapping[str, Any] | None = None,
) -> FitResult:
    type_map = cache.type_map()
    energy = mm_energy_with_lj
    if energy is None:
        raise ValueError("mm_energy_with_lj is required to re-evaluate E_mm(theta)")
    sp = state_point_for_cache(cache, molecule, e_gas=e_gas, ref=ref)
    energy_fn = energy_fn_from_cache(energy, type_map)
    return run_fit(
        sp,
        energy_fn,
        type_map,
        n_steps=n_steps,
        lr=lr,
        prior_weight=prior_weight,
        fit_lambda=fit_lambda,
    )


def dry_run(
    frames: FrameSet,
    molecule: str,
    *,
    e_gas: float | None = None,
    n_steps: int = 2,
    lr: float = 0.02,
    cache_path: str | Path | None = None,
    overwrite: bool = False,
    ref: Mapping[str, Any] | None = None,
) -> FitResult:
    """Load-free (already-loaded frames) toy decompose + check + a few Adam steps."""
    handles = toy_hybrid_handles(frames)
    cache = decompose_frames(
        frames, handles, cache_path=cache_path, overwrite=overwrite, toy=True
    )
    return fit_from_cache(
        cache,
        molecule,
        e_gas=e_gas,
        n_steps=n_steps,
        lr=lr,
        mm_energy_with_lj=handles.update_fn.energy_with_lj,
        ref=ref,
    )
