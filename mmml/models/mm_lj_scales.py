"""Per-type CGenFF LJ σ/ε scales for hybrid MM training and MD.

Learnable multiplicative scales on the fixed master tables::

    σ_eff[t] = master_sigmas[t] * sigma_scale[t]
    ε_eff[t] = master_epsilons[t] * epsilon_scale[t]

Training stores the arrays as top-level leaves on the Optax params pytree
(alongside Flax ``params``).  MD remaps by atom-type name onto CHARMM ATC
order for :func:`mm_energy_forces.build_mm_energy_forces_fn` ``ep_scale`` /
``sig_scale``.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Mapping

import numpy as np

__all__ = [
    "MM_LJ_EPSILON_SCALE_BOUNDS",
    "MM_LJ_EPSILON_SCALE_KEY",
    "MM_LJ_SIGMA_SCALE_BOUNDS",
    "MM_LJ_SIGMA_SCALE_KEY",
    "apply_mm_lj_scales",
    "attach_mm_lj_scales",
    "cgenff_type_names_from_prm",
    "clip_mm_lj_scale_params",
    "find_learnable_lj_scales_sidecar",
    "lj_scales_sidecar_candidates",
    "load_mm_lj_scales_sidecar",
    "mm_lj_scales_metadata",
    "out_of_bounds_mm_lj_scales",
    "resolve_md_lj_scales",
    "scales_to_atc",
    "split_mm_lj_scale_params",
    "write_mm_lj_scales_into_hybrid_mm_json",
]

MM_LJ_SIGMA_SCALE_KEY = "mm_lj_sigma_scale"
MM_LJ_EPSILON_SCALE_KEY = "mm_lj_epsilon_scale"

# Bounds the scales are projected into after every optimizer step.
#
# These are not cosmetic. ε enters the CHARMM combining rule as
# ``sqrt(eps_i * eps_j)``, so the instant one type's scale crosses zero every
# pair mixing it with a positive type evaluates ``sqrt`` of a negative number
# and the whole loss goes NaN. σ is worse behaved still: it multiplies Rmin
# inside r^-12, so a scale that drifts up drags the repulsive wall into
# separations the data actually samples.
#
# Unconstrained, drift is the default outcome rather than an edge case. Adam's
# step size is bounded near the learning rate regardless of gradient magnitude,
# so a scale accumulates roughly ``lr`` of travel per step -- tens of units over
# a realistic run, against a distance of 1.0 from the init to the singularity.
#
# The widths reflect how well each parameter is determined: Rmin is pinned
# tightly by packing and the repulsive wall (a correction beyond a few percent
# is compensation for something else), while well depths are genuinely uncertain
# to a factor of a few.
MM_LJ_SIGMA_SCALE_BOUNDS = (0.95, 1.05)
MM_LJ_EPSILON_SCALE_BOUNDS = (0.25, 4.0)


def cgenff_type_names_from_prm(prm_path: str | Path | None = None) -> list[str]:
    """Type names in the same order as ``cgenff_master_sigmas`` / epsilons.

    Uses :func:`mmml.data.cgenff_dataset.load_reference` so the list includes
    additive stream types (e.g. ``toppar_water_ions.str``) that extend the
    master LJ tables beyond the bare CGenFF ``.prm``.
    """
    from mmml.data.cgenff_dataset import DEF_PRM_PATH, DEF_RTF_PATH, load_reference

    prm = str(prm_path) if prm_path is not None else str(DEF_PRM_PATH)
    ref = load_reference(prm, str(DEF_RTF_PATH))
    names = [""] * len(ref.nb_map)
    for name, idx in ref.nb_map.items():
        names[int(idx)] = str(name)
    if any(not n for n in names):
        raise RuntimeError(f"incomplete CGenFF type map from {prm}")
    return names


def split_mm_lj_scale_params(params: Any) -> tuple[Any, Any | None, Any | None]:
    """Pop LJ scale leaves; return ``(model_params, sigma_scale, epsilon_scale)``."""
    if not isinstance(params, dict):
        return params, None, None
    if (
        MM_LJ_SIGMA_SCALE_KEY not in params
        and MM_LJ_EPSILON_SCALE_KEY not in params
    ):
        return params, None, None
    model_params = {
        k: v
        for k, v in params.items()
        if k not in (MM_LJ_SIGMA_SCALE_KEY, MM_LJ_EPSILON_SCALE_KEY)
    }
    return (
        model_params,
        params.get(MM_LJ_SIGMA_SCALE_KEY),
        params.get(MM_LJ_EPSILON_SCALE_KEY),
    )


def attach_mm_lj_scales(
    params: Mapping[str, Any],
    n_types: int,
    *,
    sigma_scale: np.ndarray | None = None,
    epsilon_scale: np.ndarray | None = None,
    dtype=None,
) -> dict[str, Any]:
    """Return a copy of ``params`` with unit (or provided) per-type LJ scales."""
    import jax.numpy as jnp

    n = int(n_types)
    if n <= 0:
        raise ValueError(f"n_types must be positive, got {n_types!r}")
    dt = dtype if dtype is not None else jnp.float32
    if sigma_scale is None:
        sig = jnp.ones((n,), dtype=dt)
    else:
        sig = jnp.asarray(sigma_scale, dtype=dt).reshape(n)
    if epsilon_scale is None:
        eps = jnp.ones((n,), dtype=dt)
    else:
        eps = jnp.asarray(epsilon_scale, dtype=dt).reshape(n)
    out = dict(params)
    out[MM_LJ_SIGMA_SCALE_KEY] = sig
    out[MM_LJ_EPSILON_SCALE_KEY] = eps
    return out


# Last-resort sign invariant for callers that never went through training.
#
# The LJ combining rule is a *geometric* mean, ``eps_ij = sqrt(eps_i * eps_j)``
# (cgenff_mm.cgenff_mm_energy). Its sign only cancels while every per-type
# epsilon shares a sign. A scale that crosses zero flips one type's sign, so
# every mixed pair involving it evaluates ``sqrt(negative)``.
#
# During training this floor never binds: :func:`clip_mm_lj_scale_params`
# projects into the far tighter :data:`MM_LJ_SIGMA_SCALE_BOUNDS` /
# :data:`MM_LJ_EPSILON_SCALE_BOUNDS` after every step. It covers the paths that
# bypass the optimizer -- a hand-edited sidecar, or a direct call -- where
# :func:`load_mm_lj_scales_sidecar` warns and this keeps the arithmetic sane.
MM_LJ_MIN_SCALE = 1e-3


def clip_mm_lj_scale_params(
    params: Any,
    *,
    sigma_bounds: tuple[float, float] = MM_LJ_SIGMA_SCALE_BOUNDS,
    epsilon_bounds: tuple[float, float] = MM_LJ_EPSILON_SCALE_BOUNDS,
    trainable_mask=None,
) -> Any:
    """Project the LJ-scale leaves of ``params`` back into their bounds.

    Applied after every optimizer step (see
    :func:`mmml.models.physnetjax.physnetjax.training.trainstep.train_step`), so
    the scales cannot wander to the values that make ``E_MM`` diverge or NaN --
    see :data:`MM_LJ_SIGMA_SCALE_BOUNDS`.

    Traceable: takes and returns JAX arrays, and is a no-op on pytrees without
    the scale leaves, so it is safe to call unconditionally inside ``jit``.
    """
    if not isinstance(params, dict):
        return params
    if (
        MM_LJ_SIGMA_SCALE_KEY not in params
        and MM_LJ_EPSILON_SCALE_KEY not in params
    ):
        return params

    import jax.numpy as jnp

    out = dict(params)
    for key, (lo, hi) in (
        (MM_LJ_SIGMA_SCALE_KEY, sigma_bounds),
        (MM_LJ_EPSILON_SCALE_KEY, epsilon_bounds),
    ):
        if key in out and out[key] is not None:
            value = jnp.clip(out[key], float(lo), float(hi))
            if trainable_mask is not None:
                mask = jnp.asarray(trainable_mask, dtype=bool).reshape(value.shape)
                value = jnp.where(mask, value, jnp.ones_like(value))
            out[key] = value
    return out


def out_of_bounds_mm_lj_scales(
    type_names,
    sigma_scale,
    epsilon_scale,
    *,
    sigma_bounds: tuple[float, float] = MM_LJ_SIGMA_SCALE_BOUNDS,
    epsilon_bounds: tuple[float, float] = MM_LJ_EPSILON_SCALE_BOUNDS,
) -> list[str]:
    """Human-readable descriptions of scales outside the physical bounds.

    Empty for a sidecar written by a bounded training run. Non-empty means the
    file was hand-edited or produced before the bounds existed, in which case the
    LJ it deploys may not be the LJ that was fitted.
    """
    sig = np.asarray(sigma_scale, dtype=np.float64).reshape(-1)
    eps = np.asarray(epsilon_scale, dtype=np.float64).reshape(-1)
    names = [str(n) for n in type_names]
    problems: list[str] = []
    for label, values, (lo, hi) in (
        ("sigma", sig, sigma_bounds),
        ("epsilon", eps, epsilon_bounds),
    ):
        for i, value in enumerate(values):
            if not (lo <= float(value) <= hi):
                name = names[i] if i < len(names) else f"type_{i}"
                problems.append(
                    f"{label} scale {value:.4f} for {name} outside [{lo:g}, {hi:g}]"
                )
    return problems


def apply_mm_lj_scales(
    master_sigmas,
    master_epsilons,
    sigma_scale=None,
    epsilon_scale=None,
    *,
    include_lj: bool = True,
    min_scale: float = MM_LJ_MIN_SCALE,
):
    """Return effective ``(sigmas, epsilons)`` after optional per-type scales.

    Scales are floored at ``min_scale`` so they cannot change sign; see
    :data:`MM_LJ_MIN_SCALE`. Pass ``min_scale=0.0`` to disable (tests only).
    """
    import jax.numpy as jnp

    sig = jnp.asarray(master_sigmas)
    eps = jnp.asarray(master_epsilons)
    if not include_lj:
        eps = jnp.zeros_like(eps)
    if sigma_scale is not None:
        s = jnp.asarray(sigma_scale).reshape(-1)
        sig = sig * jnp.maximum(s, min_scale)
    if epsilon_scale is not None and include_lj:
        e = jnp.asarray(epsilon_scale).reshape(-1)
        eps = eps * jnp.maximum(e, min_scale)
    return sig, eps


def scales_to_atc(
    type_names: list[str] | tuple[str, ...],
    sigma_scale: np.ndarray | list[float],
    epsilon_scale: np.ndarray | list[float],
    atc_names: list[str] | tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Map master-table scales onto CHARMM ``param.get_atc()`` order.

    Returns ``(ep_scale, sig_scale)`` length ``len(atc_names)``, defaulting to
    1.0 for ATC entries missing from the training type list.
    """
    names = [str(n) for n in type_names]
    sig = np.asarray(sigma_scale, dtype=np.float64).reshape(-1)
    eps = np.asarray(epsilon_scale, dtype=np.float64).reshape(-1)
    if sig.shape[0] != len(names) or eps.shape[0] != len(names):
        raise ValueError(
            f"scale length mismatch: {len(names)} names vs "
            f"sigma={sig.shape[0]} epsilon={eps.shape[0]}"
        )
    by_name = {names[i]: (float(sig[i]), float(eps[i])) for i in range(len(names))}
    ep_out = np.ones(len(atc_names), dtype=np.float64)
    sig_out = np.ones(len(atc_names), dtype=np.float64)
    for i, atc in enumerate(atc_names):
        pair = by_name.get(str(atc))
        if pair is None:
            continue
        sig_out[i], ep_out[i] = pair
    return ep_out, sig_out


def mm_lj_scales_metadata(
    *,
    learn_mm_lj_scales: bool,
    type_names: list[str] | None = None,
    sigma_scale=None,
    epsilon_scale=None,
    sigma_bounds: tuple[float, float] = MM_LJ_SIGMA_SCALE_BOUNDS,
    epsilon_bounds: tuple[float, float] = MM_LJ_EPSILON_SCALE_BOUNDS,
    trainable_mask=None,
    type_frame_counts=None,
) -> dict[str, Any]:
    """Serializable LJ-scale block for ``hybrid_mm.json``."""
    out: dict[str, Any] = {"learn_mm_lj_scales": bool(learn_mm_lj_scales)}
    if not learn_mm_lj_scales:
        return out
    if type_names is not None:
        out["cgenff_type_names"] = [str(n) for n in type_names]
    out["mm_lj_sigma_scale_bounds"] = [float(x) for x in sigma_bounds]
    out["mm_lj_epsilon_scale_bounds"] = [float(x) for x in epsilon_bounds]
    if trainable_mask is not None:
        out["mm_lj_trainable_mask"] = [bool(x) for x in trainable_mask]
    if type_frame_counts is not None:
        out["mm_lj_type_frame_counts"] = [int(x) for x in type_frame_counts]
    if sigma_scale is not None:
        out["mm_lj_sigma_scale"] = [float(x) for x in np.asarray(sigma_scale).reshape(-1)]
    if epsilon_scale is not None:
        out["mm_lj_epsilon_scale"] = [
            float(x) for x in np.asarray(epsilon_scale).reshape(-1)
        ]
    return out


def load_mm_lj_scales_sidecar(path: str | Path) -> dict[str, Any] | None:
    """Load LJ scales from ``hybrid_mm.json`` (or a dedicated scales JSON).

    Returns ``None`` when the file is missing the learnable-scale payload.
    """
    import json

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"LJ scale sidecar not found: {p}")
    with p.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {p}")
    if not data.get("learn_mm_lj_scales", False):
        return None
    names = data.get("cgenff_type_names")
    sig = data.get("mm_lj_sigma_scale")
    eps = data.get("mm_lj_epsilon_scale")
    if names is None or sig is None or eps is None:
        raise ValueError(
            f"{p} has learn_mm_lj_scales=true but is missing "
            "cgenff_type_names / mm_lj_sigma_scale / mm_lj_epsilon_scale"
        )
    payload = {
        "cgenff_type_names": [str(n) for n in names],
        "mm_lj_sigma_scale": np.asarray(sig, dtype=np.float64),
        "mm_lj_epsilon_scale": np.asarray(eps, dtype=np.float64),
    }
    sigma_bounds = tuple(data.get("mm_lj_sigma_scale_bounds", MM_LJ_SIGMA_SCALE_BOUNDS))
    epsilon_bounds = tuple(data.get("mm_lj_epsilon_scale_bounds", MM_LJ_EPSILON_SCALE_BOUNDS))
    problems = out_of_bounds_mm_lj_scales(
        payload["cgenff_type_names"],
        payload["mm_lj_sigma_scale"],
        payload["mm_lj_epsilon_scale"],
        sigma_bounds=sigma_bounds,
        epsilon_bounds=epsilon_bounds,
    )
    if problems:
        # Warn rather than raise: sidecars predating the bounds still load, and
        # refusing them would strand existing runs.
        warnings.warn(
            f"{p} carries LJ scales outside the trainable bounds "
            f"({len(problems)} entries, e.g. {problems[0]})",
            RuntimeWarning,
            stacklevel=2,
        )
    return payload


def lj_scales_sidecar_candidates(
    *,
    scales_file: str | Path | None = None,
    checkpoint: str | Path | None = None,
) -> list[Path]:
    """Sidecar JSON paths to try, in priority order.

    1. Explicit ``scales_file``
    2. ``<checkpoint>/hybrid_mm.json`` when ``checkpoint`` is a directory
    3. ``<checkpoint.parent>/hybrid_mm.json`` when ``checkpoint`` is a file
    """
    candidates: list[Path] = []
    if scales_file is not None:
        candidates.append(Path(scales_file).expanduser())
    if checkpoint is not None:
        ckpt = Path(checkpoint).expanduser()
        if ckpt.is_dir():
            candidates.append(ckpt / "hybrid_mm.json")
        else:
            candidates.append(ckpt.parent / "hybrid_mm.json")
            # Orbax epoch dirs live under the run root that owns hybrid_mm.json.
            if ckpt.parent.parent != ckpt.parent:
                candidates.append(ckpt.parent.parent / "hybrid_mm.json")
    return candidates


def find_learnable_lj_scales_sidecar(
    *,
    scales_file: str | Path | None = None,
    checkpoint: str | Path | None = None,
) -> Path | None:
    """Path of the first sidecar carrying learnable scales, else ``None``.

    Unlike :func:`resolve_md_lj_scales` this never needs CHARMM ATC names, so
    callers can ask "are there trained LJ scales here?" before deciding whether
    they are applicable — used to reject a request that would be silently
    dropped (JAX ``doMM`` off).
    """
    for path in lj_scales_sidecar_candidates(
        scales_file=scales_file, checkpoint=checkpoint
    ):
        if not path.is_file():
            continue
        try:
            if load_mm_lj_scales_sidecar(path) is not None:
                return path
        except Exception:  # pragma: no cover - malformed sidecar
            continue
    return None


def resolve_md_lj_scales(
    *,
    scales_file: str | Path | None = None,
    checkpoint: str | Path | None = None,
    atc_names: list[str] | tuple[str, ...] | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Resolve ``(ep_scale, sig_scale)`` for CHARMM ATC from a hybrid_mm sidecar.

    Search order is :func:`lj_scales_sidecar_candidates`.

    Returns ``(None, None)`` when no learnable scales are present.
    """
    candidates = lj_scales_sidecar_candidates(
        scales_file=scales_file, checkpoint=checkpoint
    )

    payload = None
    last_err: Exception | None = None
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = load_mm_lj_scales_sidecar(path)
        except Exception as exc:  # pragma: no cover - defensive
            last_err = exc
            continue
        if payload is not None:
            break
    if payload is None:
        if last_err is not None and scales_file is not None:
            raise last_err
        return None, None

    names = list(atc_names) if atc_names is not None else None
    if names is None:
        try:
            import pycharmm.param as param

            names = [str(x) for x in param.get_atc()]
        except Exception as exc:
            raise RuntimeError(
                "MM LJ scales require CHARMM ATC names (param.get_atc); "
                "load CGenFF toppar before resolving scales"
            ) from exc

    ep_scale, sig_scale = scales_to_atc(
        payload["cgenff_type_names"],
        payload["mm_lj_sigma_scale"],
        payload["mm_lj_epsilon_scale"],
        names,
    )
    return ep_scale, sig_scale


def write_mm_lj_scales_into_hybrid_mm_json(
    path: str | Path,
    *,
    type_names: list[str],
    sigma_scale,
    epsilon_scale,
    sigma_bounds: tuple[float, float] = MM_LJ_SIGMA_SCALE_BOUNDS,
    epsilon_bounds: tuple[float, float] = MM_LJ_EPSILON_SCALE_BOUNDS,
    trainable_mask=None,
    type_frame_counts=None,
) -> None:
    """Merge final scale vectors into an existing ``hybrid_mm.json``."""
    import json

    p = Path(path)
    data: dict[str, Any] = {}
    if p.is_file():
        with p.open(encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict):
            data = loaded
    data.update(
        mm_lj_scales_metadata(
            learn_mm_lj_scales=True,
            type_names=type_names,
            sigma_scale=sigma_scale,
            epsilon_scale=epsilon_scale,
            sigma_bounds=sigma_bounds,
            epsilon_bounds=epsilon_bounds,
            trainable_mask=trainable_mask,
            type_frame_counts=type_frame_counts,
        )
    )
    problems = out_of_bounds_mm_lj_scales(
        type_names, sigma_scale, epsilon_scale,
        sigma_bounds=sigma_bounds, epsilon_bounds=epsilon_bounds,
    )
    if problems:
        # Training projects the scales every step, so this should be unreachable
        # from a normal run. Reaching it means the sidecar is not deployable and
        # the run needs to be understood before anyone uses it.
        warnings.warn(
            f"writing {len(problems)} LJ scale(s) outside the trainable bounds to "
            f"{p} (e.g. {problems[0]}); this run's scales are not deployable",
            RuntimeWarning,
            stacklevel=2,
        )
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
