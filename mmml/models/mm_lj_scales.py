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

from pathlib import Path
from typing import Any, Mapping

import numpy as np

__all__ = [
    "MM_LJ_EPSILON_SCALE_KEY",
    "MM_LJ_SIGMA_SCALE_KEY",
    "apply_mm_lj_scales",
    "attach_mm_lj_scales",
    "cgenff_type_names_from_prm",
    "load_mm_lj_scales_sidecar",
    "mm_lj_scales_metadata",
    "resolve_md_lj_scales",
    "scales_to_atc",
    "split_mm_lj_scale_params",
    "write_mm_lj_scales_into_hybrid_mm_json",
]

MM_LJ_SIGMA_SCALE_KEY = "mm_lj_sigma_scale"
MM_LJ_EPSILON_SCALE_KEY = "mm_lj_epsilon_scale"


def cgenff_type_names_from_prm(prm_path: str | Path | None = None) -> list[str]:
    """Type names in the same order as ``cgenff_master_sigmas`` / epsilons."""
    from mmml.data.cgenff_dataset import DEF_PRM_PATH, load_cgenff_nonbonded_table

    path = Path(prm_path) if prm_path is not None else DEF_PRM_PATH
    type_map, _, _ = load_cgenff_nonbonded_table(path)
    names = [""] * len(type_map)
    for name, idx in type_map.items():
        names[int(idx)] = str(name)
    if any(not n for n in names):
        raise RuntimeError(f"incomplete CGenFF type map from {path}")
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


def apply_mm_lj_scales(
    master_sigmas,
    master_epsilons,
    sigma_scale=None,
    epsilon_scale=None,
    *,
    include_lj: bool = True,
):
    """Return effective ``(sigmas, epsilons)`` after optional per-type scales."""
    import jax.numpy as jnp

    sig = jnp.asarray(master_sigmas)
    eps = jnp.asarray(master_epsilons)
    if not include_lj:
        eps = jnp.zeros_like(eps)
    if sigma_scale is not None:
        sig = sig * jnp.asarray(sigma_scale).reshape(-1)
    if epsilon_scale is not None and include_lj:
        eps = eps * jnp.asarray(epsilon_scale).reshape(-1)
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
) -> dict[str, Any]:
    """Serializable LJ-scale block for ``hybrid_mm.json``."""
    out: dict[str, Any] = {"learn_mm_lj_scales": bool(learn_mm_lj_scales)}
    if not learn_mm_lj_scales:
        return out
    if type_names is not None:
        out["cgenff_type_names"] = [str(n) for n in type_names]
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
    return {
        "cgenff_type_names": [str(n) for n in names],
        "mm_lj_sigma_scale": np.asarray(sig, dtype=np.float64),
        "mm_lj_epsilon_scale": np.asarray(eps, dtype=np.float64),
    }


def resolve_md_lj_scales(
    *,
    scales_file: str | Path | None = None,
    checkpoint: str | Path | None = None,
    atc_names: list[str] | tuple[str, ...] | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Resolve ``(ep_scale, sig_scale)`` for CHARMM ATC from a hybrid_mm sidecar.

    Search order for the JSON:

    1. Explicit ``scales_file``
    2. ``<checkpoint>/hybrid_mm.json`` when ``checkpoint`` is a directory
    3. ``<checkpoint.parent>/hybrid_mm.json`` when ``checkpoint`` is a file

    Returns ``(None, None)`` when no learnable scales are present.
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
        )
    )
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
