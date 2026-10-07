"""Central ML/JAX compute dtype for hybrid MMML calculator paths.

JAX/NumPy are imported lazily so CLI ``--help`` paths that only need
:func:`add_ml_compute_dtype_args` do not pay for JAX startup.
"""

from __future__ import annotations

import argparse
import os
import warnings
from typing import Any, Optional

_ALIAS_KEYS = {
    "float32": "float32",
    "f32": "float32",
    "32": "float32",
    "float64": "float64",
    "f64": "float64",
    "64": "float64",
}

_aliases_cache: dict[str, Any] | None = None


def _jax_stack():
    import jax
    import jax.numpy as jnp
    import numpy as np

    return jax, jnp, np


def _aliases() -> dict[str, Any]:
    global _aliases_cache
    if _aliases_cache is None:
        _, jnp, _ = _jax_stack()
        _aliases_cache = {
            "float32": jnp.float32,
            "f32": jnp.float32,
            "32": jnp.float32,
            "float64": jnp.float64,
            "f64": jnp.float64,
            "64": jnp.float64,
        }
    return _aliases_cache


def _parse_dtype_name(raw: Optional[str]):
    if raw is None:
        return None
    key = str(raw).strip().lower()
    if not key:
        return None
    if key not in _ALIAS_KEYS:
        raise ValueError(
            f"Unsupported ML compute dtype {raw!r}; use float32 or float64 "
            "(or MMML_ML_DTYPE / --ml-compute-dtype)."
        )
    return _aliases()[key]


def resolve_ml_compute_dtype(explicit: Optional[str] = None):
    """Return jnp.float32 or jnp.float64 for ML/MM JAX interior evaluation.

    Precedence: explicit argument > ``MMML_ML_DTYPE`` env > ``JAX_ENABLE_X64=1`` > float32.

    float64 requires ``JAX_ENABLE_X64=1`` before Python imports JAX (e.g. in the shell
    or mpirun wrapper). Otherwise a warning is emitted and float32 is used.
    """
    jax, jnp, _ = _jax_stack()
    requested = _parse_dtype_name(explicit)
    if requested is None:
        requested = _parse_dtype_name(os.environ.get("MMML_ML_DTYPE"))
    if requested is None:
        x64_env = os.environ.get("JAX_ENABLE_X64", "").strip().lower()
        if x64_env in ("1", "true", "yes", "on"):
            requested = jnp.float64
    if requested is None:
        requested = jnp.float32

    if requested == jnp.float64 and not bool(jax.config.read("jax_enable_x64")):
        warnings.warn(
            "ML compute dtype float64 was requested (MMML_ML_DTYPE or JAX_ENABLE_X64) "
            "but jax_enable_x64 is False. Set JAX_ENABLE_X64=1 before launching Python. "
            "Using float32 for ML evaluation.",
            stacklevel=2,
        )
        return jnp.float32
    return requested


def ml_numpy_dtype(jnp_dtype) -> Any:
    _, jnp, np = _jax_stack()
    return np.dtype("float64" if jnp_dtype == jnp.float64 else "float32")


def as_ml_array(x: Any, *, dtype=None):
    _, jnp, _ = _jax_stack()
    return jnp.asarray(x, dtype=dtype or resolve_ml_compute_dtype())


def ml_scalar(value: Any, *, dtype=None):
    _, jnp, _ = _jax_stack()
    return jnp.array(value, dtype=dtype or resolve_ml_compute_dtype())


def ml_zeros(shape: Any, *, dtype=None):
    _, jnp, _ = _jax_stack()
    return jnp.zeros(shape, dtype=dtype or resolve_ml_compute_dtype())


def cast_pytree_to_ml_dtype(tree: Any, *, dtype=None) -> Any:
    """Cast floating leaves in a checkpoint/params pytree to the ML compute dtype."""
    jax, jnp, np = _jax_stack()
    target = dtype or resolve_ml_compute_dtype()

    def _cast(x: Any) -> Any:
        if isinstance(x, (np.ndarray, jnp.ndarray)):
            arr = np.asarray(x)
            if np.issubdtype(arr.dtype, np.floating):
                return jnp.asarray(arr, dtype=target)
        return x

    return jax.tree_util.tree_map(_cast, tree)


def json_tree_to_jax_params(obj: Any, *, dtype=None) -> Any:
    """Convert JSON checkpoint lists to JAX arrays at ``dtype`` for floats."""
    _, jnp, np = _jax_stack()
    target = dtype or resolve_ml_compute_dtype()
    np_target = ml_numpy_dtype(target)

    if isinstance(obj, dict):
        return {k: json_tree_to_jax_params(v, dtype=target) for k, v in obj.items()}
    if isinstance(obj, list):
        if len(obj) > 0 and isinstance(obj[0], (list, int, float)):
            arr = np.array(obj)
            if np.issubdtype(arr.dtype, np.floating):
                arr = arr.astype(np_target, copy=False)
            return jnp.asarray(arr, dtype=target)
        return [json_tree_to_jax_params(item, dtype=target) for item in obj]
    return obj


def add_ml_compute_dtype_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ml-compute-dtype",
        choices=("float32", "float64"),
        default=None,
        help=(
            "JAX dtype for ML/MM hybrid interior (default: float32, or MMML_ML_DTYPE / "
            "JAX_ENABLE_X64=1 → float64). CHARMM I/O stays float64."
        ),
    )
