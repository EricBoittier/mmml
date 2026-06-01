"""JAX device selection for MLpot with OpenMPI-linked DOMDEC CHARMM."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator


def _truthy(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in ("1", "yes", "true")


def mlpot_jax_device_name() -> str:
    """``cpu`` or ``gpu`` for MLpot energy/force evaluation."""
    mode = (os.environ.get("MMML_MLPOT_DEVICE") or "gpu").strip().lower()
    if mode in ("cpu", "gpu"):
        return mode
    if mode == "auto":
        return "gpu"
    return "gpu"


def apply_mlpot_jax_platform_env(*, quiet: bool = False) -> str:
    """Set ``JAX_PLATFORMS`` before the first ``import jax`` for MLpot."""
    device = mlpot_jax_device_name()
    os.environ.setdefault("JAX_PLATFORMS", device)
    if not quiet and not _truthy("MMML_QUIET") and device == "cpu":
        print(
            "mmml: MLpot JAX runs on CPU (MMML_MLPOT_DEVICE=cpu). "
            "Unset or set MMML_MLPOT_DEVICE=gpu for GPU.",
            flush=True,
        )
    return device


def jax_warmup_device_name() -> str:
    """Warmup backend; follows :func:`mlpot_jax_device_name` unless overridden."""
    mode = (os.environ.get("MMML_JAX_WARMUP_DEVICE") or "auto").strip().lower()
    if mode in ("cpu", "gpu"):
        return mode
    if mode == "auto":
        return mlpot_jax_device_name()
    return "gpu"


@contextmanager
def mlpot_jax_device_context() -> Iterator[Any]:
    """Run MLpot JAX work on the selected device."""
    import jax

    name = mlpot_jax_device_name()
    devices = jax.devices(name)
    if not devices:
        devices = jax.devices("cpu")
    with jax.default_device(devices[0]):
        yield devices[0]
