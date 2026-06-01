"""JAX device selection for MLpot with OpenMPI-linked DOMDEC CHARMM."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator


def _truthy(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in ("1", "yes", "true")


def mlpot_jax_device_name() -> str:
    """``cpu`` or ``gpu`` for MLpot energy/force evaluation."""
    mode = (os.environ.get("MMML_MLPOT_DEVICE") or "auto").strip().lower()
    if mode in ("cpu", "gpu"):
        return mode
    if mode == "auto":
        try:
            from mmml.interfaces.pycharmmInterface.charmm_mpi import charmm_lib_links_mpi

            if charmm_lib_links_mpi():
                return "cpu"
        except Exception:
            pass
    return "gpu"


def apply_mlpot_jax_platform_env(*, quiet: bool = False) -> str:
    """Set ``JAX_PLATFORMS`` before the first ``import jax`` when MLpot must avoid CUDA+MPI."""
    device = mlpot_jax_device_name()
    if device == "cpu":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        if not quiet and not _truthy("MMML_QUIET"):
            print(
                "mmml: OpenMPI-linked CHARMM — MLpot JAX runs on CPU "
                "(CUDA after MPI breaks SD barriers). "
                "Set MMML_MLPOT_DEVICE=gpu to override (experimental).",
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
