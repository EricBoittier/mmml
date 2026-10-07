"""Verlet-skin reuse for host-built neighbor lists.

``mm_energy_forces.update_mm_pairs`` already skips MM pair rebuilds while every
atom has moved less than ``skin / 2``, and does that check on device so only a
scalar crosses the PCIe bus. The ``mmml.md`` driver's ``neighbor_fn``
(:mod:`mmml.md.neighbors`) had no such reuse: every block boundary paid a full
host rebuild plus a full position download.

:func:`with_verlet_skin` wraps any ``neighbor_fn(pos, box)`` with that policy.
The wrapped list must be built at ``cutoff + skin`` for reuse to be sound — the
callers in :mod:`mmml.md.neighbors` do this. Pairs between ``cutoff`` and
``cutoff + skin`` are harmless for ``mm_nonbonded``, which zeroes every pair
beyond ``ctofnb`` explicitly.

The wrapper sets ``device_native = True`` so
:class:`~mmml.md.drivers.JaxmdDriver` hands it device arrays directly; the
download then happens only on an actual rebuild.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np

from mmml.md.nl_cadence import verlet_reuse_displacement_limit_A

__all__ = ["with_verlet_skin", "NeighborCacheStats"]


class NeighborCacheStats:
    """Rebuild/reuse counters for a wrapped ``neighbor_fn`` (diagnostics only)."""

    __slots__ = ("calls", "rebuilds", "reused", "host_syncs", "device_skin_checks")

    def __init__(self) -> None:
        self.calls = 0
        self.rebuilds = 0
        self.reused = 0
        self.host_syncs = 0
        self.device_skin_checks = 0

    def as_dict(self) -> dict[str, int]:
        return {slot: getattr(self, slot) for slot in self.__slots__}

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"NeighborCacheStats({self.as_dict()})"


def _is_device_array(x: Any) -> bool:
    return x is not None and hasattr(x, "__dlpack_device__")


def _max_displacement(pos, last_pos) -> float:
    """Max per-atom displacement; computed on device when both are device arrays."""
    if _is_device_array(pos) and _is_device_array(last_pos):
        import jax
        import jax.numpy as jnp

        return float(
            jax.device_get(jnp.max(jnp.linalg.norm(jnp.asarray(pos) - last_pos, axis=1)))
        )
    a = np.asarray(pos, dtype=np.float64)
    b = np.asarray(last_pos, dtype=np.float64)
    return float(np.max(np.linalg.norm(a - b, axis=1)))


def _box_changed(box, last_box, tol: float = 1e-8) -> bool:
    if box is None and last_box is None:
        return False
    if box is None or last_box is None:
        return True
    return bool(
        np.max(np.abs(np.asarray(box, dtype=np.float64) - np.asarray(last_box, dtype=np.float64)))
        > tol
    )


def with_verlet_skin(
    neighbor_fn: Callable[[Any, Any], Mapping[str, Any]],
    *,
    skin_A: float,
    to_device: bool = True,
) -> Callable[[Any, Any], Mapping[str, Any]]:
    """Wrap ``neighbor_fn`` so it rebuilds only when the Verlet skin is spent.

    ``neighbor_fn`` must build its list at ``cutoff + skin_A``. It is always
    called with **host** arrays; the wrapper downloads positions only when a
    rebuild is actually needed, so a reused block costs one scalar sync.

    A non-positive ``skin_A`` disables caching and returns ``neighbor_fn``
    unchanged, which keeps the "rebuild every block" behavior available.
    """
    skin = float(skin_A)
    if skin <= 0.0:
        return neighbor_fn

    limit = verlet_reuse_displacement_limit_A(skin)
    stats = NeighborCacheStats()
    cache: dict[str, Any] = {"pos": None, "box": None, "result": None}

    def cached_neighbor_fn(pos, box):
        stats.calls += 1
        have_cache = cache["result"] is not None

        if have_cache and not _box_changed(box, cache["box"]):
            stats.device_skin_checks += 1
            if _max_displacement(pos, cache["pos"]) <= limit:
                stats.reused += 1
                return cache["result"]

        # Rebuild: this is the only path that needs positions on the host.
        if _is_device_array(pos):
            import jax

            stats.host_syncs += 1
            pos_np = np.asarray(jax.device_get(pos), dtype=np.float64)
        else:
            pos_np = np.asarray(pos, dtype=np.float64)
        box_np = None if box is None else np.asarray(box, dtype=np.float64)

        result = neighbor_fn(pos_np, box_np)
        if not isinstance(result, Mapping):
            raise TypeError("neighbor_fn must return a mapping of energy keyword arrays")

        if to_device:
            import jax.numpy as jnp

            result = {key: jnp.asarray(value) for key, value in dict(result).items()}
        else:
            result = dict(result)

        # Keep the reference frame in whatever space the caller works in, so the
        # next skin check stays on device for device callers.
        cache["pos"] = pos if _is_device_array(pos) else pos_np.copy()
        cache["box"] = None if box_np is None else box_np.copy()
        cache["result"] = result
        stats.rebuilds += 1
        return result

    # Tells JaxmdDriver.refresh_real to skip its own device_get and hand us the
    # device array, so the skin check stays on device.
    cached_neighbor_fn.device_native = True  # type: ignore[attr-defined]
    cached_neighbor_fn.stats = stats  # type: ignore[attr-defined]
    cached_neighbor_fn.skin_A = skin  # type: ignore[attr-defined]
    return cached_neighbor_fn
