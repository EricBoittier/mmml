"""Integrator restart payload for the shared jax-md driver.

A CHARMM DCD holds coordinates. Continuing a trajectory also needs the
particle momenta, the Nose–Hoover chain (thermostat and, for NPT, barostat),
and the Langevin random-number state. Those live in this payload, written
beside the coordinates into ``trajectory.npz`` and the campaign handoff.

Positions and the current cell are *not* taken from the payload. The geometry
handoff owns them, and the driver recomputes forces from those coordinates.
NPT ``box_position`` stays at the value ``init_fn`` derives from the handed
cell; the piston momentum is restored so the volume keeps its rate of change.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

__all__ = [
    "ang_ps_from_momenta",
    "momenta_from_ang_ps",
    "snapshot_integrator",
    "apply_integrator_restart",
    "flatten_restart",
    "unflatten_restart",
    "write_position_dcd",
]

_ARRAY_KEYS = (
    "momentum",
    "mass",
    "rng",
    "box_momentum",
    "velocities_ang_ps",
)
_CHAIN_GROUPS = ("chain", "thermostat", "barostat")
_CHAIN_FIELDS = (
    "position",
    "momentum",
    "mass",
    "tau",
    "kinetic_energy",
    "degrees_of_freedom",
)


def _metal_velocity_scale() -> float:
    """Å/ps per jax-md metal velocity unit (``1000 * ase.units.fs``)."""
    from ase import units

    return 1000.0 * float(units.fs)


def ang_ps_from_momenta(momentum, mass) -> np.ndarray:
    """Metal momenta ``(N, 3)`` and masses ``(N,)`` → velocities in Å/ps."""
    p = np.asarray(momentum, dtype=np.float64)
    m = np.asarray(mass, dtype=np.float64).reshape(-1)
    if p.ndim != 2 or p.shape[1] != 3 or p.shape[0] != m.shape[0]:
        raise ValueError(
            f"momentum {p.shape} and mass {m.shape} must describe the same N atoms"
        )
    if not np.all(m > 0):
        raise ValueError("masses must be strictly positive")
    return (p / m[:, None]) * _metal_velocity_scale()


def momenta_from_ang_ps(velocities_ang_ps, mass) -> np.ndarray:
    """Å/ps velocities and masses → jax-md metal momenta."""
    v = np.asarray(velocities_ang_ps, dtype=np.float64).reshape(-1, 3)
    m = np.asarray(mass, dtype=np.float64).reshape(-1)
    if v.shape[0] != m.shape[0]:
        raise ValueError(
            f"velocities {v.shape} and mass {m.shape} must describe the same N atoms"
        )
    if not np.all(m > 0):
        raise ValueError("masses must be strictly positive")
    return (v / _metal_velocity_scale()) * m[:, None]


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    try:
        import jax

        value = jax.device_get(value)
    except Exception:
        pass
    return np.asarray(value)


def _snapshot_chain(chain: Any) -> dict[str, Any]:
    block: dict[str, Any] = {}
    for key in _CHAIN_FIELDS:
        if not hasattr(chain, key):
            continue
        value = getattr(chain, key)
        if key == "degrees_of_freedom":
            block[key] = int(value)
        else:
            block[key] = _to_numpy(value)
    return block


def snapshot_integrator(state: Any) -> dict[str, Any]:
    """Copy the restart-relevant leaves of a jax-md integrator state."""
    if hasattr(state, "thermostat") and hasattr(state, "barostat"):
        kind = "npt_nose_hoover"
    elif hasattr(state, "rng"):
        kind = "nvt_langevin"
    elif hasattr(state, "chain"):
        kind = "nvt_nose_hoover"
    elif getattr(state, "momentum", None) is not None:
        kind = "nve"
    else:
        kind = "min"

    snap: dict[str, Any] = {"kind": kind}
    momentum = getattr(state, "momentum", None)
    if momentum is not None:
        snap["momentum"] = _to_numpy(momentum)
    mass = getattr(state, "mass", None)
    if mass is not None:
        snap["mass"] = _to_numpy(mass).reshape(-1)
    rng = getattr(state, "rng", None)
    if rng is not None:
        snap["rng"] = _to_numpy(rng)
    box_momentum = getattr(state, "box_momentum", None)
    if box_momentum is not None:
        snap["box_momentum"] = _to_numpy(box_momentum)
    for group in _CHAIN_GROUPS:
        chain = getattr(state, group, None)
        if chain is not None and hasattr(chain, "position"):
            snap[group] = _snapshot_chain(chain)
    return snap


def _match_array(existing: Any, value: Any) -> Any:
    """Cast ``value`` onto the dtype and shape of ``existing``."""
    arr = np.asarray(value)
    shape = getattr(existing, "shape", None)
    if shape is not None and arr.shape != tuple(shape):
        arr = arr.reshape(shape)
    try:
        import jax.numpy as jnp

        if hasattr(existing, "dtype"):
            return jnp.asarray(arr, dtype=existing.dtype)
    except ImportError:
        pass
    dtype = getattr(existing, "dtype", None)
    if dtype is None:
        return arr
    return np.asarray(arr, dtype=dtype)


def _restore_chain(chain: Any, block: Mapping[str, Any]) -> Any:
    updates: dict[str, Any] = {}
    for key, value in block.items():
        if key == "degrees_of_freedom":
            updates[key] = int(value)
            continue
        if not hasattr(chain, key):
            continue
        updates[key] = _match_array(getattr(chain, key), value)
    if hasattr(chain, "set"):
        return chain.set(**updates)
    from dataclasses import replace

    return replace(chain, **updates)


def apply_integrator_restart(state: Any, snapshot: Mapping[str, Any]) -> Any:
    """Overwrite momenta, chains, piston momentum, and RNG on ``state``.

    Coordinates, forces, and the NPT box position are left as ``init_fn``
    built them from the handed geometry.
    """
    updates: dict[str, Any] = {}
    for key in _ARRAY_KEYS:
        if key not in snapshot or snapshot[key] is None or not hasattr(state, key):
            continue
        if key == "velocities_ang_ps":
            continue
        current = getattr(state, key)
        if current is None:
            continue
        updates[key] = _match_array(current, snapshot[key])
    if "momentum" not in updates and snapshot.get("velocities_ang_ps") is not None:
        mass = snapshot.get("mass", getattr(state, "mass", None))
        if mass is None:
            raise ValueError("restart velocities need masses")
        updates["momentum"] = _match_array(
            state.momentum, momenta_from_ang_ps(snapshot["velocities_ang_ps"], mass)
        )
    for group in _CHAIN_GROUPS:
        block = snapshot.get(group)
        chain = getattr(state, group, None)
        if block and chain is not None:
            updates[group] = _restore_chain(chain, block)
    if not updates:
        return state
    if hasattr(state, "set"):
        return state.set(**updates)
    from dataclasses import replace

    return replace(state, **updates)


def flatten_restart(snapshot: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Nested restart dict → arrays safe for ``np.savez``."""
    flat: dict[str, np.ndarray] = {}
    kind = snapshot.get("kind")
    if kind is not None:
        flat["kind"] = np.asarray(str(kind))
    for key in ("momentum", "mass", "rng", "box_momentum", "velocities_ang_ps"):
        if snapshot.get(key) is not None:
            flat[key] = np.asarray(snapshot[key])
    for group in _CHAIN_GROUPS:
        block = snapshot.get(group) or {}
        for sub, value in block.items():
            flat[f"{group}_{sub}"] = np.asarray(value)
    return flat


def unflatten_restart(flat: Mapping[str, Any]) -> dict[str, Any]:
    """Inverse of :func:`flatten_restart`.

    Keys may carry an ``integrator_`` prefix, as stored on the handoff NPZ.
    """
    out: dict[str, Any] = {}
    groups: dict[str, dict[str, Any]] = {name: {} for name in _CHAIN_GROUPS}
    for raw_key, value in flat.items():
        name = raw_key[len("integrator_") :] if str(raw_key).startswith("integrator_") else str(raw_key)
        if name == "kind":
            out["kind"] = str(np.asarray(value).reshape(-1)[0])
            continue
        placed = False
        for group in _CHAIN_GROUPS:
            prefix = group + "_"
            if name.startswith(prefix):
                sub = name[len(prefix) :]
                arr = np.asarray(value)
                if sub == "degrees_of_freedom":
                    groups[group][sub] = int(arr.reshape(-1)[0])
                else:
                    groups[group][sub] = arr
                placed = True
                break
        if not placed:
            out[name] = np.asarray(value)
    for group, block in groups.items():
        if block:
            out[group] = block
    return out


def write_position_dcd(
    path,
    positions,
    *,
    boxes=None,
    dt_fs: float,
    record_every: int,
) -> None:
    """Write recorded coordinate frames to a CHARMM DCD.

    The header stride is ``record_every`` integration steps. Velocities and
    the thermostat are not part of the DCD format; they stay in the NPZ.
    """
    from karml.utils.dcd_writer import DCDTrajectoryWriter

    frames = np.asarray(positions, dtype=np.float64)
    if frames.ndim != 3 or frames.shape[-1] != 3:
        raise ValueError(f"positions must be (n_frames, n_atoms, 3), got {frames.shape}")
    box_list = None
    if boxes is not None:
        box_list = [None if box is None else np.asarray(box, dtype=np.float64) for box in boxes]
        if not any(box is not None for box in box_list):
            box_list = None
    stride = max(1, int(record_every))
    with DCDTrajectoryWriter(
        path,
        n_atoms=int(frames.shape[1]),
        dt_ps=float(dt_fs) * 1.0e-3 * stride,
        steps_per_frame=stride,
        has_unitcell=box_list is not None,
    ) as writer:
        for index, frame in enumerate(frames):
            box = None if box_list is None else box_list[index]
            writer.write(frame, box=box)
