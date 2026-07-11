"""Shared jax-md propagation driver.

The driver deliberately owns integration and neighbor-list rebuild cadence;
energy terms only consume the padded arrays passed in ``dynamic_kwargs``.
Heavy jax/jax-md imports remain inside :meth:`JaxmdDriver.run` so importing the
shared MD schema does not require either optional dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from mmml.md.config import EnsembleSpec
from mmml.md.energy.registry import HybridEnergy
from mmml.md.results import Trajectory
from mmml.md.system import MolecularSystem

__all__ = ["JaxmdDriver"]


@dataclass(frozen=True)
class JaxmdDriver:
    """Propagate a :class:`MolecularSystem` with jax-md.

    ``neighbor_fn`` is called at block boundaries as ``fn(position, box)`` and
    returns keyword arrays routed to the hybrid energy (for example
    ``pair_i``, ``pair_j`` and ``pair_mask``). This keeps decision B intact:
    each term owns its capacity and the driver merely refreshes its buffers.
    """

    record_every: int = 100
    block_size: int | None = None
    neighbor_fn: Callable[[np.ndarray, np.ndarray | None], Mapping[str, Any]] | None = None
    output_path: Path | None = None
    name: str = "jaxmd"

    def run(
        self,
        system: MolecularSystem,
        energy: HybridEnergy,
        ensemble: EnsembleSpec,
        *,
        on_overlap: Callable[..., Any] | None = None,
    ) -> Trajectory:
        if ensemble.n_steps < 0:
            raise ValueError("n_steps must be non-negative")
        if self.record_every <= 0:
            raise ValueError("record_every must be positive")
        if self.block_size is not None and self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if ensemble.dt_fs <= 0:
            raise ValueError("dt_fs must be positive")
        if ensemble.ensemble not in {"min", "nve", "nvt", "npt"}:
            raise NotImplementedError("JaxmdDriver currently supports min, nve, nvt, and npt")

        try:
            import jax
            import jax.numpy as jnp
            from jax_md import minimize, simulate, space, units
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("JaxmdDriver requires the optional jax and jax-md packages") from exc

        options = dict(ensemble.params)
        dtype = jnp.float64 if bool(options.get("float64", False)) else jnp.float32
        position = jnp.asarray(system.R, dtype=dtype)
        masses = jnp.asarray(options.get("masses", np.ones(system.n_atoms)), dtype=dtype)
        if masses.shape != (system.n_atoms,):
            raise ValueError(f"masses must have shape ({system.n_atoms},), got {masses.shape}")
        if not bool(np.all(np.asarray(masses) > 0)):
            raise ValueError("masses must be strictly positive")

        is_npt = ensemble.ensemble == "npt"
        if is_npt and system.box is None:
            raise ValueError("npt requires a periodic box")

        if system.box is None:
            _, shift_fn = space.free()
            box = None
        elif is_npt:
            # NPT evolves the box; jax-md works in fractional coordinates. We keep
            # our real-space, box-aware terms and bridge here.
            box = jnp.asarray(system.box, dtype=dtype)
            _, shift_fn = space.periodic_general(box, fractional_coordinates=True)
        else:
            box = jnp.asarray(system.box, dtype=dtype)
            _, shift_fn = space.periodic_general(box)

        hybrid_fn = energy.as_jax_energy_fn()

        if is_npt:
            inv_box0 = jnp.linalg.inv(box)
            init_position = jnp.asarray(system.R, dtype=dtype) @ jnp.transpose(inv_box0)

            def _to_real(frac_R, box_curr):
                return frac_R @ jnp.transpose(box_curr)

            def energy_fn(frac_R, box=box, **kw):
                # jax-md passes the current box each step; convert to real space
                # and thread the box into the (box-aware) terms.
                return hybrid_fn(_to_real(frac_R, box), box=box, **kw)

            def _box_of(state):
                return simulate.npt_box(state)

            def _real_of(state):
                return _to_real(state.position, _box_of(state))
        else:
            init_position = position
            energy_fn = hybrid_fn

            def _box_of(state):  # noqa: ARG001 - fixed box
                return box

            def _real_of(state):
                return state.position

        dynamic_kwargs: Mapping[str, Any] = {}

        def refresh_real(real_pos_jax, box_jax):
            if self.neighbor_fn is None:
                return {}
            real_pos = np.asarray(jax.device_get(real_pos_jax))
            box_arr = None if box_jax is None else np.asarray(jax.device_get(box_jax))
            result = self.neighbor_fn(real_pos, box_arr)
            if not isinstance(result, Mapping):
                raise TypeError("neighbor_fn must return a mapping of energy keyword arrays")
            return {key: jnp.asarray(value) for key, value in dict(result).items()}

        def refresh(state):
            return refresh_real(_real_of(state), _box_of(state))

        def _record_energy(state, dyn):
            return float(jax.device_get(energy_fn(state.position, **dyn)))

        dt_ps = float(ensemble.dt_fs) * 1.0e-3
        seed = int(options.get("seed", 0))
        unit_system = units.metal_unit_system()

        def _init_state(pos, dyn, offset=0):
            if ensemble.ensemble == "min":
                return init_fn(pos, mass=masses, **dyn)
            if ensemble.ensemble == "nve":
                return init_fn(jax.random.PRNGKey(seed + offset), pos, kT, mass=masses, **dyn)
            if is_npt:
                return init_fn(jax.random.PRNGKey(seed + offset), pos, box=box, mass=masses, **dyn)
            return init_fn(jax.random.PRNGKey(seed + offset), pos, mass=masses, **dyn)

        kT = None
        if ensemble.ensemble == "min":
            init_fn, step_fn = minimize.fire_descent(energy_fn, shift_fn, dt_start=dt_ps)
        else:
            kT = float(ensemble.temperature_K) * unit_system["temperature"]
            if ensemble.ensemble == "nvt":
                init_fn, step_fn = simulate.nvt_nose_hoover(
                    energy_fn, shift_fn, dt_ps, kT,
                    thermostat_kwargs=options.get("thermostat_kwargs", {}),
                )
            elif is_npt:
                pressure = float(ensemble.pressure_bar) * unit_system["pressure"]
                init_fn, step_fn = simulate.npt_nose_hoover(
                    energy_fn, shift_fn, dt_ps, pressure, kT,
                    barostat_kwargs=options.get("barostat_kwargs", {}),
                    thermostat_kwargs=options.get("thermostat_kwargs", {}),
                )
            else:
                init_fn, step_fn = simulate.nve(energy_fn, shift_fn, dt_ps)

        # bootstrap: build the first neighbor list from the initial real positions
        real_init = _to_real(init_position, box) if is_npt else init_position
        dynamic_kwargs = refresh_real(real_init, box)
        state = _init_state(init_position, dynamic_kwargs)

        frames = [np.asarray(jax.device_get(_real_of(state)))]
        boxes = [None if box is None else np.asarray(jax.device_get(_box_of(state)))]
        energies = [_record_energy(state, dynamic_kwargs)]
        block_size = int(self.record_every if self.block_size is None else self.block_size)

        completed = 0
        next_record = min(self.record_every, ensemble.n_steps)
        while completed < ensemble.n_steps:
            dynamic_kwargs = refresh(state)
            count = min(
                block_size,
                ensemble.n_steps - completed,
                next_record - completed,
            )
            for _ in range(count):
                state = step_fn(state, **dynamic_kwargs)
            state.position.block_until_ready()
            completed += count

            if on_overlap is not None:
                repaired = on_overlap(
                    np.asarray(jax.device_get(_real_of(state))),
                    None if box is None else np.asarray(jax.device_get(_box_of(state))),
                )
                if repaired is not None:
                    repaired = np.asarray(repaired, dtype=float)
                    if repaired.shape != (system.n_atoms, 3):
                        raise ValueError(
                            "on_overlap returned positions with shape "
                            f"{repaired.shape}; expected {(system.n_atoms, 3)}"
                        )
                    if is_npt:
                        box_now = np.asarray(jax.device_get(_box_of(state)))
                        pos_reinit = jnp.asarray(repaired @ np.linalg.inv(box_now).T, dtype=dtype)
                        real_reinit = jnp.asarray(repaired, dtype=dtype)
                        box_reinit = jnp.asarray(box_now, dtype=dtype)
                    else:
                        pos_reinit = jnp.asarray(repaired, dtype=dtype)
                        real_reinit = pos_reinit
                        box_reinit = box
                    dynamic_kwargs = refresh_real(real_reinit, box_reinit)
                    state = _init_state(pos_reinit, dynamic_kwargs, offset=completed)

            if completed == next_record:
                frames.append(np.asarray(jax.device_get(_real_of(state))))
                boxes.append(None if box is None else np.asarray(jax.device_get(_box_of(state))))
                energies.append(_record_energy(state, dynamic_kwargs))
                next_record = min(completed + self.record_every, ensemble.n_steps)

        frames = [np.asarray(f) for f in frames]
        path = Path(self.output_path) if self.output_path is not None else None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, positions=np.asarray(frames), energies=np.asarray(energies))

        metadata: dict[str, Any] = {
            "steps": completed,
            "positions": np.asarray(frames),
            "energies": np.asarray(energies),
        }
        if is_npt:
            metadata["boxes"] = np.asarray(boxes)
        return Trajectory(path=path, n_frames=len(frames), metadata=metadata)
