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
        if ensemble.ensemble not in {"min", "nve", "nvt"}:
            raise NotImplementedError("JaxmdDriver currently supports min, nve, and nvt")

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

        if system.box is None:
            _, shift_fn = space.free()
            box = None
        else:
            box = jnp.asarray(system.box, dtype=dtype)
            _, shift_fn = space.periodic_general(box)

        energy_fn = energy.as_jax_energy_fn()
        dynamic_kwargs: Mapping[str, Any] = {}

        def refresh(pos):
            if self.neighbor_fn is None:
                return {}
            result = self.neighbor_fn(np.asarray(jax.device_get(pos)), None if box is None else np.asarray(box))
            if not isinstance(result, Mapping):
                raise TypeError("neighbor_fn must return a mapping of energy keyword arrays")
            return {key: jnp.asarray(value) for key, value in dict(result).items()}

        dynamic_kwargs = refresh(position)
        dt_ps = float(ensemble.dt_fs) * 1.0e-3
        seed = int(options.get("seed", 0))

        if ensemble.ensemble == "min":
            init_fn, step_fn = minimize.fire_descent(energy_fn, shift_fn, dt_start=dt_ps)
            state = init_fn(position, mass=masses, **dynamic_kwargs)
        else:
            kT = float(ensemble.temperature_K) * units.metal_unit_system()["temperature"]
            if ensemble.ensemble == "nvt":
                init_fn, step_fn = simulate.nvt_nose_hoover(
                    energy_fn, shift_fn, dt_ps, kT,
                    thermostat_kwargs=options.get("thermostat_kwargs", {}),
                )
                state = init_fn(jax.random.PRNGKey(seed), position, mass=masses, **dynamic_kwargs)
            else:
                init_fn, step_fn = simulate.nve(energy_fn, shift_fn, dt_ps)
                state = init_fn(
                    jax.random.PRNGKey(seed), position, kT, mass=masses, **dynamic_kwargs
                )

        frames = [np.asarray(jax.device_get(state.position))]
        energies = [float(jax.device_get(energy_fn(state.position, **dynamic_kwargs)))]
        block_size = int(self.record_every if self.block_size is None else self.block_size)

        completed = 0
        next_record = min(self.record_every, ensemble.n_steps)
        while completed < ensemble.n_steps:
            dynamic_kwargs = refresh(state.position)
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
                repaired = on_overlap(np.asarray(jax.device_get(state.position)), None if box is None else np.asarray(box))
                if repaired is not None:
                    position = jnp.asarray(repaired, dtype=dtype)
                    if position.shape != state.position.shape:
                        raise ValueError(
                            "on_overlap returned positions with shape "
                            f"{position.shape}; expected {state.position.shape}"
                        )
                    dynamic_kwargs = refresh(position)
                    if ensemble.ensemble == "min":
                        state = init_fn(position, mass=masses, **dynamic_kwargs)
                    elif ensemble.ensemble == "nvt":
                        state = init_fn(
                            jax.random.PRNGKey(seed + completed),
                            position,
                            mass=masses,
                            **dynamic_kwargs,
                        )
                    else:
                        state = init_fn(
                            jax.random.PRNGKey(seed + completed),
                            position,
                            kT,
                            mass=masses,
                            **dynamic_kwargs,
                        )

            if completed == next_record:
                frames.append(np.asarray(jax.device_get(state.position)))
                energies.append(float(jax.device_get(energy_fn(state.position, **dynamic_kwargs))))
                next_record = min(completed + self.record_every, ensemble.n_steps)

        path = Path(self.output_path) if self.output_path is not None else None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, positions=np.asarray(frames), energies=np.asarray(energies))

        return Trajectory(
            path=path,
            n_frames=len(frames),
            metadata={"steps": completed, "positions": np.asarray(frames), "energies": np.asarray(energies)},
        )
