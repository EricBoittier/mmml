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
    # When set, print ``step i/N`` after blocks that cross this cadence (flush).
    progress_every: int | None = None
    # Abort as soon as a recorded frame has non-finite positions/energy (hybrid
    # umbrella windows can otherwise burn hours after a silent blow-up).
    abort_nonfinite: bool = False

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
            from jax_md import minimize, quantity, simulate, space, units
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("JaxmdDriver requires the optional jax and jax-md packages") from exc

        options = dict(ensemble.params)
        dtype = jnp.float64 if bool(options.get("float64", False)) else jnp.float32
        position = jnp.asarray(system.R, dtype=dtype)
        if "masses" in options:
            masses_np = np.asarray(options["masses"], dtype=float)
        else:
            try:
                from ase.data import atomic_masses

                masses_np = np.asarray(
                    [float(atomic_masses[int(z)]) for z in np.asarray(system.Z)],
                    dtype=float,
                )
            except Exception:
                masses_np = np.ones(system.n_atoms, dtype=float)
        masses = jnp.asarray(masses_np, dtype=dtype)
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
            # NVE/NVT with a fixed box: positions stay real-space (Å), matching
            # the box-aware energy terms (mic_displacement against the real cell
            # matrix). ``periodic_general`` defaults to fractional coordinates,
            # which would silently wrap real-space Å positions modulo 1 and
            # produce nonsensical (and unstable) dynamics.
            box = jnp.asarray(system.box, dtype=dtype)
            _, shift_fn = space.periodic_general(box, fractional_coordinates=False)

        hybrid_fn = energy.as_jax_energy_fn()

        if is_npt:
            inv_box0 = jnp.linalg.inv(box)
            init_position = jnp.asarray(system.R, dtype=dtype) @ jnp.transpose(inv_box0)

            def _to_real(frac_R, box_curr):
                return frac_R @ jnp.transpose(box_curr)

            def energy_fn(frac_R, box=box, perturbation=1.0, **kw):
                # jax-md NPT force_stress_fn differentiates through
                # ``perturbation=(1+eps)`` for the virial (dUdV). Ignoring it
                # yields dUdV≈0 → kinetic-only pressure → violent expansion.
                # Isotropic scale matches jax-md pair potentials / quantity.pressure.
                box_p = jnp.asarray(box) * perturbation
                return hybrid_fn(_to_real(frac_R, box_p), box=box_p, **kw)

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

        # A neighbor_fn may declare itself device-native, meaning it takes and
        # returns device arrays. Then the per-block host round trip disappears:
        # for a 1767-atom solvated system the numpy rebuild cost 4.07 s against
        # 0.02 s of actual gradient evaluation, which is what pinned such runs
        # at 100 % CPU and 0 % GPU.
        device_native = bool(getattr(self.neighbor_fn, "device_native", False))

        def refresh_real(real_pos_jax, box_jax):
            if self.neighbor_fn is None:
                return {}
            if device_native:
                result = self.neighbor_fn(real_pos_jax, box_jax)
                if not isinstance(result, Mapping):
                    raise TypeError(
                        "neighbor_fn must return a mapping of energy keyword arrays"
                    )
                return dict(result)
            real_pos = np.asarray(jax.device_get(real_pos_jax))
            box_arr = None if box_jax is None else np.asarray(jax.device_get(box_jax))
            result = self.neighbor_fn(real_pos, box_arr)
            if not isinstance(result, Mapping):
                raise TypeError("neighbor_fn must return a mapping of energy keyword arrays")
            return {key: jnp.asarray(value) for key, value in dict(result).items()}

        def refresh(state):
            return refresh_real(_real_of(state), _box_of(state))

        # Recording used the raw (un-jitted) energy function, which cost ~0.5 s
        # per frame on a 1767-atom solvated system -- more than the dynamics
        # between frames. Compile it once instead.
        _energy_jit = jax.jit(energy_fn)

        def _record_energy(state, dyn):
            if is_npt:
                return float(
                    jax.device_get(
                        _energy_jit(state.position, box=_box_of(state), **dyn)
                    )
                )
            return float(jax.device_get(_energy_jit(state.position, **dyn)))

        def _record_kinetic(state):
            """Kinetic energy (eV) of the integrator state.

            ``energies`` alone is the *potential* surface, so it oscillates under
            NVE and cannot show whether the integrator conserves energy. Record
            KE too so ``potential + kinetic`` is available to drift diagnostics.
            All four ensembles carry ``momentum`` (fire_descent included, where it
            starts at zero); the guard returns NaN only for a state that has none,
            so ``total_energies`` degrades visibly rather than silently.
            """
            momentum = getattr(state, "momentum", None)
            if momentum is None:
                return float("nan")
            return float(
                jax.device_get(
                    quantity.kinetic_energy(momentum=momentum, mass=state.mass)
                )
            )

        seed = int(options.get("seed", 0))
        unit_system = units.metal_unit_system()

        def _volume_A3(box_arr) -> float:
            return float(abs(np.linalg.det(np.asarray(box_arr, dtype=np.float64))))

        def _record_pressure_parts_bar(
            state, dyn, kinetic_eV: float, volume_A3: float
        ) -> tuple[float, float, float]:
            """Return ``(P_total, P_kin, P_virial)`` in bar.

            jax-md: ``P = (2 KE - dUdV) / (d V)``. Split so dilute-box
            sense-checks can compare ``P_kin`` to ``N kT / V``.
            """
            box_now = _box_of(state)
            ke = jnp.asarray(kinetic_eV, dtype=dtype)
            # First call compiles AD through the hybrid energy (virial); can take
            # minutes with GPU idle / host CPU busy — not "slow MD".
            p_metal = quantity.pressure(
                _energy_jit,
                state.position,
                box_now,
                kinetic_energy=ke,
                **dyn,
            )
            scale = float(unit_system["pressure"])
            p_tot = float(jax.device_get(p_metal)) / scale
            dim = 3.0
            vol = max(float(volume_A3), 1.0e-30)
            p_kin = (2.0 * float(kinetic_eV)) / (dim * vol) / scale
            p_vir = p_tot - p_kin
            return p_tot, p_kin, p_vir

        def _record_pressure_bar(state, dyn, kinetic_eV: float) -> float:
            """Instantaneous total pressure (bar) via jax-md ``quantity.pressure``."""
            vol = _volume_A3(_box_of(state))
            p_tot, _p_kin, _p_vir = _record_pressure_parts_bar(
                state, dyn, kinetic_eV, vol
            )
            return p_tot

        # jax-md's metal system (Å, eV, amu) measures time in Å·sqrt(amu/eV) =
        # 10.18 fs, and ``unit_system["time"]`` is how many of those make one
        # picosecond. Handing the integrator a raw picosecond value silently
        # integrates ~98x too finely, so every run covered ~1% of its nominal
        # duration while looking perfectly well-behaved.
        dt_ps = float(ensemble.dt_fs) * 1.0e-3 * unit_system["time"]
        # Cast dt to the run dtype for the same reason as kT/pressure below: a
        # Python-float dt is float64 under JAX_ENABLE_X64, and jax-md threads it
        # through the Nose-Hoover chain update, so a float32 chain state comes
        # back float64 and the integrator scan rejects the mixed carry
        # (``carry component cs[0] has type float32[] ... output ... float64[]``).
        # ``minimize.fire_descent`` takes plain floats, so it keeps ``dt_ps``.
        dt = jnp.asarray(dt_ps, dtype=dtype)

        def _init_state(pos, dyn, offset=0):
            if ensemble.ensemble == "min":
                return init_fn(pos, mass=masses, **dyn)
            if ensemble.ensemble == "nve":
                return init_fn(jax.random.PRNGKey(seed + offset), pos, kT, mass=masses, **dyn)
            if is_npt:
                return init_fn(jax.random.PRNGKey(seed + offset), pos, box=box, mass=masses, **dyn)
            return init_fn(jax.random.PRNGKey(seed + offset), pos, mass=masses, **dyn)

        schedule = ensemble.temperature_schedule
        target_temperatures: list[float] = []

        def _target_temperature(step: int) -> float:
            if schedule is None:
                return float(ensemble.temperature_K)
            return float(schedule.temperature_at(step, ensemble.n_steps))

        kT = None
        if ensemble.ensemble == "min":
            # FIRE grows its step adaptively up to ``dt_max``, so starting from a
            # freshly packed box (close contacts, |F| ~ 100 eV/A) it can take a
            # step large enough to produce NaN before the geometry relaxes.
            # Default to a conservative fraction of the dynamics step and let
            # ``dt_max`` be capped explicitly; both are overridable.
            fire_kwargs = {
                "dt_start": 0.1 * dt_ps,
                "dt_max": dt_ps,
                **dict(options.get("fire_kwargs", {})),
            }
            init_fn, step_fn = minimize.fire_descent(
                energy_fn, shift_fn, **fire_kwargs
            )
        else:
            # Cast kT to the run dtype: under JAX_ENABLE_X64 a Python-float kT is
            # float64, which makes jax-md build the (NPT) barostat state in
            # float64 while positions/box are float32 -- the integrator scan then
            # rejects the mixed carry. Matching kT to ``dtype`` keeps all
            # integrator state in one precision (NVE/NVT benefit too).
            kT = jnp.asarray(_target_temperature(0) * unit_system["temperature"], dtype=dtype)
            if ensemble.ensemble == "nvt":
                thermo = str(ensemble.thermostat or "nhc").strip().lower()
                if thermo in {"langevin", "lgv"}:
                    # Prefer Langevin for packed / hybrid umbrellas: NHC couples
                    # degrees of freedom and one hot window can runaway.
                    gamma = float(
                        options.get(
                            "langevin_gamma",
                            options.get("gamma", 0.1),
                        )
                    )
                    init_fn, step_fn = simulate.nvt_langevin(
                        energy_fn,
                        shift_fn,
                        dt,
                        kT,
                        gamma=gamma,
                        center_velocity=bool(options.get("center_velocity", False)),
                    )
                else:
                    init_fn, step_fn = simulate.nvt_nose_hoover(
                        energy_fn, shift_fn, dt, kT,
                        thermostat_kwargs=options.get("thermostat_kwargs", {}),
                    )
            elif is_npt:
                pressure = jnp.asarray(
                    float(ensemble.pressure_bar) * unit_system["pressure"], dtype=dtype
                )
                init_fn, step_fn = simulate.npt_nose_hoover(
                    energy_fn, shift_fn, dt, pressure, kT,
                    barostat_kwargs=options.get("barostat_kwargs", {}),
                    thermostat_kwargs=options.get("thermostat_kwargs", {}),
                )
            else:
                init_fn, step_fn = simulate.nve(energy_fn, shift_fn, dt)

        # Match legacy jaxmd_runner: without jit, NPT force+stress AD is traced
        # every Python step (GPU util ~0, host CPU pegged for tens of minutes).
        step_fn = jax.jit(step_fn)

        # bootstrap: build the first neighbor list from the initial real positions
        real_init = _to_real(init_position, box) if is_npt else init_position
        dynamic_kwargs = refresh_real(real_init, box)
        if is_npt:
            print(f"    [{self.name}] NPT: init state + E0/P0 (first compile)…", flush=True)
        state = _init_state(init_position, dynamic_kwargs)

        frames = [np.asarray(jax.device_get(_real_of(state)))]
        boxes = [None if box is None else np.asarray(jax.device_get(_box_of(state)))]
        energies = [_record_energy(state, dynamic_kwargs)]
        kinetic_energies = [_record_kinetic(state)]
        volumes_A3: list[float] = []
        pressures_bar: list[float] = []
        pressures_kin_bar: list[float] = []
        pressures_vir_bar: list[float] = []
        if is_npt:
            volumes_A3.append(_volume_A3(boxes[0]))
            print(f"    [{self.name}] NPT: compiling pressure/virial AD…", flush=True)
            p_tot, p_kin, p_vir = _record_pressure_parts_bar(
                state, dynamic_kwargs, kinetic_energies[0], volumes_A3[0]
            )
            pressures_bar.append(p_tot)
            pressures_kin_bar.append(p_kin)
            pressures_vir_bar.append(p_vir)
            print(
                f"    [{self.name}] NPT: P0={p_tot:.4g} bar "
                f"(Pkin={p_kin:.4g}, Pvir={p_vir:.4g}); starting dynamics…",
                flush=True,
            )
        target_temperatures.append(_target_temperature(0))
        block_size = int(self.record_every if self.block_size is None else self.block_size)

        completed = 0
        next_record = min(self.record_every, ensemble.n_steps)
        # NPT smokes: surface progress even when progress_every is unset.
        if is_npt and (self.progress_every is None or int(self.progress_every) <= 0):
            self_progress_every = max(1, min(block_size, ensemble.n_steps // 4 or 1))
        else:
            self_progress_every = self.progress_every
        while completed < ensemble.n_steps:
            dynamic_kwargs = refresh(state)
            count = min(
                block_size,
                ensemble.n_steps - completed,
                next_record - completed,
            )
            # jax-md captures kT in the thermostat step closure. Rebuilding the
            # closure at block boundaries changes the target without resetting
            # positions, momenta, or thermostat/barostat state.
            if schedule is not None and ensemble.ensemble in {"nvt", "npt"}:
                block_kT = jnp.asarray(
                    _target_temperature(completed) * unit_system["temperature"], dtype=dtype
                )
                if is_npt:
                    _, step_fn = simulate.npt_nose_hoover(
                        energy_fn, shift_fn, dt, pressure, block_kT,
                        barostat_kwargs=options.get("barostat_kwargs", {}),
                        thermostat_kwargs=options.get("thermostat_kwargs", {}),
                    )
                else:
                    thermo = str(ensemble.thermostat or "nhc").strip().lower()
                    if thermo in {"langevin", "lgv"}:
                        gamma = float(
                            options.get(
                                "langevin_gamma",
                                options.get("gamma", 0.1),
                            )
                        )
                        _, step_fn = simulate.nvt_langevin(
                            energy_fn,
                            shift_fn,
                            dt,
                            block_kT,
                            gamma=gamma,
                            center_velocity=bool(options.get("center_velocity", False)),
                        )
                    else:
                        _, step_fn = simulate.nvt_nose_hoover(
                            energy_fn, shift_fn, dt, block_kT,
                            thermostat_kwargs=options.get("thermostat_kwargs", {}),
                        )
                step_fn = jax.jit(step_fn)
            for _ in range(count):
                state = step_fn(state, **dynamic_kwargs)
            state.position.block_until_ready()
            completed += count
            prog = self_progress_every
            if prog is not None and int(prog) > 0:
                # Print when this block crossed a progress boundary (or finished).
                prev = completed - count
                crossed = (prev // int(prog)) < (completed // int(prog))
                if crossed or completed >= ensemble.n_steps:
                    print(
                        f"    [{self.name}] step {completed}/{ensemble.n_steps}",
                        flush=True,
                    )

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
                kinetic_energies.append(_record_kinetic(state))
                if self.abort_nonfinite:
                    e_now = float(energies[-1])
                    kin_now = float(kinetic_energies[-1])
                    if (
                        not np.isfinite(e_now)
                        or not np.isfinite(kin_now)
                        or not np.all(np.isfinite(frames[-1]))
                    ):
                        raise RuntimeError(
                            f"{self.name}: non-finite state at step {completed}/"
                            f"{ensemble.n_steps} (E={e_now}, K={kin_now}). "
                            "Try a smaller timestep (e.g. 0.25 fs) or softer seeds."
                        )
                if is_npt:
                    volumes_A3.append(_volume_A3(boxes[-1]))
                    p_tot, p_kin, p_vir = _record_pressure_parts_bar(
                        state,
                        dynamic_kwargs,
                        kinetic_energies[-1],
                        volumes_A3[-1],
                    )
                    pressures_bar.append(p_tot)
                    pressures_kin_bar.append(p_kin)
                    pressures_vir_bar.append(p_vir)
                target_temperatures.append(_target_temperature(completed))
                next_record = min(completed + self.record_every, ensemble.n_steps)

        frames = [np.asarray(f) for f in frames]
        # Z (atomic numbers) and box are needed downstream to reconstruct ASE
        # Atoms for structural analysis (bonds/angles/dihedrals/RDF via
        # mmml.utils.plotting.trajectory_structure) without re-running the
        # simulation -- topology is static per run, so saving it once here is
        # cheap and avoids every analysis script needing to rebuild the system.
        path = Path(self.output_path) if self.output_path is not None else None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            npz_kwargs: dict[str, Any] = dict(
                positions=np.asarray(frames),
                energies=np.asarray(energies),
                kinetic_energies=np.asarray(kinetic_energies),
                total_energies=np.asarray(energies) + np.asarray(kinetic_energies),
                target_temperatures_K=np.asarray(target_temperatures),
                Z=np.asarray(system.Z),
            )
            if box is not None and not is_npt:
                npz_kwargs["box"] = np.asarray(box)
            if is_npt:
                npz_kwargs["boxes"] = np.asarray(boxes)
                npz_kwargs["volumes_A3"] = np.asarray(volumes_A3, dtype=np.float64)
                npz_kwargs["pressures_bar"] = np.asarray(pressures_bar, dtype=np.float64)
                npz_kwargs["pressures_kin_bar"] = np.asarray(
                    pressures_kin_bar, dtype=np.float64
                )
                npz_kwargs["pressures_vir_bar"] = np.asarray(
                    pressures_vir_bar, dtype=np.float64
                )
                npz_kwargs["target_pressure_bar"] = float(ensemble.pressure_bar)
            np.savez(path, **npz_kwargs)

        metadata: dict[str, Any] = {
            "steps": completed,
            "positions": np.asarray(frames),
            "energies": np.asarray(energies),
            "kinetic_energies": np.asarray(kinetic_energies),
            "total_energies": np.asarray(energies) + np.asarray(kinetic_energies),
            "target_temperatures_K": np.asarray(target_temperatures),
        }
        if is_npt:
            metadata["boxes"] = np.asarray(boxes)
            metadata["volumes_A3"] = np.asarray(volumes_A3, dtype=np.float64)
            metadata["pressures_bar"] = np.asarray(pressures_bar, dtype=np.float64)
            metadata["pressures_kin_bar"] = np.asarray(
                pressures_kin_bar, dtype=np.float64
            )
            metadata["pressures_vir_bar"] = np.asarray(
                pressures_vir_bar, dtype=np.float64
            )
            metadata["target_pressure_bar"] = float(ensemble.pressure_bar)
        return Trajectory(path=path, n_frames=len(frames), metadata=metadata)
