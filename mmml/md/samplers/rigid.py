"""Rigid-body Metropolis Monte Carlo sampler.

A :class:`Sampler` peer of the MD drivers (decision, §10): whole monomers move as
rigid bodies — a COM translation plus a rotation represented as a **unit
quaternion** — and moves are accepted with the Metropolis criterion under the
system's :class:`~mmml.md.energy.registry.HybridEnergy`. Rigid moves preserve
every intramolecular distance exactly, so this samples configurational space
without intramolecular forces.

Rigid groups are ``system.monomer_indices`` (all atoms as one body if empty).
Positions are real-space Å; the energy terms handle PBC. Monomers are assumed
un-wrapped across the periodic boundary (COM is taken directly); wrap-aware
unfolding is a future refinement.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from mmml.md.config import RunConfig
from mmml.md.energy.registry import HybridEnergy
from mmml.md.results import Trajectory
from mmml.md.system import MolecularSystem

__all__ = ["RigidBodySampler", "quat_from_axis_angle", "quat_to_matrix"]

# Boltzmann constant in eV/K (energies from HybridEnergy are in eV).
_KB_EV_PER_K = 8.617333262e-5


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Unit quaternion (w, x, y, z) for a rotation of ``angle`` about ``axis``."""
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = axis / norm
    half = 0.5 * float(angle)
    return np.concatenate([[np.cos(half)], np.sin(half) * axis])


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix (3, 3) for a unit quaternion (w, x, y, z)."""
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


@dataclass(frozen=True)
class RigidBodySampler:
    """Metropolis MC over rigid-body translation + rotation of each monomer.

    ``neighbor_fn`` mirrors :class:`~mmml.md.drivers.JaxmdDriver`'s: called as
    ``fn(position, box)`` at ``neighbor_refresh_every``-sweep boundaries,
    returning keyword arrays routed into the energy (e.g. ``pair_i`` /
    ``pair_j`` / ``pair_mask`` for ``mm_nonbonded``). Required for any term that
    declares an intermolecular :class:`~mmml.md.energy.registry.NeighborRequest`
    — without it, the term's host pair-build path is not jit-compatible and
    raises under ``jax.jit`` (decision B: terms own their pair-list capacity,
    the sampler/driver only owns the rebuild cadence). Rebuilt once per
    ``neighbor_refresh_every`` sweeps rather than per proposed move, since
    rebuilding on the host for every trial move would dominate runtime.
    """

    record_every: int = 100
    max_translation_A: float = 0.2
    max_rotation_rad: float = 0.2
    neighbor_fn: Callable[[np.ndarray, np.ndarray | None], Mapping[str, Any]] | None = None
    neighbor_refresh_every: int = 1
    output_path: Path | None = None
    name: str = "rigid"

    def run(
        self,
        system: MolecularSystem,
        energy: HybridEnergy,
        config: RunConfig,
    ) -> Trajectory:
        import jax
        import jax.numpy as jnp

        n_sweeps = int(config.ensemble.n_steps)
        if n_sweeps < 0:
            raise ValueError("n_steps must be non-negative")
        if self.record_every <= 0:
            raise ValueError("record_every must be positive")
        if self.neighbor_refresh_every <= 0:
            raise ValueError("neighbor_refresh_every must be positive")

        rng = np.random.default_rng(int(config.seed))
        temperature = float(config.ensemble.temperature_K)
        kT = _KB_EV_PER_K * temperature
        if kT <= 0:
            raise ValueError("temperature_K must be positive for Metropolis MC")

        groups = system.monomer_indices or [np.arange(system.n_atoms)]
        groups = [np.asarray(g, dtype=int) for g in groups]

        energy_fn = jax.jit(energy.as_jax_energy_fn())
        box = None if system.box is None else np.asarray(system.box)

        def refresh(positions: np.ndarray) -> Mapping[str, Any]:
            if self.neighbor_fn is None:
                return {}
            result = self.neighbor_fn(positions, box)
            if not isinstance(result, Mapping):
                raise TypeError("neighbor_fn must return a mapping of energy keyword arrays")
            return dict(result)

        def E(positions: np.ndarray, dyn: Mapping[str, Any]) -> float:
            return float(energy_fn(jnp.asarray(positions), **dyn))

        pos = np.array(system.R, dtype=float)
        dynamic_kwargs = refresh(pos)
        e_curr = E(pos, dynamic_kwargs)
        frames = [pos.copy()]
        energies = [e_curr]
        accepted = 0
        attempted = 0

        for step in range(n_sweeps):
            if step > 0 and step % self.neighbor_refresh_every == 0:
                dynamic_kwargs = refresh(pos)
                e_curr = E(pos, dynamic_kwargs)  # keep the running energy consistent

            for _ in range(len(groups)):  # one sweep ≈ one attempted move per body
                idx = groups[rng.integers(len(groups))]
                sub = pos[idx]
                com = sub.mean(axis=0)

                # rotation as a unit quaternion about a random axis
                axis = rng.normal(size=3)
                angle = rng.uniform(-self.max_rotation_rad, self.max_rotation_rad)
                rot = quat_to_matrix(quat_from_axis_angle(axis, angle))
                translation = rng.normal(scale=self.max_translation_A, size=3)

                trial = pos.copy()
                trial[idx] = (sub - com) @ rot.T + com + translation

                e_trial = E(trial, dynamic_kwargs)
                attempted += 1
                delta = e_trial - e_curr
                if delta <= 0.0 or rng.random() < np.exp(-delta / kT):
                    pos = trial
                    e_curr = e_trial
                    accepted += 1

            if (step + 1) % self.record_every == 0:
                frames.append(pos.copy())
                energies.append(e_curr)

        frames_arr = np.asarray(frames)
        energies_arr = np.asarray(energies)
        path = Path(self.output_path) if self.output_path is not None else None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, positions=frames_arr, energies=energies_arr)

        acceptance = accepted / attempted if attempted else 0.0
        metadata: dict[str, Any] = {
            "sweeps": n_sweeps,
            "attempted": attempted,
            "accepted": accepted,
            "acceptance_ratio": acceptance,
            "positions": frames_arr,
            "energies": energies_arr,
        }
        return Trajectory(path=path, n_frames=len(frames), metadata=metadata)
