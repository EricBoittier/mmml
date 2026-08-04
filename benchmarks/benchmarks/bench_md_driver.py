"""End-to-end MD throughput through ``mmml.md.drivers.JaxmdDriver``.

This is the benchmark that answers the only question a simulation actually asks:
**how many nanoseconds per day**. Everything else in the suite explains this
number; this is the number.

The driver batches steps into jitted blocks and refreshes the neighbour list on
the host at each block boundary, so ``block_size`` trades device utilisation
against list staleness — :class:`MDBlockSize` measures that trade directly.
``track_ns_per_day`` re-times the same run and converts, so a result is readable
without knowing the step count.
"""

from __future__ import annotations

import time

import numpy as np

from ._common import require_jax, require_jax_md, skip, synthetic_system

CUTOFF_A = 12.0
DT_FS = 0.5


def _hybrid_mm_energy(system):
    """A ``HybridEnergy`` with the switched MM nonbonded term over the box."""
    from mmml.interfaces.pycharmmInterface.mm_system_energy import CharmmNbondSettings
    from mmml.md.energy import EnergyContext, HybridEnergy
    from mmml.md.energy.terms import MMNonbondedTerm

    settings = CharmmNbondSettings(cutnb=CUTOFF_A, ctonnb=10.0, ctofnb=CUTOFF_A)
    return HybridEnergy([MMNonbondedTerm(settings)], system, EnergyContext())


def _ns_per_day(n_steps: int, dt_fs: float, wall_s: float) -> float:
    if wall_s <= 0:
        return float("nan")
    simulated_ns = n_steps * dt_fs * 1e-6
    return simulated_ns * 86_400.0 / wall_s


class _DriverBase:
    timeout = 1800.0
    warmup_time = 0.0
    number = 1
    repeat = (2, 5, 120.0)

    #: steps per timed call — long enough to amortise the first block's compile,
    #: short enough that a full parameter sweep finishes in minutes.
    n_steps = 200

    def _build(
        self,
        n_waters: int,
        *,
        ensemble: str = "nve",
        block_size: int = 50,
        skin_A: float = 2.0,
    ):
        require_jax()
        require_jax_md()
        try:
            from mmml.md.config import EnsembleSpec
            from mmml.md.drivers import JaxmdDriver
            from mmml.md.neighbors import make_intermolecular_neighbor_fn
        except Exception as exc:  # pragma: no cover - environment-dependent
            raise skip(f"mmml.md driver stack unavailable: {exc}") from exc

        system, box = synthetic_system(int(n_waters))
        self.system = system
        self.energy = _hybrid_mm_energy(system)

        params = {"masses": box["masses"], "seed": 0}

        self.driver = JaxmdDriver(
            record_every=max(self.n_steps, 1),
            block_size=int(block_size),
            neighbor_fn=make_intermolecular_neighbor_fn(
                system, CUTOFF_A, on_overflow="warn", skin_A=float(skin_A)
            ),
        )
        self.ensemble = EnsembleSpec(
            ensemble=ensemble,
            space="pbc",
            temperature_K=300.0,
            dt_fs=DT_FS,
            n_steps=self.n_steps,
            params=params,
        )
        # Warm the block-stepper compile out of the timed region.
        self._run()

    def _run(self):
        return self.driver.run(self.system, self.energy, self.ensemble)

    def _timed_ns_per_day(self):
        start = time.perf_counter()
        self._run()
        return _ns_per_day(self.n_steps, DT_FS, time.perf_counter() - start)


class MDSystemSize(_DriverBase):
    """MM/NVE throughput vs. box size — the headline scaling curve."""

    params = [216, 512, 1000]
    param_names = ["n_waters"]

    def setup(self, n_waters):
        self._build(n_waters)

    def time_md_steps(self, n_waters):
        self._run()

    def track_ns_per_day(self, n_waters):
        return self._timed_ns_per_day()

    track_ns_per_day.unit = "ns/day"


class MDEnsemble(_DriverBase):
    """Cost of the thermostat / barostat machinery on a fixed 512-water box.

    NpT is expected to be the slow one: the strain derivative differentiates the
    energy with respect to the cell as well as the positions, so every step pays
    an extra pass over the pair list.
    """

    params = ["nve", "nvt", "npt"]
    param_names = ["ensemble"]

    def setup(self, ensemble):
        self._build(512, ensemble=str(ensemble))

    def time_md_steps(self, ensemble):
        self._run()

    def track_ns_per_day(self, ensemble):
        return self._timed_ns_per_day()

    track_ns_per_day.unit = "ns/day"


class MDBlockSize(_DriverBase):
    """Steps per jitted block — device utilisation vs. neighbour-list staleness.

    Small blocks return to the host constantly and starve the device; large
    blocks let the list go stale, which the Verlet skin has to absorb. The
    optimum is what this measures.
    """

    params = [1, 10, 50, 200]
    param_names = ["block_size"]

    def setup(self, block_size):
        self._build(512, block_size=int(block_size))

    def time_md_steps(self, block_size):
        self._run()

    def track_ns_per_day(self, block_size):
        return self._timed_ns_per_day()

    track_ns_per_day.unit = "ns/day"


class MDNeighborSkin(_DriverBase):
    """Verlet skin width, measured where it matters: on the trajectory clock.

    ``skin_A=0`` rebuilds the pair list at every block boundary. A wider skin
    trades extra in-cutoff pairs (which ``mm_nonbonded`` masks out anyway) for
    fewer host rebuilds.
    """

    params = [0.0, 1.0, 2.0, 3.0]
    param_names = ["skin_A"]

    def setup(self, skin_A):
        self._build(512, skin_A=float(skin_A))

    def time_md_steps(self, skin_A):
        self._run()

    def track_ns_per_day(self, skin_A):
        return self._timed_ns_per_day()

    track_ns_per_day.unit = "ns/day"
