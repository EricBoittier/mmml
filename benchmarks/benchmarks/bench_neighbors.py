"""Host-side neighbour construction and the Verlet-skin cache.

Neighbour rebuilds happen on the host between jitted blocks, so on a GPU run
they are pure serialisation: the device idles while NumPy walks the box. That
makes these the benchmarks most likely to explain a disappointing ns/day, and it
is why ``with_verlet_skin`` exists at all.

Three layers are measured separately:

* ``_build_pair_indices`` — the dispatcher, which prefers Vesin inside the
  unique-MIC regime (``cutoff <= L/2``) and falls back to chunked NumPy outside
  it. ``_build_pair_indices_vectorized`` is benchmarked alongside it so the
  Vesin win is a number rather than an assumption.
* ``get_intermolecular_pairs`` / ``make_intermolecular_neighbor_fn`` — the
  molecule filter and the padding + capacity-guard the driver actually calls.
* ``with_verlet_skin`` — the amortised cost across a block of steps, which is
  the only one of these that shows up in a trajectory's wall time.
"""

from __future__ import annotations

import numpy as np

from ._common import require_jax, skip, synthetic_system, water_box

CUTOFF_A = 12.0


class PairListBackends:
    """Vesin-backed dispatcher vs. the pure-NumPy chunked MIC fallback."""

    params = [512, 1000, 1728]
    param_names = ["n_waters"]
    timeout = 900.0
    number = 1
    repeat = (3, 10, 30.0)

    def setup(self, n_waters):
        try:
            from mmml.interfaces.pycharmmInterface.mm_system_energy import (
                _build_pair_indices,
                _build_pair_indices_vectorized,
            )
        except Exception as exc:  # pragma: no cover - environment-dependent
            raise skip(f"mm_system_energy unavailable: {exc}") from exc

        box = water_box(int(n_waters))
        self.R = np.asarray(box["R"], dtype=np.float64)
        self.cell = np.asarray(box["box"], dtype=np.float64)
        self.box_L = box["box_L"]
        self.excluded = frozenset()
        self._dispatch = _build_pair_indices
        self._vectorized = _build_pair_indices_vectorized

    def time_dispatch(self, n_waters):
        self._dispatch(self.R, self.cell, self.excluded, CUTOFF_A)

    def time_numpy_chunked(self, n_waters):
        self._vectorized(self.R, self.cell, self.excluded, CUTOFF_A)

    def track_box_length_A(self, n_waters):
        return self.box_L

    track_box_length_A.unit = "angstrom"


class IntermolecularPairs:
    """``get_intermolecular_pairs`` — pair build plus the ``mol_id`` filter."""

    params = [512, 1000, 1728]
    param_names = ["n_waters"]
    timeout = 900.0
    number = 1
    repeat = (3, 10, 30.0)

    def setup(self, n_waters):
        try:
            from mmml.interfaces.jaxmdInterface.hybrid_energy import (
                get_intermolecular_pairs,
            )
        except Exception as exc:  # pragma: no cover - environment-dependent
            raise skip(f"hybrid_energy unavailable: {exc}") from exc

        box = water_box(int(n_waters))
        self.fn = get_intermolecular_pairs
        self.R = np.asarray(box["R"], dtype=np.float64)
        self.cell = np.asarray(box["box"], dtype=np.float64)
        self.mol_id = np.asarray(box["mol_id"], dtype=np.int32)

    def time_intermolecular_pairs(self, n_waters):
        self.fn(self.R, self.cell, frozenset(), CUTOFF_A, self.mol_id)


class NeighborRefresh:
    """``make_intermolecular_neighbor_fn`` — one full block-boundary refresh.

    Includes the capacity check and the padding into fixed-shape arrays, i.e.
    everything the driver pays between two jitted blocks.
    """

    params = [512, 1000, 1728]
    param_names = ["n_waters"]
    timeout = 900.0
    number = 1
    repeat = (3, 10, 30.0)

    def setup(self, n_waters):
        try:
            from mmml.md.neighbors import make_intermolecular_neighbor_fn
        except Exception as exc:  # pragma: no cover - environment-dependent
            raise skip(f"mmml.md.neighbors unavailable: {exc}") from exc

        system, box = synthetic_system(int(n_waters))
        self.system = system
        self.R = np.asarray(system.R, dtype=np.float64)
        self.cell = np.asarray(system.box, dtype=np.float64)
        # on_overflow="warn": capacity is an estimate from the shell volume, and a
        # benchmark should report the cost of the real refresh, not die on a
        # density fluctuation.
        self.fn = make_intermolecular_neighbor_fn(
            system, CUTOFF_A, on_overflow="warn"
        )
        self.fn(self.R, self.cell)

    def time_refresh(self, n_waters):
        self.fn(self.R, self.cell)


class VerletSkinCache:
    """Amortised neighbour cost over a block of steps, with and without a skin.

    ``skin_A=0`` is the rebuild-every-block behaviour; a positive skin builds at
    ``cutoff + skin`` and reuses the list until some atom has moved more than
    ``skin/2``. The displacement per call here (0.01 Å) is roughly one 0.5 fs step
    of a room-temperature water, so ``n_calls`` reads as "steps between block
    boundaries".
    """

    params = [0.0, 1.0, 2.0]
    param_names = ["skin_A"]
    timeout = 900.0
    number = 1
    repeat = (3, 10, 60.0)

    #: block boundaries simulated per timed call
    n_calls = 20
    #: per-call displacement of every atom (Å)
    step_displacement_A = 0.01

    def setup(self, skin_A):
        try:
            from mmml.md.neighbors import make_intermolecular_neighbor_fn
        except Exception as exc:  # pragma: no cover - environment-dependent
            raise skip(f"mmml.md.neighbors unavailable: {exc}") from exc

        require_jax()  # with_verlet_skin uploads results with jnp.asarray

        system, _ = synthetic_system(512)
        self.fn = make_intermolecular_neighbor_fn(
            system, CUTOFF_A, on_overflow="warn", skin_A=float(skin_A)
        )
        self.cell = np.asarray(system.box, dtype=np.float64)
        rng = np.random.default_rng(7)
        base = np.asarray(system.R, dtype=np.float64)
        # A fixed drift sequence, so the reuse pattern is identical across runs
        # and the comparison between skins is not a random walk.
        drift = rng.normal(size=base.shape)
        drift /= np.linalg.norm(drift, axis=-1, keepdims=True)
        self.frames = [
            base + drift * (self.step_displacement_A * k) for k in range(self.n_calls)
        ]
        for frame in self.frames:
            self.fn(frame, self.cell)

    def time_block_of_refreshes(self, skin_A):
        for frame in self.frames:
            self.fn(frame, self.cell)

    def track_rebuild_fraction(self, skin_A):
        """Share of the block's refreshes that actually rebuilt the list."""
        stats = getattr(self.fn, "stats", None)
        if stats is None:
            return 1.0  # unskinned path rebuilds unconditionally
        calls = max(int(stats.calls), 1)
        return float(stats.rebuilds) / calls

    track_rebuild_fraction.unit = "fraction"


class NeighborCapacitySizing:
    """``pad_indices`` — padding a raw pair list into fixed-capacity arrays.

    Tiny next to a pair build, but it runs on every refresh and it is pure host
    Python, so a regression here is invisible in profiles and shows up as GPU
    idle time.
    """

    params = [10_000, 100_000, 1_000_000]
    param_names = ["n_pairs"]
    number = 100
    repeat = (3, 10, 10.0)

    def setup(self, n_pairs):
        try:
            from mmml.md.energy.capacity import pad_indices
        except Exception as exc:  # pragma: no cover - environment-dependent
            raise skip(f"capacity module unavailable: {exc}") from exc

        rng = np.random.default_rng(0)
        self.pad_indices = pad_indices
        self.pi = rng.integers(0, 5000, size=int(n_pairs)).astype(np.int32)
        self.capacity = int(n_pairs * 1.5)

    def time_pad_indices(self, n_pairs):
        self.pad_indices(self.pi, self.capacity)
