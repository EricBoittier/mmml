"""Host-side Verlet list over molecule centroids for sparse ML dimer selection.

The sparse-dimer path in ``mmml_calculator.get_ML_energy_fn`` selects the ML
dimers whose centroid separation is ``< active_radius`` (``mm_switch_on`` +
``ml_dimer_active_margin``). Without a list it evaluates that distance for all
``n(n-1)/2`` pairs every step. This module keeps a candidate list of pairs with
centroid separation ``< active_radius + skin`` (Vesin on the host; brute-force
numpy fallback) and rebuilds it only when some centroid moved more than
``skin/2`` since the last build, or when the box changed. The jitted function
then applies the *same* ``com_dist < active_radius`` test to the candidates
only, so the active set -- and hence energies and forces -- is identical to the
all-pairs path whenever the list is valid.

Contract with the jitted side (``ml_dimer_candidates=`` of the spherical
calculator):

* int32 array of fixed length ``capacity``; valid entries are global dimer ids
  (``itertools.combinations(range(n_monomers), 2)`` order) sorted ascending,
  the padding is ``n_dimers`` (out of range, masked in-graph before any gather
  that feeds forces).
* Sorted ids make ``jnp.nonzero`` over the candidates return the active pairs
  in the same order as over all pairs, so batch slots are identical too.
* Overflow never drops pairs: when a rebuild finds more candidates than the
  capacity, the capacity grows (new static shape, one retrace).

Centroid = unweighted mean of each molecule's atoms exactly as passed to the
jitted function (after ``ml_reorder_indices``), the same quantity as
``_dimer_com_dist``; separations use the minimum image. The in-graph MIC
(fractional rounding) never under-estimates the true minimum-image distance and
the host list uses the true one plus a small float tolerance, so the list is a
superset of the in-graph active set.

Env: ``MMML_ML_DIMER_CENTROID_NL`` (on/off), ``MMML_ML_DIMER_CENTROID_NL_SKIN_A``,
``MMML_ML_DIMER_CENTROID_NL_HEADROOM``.
"""

from __future__ import annotations

import math
import os
from typing import Any, Optional, Sequence

import numpy as np

CENTROID_NL_ENV = "MMML_ML_DIMER_CENTROID_NL"
CENTROID_NL_SKIN_ENV = "MMML_ML_DIMER_CENTROID_NL_SKIN_A"
CENTROID_NL_HEADROOM_ENV = "MMML_ML_DIMER_CENTROID_NL_HEADROOM"
DEFAULT_CENTROID_NL_SKIN_A = 1.0
DEFAULT_CENTROID_NL_HEADROOM = 1.3
# Å added to the list radius: host float64 vs in-graph (possibly float32) COM
# distances differ by ~1e-5 Å; this keeps the superset property with margin.
CENTROID_NL_FLOAT_TOL_A = 1e-3

_FALSE = ("0", "false", "no", "off")


def resolve_centroid_nl_enabled(flag: Optional[bool]) -> bool:
    """Explicit argument wins, then ``MMML_ML_DIMER_CENTROID_NL``, default on."""
    if flag is not None:
        return bool(flag)
    env = (os.environ.get(CENTROID_NL_ENV) or "").strip().lower()
    if env:
        return env not in _FALSE
    return True


def resolve_centroid_nl_skin_A(skin: Optional[float]) -> float:
    env = (os.environ.get(CENTROID_NL_SKIN_ENV) or "").strip()
    if env:
        return max(0.0, float(env))
    if skin is not None:
        return max(0.0, float(skin))
    return DEFAULT_CENTROID_NL_SKIN_A


def dimer_pair_ids(i: np.ndarray, j: np.ndarray, n: int) -> np.ndarray:
    """Global dimer id of pair ``(i, j)``, ``i < j``, in combinations order."""
    i = np.asarray(i, dtype=np.int64)
    j = np.asarray(j, dtype=np.int64)
    return i * (2 * n - i - 1) // 2 + (j - i - 1)


def _mic(d: np.ndarray, cell: Optional[np.ndarray], inv_cell: Optional[np.ndarray]) -> np.ndarray:
    """Same fractional-rounding MIC as ``pbc_utils_jax.mic_displacement``."""
    if cell is None:
        return d
    s = d @ inv_cell
    s = s - np.round(s)
    return s @ cell


class CentroidDimerNeighborList:
    """Skin-buffered candidate list of ML dimer pairs over molecule centroids."""

    def __init__(
        self,
        monomer_idx_arr: np.ndarray,
        monomer_atom_mask: np.ndarray,
        *,
        active_radius: float,
        skin: float = DEFAULT_CENTROID_NL_SKIN_A,
        cell: Optional[Any] = None,
        ml_perm: Optional[Sequence[int]] = None,
        headroom: Optional[float] = None,
        capacity: Optional[int] = None,
        use_vesin: bool = True,
        verbose: bool = False,
    ) -> None:
        self.monomer_idx_arr = np.asarray(monomer_idx_arr, dtype=np.int64)
        mask = np.asarray(monomer_atom_mask, dtype=bool)
        self._w = mask.astype(np.float64)
        self._count = np.maximum(self._w.sum(axis=1), 1e-10)
        self.n_monomers = int(self.monomer_idx_arr.shape[0])
        self.n_dimers = self.n_monomers * (self.n_monomers - 1) // 2
        self.active_radius = float(active_radius)
        self.skin = float(skin)
        self.list_cutoff = self.active_radius + self.skin + CENTROID_NL_FLOAT_TOL_A
        self.default_cell = None if cell is None else np.asarray(cell, dtype=np.float64)
        self.ml_perm = None if ml_perm is None else np.asarray(ml_perm, dtype=np.int64)
        if headroom is None:
            env = (os.environ.get(CENTROID_NL_HEADROOM_ENV) or "").strip()
            headroom = float(env) if env else DEFAULT_CENTROID_NL_HEADROOM
        self.headroom = max(1.0, float(headroom))
        self.capacity: Optional[int] = None if capacity is None else max(1, min(int(capacity), max(self.n_dimers, 1)))
        self.use_vesin = bool(use_vesin)
        self.verbose = bool(verbose)
        # state
        self._ref_centroids: Optional[np.ndarray] = None
        self._ref_cell: Optional[np.ndarray] = None
        self._ids: Optional[np.ndarray] = None
        self._padded_np: Optional[np.ndarray] = None
        self._padded_dev: Any = None
        self.n_calls = 0
        self.n_builds = 0
        self.n_capacity_grows = 0
        self.n_candidates = 0
        self.backend = "none"

    # ------------------------------------------------------------------ helpers
    def centroids(self, positions: Any) -> np.ndarray:
        R = np.asarray(positions, dtype=np.float64)
        if self.ml_perm is not None and self.ml_perm.shape[0] == R.shape[0]:
            R = R[self.ml_perm]
        g = R[self.monomer_idx_arr]  # (n_mono, max_atoms, 3)
        return (g * self._w[:, :, None]).sum(axis=1) / self._count[:, None]

    def _resolve_cell(self, box: Any) -> Optional[np.ndarray]:
        if box is None:
            return self.default_cell
        c = np.asarray(box, dtype=np.float64)
        if c.ndim == 0:
            return np.diag([float(c)] * 3)
        if c.ndim == 1:
            return np.diag(c)
        return c

    def _pairs_vesin(self, C: np.ndarray, cell: Optional[np.ndarray]) -> np.ndarray:
        from vesin import NeighborList

        calc = NeighborList(cutoff=self.list_cutoff, full_list=False)
        if cell is None:
            lo = C.min(axis=0)
            i, j, d = calc.compute(points=C - lo, box=np.zeros((3, 3)), periodic=False, quantities="ijd")
        else:
            i, j, d = calc.compute(points=C, box=cell, periodic=True, quantities="ijd")
        i = np.asarray(i, dtype=np.int64)
        j = np.asarray(j, dtype=np.int64)
        ok = (np.asarray(d) < self.list_cutoff) & (i != j)
        a, b = np.minimum(i[ok], j[ok]), np.maximum(i[ok], j[ok])
        return np.unique(dimer_pair_ids(a, b, self.n_monomers))

    def _pairs_numpy(self, C: np.ndarray, cell: Optional[np.ndarray]) -> np.ndarray:
        iu, ju = np.triu_indices(self.n_monomers, k=1)
        inv = None if cell is None else np.linalg.inv(cell)
        d = _mic(C[ju] - C[iu], cell, inv)
        r = np.linalg.norm(d, axis=-1)
        if cell is not None and np.count_nonzero(cell - np.diag(np.diag(cell))):
            # Skewed cell: fractional rounding is not the true minimum image;
            # take the shortest of the 27 neighbouring images (list superset).
            shifts = np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])).reshape(3, -1).T @ cell
            r = np.min(np.linalg.norm(d[:, None, :] + shifts[None], axis=-1), axis=1)
        # combinations order == triu order, so the positions in iu are the ids
        return np.nonzero(r < self.list_cutoff)[0].astype(np.int64)

    def _needs_rebuild(self, C: np.ndarray, cell: Optional[np.ndarray]) -> bool:
        if self._ref_centroids is None or self._ref_centroids.shape != C.shape:
            return True
        if (cell is None) != (self._ref_cell is None):
            return True
        if cell is not None and not np.array_equal(cell, self._ref_cell):
            return True
        inv = None if cell is None else np.linalg.inv(cell)
        disp = np.linalg.norm(_mic(C - self._ref_centroids, cell, inv), axis=-1)
        return bool(disp.max(initial=0.0) > 0.5 * self.skin)

    def _build(self, C: np.ndarray, cell: Optional[np.ndarray]) -> None:
        ids = None
        if self.use_vesin:
            try:
                ids = self._pairs_vesin(C, cell)
                self.backend = "vesin"
            except Exception as exc:  # vesin missing / unsupported cell
                if self.verbose:
                    print(f"[dimer-centroid-nl] vesin failed ({exc}); numpy fallback", flush=True)
                self.use_vesin = False
        if ids is None:
            ids = self._pairs_numpy(C, cell)
            self.backend = "numpy"
        n = int(ids.shape[0])
        if self.capacity is None or n > self.capacity:
            new_cap = min(max(self.n_dimers, 1), int(math.ceil(n * self.headroom)) + 16)
            if self.capacity is not None:
                self.n_capacity_grows += 1
                print(
                    f"[dimer-centroid-nl] {n} candidate pairs > capacity {self.capacity}; "
                    f"growing to {new_cap} (one retrace)",
                    flush=True,
                )
            self.capacity = new_cap
        padded = np.full((self.capacity,), self.n_dimers, dtype=np.int32)
        padded[:n] = ids  # np.unique / nonzero: already ascending
        self._ids = ids
        self._padded_np = padded
        self._padded_dev = None
        self._ref_centroids = C.copy()
        self._ref_cell = None if cell is None else cell.copy()
        self.n_candidates = n
        self.n_builds += 1

    # --------------------------------------------------------------------- API
    def update_numpy(self, positions: Any, box: Any = None) -> np.ndarray:
        """Return the padded candidate ids (numpy), rebuilding if needed."""
        self.n_calls += 1
        C = self.centroids(positions)
        cell = self._resolve_cell(box)
        if self._needs_rebuild(C, cell):
            self._build(C, cell)
        return self._padded_np

    def update(self, positions: Any, box: Any = None):
        """Padded candidate ids as a device array (re-used until the next rebuild)."""
        import jax.numpy as jnp

        before = self.n_builds
        padded = self.update_numpy(positions, box)
        if self._padded_dev is None or self.n_builds != before:
            self._padded_dev = jnp.asarray(padded, dtype=jnp.int32)
        return self._padded_dev

    def invalidate(self) -> None:
        self._ref_centroids = None

    def stats(self) -> dict:
        return dict(
            calls=self.n_calls,
            builds=self.n_builds,
            capacity=self.capacity,
            candidates=self.n_candidates,
            capacity_grows=self.n_capacity_grows,
            skin_A=self.skin,
            list_cutoff_A=self.list_cutoff,
            backend=self.backend,
        )
