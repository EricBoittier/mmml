"""Stable structure identities and composition / condition keys.

IDs are derived from geometry and composition so that overlapping selections
reuse cached reference calculations even when source files differ.  Provenance
(trajectory, parent configuration, frame index) is stored alongside the ID
but does not change it.

Coordinates are COM-centered and rounded before hashing so tiny numeric noise
does not mint a new identity.  The hash is *not* rotation-invariant: two
orientations of the same molecule are distinct structures for labeling.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np

COORD_DECIMALS = 4
CELL_DECIMALS = 4


def composition_key(atomic_numbers: np.ndarray, n_atoms: int | None = None) -> str:
    """Return a Hill-like composition string, e.g. ``C1H4`` or ``H2O1``."""
    z = np.asarray(atomic_numbers, dtype=np.int32).reshape(-1)
    n = int(n_atoms) if n_atoms is not None else int((z > 0).sum())
    z = z[:n]
    z = z[z > 0]
    counts: dict[int, int] = {}
    for zi in z.tolist():
        counts[int(zi)] = counts.get(int(zi), 0) + 1
    # C, H first (Hill), then remaining Z ascending.
    order = []
    if 6 in counts:
        order.append(6)
    if 1 in counts:
        order.append(1)
    order.extend(sorted(k for k in counts if k not in (1, 6)))
    parts = []
    for zi in order:
        parts.append(f"{_element_symbol(zi)}{counts[zi]}")
    return "".join(parts) if parts else "empty"


def _element_symbol(z: int) -> str:
    # Minimal table; unknown Z falls back to Z<n>.
    symbols = {
        1: "H",
        6: "C",
        7: "N",
        8: "O",
        9: "F",
        15: "P",
        16: "S",
        17: "Cl",
        35: "Br",
    }
    return symbols.get(int(z), f"Z{int(z)}")


def _canonical_coords(positions: np.ndarray, n_atoms: int) -> np.ndarray:
    r = np.asarray(positions, dtype=np.float64).reshape(-1, 3)[: int(n_atoms)]
    com = r.mean(axis=0, keepdims=True)
    centered = r - com
    return np.round(centered, COORD_DECIMALS)


def geometry_fingerprint(
    positions: np.ndarray,
    atomic_numbers: np.ndarray,
    n_atoms: int | None = None,
    *,
    cell: np.ndarray | None = None,
) -> str:
    """SHA256 of composition + COM-centered rounded coordinates (+ cell)."""
    z = np.asarray(atomic_numbers, dtype=np.int32).reshape(-1)
    n = int(n_atoms) if n_atoms is not None else int((z > 0).sum())
    payload = {
        "comp": composition_key(z, n),
        "R": _canonical_coords(positions, n).tolist(),
        "Z": z[:n].astype(int).tolist(),
    }
    if cell is not None:
        c = np.asarray(cell, dtype=np.float64).reshape(-1)
        payload["cell"] = np.round(c, CELL_DECIMALS).tolist()
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def structure_id(
    positions: np.ndarray,
    atomic_numbers: np.ndarray,
    n_atoms: int | None = None,
    *,
    cell: np.ndarray | None = None,
    prefix: str = "sid",
) -> str:
    """Stable ID used as the cache key for expensive labels."""
    fp = geometry_fingerprint(positions, atomic_numbers, n_atoms, cell=cell)
    return f"{prefix}-{fp[:16]}"


def condition_key(record: Mapping[str, Any] | None) -> str:
    """Thermodynamic-condition stratum, e.g. ``T300_P1_phase-liquid``."""
    if not record:
        return "unspecified"
    t = record.get("temperature", record.get("T", None))
    p = record.get("pressure", record.get("P", None))
    phase = record.get("phase", record.get("ensemble", None))
    parts = []
    if t is not None and np.isfinite(t):
        parts.append(f"T{int(round(float(t)))}")
    if p is not None and np.isfinite(p):
        parts.append(f"P{float(p):g}")
    if phase is not None and str(phase).strip():
        parts.append(f"phase-{str(phase).strip()}")
    return "_".join(parts) if parts else "unspecified"


def stratum_key(
    atomic_numbers: np.ndarray,
    n_atoms: int | None = None,
    record: Mapping[str, Any] | None = None,
) -> str:
    """Composition × thermodynamic-condition stratum used by the random baseline."""
    return f"{composition_key(atomic_numbers, n_atoms)}|{condition_key(record)}"


def group_key(record: Mapping[str, Any] | None, fallback_index: int) -> str:
    """Trajectory / parent-configuration group for split isolation.

    Neighboring MD frames share a group so they cannot leak across
    candidate / validation / test splits.
    """
    if not record:
        return f"ungrouped:{int(fallback_index)}"
    for key in (
        "group_id",
        "trajectory_id",
        "parent_id",
        "group_seed",
        "group_file",
        "source_path",
    ):
        val = record.get(key)
        if val is None:
            continue
        if isinstance(val, (bytes, np.bytes_)):
            val = val.decode("utf-8", errors="replace")
        text = str(val).strip()
        if text and text not in ("", "nan", "-1", "None"):
            return f"{key}:{text}"
    return f"ungrouped:{int(fallback_index)}"
