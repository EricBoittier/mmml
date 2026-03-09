"""Shared utilities for MM/ML calculator (indices, switching functions, etc.)."""

from __future__ import annotations

from itertools import combinations
from typing import Any, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import jax.numpy as jnp
    from jax import Array
except ModuleNotFoundError:
    jnp = None  # type: ignore[assignment]
    Array = Any  # type: ignore[misc,assignment]

from mmml.interfaces.pycharmmInterface.cutoffs import GAMMA_OFF, GAMMA_ON

# Re-export for convenience
__all__ = [
    "GAMMA_OFF",
    "GAMMA_ON",
    "ModelOutput",
    "debug_print",
    "dimer_permutations",
    "epsilon",
    "indices_of_monomer",
    "indices_of_pairs",
    "jax_smooth_cutoff_cosine",
    "jax_smooth_switch_linear",
    "ml_switch_simple",
    "mm_switch_simple",
    "parse_non_int",
    "_ase_cell_to_3x3",
    "_safe_den",
    "_sharpstep",
    "_smoothstep01",
]

epsilon = 10 ** (-6)


def parse_non_int(s: str) -> str:
    return "".join(ch for ch in s if ch.isalpha()).lower().capitalize()


def dimer_permutations(n_mol: int) -> List[Tuple[int, int]]:
    return list(combinations(range(n_mol), 2))


# -----------------------------------------------------------------------------
# JAX-native smooth switching utilities (avoid traced-python conditionals)
# -----------------------------------------------------------------------------
def _safe_den(x: float | Array) -> Array:
    return jnp.maximum(x, 1e-6)


def jax_smooth_switch_linear(r: Array, x0: float, x1: float) -> Array:
    t = (r - x0) / _safe_den(x1 - x0)
    return jnp.clip(t, 0.0, 1.0)


def jax_smooth_cutoff_cosine(r: Array, cutoff: float) -> Array:
    t = r / _safe_den(cutoff)
    val = 0.5 * (1.0 + jnp.cos(jnp.pi * jnp.clip(t, 0.0, 1.0)))
    return jnp.where(r < cutoff, val, 0.0)


def ml_switch_simple(r: Array, ml_cutoff: float, mm_switch_on: float) -> Array:
    """ML active at short range, tapers 1→0 over [mm_switch_on - ml_cutoff, mm_switch_on]."""
    taper_start = mm_switch_on - ml_cutoff
    t = (r - taper_start) / _safe_den(ml_cutoff)
    cosine_taper = 0.5 * (1.0 + jnp.cos(jnp.pi * jnp.clip(t, 0.0, 1.0)))
    return jnp.where(r < taper_start, 1.0, jnp.where(r < mm_switch_on, cosine_taper, 0.0))


def mm_switch_simple(r: Array, mm_switch_on: float, mm_cutoff: float) -> Array:
    """MM off at short range, ramps 0→1 over [mm_switch_on, mm_switch_on + mm_cutoff]."""
    ramp_end = mm_switch_on + mm_cutoff
    t = (r - mm_switch_on) / _safe_den(mm_cutoff)
    cosine_ramp = 0.5 * (1.0 - jnp.cos(jnp.pi * jnp.clip(t, 0.0, 1.0)))
    return jnp.where(r < mm_switch_on, 0.0, jnp.where(r < ramp_end, cosine_ramp, 1.0))


def _smoothstep01(s: Array) -> Array:
    return s * s * (3.0 - 2.0 * s)


def _sharpstep(r: Array, x0: float, x1: float, gamma: float = GAMMA_ON) -> Array:
    s = jnp.clip((r - x0) / _safe_den(x1 - x0), 0.0, 1.0)
    s = s ** gamma
    return _smoothstep01(s)


def indices_of_pairs(
    a: int,
    b: int,
    n_atoms: int = 5,
    n_mol: int = 20,
    monomer_offsets: Optional[np.ndarray] = None,
    atoms_per_monomer_list: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Return concatenated atom indices for monomers *a* and *b* (1-indexed).

    When *monomer_offsets* and *atoms_per_monomer_list* are provided the function
    supports heterogeneous monomer sizes; otherwise it falls back to the legacy
    uniform-size behaviour.
    """
    assert a < b, "by convention, res a must have a smaller index than res b"
    assert a >= 1, "res indices can't start from 1"
    assert b >= 1, "res indices can't start from 1"
    assert a != b, "pairs can't contain same residue"
    if monomer_offsets is not None and atoms_per_monomer_list is not None:
        off_a = monomer_offsets[a - 1]
        off_b = monomer_offsets[b - 1]
        n_a = atoms_per_monomer_list[a - 1]
        n_b = atoms_per_monomer_list[b - 1]
        return np.concatenate([
            np.arange(off_a, off_a + n_a),
            np.arange(off_b, off_b + n_b),
        ])
    return np.concatenate(
        [
            np.arange(0, n_atoms, 1) + (a - 1) * n_atoms,
            np.arange(0, n_atoms, 1) + (b - 1) * n_atoms,
        ]
    )


def indices_of_monomer(
    a: int,
    n_atoms: int = 5,
    n_mol: int = 20,
    monomer_offsets: Optional[np.ndarray] = None,
    atoms_per_monomer_list: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Return atom indices for monomer *a* (1-indexed).

    When *monomer_offsets* and *atoms_per_monomer_list* are provided the function
    supports heterogeneous monomer sizes.
    """
    assert a < (n_mol + 1), "monomer index outside total n molecules"
    if monomer_offsets is not None and atoms_per_monomer_list is not None:
        off = monomer_offsets[a - 1]
        n = atoms_per_monomer_list[a - 1]
        return np.arange(off, off + n)
    return np.arange(0, n_atoms, 1) + (a - 1) * n_atoms


class ModelOutput(NamedTuple):
    energy: Array  # Shape: (,), total energy in eV
    forces: Array  # Shape: (n_atoms, 3), forces in eV/Å
    dH: Array  # Shape: (,), total interaction energy in eV
    internal_E: Array  # Shape: (,) total internal energy in eV
    internal_F: Array
    mm_E: Array
    mm_F: Array
    ml_2b_E: Array
    ml_2b_F: Array


def debug_print(debug: bool, msg: str, *args: Any, **kwargs: Any) -> None:
    """Helper function for conditional debug printing."""
    if debug:
        print(msg)
        for arg in args:
            pass
        try:
            for name, value in kwargs.items():
                print(f"{name}: {value.shape}")
        except Exception:
            pass


def _ase_cell_to_3x3(atoms: Any) -> np.ndarray | None:
    """Extract 3x3 cell matrix from ASE atoms. Returns None if cell is invalid/empty."""
    try:
        import ase  # noqa: F401
    except ImportError:
        return None
    if atoms is None:
        return None
    try:
        cell = atoms.get_cell()
        if cell is None:
            return None
        arr = np.asarray(cell, dtype=np.float64)
        if arr.shape != (3, 3):
            return None
        lengths = np.linalg.norm(arr, axis=1)
        if np.any(lengths < 1e-6):
            return None
        return arr
    except Exception:
        return None
