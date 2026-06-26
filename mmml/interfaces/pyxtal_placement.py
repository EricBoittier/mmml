"""PyXtal-backed symmetry-aware structure building (optional ``mmml[chem]``).

PyXtal generates crystal structures with space-group symmetry. This module
exports ASE :class:`ase.Atoms` with periodic boundary conditions for downstream
MMML workflows (ASE/JAX-MD optimization, NPZ handoff, PyCHARMM setup).

Install PyXtal with::

    uv sync --extra chem
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PathLike = str | Path


def have_pyxtal() -> bool:
    """Return True when the optional ``pyxtal`` package is importable."""
    try:
        import pyxtal  # noqa: F401

        return True
    except ImportError:
        return False


def require_pyxtal() -> None:
    """Raise ``ImportError`` with install instructions when PyXtal is missing."""
    if not have_pyxtal():
        raise ImportError(
            "PyXtal is not installed. Install with: uv sync --extra chem  "
            "(or pip install 'mmml[chem]')."
        )


@dataclass(frozen=True)
class MolecularCrystalBuildRequest:
    """Inputs for a random molecular crystal build via PyXtal."""

    molecules: list[str]
    stoichiometry: list[int]
    dimension: int = 3
    space_group: int = 14
    factor: float = 1.0
    seed: int | None = None
    molecular: bool = True
    max_attempts: int = 10
    resort: bool = True
    center_only: bool = False

    def __post_init__(self) -> None:
        if len(self.molecules) != len(self.stoichiometry):
            raise ValueError(
                "molecules and stoichiometry must have the same length "
                f"({len(self.molecules)} vs {len(self.stoichiometry)})"
            )
        if not self.molecules:
            raise ValueError("at least one molecule specification is required")
        if any(int(z) <= 0 for z in self.stoichiometry):
            raise ValueError(f"stoichiometry entries must be positive: {self.stoichiometry}")
        if int(self.dimension) not in (0, 1, 2, 3):
            raise ValueError(f"dimension must be 0–3, got {self.dimension}")
        if int(self.space_group) <= 0:
            raise ValueError(f"space_group must be positive, got {self.space_group}")


@dataclass
class MolecularCrystalBuildResult:
    """PyXtal crystal plus exported ASE structure."""

    crystal: Any
    atoms: Any
    attempts: int = 1
    space_group: int = 0
    formula: str = ""


def _import_pyxtal():
    require_pyxtal()
    from pyxtal import pyxtal

    return pyxtal


def load_pyxtal_molecule(spec: str):
    """Load a PyXtal molecule from SMILES, formula, or a structure file path."""
    require_pyxtal()
    from pyxtal.molecule import pyxtal_molecule

    return pyxtal_molecule(spec)


def build_molecular_crystal_random(
    request: MolecularCrystalBuildRequest,
) -> MolecularCrystalBuildResult:
    """Generate a random molecular crystal; retry until PyXtal succeeds."""
    pyxtal_cls = _import_pyxtal()
    rng = np.random.default_rng(request.seed)
    last_error: Exception | None = None

    for attempt in range(1, max(1, int(request.max_attempts)) + 1):
        crystal = pyxtal_cls(molecular=bool(request.molecular))
        try:
            trial_seed = None if request.seed is None else int(rng.integers(0, 2**31 - 1))
            crystal.from_random(
                int(request.dimension),
                int(request.space_group),
                list(request.molecules),
                [int(z) for z in request.stoichiometry],
                float(request.factor),
                trial_seed,
            )
        except Exception as exc:
            last_error = exc
            continue

        atoms = pyxtal_to_ase(
            crystal,
            resort=bool(request.resort),
            center_only=bool(request.center_only),
        )
        sg = int(getattr(getattr(crystal, "group", None), "number", request.space_group))
        formula = str(getattr(crystal, "formula", "") or "")
        return MolecularCrystalBuildResult(
            crystal=crystal,
            atoms=atoms,
            attempts=attempt,
            space_group=sg,
            formula=formula,
        )

    raise RuntimeError(
        f"PyXtal failed to build a molecular crystal after "
        f"{max(1, int(request.max_attempts))} attempt(s) "
        f"(dim={request.dimension}, spg={request.space_group}, "
        f"molecules={request.molecules}, Z={request.stoichiometry})"
    ) from last_error


def pyxtal_to_ase(
    crystal: Any,
    *,
    resort: bool = True,
    center_only: bool = False,
) -> Any:
    """Convert a PyXtal object to ``ase.Atoms`` (preserves cell and PBC)."""
    if not getattr(crystal, "valid", True):
        raise ValueError("PyXtal crystal is not valid; cannot export to ASE")
    if not hasattr(crystal, "to_ase"):
        raise TypeError(f"Expected PyXtal crystal with to_ase(); got {type(crystal)}")
    return crystal.to_ase(resort=bool(resort), center_only=bool(center_only))


def ase_supercell(
    atoms: Any,
    reps: Sequence[int],
) -> Any:
    """Build an ASE supercell along ``reps`` (e.g. ``(2, 2, 2)``)."""
    from ase.build import make_supercell

    if len(reps) != 3:
        raise ValueError(f"reps must have length 3, got {reps}")
    matrix = np.diag([int(r) for r in reps])
    return make_supercell(atoms, matrix)


def write_ase_structure(
    atoms: Any,
    path: PathLike,
    *,
    format: str | None = None,
) -> Path:
    """Write ``atoms`` using ASE I/O."""
    from ase.io import write

    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    write(str(out), atoms, format=format)
    return out


def atoms_to_reference_npz(
    atoms: Any,
    path: PathLike,
    *,
    label: str = "pyxtal",
) -> Path:
    """Write a minimal NPZ for ``md_handoff`` / ``build_cluster_from_reference_npz``."""
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int32)
    positions = np.asarray(atoms.get_positions(), dtype=np.float64)
    payload: dict[str, Any] = {
        "R": positions,
        "Z": numbers,
        "source": np.array(label),
    }
    if atoms.pbc.any():
        payload["cell"] = np.asarray(atoms.get_cell(), dtype=np.float64)
        payload["pbc"] = np.asarray(atoms.pbc, dtype=bool)
    np.savez_compressed(out, **payload)
    return out


def parse_supercell_reps(text: str) -> tuple[int, int, int]:
    """Parse ``2,2,2`` or ``2x2x2`` into a supercell repetition tuple."""
    raw = str(text).strip().lower().replace("x", ",")
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(
            f"supercell must have three integers (e.g. 2,2,2 or 2x2x2), got {text!r}"
        )
    reps = tuple(int(p) for p in parts)
    if any(r <= 0 for r in reps):
        raise ValueError(f"supercell repetitions must be positive: {reps}")
    return reps  # type: ignore[return-value]


def parse_stoichiometry(
    molecules: Sequence[str],
    stoichiometry: Sequence[int] | None,
    z_values: Sequence[int] | None,
) -> list[int]:
    """Resolve per-molecule Z from ``--stoichiometry`` or repeated ``--z``."""
    n = len(molecules)
    if stoichiometry is not None:
        vals = [int(z) for z in stoichiometry]
        if len(vals) != n:
            raise ValueError(
                f"--stoichiometry length {len(vals)} must match molecule count {n}"
            )
        return vals
    if z_values is not None:
        vals = [int(z) for z in z_values]
        if len(vals) == 1 and n > 1:
            return vals * n
        if len(vals) != n:
            raise ValueError(f"--z length {len(vals)} must match molecule count {n}")
        return vals
    return [2] * n
