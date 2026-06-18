"""Medium PBC sparse-dimer cap validation helpers (500–2000 monomers).

Run before production MD on equilibrated CRDs::

    python scripts/validate_mlpot_sparse_dimers.py \\
      --crd path/to/mini_full_mlpot_TAG.crd \\
      --n-monomers 1000 --atoms-per-monomer 10 --box-size 40

Or audit a build directory::

    python scripts/audit_mlpot_cluster.py --output-dir artifacts/pycharmm_mlpot/...
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from mmml.interfaces.pycharmmInterface.cutoffs import DEFAULT_MM_SWITCH_ON
from mmml.interfaces.pycharmmInterface.mlpot.mlpot_batch_policy import resolve_ml_batch_size
from mmml.interfaces.pycharmmInterface.mlpot.mlpot_sparse_dimer_policy import (
    resolve_max_active_dimers,
    validate_sparse_dimer_cap,
)

# Reference monomer counts for medium PBC dense-liquid workflows.
MEDIUM_PBC_MONOMER_COUNTS: tuple[int, ...] = (500, 1000, 2000)


@dataclass(frozen=True)
class MediumPbcSizing:
    """Suggested knobs for a medium PBC cluster."""

    n_monomers: int
    max_active_dimers_cap: int
    physnet_systems_upper_bound: int
    ml_batch_size_gpu: Optional[int]
    ml_batch_size_cpu: Optional[int]
    expected_gpu_chunks: Optional[int]


def suggest_medium_pbc_sizing(n_monomers: int) -> MediumPbcSizing:
    """Return default sparse cap and batch sizing for ``n_monomers`` (PBC)."""
    n = int(n_monomers)
    cap = resolve_max_active_dimers(n, free_space=False)
    upper = n + cap
    batch = resolve_ml_batch_size(n, None)
    # Documented defaults when env/device policy differs (see mlpot_batch_policy).
    batch_gpu = 256 if n >= 40 else batch
    batch_cpu = 64 if n >= 40 else batch
    chunks = (upper + batch_gpu - 1) // batch_gpu if batch_gpu else None
    return MediumPbcSizing(
        n_monomers=n,
        max_active_dimers_cap=cap,
        physnet_systems_upper_bound=upper,
        ml_batch_size_gpu=batch_gpu,
        ml_batch_size_cpu=batch_cpu,
        expected_gpu_chunks=chunks,
    )


def lattice_positions_cubic_pbc(
    n_monomers: int,
    atoms_per_monomer: int,
    box_side_A: float,
    *,
    spacing_A: float,
    seed: int = 0,
) -> np.ndarray:
    """Place monomers on a cubic lattice with small jitter (synthetic validation)."""
    n = int(n_monomers)
    apm = int(atoms_per_monomer)
    side = float(box_side_A)
    rng = np.random.default_rng(seed)
    grid_n = int(np.ceil(n ** (1.0 / 3.0)))
    coords: list[np.ndarray] = []
    idx = 0
    for ix in range(grid_n):
        for iy in range(grid_n):
            for iz in range(grid_n):
                if idx >= n:
                    break
                base = np.array(
                    [
                        (ix + 0.5) * spacing_A,
                        (iy + 0.5) * spacing_A,
                        (iz + 0.5) * spacing_A,
                    ],
                    dtype=np.float64,
                )
                base += rng.normal(scale=0.05, size=3)
                base %= side
                monomer = base + rng.normal(scale=0.2, size=(apm, 3))
                monomer %= side
                coords.append(monomer.reshape(-1))
                idx += 1
    return np.concatenate(coords).reshape(n * apm, 3)


def validate_medium_pbc_geometry(
    positions: np.ndarray,
    n_monomers: int,
    atoms_per_monomer: int,
    *,
    box_side_A: float,
    mm_switch_on: float = DEFAULT_MM_SWITCH_ON,
    max_active_dimers: Optional[int] = None,
) -> dict:
    """Validate sparse cap for a medium PBC geometry (same as CLI script)."""
    return validate_sparse_dimer_cap(
        positions,
        int(n_monomers),
        atoms_per_monomer,
        mm_switch_on=float(mm_switch_on),
        box_side_A=float(box_side_A),
        max_active_dimers=max_active_dimers,
        free_space=False,
    )


def workflow_checklist(n_monomers: int) -> Sequence[str]:
    """Copy-paste checklist for medium PBC production prep."""
    sizing = suggest_medium_pbc_sizing(n_monomers)
    return (
        f"1. Equilibrate with mmml-charmm-mpirun.sh (MMML_MPI_NP=1).",
        f"2. validate_mlpot_sparse_dimers.py on mini_full_mlpot_*.crd "
        f"(n={n_monomers}, cap default {sizing.max_active_dimers_cap}).",
        f"3. If cap saturated: raise --ml-max-active-dimers or enlarge box.",
        f"4. Production: --ml-batch-size {sizing.ml_batch_size_gpu or 256} on GPU "
        f"(~{sizing.expected_gpu_chunks} PhysNet chunks/step upper bound).",
        f"5. Long NVE/NVT: JAX-MD after ASE/JAX-MD consistency tests pass.",
    )
