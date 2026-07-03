"""Shared helpers for live PyCHARMM / MLpot optimizer and dynamics tests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]

DCM1_CKPT = Path(
    "/mmhome/boittier/home/mmml_tutorial/acodcm/ckpts/dcm1/dcm1_params.json"
)
TIMESTEP_PS = 0.00025


def max_displacement(a: np.ndarray, b: np.ndarray) -> float:
    """Maximum Cartesian displacement (Å) between two ``(N, 3)`` position sets."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return float(np.max(np.linalg.norm(a - b, axis=1)))


def subset_positions(pos: np.ndarray, charmm_indexes: Sequence[int]) -> np.ndarray:
    """Extract positions for 1-based CHARMM atom indices."""
    idx = np.asarray(charmm_indexes, dtype=int) - 1
    return np.asarray(pos, dtype=float)[idx]


def resolve_live_checkpoint() -> Path | None:
    """Best-effort PhysNet JSON / bundle for live MLpot tests."""
    candidates: list[Path] = []
    env = os.environ.get("MMML_CKPT", "").strip()
    if env:
        candidates.append(Path(env).expanduser())
    candidates.extend(
        [
            DCM1_CKPT,
            REPO_ROOT / "examples/ckpts_json/DESdimers_params.json",
            REPO_ROOT / "mmml/models/physnetjax/ckpts/DESdimers",
        ]
    )
    for path in candidates:
        if path.exists():
            return path.resolve()
    return None


def can_import_pycharmm() -> bool:
    try:
        import pycharmm  # noqa: F401

        return True
    except Exception:
        return False


def setup_aco_mlpot(
    ckpt: Path,
    *,
    n_molecules: int = 2,
    spacing: float = 4.0,
) -> tuple[Any, np.ndarray, np.ndarray, int]:
    """Build ACO cluster, register MLpot; return ``(ctx, z, r, n_atoms)``."""
    import ase

    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401

    from mmml.interfaces.pycharmmInterface.mlpot import (
        load_physnet_mlpot_bundle,
        register_mlpot,
        select_all_atoms,
        setup_default_nbonds,
        sync_charmm_positions,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import build_ase_cluster

    z, r = build_ase_cluster("ACO", n_molecules, spacing)
    n_atoms = len(z)
    setup_default_nbonds()
    sync_charmm_positions(r)

    atoms = ase.Atoms(numbers=z, positions=r)
    _, _, pyCModel = load_physnet_mlpot_bundle(ckpt, n_atoms, atoms)
    ctx = register_mlpot(pyCModel, z, select_all_atoms())
    return ctx, z, r, n_atoms


def setup_dcm_mlpot(
    ckpt: Path,
    *,
    n_molecules: int = 2,
    spacing: float = 4.0,
) -> tuple[Any, np.ndarray, np.ndarray, int]:
    """Build DCM cluster, register MLpot; return ``(ctx, z, r, n_atoms)``."""
    import ase

    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401

    from mmml.interfaces.pycharmmInterface.mlpot import (
        load_physnet_mlpot_bundle,
        register_mlpot,
        select_all_atoms,
        setup_default_nbonds,
        sync_charmm_positions,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import build_ase_cluster

    z, r = build_ase_cluster("DCM", n_molecules, spacing)
    n_atoms = len(z)
    setup_default_nbonds()
    sync_charmm_positions(r)

    atoms = ase.Atoms(numbers=z, positions=r)
    _, _, pyCModel = load_physnet_mlpot_bundle(ckpt, n_atoms, atoms)
    ctx = register_mlpot(pyCModel, z, select_all_atoms())
    return ctx, z, r, n_atoms


def positions_for_resids(resids: Sequence[int]) -> np.ndarray:
    """Current CHARMM coordinates for the given residue IDs."""
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        get_charmm_positions_array,
        select_by_resids,
    )

    pos = get_charmm_positions_array()
    indexes = select_by_resids(list(resids)).get_atom_indexes()
    return subset_positions(pos, indexes)


def translate_resid_and_sync(resids: Sequence[int], delta: Sequence[float]) -> None:
    """Translate one or more monomers in place and push coordinates to CHARMM."""
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        get_charmm_positions_array,
        select_by_resids,
        sync_charmm_positions,
    )

    pos = get_charmm_positions_array().copy()
    indexes = np.asarray(select_by_resids(list(resids)).get_atom_indexes(), dtype=int) - 1
    pos[indexes] += np.asarray(delta, dtype=float)
    sync_charmm_positions(pos)


def run_short_sd(
    *,
    nstep: int = 25,
    nprint: int | None = None,
) -> None:
    """CHARMM SD with list rebuild disabled (MLpot-safe mini kwargs)."""
    import pycharmm.minimize as minimize

    minimize.run_sd(
        nstep=int(nstep),
        nprint=int(nstep if nprint is None else nprint),
        inbfrq=0,
        ihbfrq=0,
    )
