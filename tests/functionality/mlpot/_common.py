"""Shared helpers for MLpot exploration scripts."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Tuple

import numpy as np

# Repo root: tests/functionality/mlpot -> parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Defaults aligned with DESdimers / acetone cluster tests
DEFAULT_RESIDUE = "ACO"
DEFAULT_N_MOLECULES = 2
DEFAULT_SPACING = 4.0


def add_charmm_output_args(parser: argparse.ArgumentParser) -> None:
    """CLI flags for CHARMM console verbosity (scripts 04–05)."""
    group = parser.add_argument_group("CHARMM console output")
    group.add_argument(
        "--prnlev",
        type=int,
        default=5,
        help="CHARMM PRNLev (0=quiet, 5=verbose; default: 5)",
    )
    group.add_argument(
        "--warnlev",
        type=int,
        default=5,
        help="CHARMM WRNLev (default: 5)",
    )
    group.add_argument(
        "--bomlev",
        type=int,
        default=0,
        help="CHARMM BOMBlev (default: 0 = stop on errors)",
    )
    group.add_argument(
        "--nprint",
        type=int,
        default=1,
        help="Print energy/status every N minimization or dynamics steps (default: 1)",
    )
    group.add_argument(
        "--quiet",
        action="store_true",
        help="Shortcut for --prnlev 0 --warnlev 0 and coarse nprint",
    )


def apply_charmm_output_from_args(args: argparse.Namespace) -> int:
    """Apply PRNLev/WRNLev from argparse; return effective ``nprint``."""
    from mmml.interfaces.pycharmmInterface.mlpot.setup import apply_charmm_verbosity

    if getattr(args, "quiet", False):
        apply_charmm_verbosity(prnlev=0, warnlev=0, bomlev=args.bomlev)
        nstep = getattr(args, "nstep", 100)
        return max(1, nstep)
    apply_charmm_verbosity(
        prnlev=args.prnlev,
        warnlev=args.warnlev,
        bomlev=args.bomlev,
    )
    return max(1, int(args.nprint))


def add_cluster_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--residue",
        default=DEFAULT_RESIDUE,
        help="CHARMM residue name for CGenFF cluster (default: ACO)",
    )
    parser.add_argument(
        "--n-molecules",
        type=int,
        default=DEFAULT_N_MOLECULES,
        help="Number of identical residues in the cluster",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=DEFAULT_SPACING,
        help="Spacing (Å) when placing multiple residues",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint (.json or Orbax root). Default: MMML_CKPT or repo ckpts.",
    )


def resolve_checkpoint(explicit: Path | None = None) -> Path:
    if explicit is not None:
        p = explicit.expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p

    ckpt_env = os.environ.get("MMML_CKPT")
    candidates: list[Path] = []
    if ckpt_env:
        candidates.append(Path(ckpt_env))
    candidates.extend(
        [
            PROJECT_ROOT / "examples/ckpts_json/DESdimers_params.json",
            PROJECT_ROOT / "examples/ckpts_json",
            PROJECT_ROOT / "ckpts_json/DESdimers_params.json",
            PROJECT_ROOT / "mmml/models/physnetjax/ckpts/DESdimers",
            PROJECT_ROOT / "mmml/models/physnetjax/ckpts",
        ]
    )
    for ckpt in candidates:
        if ckpt.exists():
            return ckpt.resolve()
    raise FileNotFoundError(
        "No checkpoint found. Set MMML_CKPT or pass --checkpoint."
    )


def load_physnet_for_cluster(
    checkpoint: Path,
    n_atoms: int,
) -> Tuple[Any, Any]:
    """Return (params, model) for ``n_atoms`` atoms."""
    from mmml.cli.base import load_physnet_params_and_ef_model, resolve_checkpoint_paths

    if checkpoint.is_file() and checkpoint.suffix == ".json":
        return load_physnet_params_and_ef_model(checkpoint, natoms=n_atoms)

    _, epoch_dir = resolve_checkpoint_paths(checkpoint)
    from mmml.models.physnetjax.physnetjax.restart.restart import get_params_model

    params, model = get_params_model(str(epoch_dir), natoms=n_atoms)
    return params, model


def validate_cluster_geometry(
    positions: np.ndarray,
    *,
    min_axis_span: float = 0.3,
    min_monomer_extent: float = 1.5,
    n_molecules: int | None = None,
) -> dict[str, float]:
    """Raise if coordinates look collapsed or non-physical; else return summary stats."""
    r = np.asarray(positions, dtype=float)
    if r.ndim != 2 or r.shape[1] != 3:
        raise ValueError(f"positions must be (N, 3), got {r.shape}")
    span = r.max(axis=0) - r.min(axis=0)
    if float(span[1]) < min_axis_span or float(span[2]) < min_axis_span:
        raise ValueError(
            f"Cluster is nearly planar/collinear (spans Å x={span[0]:.3f} y={span[1]:.3f} z={span[2]:.3f})"
        )
    if n_molecules is not None and n_molecules > 0 and r.shape[0] % n_molecules == 0:
        n_per = r.shape[0] // n_molecules
        coms = []
        for i in range(n_molecules):
            chunk = r[i * n_per : (i + 1) * n_per]
            extent = float(np.linalg.norm(chunk.max(axis=0) - chunk.min(axis=0)))
            if extent < min_monomer_extent:
                raise ValueError(
                    f"Monomer {i + 1} extent {extent:.3f} Å < {min_monomer_extent} Å (likely bad template/ic.build)"
                )
            coms.append(chunk.mean(axis=0))
        if len(coms) > 1:
            com_sep = float(np.linalg.norm(coms[1] - coms[0]))
        else:
            com_sep = 0.0
    else:
        com_sep = float("nan")
    return {
        "span_x": float(span[0]),
        "span_y": float(span[1]),
        "span_z": float(span[2]),
        "com_sep_01": com_sep,
    }


def build_ase_cluster(
    residue: str,
    n_molecules: int,
    spacing: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build CGenFF cluster in CHARMM and return (Z, positions) without keeping CHARMM state."""
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401 — sets CHARMM env
    from mmml.cli.run.md_pbc_suite.cluster import _build_psf_ordered_cluster

    from mmml.interfaces.pycharmmInterface.mlpot.setup import sync_charmm_positions

    z, r = _build_psf_ordered_cluster(residue.upper(), n_molecules, spacing)
    sync_charmm_positions(r)
    validate_cluster_geometry(r, n_molecules=n_molecules)
    return z, r


def build_acetone_dimer_cluster(spacing: float = DEFAULT_SPACING) -> Tuple[np.ndarray, np.ndarray]:
    """20-atom acetone dimer (ACO × 2) with bundled 3D monomer template."""
    z, r = build_ase_cluster(DEFAULT_RESIDUE, DEFAULT_N_MOLECULES, spacing)
    if len(z) != 20:
        raise RuntimeError(f"Expected 20 atoms for ACO dimer, got {len(z)}")
    return z, r


def print_cluster_geometry_summary(positions: np.ndarray, n_molecules: int) -> None:
    stats = validate_cluster_geometry(positions, n_molecules=n_molecules)
    print(
        "Cluster geometry OK:"
        f" spans (Å) x={stats['span_x']:.2f} y={stats['span_y']:.2f} z={stats['span_z']:.2f}"
        f" | COM separation (1→2) = {stats['com_sep_01']:.2f} Å"
    )


def all_atom_selection():
    """PyCHARMM selection for all atoms (for MLpot on full cluster)."""
    from mmml.interfaces.pycharmmInterface.mlpot.setup import select_all_atoms

    return select_all_atoms()


def setup_charmm_nbonds() -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.setup import setup_default_nbonds

    setup_default_nbonds()


def check_mlpot_symbols() -> list[str]:
    """Return missing MLpot symbols on libcharmm, if any."""
    import pycharmm.lib as lib

    required = ("mlpot_set_func", "mlpot_set_properties", "mlpot_unset")
    missing = []
    for name in required:
        if not hasattr(lib.charmm, name):
            missing.append(name)
    return missing


def charmm_energy_row() -> dict[str, float]:
    import pycharmm.energy as energy

    df = energy.get_energy()
    row = df.iloc[0].to_dict()
    out: dict[str, float] = {}
    for key, value in row.items():
        if isinstance(value, (int, float, np.floating)):
            out[str(key)] = float(value)
    return out


def print_header(title: str) -> None:
    bar = "=" * len(title)
    print(f"\n{bar}\n{title}\n{bar}")
