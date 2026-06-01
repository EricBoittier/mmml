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


def build_ase_cluster(
    residue: str,
    n_molecules: int,
    spacing: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build CGenFF cluster in CHARMM and return (Z, positions) without keeping CHARMM state."""
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401 — sets CHARMM env
    from mmml.cli.run.md_pbc_suite.cluster import _build_psf_ordered_cluster

    z, r = _build_psf_ordered_cluster(residue.upper(), n_molecules, spacing)
    return z, r


def setup_charmm_nbonds() -> None:
    import pycharmm

    nbonds = """
    nbonds atom cutnb 14.0 ctofnb 12.0 ctonnb 10.0 -
    vswitch -
    inbfrq -1 imgfrq -1
    """
    pycharmm.lingo.charmm_script(nbonds)


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
