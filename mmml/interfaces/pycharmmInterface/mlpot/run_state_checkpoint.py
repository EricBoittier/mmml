"""Save post-workflow atomic state (Orbax or NPZ); PhysNet weights stay in ``--checkpoint``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _charmm_velocities_array() -> np.ndarray | None:
    """Best-effort CHARMM main-set velocities (N, 3) or None."""
    try:
        import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
        import pycharmm.coor as coor

        v = coor.get_velocity()
        if v is None:
            return None
        cols = [c for c in ("vx", "vy", "vz") if c in v.columns]
        if len(cols) != 3:
            return None
        return v[cols].to_numpy(dtype=float)
    except Exception:
        return None


def build_run_state_tree(
    *,
    positions: np.ndarray,
    atomic_numbers: np.ndarray,
    metadata: dict[str, Any],
    velocities: np.ndarray | None = None,
) -> dict[str, Any]:
    """PyTree for Orbax / NPZ: geometry + run metadata (not ML model params)."""
    if velocities is None:
        velocities = _charmm_velocities_array()
    tree: dict[str, Any] = {
        "positions": np.asarray(positions, dtype=np.float64),
        "atomic_numbers": np.asarray(atomic_numbers, dtype=np.int32),
        "metadata": metadata,
    }
    if velocities is not None:
        tree["velocities"] = np.asarray(velocities, dtype=np.float64)
    return tree


def save_run_state(
    path: Path,
    tree: dict[str, Any],
    *,
    quiet: bool = False,
) -> Path:
    """Save run state with Orbax when available, else ``run_state.npz``."""
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    try:
        import orbax.checkpoint as ocp

        ocp.PyTreeCheckpointer().save(path / "orbax", tree)
        (path / "format.txt").write_text("orbax\n", encoding="utf-8")
        if not quiet:
            print(f"Run state saved (orbax): {path / 'orbax'}", flush=True)
        return path / "orbax"
    except ImportError:
        np.savez(path / "run_state.npz", **tree)
        (path / "format.txt").write_text("npz\n", encoding="utf-8")
        (path / "metadata.json").write_text(
            json.dumps(tree.get("metadata", {}), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if not quiet:
            print(f"Run state saved (npz): {path / 'run_state.npz'}", flush=True)
        return path / "run_state.npz"


def maybe_save_run_state_from_workflow(
    args: Any,
    *,
    positions: np.ndarray,
    atomic_numbers: np.ndarray,
    out_dir: Path,
    tag: str,
    stages_completed: list[str],
    last_restart: Path | None,
    last_trajectory: Path | None,
) -> None:
    if not bool(getattr(args, "save_run_state", False)):
        return
    ckpt = getattr(args, "checkpoint", None)
    meta = {
        "tag": tag,
        "stages_completed": list(stages_completed),
        "composition": getattr(args, "composition", None),
        "temperature": float(getattr(args, "temperature", 300.0)),
        "physnet_checkpoint": str(Path(ckpt).resolve()) if ckpt else None,
        "last_restart": str(last_restart) if last_restart else None,
        "last_trajectory": str(last_trajectory) if last_trajectory else None,
        "note": (
            "positions/velocities only; reload PhysNet from physnet_checkpoint. "
            "CHARMM PSF/restart are separate artifacts under output-dir."
        ),
    }
    save_dir = Path(getattr(args, "run_state_dir", None) or (out_dir / "run_state"))
    tree = build_run_state_tree(
        positions=positions,
        atomic_numbers=atomic_numbers,
        metadata=meta,
    )
    save_run_state(save_dir, tree, quiet=bool(getattr(args, "quiet", False)))
