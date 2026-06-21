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


def load_run_state_tree(path: Path) -> dict[str, Any]:
    """Load geometry handoff tree from Orbax dir or ``run_state.npz``."""
    root = Path(path).expanduser().resolve()
    if (root / "format.txt").is_file():
        fmt = (root / "format.txt").read_text(encoding="utf-8").strip().lower()
        if fmt == "orbax" and (root / "orbax").is_dir():
            import orbax.checkpoint as ocp

            return ocp.PyTreeCheckpointer().restore(root / "orbax")
    npz_path = root / "run_state.npz"
    if npz_path.is_file():
        loaded = np.load(npz_path, allow_pickle=True)
        tree: dict[str, Any] = {}
        for key in loaded.files:
            if key == "metadata":
                raw = loaded[key]
                if isinstance(raw, np.ndarray) and raw.dtype == object and raw.shape == ():
                    tree[key] = raw.item()
                else:
                    tree[key] = raw
            else:
                tree[key] = loaded[key]
        if "metadata" not in tree and (root / "metadata.json").is_file():
            tree["metadata"] = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        return tree
    if root.suffix.lower() == ".npz" and root.is_file():
        loaded = np.load(root, allow_pickle=True)
        return {key: loaded[key] for key in loaded.files}
    raise FileNotFoundError(f"No run state at {root}")


def save_overlap_run_state(
    directory: Path,
    *,
    step: int,
    segment: str,
    chunk_index: int,
    positions: np.ndarray,
    velocities: np.ndarray | None = None,
    box: np.ndarray | None = None,
    restart_path: Path | None = None,
    quiet: bool = False,
) -> Path:
    """Persist overlap-chunk geometry sidecar (Orbax or NPZ)."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "step": int(step),
        "segment": str(segment),
        "chunk_index": int(chunk_index),
        "restart_path": str(restart_path) if restart_path else None,
    }
    if box is not None:
        metadata["box"] = np.asarray(box, dtype=np.float64).tolist()
    tree = build_run_state_tree(
        positions=positions,
        atomic_numbers=np.zeros(len(positions), dtype=np.int32),
        metadata=metadata,
        velocities=velocities,
    )
    chunk_dir = directory / f"chunk_{int(chunk_index):04d}"
    return save_run_state(chunk_dir, tree, quiet=quiet)


def load_overlap_run_state(directory: Path) -> dict[str, Any]:
    """Load the newest overlap run-state sidecar under ``directory``."""
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"No overlap run state directory at {root}")
    candidates = sorted(root.glob("chunk_*"))
    if not candidates:
        raise FileNotFoundError(f"No overlap chunk run state under {root}")
    return load_run_state_tree(candidates[-1])


def restore_positions_from_overlap_run_state(
    directory: Path,
    *,
    label: str = "overlap run-state recovery",
) -> bool:
    """Reload CHARMM positions from the latest overlap sidecar."""
    try:
        tree = load_overlap_run_state(directory)
    except FileNotFoundError:
        return False
    positions = tree.get("positions")
    if positions is None:
        return False
    from mmml.interfaces.pycharmmInterface.mlpot.setup import sync_charmm_positions

    sync_charmm_positions(np.asarray(positions, dtype=np.float64))
    velocities = tree.get("velocities")
    if velocities is not None:
        try:
            import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
            import pycharmm.coor as coor

            v = np.asarray(velocities, dtype=np.float64)
            coor.set_velocity(v[:, 0], v[:, 1], v[:, 2])
        except Exception:
            pass
    print(f"{label}: restored positions from {directory}", flush=True)
    return True


def maybe_save_overlap_run_state(
    directory: Path | None,
    *,
    step: int,
    segment: str,
    chunk_index: int,
    restart_path: Path | None = None,
    quiet: bool = True,
) -> Path | None:
    if directory is None:
        return None
    from mmml.interfaces.pycharmmInterface.mlpot.setup import get_charmm_positions_array

    positions = get_charmm_positions_array()
    box = None
    try:
        from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import _read_charmm_box_sides_A

        sides = _read_charmm_box_sides_A()
        if sides is not None:
            box = np.asarray(sides, dtype=np.float64)
    except Exception:
        box = None
    return save_overlap_run_state(
        Path(directory),
        step=step,
        segment=segment,
        chunk_index=chunk_index,
        positions=positions,
        box=box,
        restart_path=restart_path,
        quiet=quiet,
    )


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
