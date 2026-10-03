"""Efield-train checkpoints: JSONL history, rich metadata, optional Orbax.

JSON ``params-best-*.json`` is still written when ``save_format`` includes
``json`` so ``--restart`` keeps working. Orbax is the preferred weight dump
(no 5 MB device-to-host JSON on every improvement).
"""

from __future__ import annotations

import json
import os
import shutil
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HISTORY_NAME = "history.jsonl"
RUN_META_NAME = "run_meta.json"
BEST_VALID_PREFIX = "best-valid-"
ORBAX_SUBDIR = "orbax"
SAVE_FORMATS = frozenset({"json", "orbax", "both"})


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def to_plain(obj: Any) -> Any:
    """JSON-safe scalars / nested dicts (no JAX import)."""
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):  # NaN / inf
            return None
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(x) for x in obj]
    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return to_plain(item())
        except Exception:
            pass
    try:
        return float(obj)
    except (TypeError, ValueError):
        return str(obj)


def params_for_save(params: Any) -> Any:
    """Drop Flax ``intermediates`` (sow artifacts) before a weight dump."""
    if isinstance(params, dict) and "intermediates" in params:
        return {k: v for k, v in params.items() if k != "intermediates"}
    return params


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(to_plain(payload), indent=2, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def append_history(ckpt_dir: Path, record: dict[str, Any]) -> Path:
    """Append one epoch row to ``history.jsonl`` and fsync."""
    path = Path(ckpt_dir) / HISTORY_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(to_plain(record), sort_keys=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return path


def read_history(ckpt_dir: Path) -> list[dict[str, Any]]:
    path = Path(ckpt_dir) / HISTORY_NAME
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            rows.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return rows


def write_run_meta(ckpt_dir: Path, meta: dict[str, Any]) -> Path:
    payload = {
        "format": "mmml-efield-run-meta-v1",
        "updated_utc": utc_now_iso(),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
        **meta,
    }
    return write_json(Path(ckpt_dir) / RUN_META_NAME, payload)


def point_symlink(link: Path, target: str | Path) -> Path:
    link = Path(link)
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(Path(target).name if Path(target).parent == link.parent else target)
    return link


def best_valid_path(ckpt_dir: Path, run_uuid: str) -> Path:
    return Path(ckpt_dir) / f"{BEST_VALID_PREFIX}{run_uuid}.json"


def write_best_valid(ckpt_dir: Path, run_uuid: str, record: dict[str, Any]) -> Path:
    path = best_valid_path(ckpt_dir, run_uuid)
    write_json(path, {"format": "mmml-efield-best-valid-v1", **record, "uuid": run_uuid})
    point_symlink(Path(ckpt_dir) / "best-valid.json", path.name)
    return path


def orbax_dir(ckpt_dir: Path, name: str) -> Path:
    return Path(ckpt_dir) / ORBAX_SUBDIR / name


def is_orbax_checkpoint(path: Path) -> bool:
    path = Path(path)
    if not path.is_dir():
        return False
    if (path / "manifest.ocdbt").exists() or (path / "_CHECKPOINT_METADATA").exists():
        return True
    if (path / "checkpoint").exists():
        return True
    return (path / "params" / "manifest.ocdbt").exists()


def save_params_orbax(dest: Path, params: Any, metadata: dict[str, Any] | None = None) -> Path:
    """Overwrite ``dest`` with an Orbax PyTree of stripped params + sidecar metadata."""
    try:
        from orbax.checkpoint import PyTreeCheckpointer
    except ImportError as exc:
        raise ImportError("orbax-checkpoint is required for --save-format orbax/both") from exc

    dest = Path(dest).resolve()
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tree = params_for_save(params)
    PyTreeCheckpointer().save(str(dest), tree)
    if metadata:
        write_json(dest / "metadata.json", metadata)
    return dest


def load_params_orbax(path: Path) -> Any:
    from orbax.checkpoint import PyTreeCheckpointer

    path = Path(path).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Orbax checkpoint directory not found: {path}")
    return PyTreeCheckpointer().restore(str(path))


def wants_json(save_format: str) -> bool:
    return save_format in {"json", "both"}


def wants_orbax(save_format: str) -> bool:
    return save_format in {"orbax", "both"}


def normalize_save_format(save_format: str | None) -> str:
    value = (save_format or "both").strip().lower()
    if value not in SAVE_FORMATS:
        raise ValueError(f"save_format must be one of {sorted(SAVE_FORMATS)}, got {save_format!r}")
    return value


def epoch_record(
    *,
    epoch: int,
    run_uuid: str,
    improved: bool,
    best_epoch: int,
    best_valid_loss: float,
    patience_counter: int,
    train_loss: float,
    valid_loss: float,
    train_energy_mae_kcal: float,
    valid_energy_mae_kcal: float,
    train_forces_mae_kcal: float,
    valid_forces_mae_kcal: float,
    train_dipole_mae: float = 0.0,
    valid_dipole_mae: float = 0.0,
    train_polar_mae: float = 0.0,
    valid_polar_mae: float = 0.0,
    train_polar_mse: float = 0.0,
    valid_polar_mse: float = 0.0,
    lr_scale: float = 1.0,
    learning_rate: float = 0.0,
    n_train: int = 0,
    n_valid: int = 0,
    batch_size: int = 0,
    polar_weight: float = 0.0,
    epoch_wall_s: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rec = {
        "epoch": int(epoch),
        "uuid": run_uuid,
        "improved": bool(improved),
        "best_epoch": int(best_epoch),
        "best_valid_loss": float(best_valid_loss),
        "patience": int(patience_counter),
        "train_loss": float(train_loss),
        "valid_loss": float(valid_loss),
        "train_energy_mae_kcal": float(train_energy_mae_kcal),
        "valid_energy_mae_kcal": float(valid_energy_mae_kcal),
        "train_forces_mae_kcal": float(train_forces_mae_kcal),
        "valid_forces_mae_kcal": float(valid_forces_mae_kcal),
        "train_dipole_mae": float(train_dipole_mae),
        "valid_dipole_mae": float(valid_dipole_mae),
        "train_polar_mae_bohr3": float(train_polar_mae),
        "valid_polar_mae_bohr3": float(valid_polar_mae),
        "train_polar_mse_bohr6": float(train_polar_mse),
        "valid_polar_mse_bohr6": float(valid_polar_mse),
        "lr_scale": float(lr_scale),
        "effective_lr": float(learning_rate) * float(lr_scale),
        "n_train": int(n_train),
        "n_valid": int(n_valid),
        "batch_size": int(batch_size),
        "polar_weight": float(polar_weight),
        "epoch_wall_s": None if epoch_wall_s is None else float(epoch_wall_s),
        "timestamp_utc": utc_now_iso(),
    }
    if extra:
        rec.update(to_plain(extra))
    return rec
