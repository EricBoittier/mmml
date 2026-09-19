"""Learning-curve train subsets for efield polar jobs.

Keeps one shared valid NPZ and writes a smaller train NPZ. No downloads.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from mmml.data.spice_alpha import write_physnet_npz

DEFAULT_LC_FRACS: tuple[float, ...] = (0.01, 0.03, 0.10, 0.30)
TRAIN_NAME = "energies_forces_dipoles_train.npz"
VALID_NAME = "energies_forces_dipoles_valid.npz"


def frac_tag(frac: float) -> str:
    """``0.03`` → ``p03`` (directory / job suffix)."""
    pct = frac * 100.0
    if abs(pct - round(pct)) < 1e-9:
        return f"p{int(round(pct)):02d}"
    return f"p{pct:.2f}".replace(".", "")


def n_train_for_frac(n_full: int, frac: float, batch_size: int) -> int:
    """Floor ``frac * n_full`` onto a multiple of ``batch_size`` (drop_last)."""
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"frac must be in (0, 1], got {frac}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    n = int(n_full * frac)
    n = (n // batch_size) * batch_size
    if n < batch_size:
        raise ValueError(
            f"frac={frac} on n_full={n_full} yields n_train={n} < BATCH_SIZE={batch_size}"
        )
    return n


def _frame_count(data: Mapping[str, Any]) -> int:
    if "E" in data:
        return int(np.asarray(data["E"]).reshape(-1).shape[0])
    return int(np.asarray(data["R"]).shape[0])


def subsample_frame_npz(
    src: Path | str,
    dest: Path | str,
    *,
    n_keep: int,
    seed: int = 0,
) -> dict[str, Any]:
    """Write ``n_keep`` random frames (fixed seed). Non-frame keys are copied."""
    src_path = Path(src)
    raw = np.load(src_path, allow_pickle=True)
    data = {k: raw[k] for k in raw.files}
    n = _frame_count(data)
    if n_keep > n:
        raise ValueError(f"n_keep={n_keep} > n={n} in {src_path}")
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=n_keep, replace=False))
    part: dict[str, Any] = {}
    for key, val in data.items():
        arr = np.asarray(val)
        part[key] = arr[idx] if arr.shape[:1] == (n,) else arr
    write_physnet_npz(part, dest)
    return {"n_full": n, "n_keep": n_keep, "seed": seed, "indices": idx.tolist()}


def prepare_lc_split(
    src_splits: Path | str,
    dest_splits: Path | str,
    *,
    frac: float,
    batch_size: int = 4,
    seed: int = 0,
) -> dict[str, Any]:
    """Subsample train; symlink (or copy) the original valid NPZ."""
    src = Path(src_splits)
    dest = Path(dest_splits)
    train_src = src / TRAIN_NAME
    valid_src = src / VALID_NAME
    if not train_src.is_file() or not valid_src.is_file():
        raise FileNotFoundError(f"need {TRAIN_NAME} and {VALID_NAME} in {src}")
    raw = np.load(train_src, allow_pickle=True)
    n_full = _frame_count({k: raw[k] for k in raw.files})
    n_keep = n_train_for_frac(n_full, frac, batch_size)
    dest.mkdir(parents=True, exist_ok=True)
    train_dest = dest / TRAIN_NAME
    meta = subsample_frame_npz(train_src, train_dest, n_keep=n_keep, seed=seed)
    valid_dest = dest / VALID_NAME
    if valid_dest.exists() or valid_dest.is_symlink():
        valid_dest.unlink()
    try:
        os.symlink(valid_src.resolve(), valid_dest)
        valid_how = "symlink"
    except OSError:
        import shutil

        shutil.copy2(valid_src, valid_dest)
        valid_how = "copy"
    payload = {
        "format": "mmml-efield-lc-split-v1",
        "frac": frac,
        "tag": frac_tag(frac),
        "batch_size": batch_size,
        "seed": seed,
        "n_full_train": n_full,
        "n_train": n_keep,
        "train_npz": str(train_dest),
        "valid_npz": str(valid_dest),
        "valid_from": str(valid_src.resolve()),
        "valid_how": valid_how,
        **{k: meta[k] for k in ("n_keep", "indices") if k in meta},
    }
    (dest / "lc_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def prepare_lc_grid(
    src_splits: Path | str,
    out_root: Path | str,
    fracs: Sequence[float] = DEFAULT_LC_FRACS,
    *,
    batch_size: int = 4,
    seed: int = 0,
) -> list[dict[str, Any]]:
    rows = []
    root = Path(out_root)
    for frac in fracs:
        dest = root / f"splits_{frac_tag(frac)}"
        rows.append(
            prepare_lc_split(src_splits, dest, frac=frac, batch_size=batch_size, seed=seed)
        )
    return rows


def parse_fracs(text: str) -> list[float]:
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise ValueError("no fractions in FRACTIONS")
    return out


def last_history_row(ckpt_dir: Path | str) -> dict[str, Any] | None:
    path = Path(ckpt_dir) / "history.jsonl"
    if not path.is_file():
        return None
    last = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            last = json.loads(line)
    return last


def summarize_lc_runs(ckpt_dirs: Iterable[Path | str]) -> list[dict[str, Any]]:
    """One row per ckpt dir from ``history.jsonl`` + optional ``lc_manifest.json``."""
    rows = []
    for raw in ckpt_dirs:
        ckpt = Path(raw)
        rec = last_history_row(ckpt) or {}
        manifest = {}
        for cand in (ckpt / "lc_manifest.json", ckpt.parent / "lc_manifest.json"):
            if cand.is_file():
                manifest = json.loads(cand.read_text(encoding="utf-8"))
                break
        rows.append(
            {
                "ckpt": str(ckpt),
                "n_train": rec.get("n_train") or manifest.get("n_train"),
                "frac": manifest.get("frac"),
                "last_epoch": rec.get("epoch"),
                "valid_polar_mae_bohr3": rec.get("valid_polar_mae_bohr3"),
                "valid_loss": rec.get("valid_loss"),
                "best_epoch": rec.get("best_epoch"),
                "best_valid_loss": rec.get("best_valid_loss"),
            }
        )
    return rows


def format_lc_table(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "n_train  frac   epoch  polar_mae_bohr3          valid_loss           best_epoch  ckpt"
    ]
    for row in rows:
        lines.append(
            f"{str(row.get('n_train')):>7}  {str(row.get('frac')):>5}  "
            f"{str(row.get('last_epoch')):>5}  {row.get('valid_polar_mae_bohr3')!s:>20}  "
            f"{row.get('valid_loss')!s:>20}  {str(row.get('best_epoch')):>10}  "
            f"{Path(str(row.get('ckpt', ''))).name}"
        )
    return "\n".join(lines) + "\n"
