"""Optional CHARMM nonbond / PBC state snapshots before risky list rebuilds."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np


def nbond_debug_enabled() -> bool:
    """True when ``MMML_NBOND_DEBUG`` or ``MMML_SAVE_NBOND_SNAPSHOTS`` is set."""
    for key in ("MMML_NBOND_DEBUG", "MMML_SAVE_NBOND_SNAPSHOTS"):
        val = os.environ.get(key, "").strip().lower()
        if val in ("1", "true", "yes", "on"):
            return True
    return False


def _resolve_debug_dir(ctx: Any | None) -> Path:
    for attr in ("out_dir", "workflow_out_dir"):
        if ctx is not None:
            raw = getattr(ctx, attr, None)
            if raw:
                return Path(raw).expanduser().resolve() / "nbond_debug"
    env = os.environ.get("MMML_NBOND_DEBUG_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return Path("/tmp/mmml-nbond-debug")


def _safe_int(fn: Any, default: int = -1) -> int:
    try:
        return int(fn())
    except Exception:
        return default


def _pbc_box_side_A() -> tuple[float, float, float] | None:
    try:
        import ctypes

        import pycharmm.lib as lib

        sx = ctypes.c_double(0.0)
        sy = ctypes.c_double(0.0)
        sz = ctypes.c_double(0.0)
        lib.charmm.pbound_get_size(
            ctypes.byref(sx),
            ctypes.byref(sy),
            ctypes.byref(sz),
        )
        return float(sx.value), float(sy.value), float(sz.value)
    except Exception:
        return None


def collect_nbond_state(ctx: Any | None = None, *, context: str = "") -> dict[str, Any]:
    """Summarize CHARMM coordinates, PBC flags, and exclusion lists (no ENER)."""
    payload: dict[str, Any] = {
        "context": str(context),
        "timestamp_unix": time.time(),
    }
    try:
        import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
        import pycharmm
        import pycharmm.psf as psf

        payload["natom"] = _safe_int(psf.get_natom)
        payload["nbond"] = _safe_int(psf.get_nbond)
        payload["nres"] = _safe_int(psf.get_nres)
        try:
            import pycharmm.image as image

            payload["ntrans"] = _safe_int(image.get_ntrans)
        except Exception:
            payload["ntrans"] = None
        box = _pbc_box_side_A()
        if box is not None:
            payload["pbound_A"] = {"x": box[0], "y": box[1], "z": box[2]}
        try:
            iblo, inb = psf.get_iblo_inb()
            iblo_arr = np.asarray(iblo, dtype=int)
            inb_arr = np.asarray(inb, dtype=int)
            payload["iblo_inb"] = {
                "n_lists": int(len(iblo_arr)),
                "iblo_min": int(iblo_arr.min()) if iblo_arr.size else None,
                "iblo_max": int(iblo_arr.max()) if iblo_arr.size else None,
                "inb_min": int(inb_arr.min()) if inb_arr.size else None,
                "inb_max": int(inb_arr.max()) if inb_arr.size else None,
            }
        except Exception as exc:
            payload["iblo_inb_error"] = str(exc)
        pos = np.asarray(pycharmm.coor.get_positions(), dtype=np.float64)
        if pos.size:
            payload["positions"] = {
                "min": [float(x) for x in pos.min(axis=0)],
                "max": [float(x) for x in pos.max(axis=0)],
                "rms": float(np.sqrt(np.mean(np.square(pos)))),
            }
    except Exception as exc:
        payload["collect_error"] = str(exc)

    if ctx is not None:
        payload["ctx"] = {
            "use_pbc": bool(getattr(ctx, "use_pbc", False)),
            "registration_uses_block": bool(getattr(ctx, "registration_uses_block", False)),
            "cubic_box_side_A": getattr(ctx, "cubic_box_side_A", None),
            "charmm_cubic_box_side_A": getattr(ctx, "charmm_cubic_box_side_A", None),
        }
    try:
        from mmml.interfaces.pycharmmInterface.mlpot.cgenff_prm_swap import (
            active_cgenff_prm_mode,
        )

        payload["cgenff_prm_mode"] = active_cgenff_prm_mode()
    except Exception:
        pass
    return payload


def maybe_snapshot_nbond_state(
    ctx: Any | None = None,
    *,
    context: str = "",
    force: bool = False,
) -> Path | None:
    """Write a JSON snapshot when debug env is set (or ``force=True``)."""
    if not force and not nbond_debug_enabled():
        return None
    state = collect_nbond_state(ctx, context=context)
    out_dir = _resolve_debug_dir(ctx)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = "".join(c if c.isalnum() else "_" for c in context)[:80] or "snapshot"
    path = out_dir / f"{int(state['timestamp_unix'])}_{slug}.json"
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"MMML nbond debug: wrote {path}", flush=True)
    return path
