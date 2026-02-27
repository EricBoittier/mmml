"""
Hidden-state inspection for EF MessagePassingModel.
"""

from __future__ import annotations

import json
import functools
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import e3x
import jax
import jax.numpy as jnp
import numpy as np


def _load_params(params_path: Path):
    with open(params_path, "r") as f:
        params_dict = json.load(f)

    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            arr = np.array(obj)
            if arr.dtype == np.float64:
                return jnp.array(arr, dtype=jnp.float32)
            if arr.dtype == np.int64:
                return jnp.array(arr, dtype=jnp.int32)
            return jnp.array(arr)
        return obj

    return convert(params_dict)


def _load_config(config_path: Optional[Path], params_path: Path) -> Dict[str, Any]:
    resolved = config_path
    if resolved is None:
        stem = params_path.stem
        if stem.startswith("params-") and len(stem) > 7:
            uuid_part = stem[7:]
            cand = params_path.parent / f"config-{uuid_part}.json"
            if cand.exists():
                resolved = cand
            elif (params_path.parent / "config.json").exists():
                resolved = params_path.parent / "config.json"

    if resolved is None or not resolved.exists():
        return {
            "features": 64,
            "max_degree": 2,
            "num_iterations": 2,
            "num_basis_functions": 64,
            "cutoff": 10.0,
            "max_atomic_number": 55,
            "include_pseudotensors": True,
        }

    with open(resolved, "r") as f:
        cfg = json.load(f)

    model_keys = {
        "features",
        "max_degree",
        "num_iterations",
        "num_basis_functions",
        "cutoff",
        "max_atomic_number",
        "include_pseudotensors",
        "dipole_field_coupling",
        "field_scale",
    }
    if "model" in cfg and isinstance(cfg["model"], dict):
        model_cfg = {k: v for k, v in cfg["model"].items() if k in model_keys}
    elif "model_config" in cfg and isinstance(cfg["model_config"], dict):
        model_cfg = {k: v for k, v in cfg["model_config"].items() if k in model_keys}
    else:
        model_cfg = {k: v for k, v in cfg.items() if k in model_keys}
    return model_cfg


def _summarize_tensor(name: str, arr: np.ndarray) -> Dict[str, Any]:
    flat = arr.reshape(-1) if arr.size else np.array([])
    sample = flat[: min(128, flat.size)].tolist() if flat.size else []
    return {
        "name": name,
        "shape": list(arr.shape),
        "mean": float(np.mean(arr)) if arr.size else 0.0,
        "std": float(np.std(arr)) if arr.size else 0.0,
        "min": float(np.min(arr)) if arr.size else 0.0,
        "max": float(np.max(arr)) if arr.size else 0.0,
        "l2_norm": float(np.linalg.norm(arr)) if arr.size else 0.0,
        "sample": sample,
    }


class HiddenStateInspector:
    def __init__(self, params_path: Path, config_path: Optional[Path] = None):
        self.params_path = Path(params_path).resolve()
        self.config_path = Path(config_path).resolve() if config_path else None
        if not self.params_path.exists():
            raise FileNotFoundError(f"Model params not found: {self.params_path}")

        sys.path.insert(0, str(self.params_path.parent))
        import_exc = None
        MessagePassingModel = None
        # Prefer training.py model first, matching the checkpoint's original graph.
        try:
            old_cwd = Path.cwd()
            try:
                # Some training modules read local files at import time.
                # Import from checkpoint directory to keep relative paths valid.
                import os
                os.chdir(self.params_path.parent)
                module = __import__("training", fromlist=["MessagePassingModel"])
            finally:
                os.chdir(old_cwd)
            MessagePassingModel = getattr(module, "MessagePassingModel")
        except Exception as exc:  # pragma: no cover - fallback behavior
            import_exc = exc

        if MessagePassingModel is None:
            try:
                module = __import__("model", fromlist=["MessagePassingModel"])
                MessagePassingModel = getattr(module, "MessagePassingModel")
            except Exception as exc:  # pragma: no cover - fallback behavior
                import_exc = exc
        if MessagePassingModel is None:
            raise ImportError(
                f"Could not import MessagePassingModel from model.py/training.py near {self.params_path}"
            ) from import_exc

        self.params = _load_params(self.params_path)
        self.model = MessagePassingModel(**_load_config(self.config_path, self.params_path))

        @functools.partial(jax.jit, static_argnames=("batch_size",))
        def _apply_with_intermediates(
            params,
            atomic_numbers,
            positions,
            ef,
            dst_idx,
            src_idx,
            dst_idx_flat,
            src_idx_flat,
            batch_segments,
            batch_size,
        ):
            return self.model.apply(
                params,
                atomic_numbers,
                positions,
                ef,
                dst_idx=dst_idx,
                src_idx=src_idx,
                dst_idx_flat=dst_idx_flat,
                src_idx_flat=src_idx_flat,
                batch_segments=batch_segments,
                batch_size=batch_size,
                mutable=["intermediates"],
            )

        self._apply_with_intermediates = _apply_with_intermediates

    def inspect_frame(
        self,
        positions: Optional[list],
        atomic_numbers: Optional[list],
        electric_field: Optional[list],
    ) -> Dict[str, Any]:
        if positions is None or atomic_numbers is None:
            raise ValueError("Frame has no positions/atomic numbers.")

        pos = np.asarray(positions, dtype=np.float32)
        z = np.asarray(atomic_numbers, dtype=np.int32)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f"Invalid positions shape: {pos.shape}")
        if z.ndim != 1 or z.shape[0] != pos.shape[0]:
            raise ValueError(f"Invalid atomic numbers shape: {z.shape} for positions {pos.shape}")

        n_atoms = int(pos.shape[0])
        ef = np.asarray(electric_field if electric_field is not None else [0.0, 0.0, 0.0], dtype=np.float32).reshape(3)

        dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(n_atoms)
        dst_idx = jnp.asarray(dst_idx, dtype=jnp.int32)
        src_idx = jnp.asarray(src_idx, dtype=jnp.int32)
        batch_size = 1
        batch_segments = jnp.zeros(n_atoms, dtype=jnp.int32)
        offsets = jnp.arange(batch_size, dtype=jnp.int32) * n_atoms
        dst_idx_flat = (dst_idx[None, :] + offsets[:, None]).reshape(-1)
        src_idx_flat = (src_idx[None, :] + offsets[:, None]).reshape(-1)

        (energy, dipole), state = self._apply_with_intermediates(
            self.params,
            jnp.asarray(z[None, :], dtype=jnp.int32),
            jnp.asarray(pos[None, :, :], dtype=jnp.float32),
            jnp.asarray(ef[None, :], dtype=jnp.float32),
            dst_idx,
            src_idx,
            dst_idx_flat,
            src_idx_flat,
            batch_segments,
            batch_size,
        )

        intermediates = state.get("intermediates", {})
        summaries = []
        atomic_charges = None
        atomic_dipoles = None
        for name, value in intermediates.items():
            arr = value[-1] if isinstance(value, (list, tuple)) else value
            arr_np = np.asarray(arr)
            summaries.append(_summarize_tensor(name, arr_np))
            if name == "atomic_charges":
                atomic_charges = arr_np[0].tolist() if arr_np.ndim >= 2 else arr_np.tolist()
            elif name == "atomic_dipoles":
                atomic_dipoles = arr_np[0].tolist() if arr_np.ndim >= 3 else arr_np.tolist()

        return {
            "energy": float(np.asarray(energy).reshape(-1)[0]),
            "dipole": np.asarray(dipole).reshape(-1, 3)[0].tolist(),
            "atomic_charges": atomic_charges,
            "atomic_dipoles": atomic_dipoles,
            "summaries": summaries,
        }
