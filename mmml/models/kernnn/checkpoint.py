"""JSON checkpoint I/O for KerNN (params + config + normalization stats)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from mmml.models.kernnn.model import DualFFNet, FFNet, KerNNConfig, KerNNStats, _build_model

# Hardcoded stats from scripts/kernn KerNNCalculator (H2CO train split).
H2CO_CALCULATOR_STATS = KerNNStats(
    mean_e=-15.92673969,
    std_e=0.19340856,
    min_r=[1.20929146, 1.10203063, 1.10526860, 2.02403641, 2.01487780, 1.88028228],
    mean_k=[0.01513197, 0.02114981, 0.02089267, 0.00188985, 0.00190670, 0.00253538],
    std_k=[
        4.34973917e-04,
        2.66251061e-03,
        2.04474782e-03,
        9.74953655e-05,
        9.38079320e-05,
        1.50988708e-04,
    ],
)


def _arrays_to_jsonable_leaves(obj: Any) -> Any:
    return to_jsonable(obj)


def _json_to_arrays(obj: Any, *, dtype=np.float32) -> Any:
    if isinstance(obj, dict):
        return {k: _json_to_arrays(v, dtype=dtype) for k, v in obj.items()}
    if isinstance(obj, list):
        if len(obj) > 0 and isinstance(obj[0], (list, int, float)):
            arr = np.asarray(obj)
            if np.issubdtype(arr.dtype, np.floating):
                arr = arr.astype(dtype)
            return jnp.asarray(arr)
        return [_json_to_arrays(x, dtype=dtype) for x in obj]
    return obj


def save_checkpoint(
    path: str | Path,
    *,
    params: Mapping[str, Any],
    config: KerNNConfig | Mapping[str, Any],
    stats: KerNNStats | Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write a portable KerNN JSON checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = config if isinstance(config, KerNNConfig) else KerNNConfig.from_mapping(config)
    st = stats if isinstance(stats, KerNNStats) else KerNNStats.from_mapping(stats)
    payload = {
        "params": _arrays_to_jsonable_leaves(params),
        "config": cfg.to_dict(),
        "stats": {
            "mean_e": float(st.mean_e),
            "std_e": float(st.std_e),
            "min_r": _arrays_to_jsonable_leaves(st.min_r),
            "mean_k": _arrays_to_jsonable_leaves(st.mean_k),
            "std_k": _arrays_to_jsonable_leaves(st.std_k),
            "mean_dihedral": float(st.mean_dihedral),
            "std_dihedral": float(st.std_dihedral),
        },
    }
    if metadata:
        payload["metadata"] = _arrays_to_jsonable_leaves(dict(metadata))
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_checkpoint(
    path: str | Path,
    *,
    dtype=np.float32,
) -> tuple[dict[str, Any], KerNNConfig, KerNNStats, dict[str, Any]]:
    """Load params, config, stats, and optional metadata from a JSON checkpoint."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"KerNN checkpoint not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if "params" not in data or "stats" not in data:
        raise ValueError(
            f"KerNN checkpoint {path} must contain 'params' and 'stats' keys"
        )
    params = _json_to_arrays(data["params"], dtype=dtype)
    # Unwrap Flax {"params": ...} if the file nested it that way
    if isinstance(params, dict) and set(params.keys()) == {"params"}:
        params = params["params"]
    config = KerNNConfig.from_mapping(data.get("config"))
    stats = KerNNStats.from_mapping(data["stats"])
    metadata = dict(data.get("metadata") or {})
    return params, config, stats, metadata


def load_kernnn_model(
    checkpoint: str | Path,
) -> tuple[FFNet | DualFFNet, dict[str, Any], KerNNConfig, KerNNStats]:
    """Load Flax model + params + config + stats from a KerNN JSON checkpoint."""
    params, config, stats, _ = load_checkpoint(checkpoint)
    model = _build_model(config)
    return model, params, config, stats


def torch_state_dict_to_flax_params(
    state_dict: Mapping[str, Any],
    *,
    n_input: int = 6,
    n_hidden: int = 20,
    n_out: int = 1,
) -> dict[str, Any]:
    """Convert a Torch ``FFNet`` Sequential state_dict to Flax ``FFNet`` params.

    Torch Linear weight is ``(out, in)``; Flax Dense kernel is ``(in, out)``.
    Expected Torch keys: ``layers.0/2/4/6.weight`` and ``.bias``.
    """
    # Map Sequential indices → Flax dense names
    layer_map = {
        0: "dense_0",
        2: "dense_1",
        4: "dense_2",
        6: "dense_3",
    }
    params: dict[str, Any] = {}
    for idx, name in layer_map.items():
        w_key = f"layers.{idx}.weight"
        b_key = f"layers.{idx}.bias"
        if w_key not in state_dict or b_key not in state_dict:
            raise KeyError(
                f"Torch state_dict missing {w_key}/{b_key}; keys={list(state_dict)}"
            )
        weight = np.asarray(state_dict[w_key])
        bias = np.asarray(state_dict[b_key])
        # Torch (out, in) → Flax (in, out)
        kernel = jnp.asarray(weight.T)
        params[name] = {"kernel": kernel, "bias": jnp.asarray(bias)}

    expected = {
        "dense_0": (n_input, n_hidden),
        "dense_1": (n_hidden, n_hidden),
        "dense_2": (n_hidden, n_hidden),
        "dense_3": (n_hidden, n_out),
    }
    for name, exp in expected.items():
        shape = tuple(params[name]["kernel"].shape)
        if shape != exp:
            raise ValueError(f"layer {name} kernel shape {shape} != expected {exp}")
    return params


def import_torch_state_dict(
    path: str | Path,
    *,
    stats: KerNNStats | None = None,
    config: KerNNConfig | None = None,
    out_path: str | Path | None = None,
) -> tuple[dict[str, Any], KerNNConfig, KerNNStats]:
    """Load a Torch ``state_dict`` file and optionally write a KerNN JSON checkpoint.

    Requires ``torch`` only when this helper is called.
    """
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "import_torch_state_dict requires torch to be installed"
        ) from exc

    path = Path(path).expanduser()
    raw = torch.load(path, map_location="cpu", weights_only=True)
    # Allow either raw state_dict or {"state_dict": ...}
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    # Convert tensors → numpy
    state = {
        k: (v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v))
        for k, v in raw.items()
    }
    cfg = config or KerNNConfig()
    params = torch_state_dict_to_flax_params(
        state,
        n_input=cfg.n_input,
        n_hidden=cfg.n_hidden,
        n_out=cfg.n_out,
    )
    st = stats or H2CO_CALCULATOR_STATS
    if out_path is not None:
        save_checkpoint(
            out_path,
            params=params,
            config=cfg,
            stats=st,
            metadata={"source_torch": str(path)},
        )
    return params, cfg, st


def init_params(
    key: jax.Array,
    *,
    config: KerNNConfig | None = None,
) -> dict[str, Any]:
    """Initialize Flax model params with a dummy feature batch."""
    cfg = config or KerNNConfig()
    model = _build_model(cfg)
    dummy_k = jnp.zeros((1, cfg.n_input), dtype=jnp.float32)
    if cfg.architecture == "dual":
        dummy_d = jnp.zeros((1, 1), dtype=jnp.float32)
        return model.init(key, dummy_k, dummy_d, deterministic=True)["params"]
    return model.init(key, dummy_k)["params"]
