"""Shared configuration and checkpoint validation for the CG examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from mmml.interfaces.calculators.simple_inference import create_calculator_from_checkpoint


def load_cg_config(
    defaults: Mapping[str, Any],
    *,
    description: str,
    argv: Sequence[str] | None = None,
) -> SimpleNamespace:
    """Load example settings from defaults, optional JSON, then CLI overrides."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, help="JSON object overriding any example setting")
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint used for every ML region")
    parser.add_argument("--peptide-checkpoint", type=Path)
    parser.add_argument("--water-checkpoint", type=Path)
    parser.add_argument("--n-waters", type=int)
    parser.add_argument("--box-size", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--dt-fs", type=float)
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)

    values = dict(defaults)
    if args.config is not None:
        with args.config.expanduser().open(encoding="utf-8") as handle:
            configured = json.load(handle)
        if not isinstance(configured, dict):
            raise ValueError(f"CG config must contain a JSON object: {args.config}")
        unknown = sorted(set(configured) - set(values))
        if unknown:
            raise ValueError(f"Unknown CG config keys: {', '.join(unknown)}")
        values.update(configured)

    cli_mapping = {
        "checkpoint": "checkpoint",
        "peptide_checkpoint": "peptide_checkpoint",
        "water_checkpoint": "water_checkpoint",
        "n_waters": "n_waters",
        "box_size": "box_size",
        "seed": "seed",
        "temperature": "temperature",
        "dt_fs": "dt_fs",
        "workdir": "workdir",
        "output_dir": "output_dir",
    }
    for argument, setting in cli_mapping.items():
        value = getattr(args, argument)
        if value is not None:
            values[setting] = str(value) if isinstance(value, Path) else value

    if args.checkpoint is not None:
        checkpoint = str(args.checkpoint.expanduser())
        values["checkpoint"] = checkpoint
        if "peptide_checkpoint" in values:
            values["peptide_checkpoint"] = checkpoint
        if "water_checkpoint" in values:
            values["water_checkpoint"] = checkpoint

    return SimpleNamespace(**values)


def load_cg_checkpoint(checkpoint: str | Path) -> tuple[Any, Any, Any]:
    """Load a checkpoint and expose its calculator, model, and parameter tree."""
    path = Path(checkpoint).expanduser().resolve()
    calculator = create_calculator_from_checkpoint(path)
    model = getattr(calculator, "model", getattr(calculator, "_mmml_physnet_model", None))
    params = getattr(calculator, "params", getattr(calculator, "_mmml_physnet_params", None))
    if model is None or params is None:
        raise ValueError(f"Could not extract model or parameters from {path}")
    return calculator, model, params


def validate_supported_elements(
    model: Any,
    atomic_numbers: Sequence[int] | np.ndarray,
    *,
    label: str,
) -> None:
    """Validate atomic numbers without imposing a fixed molecule or padding size."""
    numbers = np.asarray(atomic_numbers, dtype=np.int32).reshape(-1)
    if numbers.size == 0 or np.any(numbers <= 0):
        raise ValueError(f"{label}: atomic numbers must be positive")
    maximum = getattr(model, "max_atomic_number", None)
    if maximum is not None and int(numbers.max()) > int(maximum):
        unsupported = sorted({int(z) for z in numbers if int(z) > int(maximum)})
        raise ValueError(
            f"{label}: checkpoint supports Z <= {int(maximum)}, "
            f"but the structure contains {unsupported}"
        )


def probe_charge_output(
    model: Any,
    params: Any,
    atomic_numbers: Sequence[int] | np.ndarray,
    positions: np.ndarray,
    *,
    charge: float,
    spin: float,
    label: str,
) -> np.ndarray:
    """Run one model forward pass and require finite per-atom charge output."""
    numbers = np.asarray(atomic_numbers, dtype=np.int32).reshape(-1)
    coordinates = np.asarray(positions, dtype=np.float32).reshape((-1, 3))
    if coordinates.shape[0] != numbers.size:
        raise ValueError(f"{label}: positions and atomic numbers have different lengths")
    validate_supported_elements(model, numbers, label=label)
    if not bool(getattr(model, "charges", False)):
        raise ValueError(f"{label}: checkpoint was created with charge prediction disabled")

    dst_idx, src_idx = np.where(~np.eye(numbers.size, dtype=bool))
    apply_kwargs = {
        "atomic_numbers": jnp.asarray(numbers),
        "positions": jnp.asarray(coordinates),
        "dst_idx": jnp.asarray(dst_idx, dtype=jnp.int32),
        "src_idx": jnp.asarray(src_idx, dtype=jnp.int32),
        "compute_forces": False,
    }
    is_spooky = "spooky" in type(model).__module__.lower() or "spooky" in type(
        model
    ).__name__.lower()
    if is_spooky:
        apply_kwargs["charges"] = jnp.full(
            (numbers.size, 1), float(charge), dtype=jnp.float32
        )
        apply_kwargs["spins"] = jnp.full(
            (numbers.size, 1), float(spin), dtype=jnp.float32
        )
    output = model.apply(params, **apply_kwargs)
    charge_values = output.get("charges_as_mono", output.get("charges"))
    if charge_values is None:
        raise ValueError(f"{label}: model output contains no atomic charges")
    result = np.asarray(jax.device_get(charge_values), dtype=np.float64).reshape(-1)
    if result.size < numbers.size:
        raise ValueError(
            f"{label}: model returned {result.size} charges for {numbers.size} atoms"
        )
    result = result[: numbers.size]
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label}: model returned non-finite atomic charges")
    print(
        f"{label}: checkpoint element and charge-output probe passed "
        f"(Zmax={int(numbers.max())}, charge sum={float(result.sum()):.6f})"
    )
    return result
