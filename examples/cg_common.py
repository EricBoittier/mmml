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
from ase.io.trajectory import Trajectory

from mmml.interfaces.calculators.simple_inference import create_calculator_from_checkpoint
from mmml.utils.dcd_writer import DCDTrajectoryWriter


class DualTrajectoryWriter:
    """Write each ASE frame to both an ASE trajectory and a DCD."""

    def __init__(
        self,
        traj_path: str | Path,
        atoms: Any,
        *,
        write_dcd: bool = True,
        dt_ps: float = 1.0,
        steps_per_frame: int = 1,
    ) -> None:
        self.traj_path = Path(traj_path)
        self.ase = Trajectory(str(self.traj_path), "w", atoms)
        self.dcd = (
            DCDTrajectoryWriter(
                self.traj_path.with_suffix(".dcd"),
                len(atoms),
                dt_ps=dt_ps,
                steps_per_frame=steps_per_frame,
                has_unitcell=True,
            )
            if write_dcd
            else None
        )

    def write(self, atoms: Any) -> None:
        self.ase.write(atoms)
        if self.dcd is not None:
            self.dcd.write(
                atoms.get_positions(),
                box=np.asarray(atoms.get_cell().array, dtype=np.float64),
            )

    def close(self) -> None:
        self.ase.close()
        if self.dcd is not None:
            self.dcd.close()


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
    parser.add_argument("--sequence")
    parser.add_argument("--first-patch")
    parser.add_argument("--last-patch")
    parser.add_argument("--peptide-patch", action="append", dest="peptide_patches")
    parser.add_argument("--initial-peptide-pdb", type=Path)
    parser.add_argument("--pdb-id")
    parser.add_argument("--pdb-chain")
    parser.add_argument("--n-waters", type=int)
    parser.add_argument("--box-size", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--dt-fs", type=float)
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--no-dcd",
        action="store_true",
        help="Disable simultaneous CHARMM DCD output",
    )
    parser.add_argument(
        "--no-ml",
        action="store_true",
        help="Mock ML internals with classical MM bonded terms",
    )
    args, unknown_args = parser.parse_known_args(argv)

    values = dict(defaults)
    configured: dict[str, Any] = {}
    if args.config is not None:
        with args.config.expanduser().open(encoding="utf-8") as handle:
            if args.config.suffix in (".yaml", ".yml"):
                import yaml
                configured = yaml.safe_load(handle)
            else:
                configured = json.load(handle)
        if not isinstance(configured, dict):
            raise ValueError(f"CG config must contain a dictionary object: {args.config}")
        unknown = sorted(set(configured) - set(values))
        if unknown:
            raise ValueError(f"Unknown CG config keys: {', '.join(unknown)}")
        values.update(configured)
        if "checkpoint" in configured:
            if "peptide_checkpoint" in values and "peptide_checkpoint" not in configured:
                values["peptide_checkpoint"] = configured["checkpoint"]
            if "water_checkpoint" in values and "water_checkpoint" not in configured:
                values["water_checkpoint"] = configured["checkpoint"]

    cli_mapping = {
        "checkpoint": "checkpoint",
        "peptide_checkpoint": "peptide_checkpoint",
        "water_checkpoint": "water_checkpoint",
        "sequence": "sequence",
        "first_patch": "first_patch",
        "last_patch": "last_patch",
        "peptide_patches": "peptide_patches",
        "initial_peptide_pdb": "initial_peptide_pdb",
        "pdb_id": "pdb_id",
        "pdb_chain": "pdb_chain",
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
    if args.no_dcd and "write_dcd" in values:
        values["write_dcd"] = False
    if args.no_ml:
        values["use_ml_intramolecular"] = False

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


def parse_temp_schedule(schedule_str, total_steps):
    """
    Parses a temperature schedule string and returns a function T(step) -> Kelvin.
    Supported formats:
      1. A single float: "298.0" -> constant 298 K
      2. "T1->T2" -> linear ramp from T1 to T2 over total_steps
      3. Complex staged schedule: "298.0->398.0:0.25, 398.0:0.5, 398.0->298.0:0.25"
    """
    schedule_str = str(schedule_str).strip()
    try:
        val = float(schedule_str)
        return lambda step: val
    except ValueError:
        pass
    if "->" in schedule_str and ":" not in schedule_str:
        parts = schedule_str.split("->")
        T1 = float(parts[0])
        T2 = float(parts[1])
        return lambda step: T1 + (T2 - T1) * (step / max(1, total_steps))
    stages = []
    accum_frac = 0.0
    for part in schedule_str.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid schedule segment (missing fraction): {part}")
        expr, frac_str = part.split(":")
        frac = float(frac_str)
        start_frac = accum_frac
        end_frac = accum_frac + frac
        accum_frac = end_frac
        if "->" in expr:
            t_parts = expr.split("->")
            T1 = float(t_parts[0])
            T2 = float(t_parts[1])
            stages.append((start_frac, end_frac, T1, T2, True))
        else:
            T = float(expr)
            stages.append((start_frac, end_frac, T, T, False))
    if abs(accum_frac - 1.0) > 1e-4:
        stages = [(sf/accum_frac, ef/accum_frac, T1, T2, is_ramp) for sf, ef, T1, T2, is_ramp in stages]
    def T_func(step):
        frac = step / max(1, total_steps)
        frac = min(max(frac, 0.0), 1.0)
        for sf, ef, T1, T2, is_ramp in stages:
            if sf <= frac <= ef:
                if sf == ef:
                    return T1
                if is_ramp:
                    segment_frac = (frac - sf) / (ef - sf)
                    return T1 + (T2 - T1) * segment_frac
                else:
                    return T1
        return stages[-1][2]
    return T_func
