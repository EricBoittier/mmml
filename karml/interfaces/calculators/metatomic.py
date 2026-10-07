"""Optional metatomic ASE calculator loader.

Metatomic models are TorchScript ``AtomisticModel`` files (typically ``.pt``).
This module does not import torch or metatomic at package import time.

Install: ``uv sync --extra metatomic`` (pulls ``metatomic-ase`` + torch).
Device: ``MMML_METATOMIC_DEVICE`` (default ``cpu``); do not set CUDA here at
import time.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ase.calculators.calculator import Calculator

METATOMIC_CHECKPOINT_SUFFIXES = frozenset({".pt", ".pth"})
METATOMIC_DIR_FILENAMES = (
    "model.pt",
    "exported.pt",
    "metatomic.pt",
    "atomistic-model.pt",
)
DEFAULT_METATOMIC_DEVICE = "cpu"
METATOMIC_DEVICE_ENV = "MMML_METATOMIC_DEVICE"


def have_metatomic() -> bool:
    """Return True when a metatomic ASE calculator class can be imported."""
    try:
        import metatomic_ase  # noqa: F401
    except ImportError:
        try:
            import metatomic.torch.ase_calculator  # noqa: F401
        except ImportError:
            return False
    return True


def metatomic_device_name(*, device: str | None = None) -> str:
    """Resolve torch device without mutating the process environment."""
    if device is not None and str(device).strip():
        return str(device).strip()
    raw = os.environ.get(METATOMIC_DEVICE_ENV, "").strip()
    return raw or DEFAULT_METATOMIC_DEVICE


def is_metatomic_checkpoint(path: Path | str | None) -> bool:
    """True when ``path`` looks like a metatomic AtomisticModel file or export dir."""
    if path is None:
        return False
    candidate = Path(path).expanduser()
    if candidate.is_file():
        return candidate.suffix.lower() in METATOMIC_CHECKPOINT_SUFFIXES
    if candidate.is_dir():
        return any((candidate / name).is_file() for name in METATOMIC_DIR_FILENAMES)
    return candidate.suffix.lower() in METATOMIC_CHECKPOINT_SUFFIXES


def resolve_metatomic_model_path(path: Path | str) -> Path:
    """Return the ``.pt`` file to pass to ``MetatomicCalculator``."""
    candidate = Path(path).expanduser().resolve()
    if candidate.is_file():
        return candidate
    if candidate.is_dir():
        for name in METATOMIC_DIR_FILENAMES:
            nested = candidate / name
            if nested.is_file():
                return nested.resolve()
        raise FileNotFoundError(
            f"metatomic checkpoint directory {candidate} has none of {METATOMIC_DIR_FILENAMES}"
        )
    raise FileNotFoundError(f"metatomic checkpoint not found: {candidate}")


def _import_metatomic_calculator_cls() -> type[Calculator]:
    try:
        from metatomic_ase import MetatomicCalculator
    except ImportError:
        try:
            from metatomic.torch.ase_calculator import MetatomicCalculator
        except ImportError as exc:
            raise ModuleNotFoundError(
                "metatomic ASE calculator is not installed. "
                "Install with: uv sync --extra metatomic"
            ) from exc
    return MetatomicCalculator


def load_metatomic_calculator(
    checkpoint: Path | str,
    *,
    device: str | None = None,
    extra_kwargs: Mapping[str, Any] | None = None,
) -> Calculator:
    """Load a metatomic model as an ASE calculator (energy eV, forces eV/Å)."""
    model_path = resolve_metatomic_model_path(checkpoint)
    calc_cls = _import_metatomic_calculator_cls()
    kwargs = dict(extra_kwargs) if extra_kwargs is not None else {}
    return calc_cls(str(model_path), device=metatomic_device_name(device=device), **kwargs)
