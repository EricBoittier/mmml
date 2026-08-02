"""Teacher-distillation helpers for the SpookyPhysNet extxyz trainer.

The blending arithmetic itself is shared with the generic PhysNet distillation
pipeline (:mod:`mmml.models.physnetjax.physnetjax.training.distill`); this
module adds the two pieces that are specific to distilling *between Spooky
checkpoints*:

* rebuilding the teacher's architecture from its own checkpoint, so a teacher
  can never silently inherit (or overwrite) the student's hyperparameters, and
* aligning the teacher's energy zero onto the student's reference scale.

The zero alignment matters because teacher and student are generally trained on
different caches with different ``use_energy_bias`` settings, so their absolute
energies are offset by an arbitrary constant plus a per-element atomic
reference.  Forces are unaffected by either (both vanish under differentiation),
so only the energy channel needs correcting.

Everything here is deliberately free of JAX tracing and of the training loop, so
it can be unit-tested directly.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from mmml.models.physnetjax.physnetjax.training.distill import (
    blend_component_loss,
    blend_regression_loss,
)

__all__ = [
    "EnergyAlignment",
    "TeacherArchitecture",
    "blend_component_loss",
    "blend_regression_loss",
    "checkpoint_fingerprint",
    "fit_energy_alignment",
    "parse_spooky_distill_targets",
    "teacher_architecture_from_checkpoint",
]

# Teacher targets the Spooky trainer knows how to distil.  Charges and dipoles
# are deliberately excluded: the student's charge head is the whole point of the
# charge-aware campaign, and it stays supervised by reference data only.
SUPPORTED_DISTILL_TARGETS = ("energy", "forces")
REJECTED_DISTILL_TARGETS = ("dipole", "dipoles", "charge", "charges")


def parse_spooky_distill_targets(targets: Any) -> tuple[bool, bool]:
    """Return ``(distill_energy, distill_forces)`` for a sequence of target names.

    ``None`` means "both", matching the generic helper's default.  Charge and
    dipole targets are rejected rather than ignored: silently dropping them
    would let a run claim charge distillation it never performed.
    """
    if targets is None:
        return True, True
    normalized = {str(t).strip().lower() for t in targets if str(t).strip()}
    if not normalized:
        raise ValueError("--distill-targets was given no usable target names")
    rejected = sorted(normalized & set(REJECTED_DISTILL_TARGETS))
    if rejected:
        raise ValueError(
            f"Cannot distil {', '.join(rejected)} from a Spooky teacher: the "
            "student's charge/dipole heads stay supervised by reference data "
            f"only. Supported targets are {', '.join(SUPPORTED_DISTILL_TARGETS)}."
        )
    unknown = sorted(normalized - set(SUPPORTED_DISTILL_TARGETS))
    if unknown:
        raise ValueError(
            f"Unknown distillation target(s): {', '.join(unknown)}. "
            f"Supported targets are {', '.join(SUPPORTED_DISTILL_TARGETS)}."
        )
    return "energy" in normalized, "forces" in normalized


# ---------------------------------------------------------------------------
# Teacher architecture
# ---------------------------------------------------------------------------

# SpookyPhysNet constructor defaults, replicated here so a checkpoint that
# predates a field still rebuilds exactly as it was trained rather than picking
# up whatever the *current* default happens to be.
_SPOOKY_DEFAULTS: dict[str, Any] = {
    "features": 64,
    "max_degree": 1,
    "num_iterations": 2,
    "num_basis_functions": 32,
    "cutoff": 6.0,
    "max_atomic_number": 87,
    "charges": False,
    "total_charge": 0,
    "n_refinement_blocks": 2,
    "zbl": True,
    "trainable_zbl": False,
    "zbl_cuton": 0.1,
    "zbl_cutoff": 0.6,
    "efa": False,
    "use_energy_bias": False,
    "electrostatics_damping_sigma": 4.0,
    "switch_start": 1.0,
    "switch_end": 10.0,
    "electrostatics_off_start": 8.0,
    "electrostatics_off_end": 10.0,
    "learn_cgenff_vdw_scale": True,
    "predict_atomic_vdw_scale": True,
    "interaction_trust_map": False,
}

# How to read each architecture field out of a run's argparse ``config`` blob,
# which stores CLI flags rather than model fields.
_CONFIG_ALIASES: dict[str, tuple[str, Any]] = {
    "charges": ("predict_charges", lambda v: bool(v)),
    "n_refinement_blocks": ("n_res", lambda v: int(v)),
    "zbl": ("no_zbl", lambda v: not bool(v)),
    "learn_cgenff_vdw_scale": ("fixed_cgenff_vdw", lambda v: not bool(v)),
    "predict_atomic_vdw_scale": ("fixed_cgenff_vdw", lambda v: not bool(v)),
}


@dataclass(frozen=True)
class TeacherArchitecture:
    """A teacher's SpookyPhysNet kwargs plus where each value came from."""

    kwargs: dict[str, Any]
    source: str
    missing_fields: tuple[str, ...] = ()

    def differing_fields(self, student_kwargs: Mapping[str, Any]) -> dict[str, tuple[Any, Any]]:
        """Fields where teacher and student disagree, as ``{field: (teacher, student)}``."""
        out: dict[str, tuple[Any, Any]] = {}
        for key, value in self.kwargs.items():
            if key == "max_padded_atoms":
                continue
            if key in student_kwargs and student_kwargs[key] != value:
                out[key] = (value, student_kwargs[key])
        return out


def teacher_architecture_from_checkpoint(
    restored: Mapping[str, Any],
    *,
    max_padded_atoms: int,
) -> TeacherArchitecture:
    """Rebuild a teacher's SpookyPhysNet kwargs from its own checkpoint.

    ``model_attributes`` is preferred because it is serialized from the model
    object itself; a params-only JSON export carries the run's ``config``
    instead, which we map through :data:`_CONFIG_ALIASES`.  Nothing is ever
    taken from the student.

    ``max_padded_atoms`` is the one exception: it is a batch-shape property, not
    a learned one, so it must follow the student's padding for the teacher to be
    applicable to the same batch.
    """
    attributes = restored.get("model_attributes")
    config = restored.get("config")

    if isinstance(attributes, Mapping) and attributes:
        source = "model_attributes"
        lookup: Mapping[str, Any] = attributes
        aliases: dict[str, tuple[str, Any]] = {}
    elif isinstance(config, Mapping) and config:
        source = "config"
        lookup = config
        aliases = _CONFIG_ALIASES
    else:
        raise ValueError(
            "Teacher checkpoint carries neither 'model_attributes' nor 'config'; "
            "its architecture cannot be confirmed, so it will not be used."
        )

    kwargs: dict[str, Any] = {}
    missing: list[str] = []
    for field_name, default in _SPOOKY_DEFAULTS.items():
        if field_name in lookup:
            kwargs[field_name] = _coerce_like(default, lookup[field_name])
            continue
        alias = aliases.get(field_name)
        if alias is not None and alias[0] in lookup:
            kwargs[field_name] = alias[1](lookup[alias[0]])
            continue
        kwargs[field_name] = default
        missing.append(field_name)

    kwargs["max_padded_atoms"] = int(max_padded_atoms)
    return TeacherArchitecture(
        kwargs=kwargs, source=source, missing_fields=tuple(sorted(missing))
    )


def _coerce_like(default: Any, value: Any) -> Any:
    """Coerce a restored scalar to the type of the corresponding model default.

    Orbax round-trips scalars as 0-d arrays, and JSON turns bools into ints, so
    a raw restore would hand flax a ``np.int64`` where it expects ``bool`` and
    silently change module behaviour.
    """
    if isinstance(value, np.ndarray):
        value = value.item() if value.ndim == 0 else value.tolist()
    if isinstance(default, bool):
        return bool(value)
    if isinstance(default, int) and not isinstance(default, bool):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value


def checkpoint_fingerprint(path: Path) -> dict[str, Any]:
    """Identify a checkpoint by content hash so a run records *which* teacher it used."""
    path = Path(path)
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        size = path.stat().st_size
        n_files = 1
    else:
        files = sorted(p for p in path.rglob("*") if p.is_file())
        size = 0
        for item in files:
            digest.update(str(item.relative_to(path)).encode())
            digest.update(str(item.stat().st_size).encode())
            size += item.stat().st_size
        n_files = len(files)
    return {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "size_bytes": int(size),
        "n_files": int(n_files),
        "hashed": "contents" if path.is_file() else "file-tree",
    }


# ---------------------------------------------------------------------------
# Energy-zero alignment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnergyAlignment:
    """A recorded, explicit map from teacher energies onto the reference scale.

    ``E_aligned = E_teacher + scalar_offset + counts @ element_offsets``
    where ``counts[s, z]`` is how many atoms of atomic number ``z`` structure
    ``s`` has.  ``element_offsets`` is indexed by atomic number so the training
    loop can apply it as a per-atom lookup and segment-sum.
    """

    mode: str
    scalar_offset: float
    element_offsets: np.ndarray
    n_samples: int
    rms_before_eV: float
    rms_after_eV: float
    mean_abs_shift_eV: float
    fallback_reason: str | None = None
    requested_mode: str = ""
    elements_present: tuple[int, ...] = field(default=())

    def apply(self, teacher_energy: np.ndarray, counts: np.ndarray) -> np.ndarray:
        return (
            np.asarray(teacher_energy, dtype=np.float64)
            + self.scalar_offset
            + np.asarray(counts, dtype=np.float64) @ self.element_offsets
        )

    def to_metadata(self) -> dict[str, Any]:
        data = asdict(self)
        data["element_offsets"] = np.asarray(self.element_offsets, dtype=np.float64).tolist()
        data["elements_present"] = list(self.elements_present)
        return data


def fit_energy_alignment(
    teacher_energy: np.ndarray,
    reference_energy: np.ndarray,
    counts: np.ndarray,
    *,
    mode: str = "atomic",
) -> EnergyAlignment:
    """Fit the teacher→reference energy-zero correction on a calibration sample.

    ``mode``:

    * ``"none"``   – no correction (only sane when both were trained on the same
      cache with the same energy bias); the residual is still measured and
      recorded so the choice can be judged after the fact.
    * ``"scalar"`` – a single constant shift.
    * ``"atomic"`` – least-squares per-element atomic references, which is what
      actually differs between two caches.  Falls back to ``"scalar"`` — with
      the reason recorded — when the sample cannot support the fit or when the
      fit fails to beat it.
    """
    teacher_energy = np.asarray(teacher_energy, dtype=np.float64).reshape(-1)
    reference_energy = np.asarray(reference_energy, dtype=np.float64).reshape(-1)
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 2:
        raise ValueError(f"counts must be 2-D (n_structures, n_z), got shape {counts.shape}")
    if not (teacher_energy.shape[0] == reference_energy.shape[0] == counts.shape[0]):
        raise ValueError(
            "teacher_energy, reference_energy and counts must agree on the number "
            f"of structures: {teacher_energy.shape[0]}, {reference_energy.shape[0]}, "
            f"{counts.shape[0]}"
        )
    if mode not in ("none", "scalar", "atomic"):
        raise ValueError(f"Unknown energy-alignment mode {mode!r}")

    n_z = counts.shape[1]
    n_samples = teacher_energy.shape[0]
    residual = reference_energy - teacher_energy
    rms_before = float(np.sqrt(np.mean(residual**2))) if n_samples else 0.0
    present = tuple(int(z) for z in np.flatnonzero(counts.any(axis=0)))

    def _build(
        mode_used: str,
        scalar: float,
        element: np.ndarray,
        reason: str | None,
    ) -> EnergyAlignment:
        shift = scalar + counts @ element
        after = residual - shift
        return EnergyAlignment(
            mode=mode_used,
            scalar_offset=float(scalar),
            element_offsets=np.asarray(element, dtype=np.float64),
            n_samples=int(n_samples),
            rms_before_eV=rms_before,
            rms_after_eV=float(np.sqrt(np.mean(after**2))) if n_samples else 0.0,
            mean_abs_shift_eV=float(np.mean(np.abs(shift))) if n_samples else 0.0,
            fallback_reason=reason,
            requested_mode=mode,
            elements_present=present,
        )

    zeros = np.zeros(n_z, dtype=np.float64)
    if mode == "none" or n_samples == 0:
        reason = "no calibration samples" if (n_samples == 0 and mode != "none") else None
        return _build("none", 0.0, zeros, reason)

    scalar_fit = _build("scalar", float(np.mean(residual)), zeros, None)
    if mode == "scalar":
        return scalar_fit

    if n_samples < 2 * max(1, len(present)):
        return _build(
            "scalar",
            scalar_fit.scalar_offset,
            zeros,
            f"atomic fit needs >= {2 * max(1, len(present))} calibration structures "
            f"for {len(present)} elements, got {n_samples}",
        )

    try:
        solution, *_ = np.linalg.lstsq(counts, residual, rcond=None)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - numerically rare
        return _build("scalar", scalar_fit.scalar_offset, zeros, f"lstsq failed: {exc}")
    if not np.all(np.isfinite(solution)):
        return _build("scalar", scalar_fit.scalar_offset, zeros, "lstsq produced non-finite offsets")

    atomic_fit = _build("atomic", 0.0, solution, None)
    if atomic_fit.rms_after_eV > scalar_fit.rms_after_eV:
        return _build(
            "scalar",
            scalar_fit.scalar_offset,
            zeros,
            "atomic fit did not improve on the scalar shift "
            f"({atomic_fit.rms_after_eV:.6g} vs {scalar_fit.rms_after_eV:.6g} eV RMS)",
        )
    return atomic_fit


def element_counts_from_atomic_numbers(
    atomic_numbers: np.ndarray,
    atom_mask: np.ndarray,
    n_z: int,
) -> np.ndarray:
    """Per-structure counts of each atomic number, ignoring padded atoms.

    ``atomic_numbers`` and ``atom_mask`` are ``(n_structures, n_atoms)``.
    """
    atomic_numbers = np.asarray(atomic_numbers)
    atom_mask = np.asarray(atom_mask).astype(bool)
    if atomic_numbers.shape != atom_mask.shape:
        raise ValueError(
            f"atomic_numbers {atomic_numbers.shape} and atom_mask {atom_mask.shape} "
            "must have the same shape"
        )
    counts = np.zeros((atomic_numbers.shape[0], n_z), dtype=np.float64)
    for row, (z_row, mask_row) in enumerate(zip(atomic_numbers, atom_mask, strict=True)):
        valid = z_row[mask_row].astype(np.int64)
        valid = valid[(valid >= 0) & (valid < n_z)]
        if valid.size:
            counts[row] = np.bincount(valid, minlength=n_z)[:n_z]
    return counts
