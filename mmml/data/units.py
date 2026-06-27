"""
Central unit conversion constants and helpers for MMML.

All conversion factors are defined here to avoid magic numbers and ensure
consistency across train_joint, fix_and_split, DCMNet, PhysNet, and calculators.

Reference: CODATA 2018 / NIST

Canonical ML / hybrid inference units: energy eV, forces eV/Angstrom, coords Angstrom.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, MutableMapping, Sequence

import numpy as np

# -----------------------------------------------------------------------------
# Length
# -----------------------------------------------------------------------------
BOHR_TO_ANGSTROM = 0.529177
ANGSTROM_TO_BOHR = 1.88973

# -----------------------------------------------------------------------------
# Energy
# -----------------------------------------------------------------------------
HARTREE_TO_EV = 27.211386
EV_TO_HARTREE = 1.0 / HARTREE_TO_EV
EV_TO_KCAL_MOL = 23.060549
HARTREE_TO_KCAL_MOL = 627.509474
KCAL_MOL_TO_EV = 1.0 / EV_TO_KCAL_MOL
KCAL_MOL_TO_HARTREE = 1.0 / HARTREE_TO_KCAL_MOL

# -----------------------------------------------------------------------------
# Forces
# -----------------------------------------------------------------------------
HARTREE_BOHR_TO_EV_ANGSTROM = HARTREE_TO_EV / BOHR_TO_ANGSTROM
EV_ANGSTROM_TO_HARTREE_BOHR = BOHR_TO_ANGSTROM / HARTREE_TO_EV

# -----------------------------------------------------------------------------
# Dipole moment
# -----------------------------------------------------------------------------
DEBYE_TO_EANGSTROM = 0.208194
EANGSTROM_TO_DEBYE = 1.0 / DEBYE_TO_EANGSTROM

# Aliases used in some CLI code paths
HARTREE_BOHR_TO_EV_ANG = HARTREE_BOHR_TO_EV_ANGSTROM
EV_ANG_TO_HARTREE_BOHR = EV_ANGSTROM_TO_HARTREE_BOHR
BOHR_TO_ANG = BOHR_TO_ANGSTROM

EnergyUnit = Literal["ev", "hartree", "kcal_mol"]
ForceUnit = Literal["ev_angstrom", "hartree_bohr", "kcal_mol_angstrom"]
LengthUnit = Literal["angstrom", "bohr"]
DipoleUnit = Literal["debye", "e_angstrom"]
EspUnit = Literal["hartree_per_e"]

CANONICAL_ENERGY_UNIT: EnergyUnit = "ev"
CANONICAL_FORCE_UNIT: ForceUnit = "ev_angstrom"
CANONICAL_LENGTH_UNIT: LengthUnit = "angstrom"
CANONICAL_DIPOLE_UNIT: DipoleUnit = "e_angstrom"

CALCULATOR_UNITS: dict[str, str] = {
    "energy": "eV",
    "forces": "eV/Angstrom",
    "coords": "Angstrom",
    "dipole": "e*Angstrom",
}

TRAINING_UNITS: dict[str, str] = {
    "energy": "eV",
    "forces": "eV/Angstrom",
    "coords": "Angstrom",
}

PYSCF_NATIVE_UNITS: dict[str, str] = {
    "R": "angstrom",
    "E": "hartree",
    "F": "hartree_bohr",
    "Dxyz": "debye",
    "esp": "hartree_per_e",
    "esp_grid": "bohr",
}

_ENERGY_ALIASES = {
    "ev": "ev",
    "eV".lower(): "ev",
    "hartree": "hartree",
    "ha": "hartree",
    "kcal": "kcal_mol",
    "kcal/mol": "kcal_mol",
    "kcal_mol": "kcal_mol",
}

_FORCE_ALIASES = {
    "ev_angstrom": "ev_angstrom",
    "ev/angstrom": "ev_angstrom",
    "ev/ang": "ev_angstrom",
    "ev/a": "ev_angstrom",
    "ev_ang": "ev_angstrom",
    "hartree_bohr": "hartree_bohr",
    "hartree/bohr": "hartree_bohr",
    "ha/bohr": "hartree_bohr",
    "kcal_mol_angstrom": "kcal_mol_angstrom",
    "kcal/mol/angstrom": "kcal_mol_angstrom",
    "kcal/mol/ang": "kcal_mol_angstrom",
}

_LENGTH_ALIASES = {
    "angstrom": "angstrom",
    "ang": "angstrom",
    "a": "angstrom",
    "bohr": "bohr",
    "au": "bohr",
}

_DIPOLE_ALIASES = {
    "debye": "debye",
    "d": "debye",
    "e_angstrom": "e_angstrom",
    "e-angstrom": "e_angstrom",
    "e*angstrom": "e_angstrom",
}


def normalize_energy_unit(unit: str) -> EnergyUnit:
    key = str(unit).strip().lower().replace("-", "_").replace(" ", "")
    if key not in _ENERGY_ALIASES:
        raise ValueError(f"Unsupported energy unit: {unit!r}")
    return _ENERGY_ALIASES[key]  # type: ignore[return-value]


def normalize_force_unit(unit: str) -> ForceUnit:
    key = str(unit).strip().lower().replace("-", "_").replace(" ", "").replace("/", "_")
    if key not in _FORCE_ALIASES:
        raise ValueError(f"Unsupported force unit: {unit!r}")
    return _FORCE_ALIASES[key]  # type: ignore[return-value]


def normalize_length_unit(unit: str) -> LengthUnit:
    key = str(unit).strip().lower().replace("-", "_").replace(" ", "")
    if key not in _LENGTH_ALIASES:
        raise ValueError(f"Unsupported length unit: {unit!r}")
    return _LENGTH_ALIASES[key]  # type: ignore[return-value]


def normalize_dipole_unit(unit: str) -> DipoleUnit:
    key = str(unit).strip().lower().replace("-", "_").replace(" ", "").replace("*", "_")
    if key not in _DIPOLE_ALIASES:
        raise ValueError(f"Unsupported dipole unit: {unit!r}")
    return _DIPOLE_ALIASES[key]  # type: ignore[return-value]


def convert_energy(
    values: np.ndarray | float,
    from_unit: str,
    to_unit: str,
) -> np.ndarray | float:
    """Convert energy values between supported units."""
    src = normalize_energy_unit(from_unit)
    dst = normalize_energy_unit(to_unit)
    arr = np.asarray(values, dtype=np.float64)
    scalar = arr.ndim == 0
    if src == dst:
        out = arr
    else:
        ha = arr
        if src == "ev":
            ha = arr * EV_TO_HARTREE
        elif src == "kcal_mol":
            ha = arr * KCAL_MOL_TO_HARTREE
        if dst == "hartree":
            out = ha
        elif dst == "ev":
            out = ha * HARTREE_TO_EV
        elif dst == "kcal_mol":
            out = ha * HARTREE_TO_KCAL_MOL
        else:  # pragma: no cover
            raise ValueError(f"Unsupported target energy unit: {to_unit}")
    return float(out) if scalar else out


def format_energy_ev_kcal(
    energy_ev: float,
    *,
    ev_digits: int = 6,
    kcal_digits: int = 4,
) -> str:
    """Format hybrid/ML energy as ``X eV (Y kcal/mol)``."""
    e_ev = float(energy_ev)
    e_kcal = e_ev * EV_TO_KCAL_MOL
    return f"{e_ev:.{ev_digits}f} eV ({e_kcal:.{kcal_digits}f} kcal/mol)"


def format_energy_kcal_ev(
    energy_kcal: float,
    *,
    kcal_digits: int = 4,
    ev_digits: int = 6,
) -> str:
    """Format CHARMM energy as ``X kcal/mol (Y eV)``."""
    e_kcal = float(energy_kcal)
    e_ev = e_kcal * KCAL_MOL_TO_EV
    return f"{e_kcal:.{kcal_digits}f} kcal/mol ({e_ev:.{ev_digits}f} eV)"


def format_grms_kcal_ev_a(
    grms_kcal_mol_a: float,
    *,
    kcal_digits: int = 4,
    ev_digits: int = 4,
) -> str:
    """Format GRMS as ``X kcal/mol/Å (Y eV/Å)``."""
    g_kcal = float(grms_kcal_mol_a)
    g_ev = g_kcal * KCAL_MOL_TO_EV
    return f"{g_kcal:.{kcal_digits}f} kcal/mol/Å ({g_ev:.{ev_digits}f} eV/Å)"


def format_fmax_ev_kcal_a(
    fmax_ev_a: float,
    *,
    ev_digits: int = 4,
    kcal_digits: int = 4,
) -> str:
    """Format max force as ``X eV/Å (Y kcal/mol/Å)``."""
    f_ev = float(fmax_ev_a)
    f_kcal = f_ev * EV_TO_KCAL_MOL
    return f"{f_ev:.{ev_digits}f} eV/Å ({f_kcal:.{kcal_digits}f} kcal/mol/Å)"


def convert_forces(
    values: np.ndarray | float,
    from_unit: str,
    to_unit: str,
) -> np.ndarray | float:
    """Convert force values between supported units."""
    src = normalize_force_unit(from_unit)
    dst = normalize_force_unit(to_unit)
    arr = np.asarray(values, dtype=np.float64)
    scalar = arr.ndim == 0
    if src == dst:
        out = arr
    else:
        ev_ang = arr
        if src == "hartree_bohr":
            ev_ang = arr * HARTREE_BOHR_TO_EV_ANGSTROM
        elif src == "kcal_mol_angstrom":
            ev_ang = arr * KCAL_MOL_TO_EV
        if dst == "ev_angstrom":
            out = ev_ang
        elif dst == "hartree_bohr":
            out = ev_ang * EV_ANGSTROM_TO_HARTREE_BOHR
        elif dst == "kcal_mol_angstrom":
            out = ev_ang * EV_TO_KCAL_MOL
        else:  # pragma: no cover
            raise ValueError(f"Unsupported target force unit: {to_unit}")
    return float(out) if scalar else out


def convert_coords(
    values: np.ndarray | float,
    from_unit: str,
    to_unit: str,
) -> np.ndarray | float:
    """Convert coordinate values between Angstrom and Bohr."""
    src = normalize_length_unit(from_unit)
    dst = normalize_length_unit(to_unit)
    arr = np.asarray(values, dtype=np.float64)
    scalar = arr.ndim == 0
    if src == dst:
        out = arr
    elif src == "angstrom" and dst == "bohr":
        out = arr * ANGSTROM_TO_BOHR
    elif src == "bohr" and dst == "angstrom":
        out = arr * BOHR_TO_ANGSTROM
    else:  # pragma: no cover
        raise ValueError(f"Unsupported coordinate conversion: {from_unit} -> {to_unit}")
    return float(out) if scalar else out


def convert_dipole(
    values: np.ndarray | float,
    from_unit: str,
    to_unit: str,
) -> np.ndarray | float:
    """Convert dipole values between Debye and e·Angstrom."""
    src = normalize_dipole_unit(from_unit)
    dst = normalize_dipole_unit(to_unit)
    arr = np.asarray(values, dtype=np.float64)
    scalar = arr.ndim == 0
    if src == dst:
        out = arr
    elif src == "debye" and dst == "e_angstrom":
        out = arr * DEBYE_TO_EANGSTROM
    elif src == "e_angstrom" and dst == "debye":
        out = arr * EANGSTROM_TO_DEBYE
    else:  # pragma: no cover
        raise ValueError(f"Unsupported dipole conversion: {from_unit} -> {to_unit}")
    return float(out) if scalar else out


def energy_to_ev(values: np.ndarray | float, unit: str) -> np.ndarray | float:
    """Convert energy to eV (convenience wrapper)."""
    return convert_energy(values, unit, "ev")


def _as_numeric_energy_array(values: np.ndarray) -> np.ndarray:
    """Coerce NPZ ``E`` values to float64, rejecting unit-label strings."""
    raw = np.asarray(values, dtype=object).reshape(-1)
    out = np.empty(int(raw.shape[0]), dtype=np.float64)
    for i, val in enumerate(raw):
        if isinstance(val, (str, bytes)):
            text = val.decode("utf-8") if isinstance(val, bytes) else str(val)
            raise ValueError(
                f"Reference energy index {i} is non-numeric ({text!r}). "
                "Use numeric 'E' or canonical 'E_eV' arrays."
            )
        out[i] = float(val)
    return out


def _load_reference_energy_array(data: Any) -> np.ndarray | None:
    """Load per-frame reference energies from NPZ data (no unit inference)."""
    files = list(getattr(data, "files", data.keys()))
    if "E_eV" in files:
        return np.asarray(data["E_eV"], dtype=np.float64).reshape(-1)
    if "E" not in files:
        return None
    try:
        return np.asarray(data["E"], dtype=np.float64).reshape(-1)
    except (ValueError, TypeError):
        return _as_numeric_energy_array(np.asarray(data["E"]))


def load_reference_energies_from_npz(
    data: Any,
    *,
    path: Path | str | None = None,
) -> tuple[np.ndarray | None, str]:
    """Load per-frame reference energies and their unit from an NPZ."""
    files = list(getattr(data, "files", data.keys()))
    if "E_eV" in files:
        return _load_reference_energy_array(data), "ev"
    if "E" not in files:
        return None, infer_reference_energy_unit(path)
    return _load_reference_energy_array(data), infer_reference_energy_unit(path)


def reference_energy_ev_at_frame(
    data: Any,
    frame: int,
    *,
    path: Path | str | None = None,
    energy_unit: str | None = None,
) -> tuple[float, str, float]:
    """Return ``(energy_eV, unit, raw_value)`` for one reference frame."""
    energies, default_unit = load_reference_energies_from_npz(data, path=path)
    if energies is None:
        raise KeyError("Reference NPZ has no numeric 'E' or 'E_eV' energies")
    unit = str(energy_unit or default_unit)
    raw = float(energies[int(frame)])
    if normalize_energy_unit(unit) == "ev":
        return raw, unit, raw
    ev = float(energy_to_ev(raw, unit))
    return ev, unit, raw


def forces_to_ev_angstrom(values: np.ndarray | float, unit: str) -> np.ndarray | float:
    """Convert forces to eV/Angstrom (convenience wrapper)."""
    return convert_forces(values, unit, "ev_angstrom")


def subtract_atom_refs(
    energies: np.ndarray,
    atomic_numbers: np.ndarray,
    *,
    energy_unit: str = "ev",
    level: str | None = None,
    charge_state: int = 0,
) -> np.ndarray:
    """Subtract per-atom reference energies from total energies."""
    from mmml.data.atomic_references import (
        DEFAULT_REFERENCE_LEVEL,
        get_atomic_reference_array,
    )

    unit = normalize_energy_unit(energy_unit)
    ref_unit = "ev" if unit == "ev" else "hartree"
    ref_level = level or DEFAULT_REFERENCE_LEVEL
    ref_array = get_atomic_reference_array(
        level=ref_level,
        charge_state=charge_state,
        unit=ref_unit,
    )
    z = np.asarray(atomic_numbers, dtype=np.int32)
    e = np.asarray(energies, dtype=np.float64)
    if z.ndim == 1:
        atom_refs = ref_array[z].sum()
        return e - atom_refs
    atom_refs = ref_array[z].sum(axis=-1)
    if e.ndim == 1:
        return e - atom_refs
    return e - atom_refs.reshape(e.shape)


@dataclass
class UnitsManifestV2:
    """Recorded in units_manifest.json for downstream loaders (schema v2)."""

    schema_version: int = 2
    canonical: dict[str, str] = field(
        default_factory=lambda: {
            "energy": CANONICAL_ENERGY_UNIT,
            "force": CANONICAL_FORCE_UNIT,
            "coords": CANONICAL_LENGTH_UNIT,
        }
    )
    arrays: dict[str, str] = field(default_factory=dict)
    preserve_units: bool = False
    notes: list[str] = field(default_factory=list)
    # v1 compatibility fields (optional)
    coords_in: str | None = None
    coords_out: str | None = None
    coords_detected: str | None = None
    energy_in: str | None = None
    energy_out: str | None = None
    force_in: str | None = None
    force_out: str | None = None
    dipole_in: str | None = None
    dipole_out: str | None = None
    grid_coords_in: str | None = None
    grid_coords_out: str | None = None
    esp_values: str | None = "hartree/e"
    flip_forces: bool = False

    def array_unit(self, key: str, default: str | None = None) -> str | None:
        if key in self.arrays:
            return self.arrays[key]
        return default

    def energy_unit(self) -> str:
        return self.arrays.get("E") or self.energy_out or CANONICAL_ENERGY_UNIT

    def force_unit(self) -> str:
        return self.arrays.get("F") or self.force_out or CANONICAL_FORCE_UNIT

    def coords_unit(self) -> str:
        return self.arrays.get("R") or self.coords_out or CANONICAL_LENGTH_UNIT

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> UnitsManifestV2:
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: data[k] for k in known if k in data}
        if "arrays" not in kwargs:
            kwargs["arrays"] = _arrays_from_v1_fields(data)
        if kwargs.get("schema_version", 1) < 2:
            kwargs["schema_version"] = 2
            kwargs.setdefault("arrays", _arrays_from_v1_fields(data))
            kwargs.setdefault(
                "canonical",
                {
                    "energy": kwargs.get("energy_out") or CANONICAL_ENERGY_UNIT,
                    "force": kwargs.get("force_out") or CANONICAL_FORCE_UNIT,
                    "coords": kwargs.get("coords_out") or CANONICAL_LENGTH_UNIT,
                },
            )
        return cls(**kwargs)


def _arrays_from_v1_fields(data: Mapping[str, Any]) -> dict[str, str]:
    arrays: dict[str, str] = {}
    if "coords_out" in data and data["coords_out"]:
        arrays["R"] = str(data["coords_out"])
    if "energy_out" in data and data["energy_out"]:
        arrays["E"] = str(data["energy_out"])
        for key in ("efield_energy", "efield_scf_energy"):
            arrays[key] = str(data["energy_out"])
    if "force_out" in data and data["force_out"]:
        arrays["F"] = str(data["force_out"])
        arrays["efield_scf_F"] = str(data["force_out"])
    if "dipole_out" in data and data.get("dipole_out"):
        arrays["Dxyz"] = str(data["dipole_out"])
        for key in ("D", "efield_scf_D", "efield_D"):
            arrays[key] = str(data["dipole_out"])
    if "grid_coords_out" in data and data.get("grid_coords_out"):
        for key in ("vdw_surface", "vdw_grid", "esp_grid"):
            arrays[key] = str(data["grid_coords_out"])
    if data.get("esp_values"):
        arrays["esp"] = str(data["esp_values"]).replace("/", "_per_").replace("hartree/e", "hartree_per_e")
    for key in ("Ef", "efield_Ef", "efield_scf_Ef"):
        arrays[key] = "atomic_unit_field"
    return arrays


def load_units_manifest(path: Path | str) -> UnitsManifestV2 | None:
    """Load units_manifest.json from a file or directory."""
    p = Path(path).expanduser()
    if p.is_dir():
        p = p / "units_manifest.json"
    if not p.is_file():
        return None
    with p.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return UnitsManifestV2.from_dict(data)


def find_units_manifest(npz_path: Path | str) -> UnitsManifestV2 | None:
    """Search for units_manifest.json near an NPZ file."""
    p = Path(npz_path).expanduser().resolve()
    for candidate in (p.parent, p.parent.parent):
        manifest = load_units_manifest(candidate)
        if manifest is not None:
            return manifest
    return None


def units_from_npz(npz_path: Path | str) -> UnitsManifestV2 | None:
    """Read embedded _mmml_units from NPZ if present, else nearby manifest."""
    p = Path(npz_path).expanduser()
    try:
        with np.load(p, allow_pickle=True) as data:
            if "_mmml_units" in data.files:
                raw = data["_mmml_units"]
                if isinstance(raw, np.ndarray) and raw.shape == ():
                    raw = raw.item()
                if isinstance(raw, (bytes, str)):
                    payload = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8"))
                    if "arrays" in payload or "energy_out" in payload:
                        return UnitsManifestV2.from_dict(payload)
                    return UnitsManifestV2(
                        arrays={k: str(v) for k, v in payload.items()},
                        preserve_units=False,
                    )
    except Exception:
        pass
    return find_units_manifest(p)


def _units_from_npz_metadata_json(raw: Any) -> tuple[str | None, str | None]:
    if raw is None:
        return None, None
    if isinstance(raw, np.ndarray) and raw.dtype == object and raw.shape == ():
        raw = raw.item()
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None, None
    elif isinstance(raw, Mapping):
        payload = dict(raw)
    else:
        return None, None
    arrays = payload.get("arrays") if isinstance(payload.get("arrays"), Mapping) else payload
    if not isinstance(arrays, Mapping):
        return None, None
    e_unit = arrays.get("E") or payload.get("energy_out") or payload.get("energy_unit")
    f_unit = arrays.get("F") or payload.get("force_out") or payload.get("force_unit")
    return (
        str(e_unit) if e_unit is not None else None,
        str(f_unit) if f_unit is not None else None,
    )


def _infer_reference_units_from_arrays(npz_path: Path | str) -> tuple[str | None, str | None]:
    """Guess reference units from NPZ arrays when no manifest is embedded."""
    p = Path(npz_path).expanduser()
    try:
        with np.load(p, allow_pickle=True) as data:
            if "E_eV" in data.files:
                return "ev", "ev_angstrom"
            meta_e, meta_f = _units_from_npz_metadata_json(data.get("metadata"))
            if meta_e is not None or meta_f is not None:
                return meta_e, meta_f
            if "E" not in data.files:
                return None, None
            try:
                e = _load_reference_energy_array(data)
            except ValueError:
                return None, None
            if e is None or e.size == 0:
                return None, None
            e_sample = e[np.isfinite(e)][:32]
            if e_sample.size == 0:
                return None, None
            if "F" in data.files and np.size(data["F"]) > 0:
                f = np.asarray(data["F"], dtype=np.float64)
                f_mag = float(np.nanmax(np.abs(f)))
                if f_mag >= 0.5:
                    return "ev", "ev_angstrom"
                if f_mag > 0.0:
                    return "hartree", "hartree_bohr"
            if float(np.nanmedian(np.abs(e_sample))) > 150.0:
                return "ev", "ev_angstrom"
    except Exception:
        return None, None
    return None, None


def infer_reference_energy_unit(
    npz_path: Path | str | None = None,
    *,
    manifest: UnitsManifestV2 | None = None,
    default: str = "hartree",
) -> str:
    """Best-effort energy unit for a reference NPZ."""
    if manifest is None and npz_path is not None:
        manifest = units_from_npz(npz_path)
    if manifest is not None:
        return manifest.energy_unit()
    if npz_path is not None:
        inferred_e, _ = _infer_reference_units_from_arrays(npz_path)
        if inferred_e is not None:
            return inferred_e
    return default


def infer_reference_force_unit(
    npz_path: Path | str | None = None,
    *,
    manifest: UnitsManifestV2 | None = None,
    default: str = "hartree_bohr",
) -> str:
    if manifest is None and npz_path is not None:
        manifest = units_from_npz(npz_path)
    if manifest is not None:
        return manifest.force_unit()
    if npz_path is not None:
        _, inferred_f = _infer_reference_units_from_arrays(npz_path)
        if inferred_f is not None:
            return inferred_f
    return default


def normalize_to_canonical(
    data: MutableMapping[str, Any],
    manifest: UnitsManifestV2 | None = None,
    *,
    allow_hartree: bool = False,
) -> MutableMapping[str, Any]:
    """Convert NPZ-like dict arrays to canonical eV/eV-Å/Å where applicable."""
    out = dict(data)
    energy_unit = manifest.energy_unit() if manifest else CANONICAL_ENERGY_UNIT
    force_unit = manifest.force_unit() if manifest else CANONICAL_FORCE_UNIT
    coords_unit = manifest.coords_unit() if manifest else CANONICAL_LENGTH_UNIT

    if normalize_energy_unit(energy_unit) != CANONICAL_ENERGY_UNIT:
        if not allow_hartree:
            warnings.warn(
                f"Energy unit {energy_unit!r} is not canonical eV; "
                "re-run fix-and-split without --preserve-units or pass allow_hartree=True",
                stacklevel=2,
            )
        if "E" in out:
            out["E"] = convert_energy(out["E"], energy_unit, CANONICAL_ENERGY_UNIT)
        for key in ("efield_energy", "efield_scf_energy"):
            if key in out:
                out[key] = convert_energy(out[key], energy_unit, CANONICAL_ENERGY_UNIT)

    if normalize_force_unit(force_unit) != CANONICAL_FORCE_UNIT:
        if "F" in out:
            out["F"] = convert_forces(out["F"], force_unit, CANONICAL_FORCE_UNIT)
        if "efield_scf_F" in out:
            out["efield_scf_F"] = convert_forces(
                out["efield_scf_F"], force_unit, CANONICAL_FORCE_UNIT
            )

    if normalize_length_unit(coords_unit) != CANONICAL_LENGTH_UNIT:
        if "R" in out:
            out["R"] = convert_coords(out["R"], coords_unit, CANONICAL_LENGTH_UNIT)

    if manifest and manifest.dipole_out and "Dxyz" in out:
        dip_out = manifest.dipole_out
        if dip_out and dip_out != CANONICAL_DIPOLE_UNIT:
            out["Dxyz"] = convert_dipole(out["Dxyz"], dip_out, CANONICAL_DIPOLE_UNIT)

    return out


def pyscf_units_metadata() -> dict[str, str]:
    """Native PySCF export units for embedding in NPZ."""
    return dict(PYSCF_NATIVE_UNITS)


def pyscf_units_json() -> str:
    return json.dumps(pyscf_units_metadata())


def attach_units_to_npz_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return payload copy with embedded _mmml_units JSON string."""
    out = dict(payload)
    out["_mmml_units"] = np.array(pyscf_units_json())
    return out


def calculator_results_units() -> dict[str, str]:
    """Standard unit metadata for hybrid calculator ASE results."""
    return dict(CALCULATOR_UNITS)
