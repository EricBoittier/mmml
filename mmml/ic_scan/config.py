"""Serializable configuration for internal-coordinate (bond/angle/dihedral) scans."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Literal


RESULT_SCHEMA_VERSION = "1.0"
CONFIG_SCHEMA_VERSION = "1.0"

DofKind = Literal["bond", "angle", "dihedral"]
ScanMode = Literal["product", "individual"]
GeometryMode = Literal["rigid"]
FailurePolicy = Literal["fail", "allow_partial"]
EvaluateMode = Literal["none", "energy"]


def _finite(value: float, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def build_grid(
    *,
    start: float | None = None,
    stop: float | None = None,
    n_points: int | None = None,
    values: tuple[float, ...] | None = None,
    kind: DofKind,
) -> tuple[float, ...]:
    """Build an inclusive scan grid from either explicit values or start/stop/n_points."""

    if values is not None:
        if start is not None or stop is not None or n_points is not None:
            raise ValueError("provide either values or start/stop/n_points, not both")
        grid = tuple(_finite(v, name="grid value") for v in values)
        if not grid:
            raise ValueError("values must contain at least one point")
        return grid
    if start is None or stop is None or n_points is None:
        raise ValueError("each DoF needs values=... or start/stop/n_points")
    count = int(n_points)
    if count < 1:
        raise ValueError("n_points must be >= 1")
    start_f = _finite(start, name="start")
    stop_f = _finite(stop, name="stop")
    if count == 1:
        grid = (start_f,)
    else:
        grid = tuple(
            float(v) for v in __import__("numpy").linspace(start_f, stop_f, count)
        )
    if kind == "bond" and any(v <= 0.0 for v in grid):
        raise ValueError("bond grid values must be positive")
    return grid


@dataclass(frozen=True)
class DegreeOfFreedom:
    """One scanned internal coordinate defined by atom indices."""

    name: str
    kind: DofKind
    atoms: tuple[int, ...]
    values: tuple[float, ...]
    mask: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("DoF name must not be empty")
        kind = str(self.kind).lower()
        if kind not in ("bond", "angle", "dihedral"):
            raise ValueError(f"unsupported DoF kind: {self.kind!r}")
        expected = {"bond": 2, "angle": 3, "dihedral": 4}[kind]
        atoms = tuple(int(i) for i in self.atoms)
        if len(atoms) != expected:
            raise ValueError(f"{kind} DoF requires exactly {expected} atom indices")
        if len(set(atoms)) != len(atoms):
            raise ValueError(f"{name}: atom indices must be distinct")
        if any(i < 0 for i in atoms):
            raise ValueError(f"{name}: atom indices must be non-negative")
        values = tuple(_finite(v, name=f"{name} value") for v in self.values)
        if not values:
            raise ValueError(f"{name}: values must contain at least one point")
        if kind == "bond" and any(v <= 0.0 for v in values):
            raise ValueError(f"{name}: bond lengths must be positive")
        mask = None if self.mask is None else tuple(int(i) for i in self.mask)
        if mask is not None:
            if any(i < 0 for i in mask):
                raise ValueError(f"{name}: mask indices must be non-negative")
            if len(set(mask)) != len(mask):
                raise ValueError(f"{name}: mask indices must be unique")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "atoms", atoms)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "mask", mask)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "name": self.name,
            "kind": self.kind,
            "atoms": list(self.atoms),
            "values": list(self.values),
        }
        if self.mask is not None:
            data["mask"] = list(self.mask)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DegreeOfFreedom:
        payload = dict(data)
        kind = str(payload.get("kind", "bond")).lower()
        if "values" in payload:
            for key in ("start", "stop", "n_points"):
                payload.pop(key, None)
            payload["values"] = tuple(payload["values"])
        else:
            payload["values"] = build_grid(
                start=payload.pop("start", None),
                stop=payload.pop("stop", None),
                n_points=payload.pop("n_points", None),
                kind=kind,  # type: ignore[arg-type]
            )
        payload["atoms"] = tuple(payload["atoms"])
        if payload.get("mask") is not None:
            payload["mask"] = tuple(payload["mask"])
        return cls(**payload)


@dataclass(frozen=True)
class ScanSpec:
    """One named scan job selecting a subset of configured DoFs."""

    name: str
    dofs: tuple[str, ...]

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("scan name must not be empty")
        dofs = tuple(str(item).strip() for item in self.dofs)
        if not dofs:
            raise ValueError(f"scan {name!r} must select at least one DoF")
        if len(set(dofs)) != len(dofs):
            raise ValueError(f"scan {name!r} has duplicate DoF names")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "dofs", dofs)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "dofs": list(self.dofs)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScanSpec:
        return cls(name=data["name"], dofs=tuple(data["dofs"]))


@dataclass(frozen=True)
class IcScanConfig:
    """Complete scientific definition of an internal-coordinate scan campaign."""

    structure: Path
    dofs: tuple[DegreeOfFreedom, ...]
    calculator: str | None = None
    checkpoint: Path | None = None
    scan_mode: ScanMode = "product"
    scans: tuple[ScanSpec, ...] | None = None
    geometry_mode: GeometryMode = "rigid"
    evaluate: EvaluateMode = "energy"
    failure_policy: FailurePolicy = "fail"
    reference: dict[str, float] = field(default_factory=dict)
    charge: float | None = None
    spin: float | None = None
    method: str | None = None
    basis: str | None = None
    xc: str | None = None
    calculator_config: Path | None = None
    electric_field_au: tuple[float, float, float] | None = None
    slako_dir: Path | None = None
    workdir: Path | None = None
    executable: str | None = None
    multipole_force_step_angstrom: float = 1.0e-4
    seed: int = 0
    config_schema_version: str = CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        structure = Path(self.structure).expanduser()
        dofs = tuple(self.dofs)
        if not dofs:
            raise ValueError("dofs must contain at least one degree of freedom")
        names = [dof.name for dof in dofs]
        if len(set(names)) != len(names):
            raise ValueError("DoF names must be unique")
        scan_mode = str(self.scan_mode).lower()
        if scan_mode not in ("product", "individual"):
            raise ValueError("scan_mode must be 'product' or 'individual'")
        geometry_mode = str(self.geometry_mode).lower()
        if geometry_mode not in ("rigid",):
            raise ValueError("geometry_mode currently supports only 'rigid'")
        evaluate = str(self.evaluate).lower()
        if evaluate not in ("none", "energy"):
            raise ValueError("evaluate must be 'none' or 'energy'")
        if evaluate == "energy" and not self.calculator:
            raise ValueError("calculator is required when evaluate='energy'")
        if evaluate == "none" and self.calculator is not None:
            # Allowed: prepare-only with calculator recorded for later use.
            pass
        failure_policy = str(self.failure_policy).lower()
        if failure_policy not in ("fail", "allow_partial"):
            raise ValueError("failure_policy must be 'fail' or 'allow_partial'")
        reference = {
            str(key): _finite(value, name=f"reference[{key}]")
            for key, value in dict(self.reference).items()
        }
        unknown = set(reference) - set(names)
        if unknown:
            raise ValueError(f"reference keys unknown DoFs: {sorted(unknown)}")
        scans = self.scans
        if scans is not None:
            scans = tuple(scans)
            scan_names = [scan.name for scan in scans]
            if len(set(scan_names)) != len(scan_names):
                raise ValueError("scan names must be unique")
            known = set(names)
            for scan in scans:
                missing = set(scan.dofs) - known
                if missing:
                    raise ValueError(
                        f"scan {scan.name!r} references unknown DoFs: {sorted(missing)}"
                    )
        calculator = (
            None if self.calculator is None else str(self.calculator).lower()
        )
        object.__setattr__(self, "structure", structure)
        object.__setattr__(self, "dofs", dofs)
        object.__setattr__(self, "scan_mode", scan_mode)
        object.__setattr__(self, "geometry_mode", geometry_mode)
        object.__setattr__(self, "evaluate", evaluate)
        object.__setattr__(self, "failure_policy", failure_policy)
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "scans", scans)
        object.__setattr__(self, "calculator", calculator)
        if self.checkpoint is not None:
            object.__setattr__(self, "checkpoint", Path(self.checkpoint).expanduser())
        for name in ("calculator_config", "slako_dir", "workdir"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value).expanduser())
        if self.electric_field_au is not None:
            field = tuple(float(v) for v in self.electric_field_au)
            if len(field) != 3 or any(not math.isfinite(v) for v in field):
                raise ValueError("electric_field_au must contain three finite values")
            object.__setattr__(self, "electric_field_au", field)
        if self.multipole_force_step_angstrom <= 0.0:
            raise ValueError("multipole_force_step_angstrom must be positive")

    def dof_map(self) -> dict[str, DegreeOfFreedom]:
        return {dof.name: dof for dof in self.dofs}

    def resolved_scans(self) -> tuple[ScanSpec, ...]:
        """Return explicit scan jobs, synthesizing from scan_mode when needed."""

        if self.scans is not None:
            return self.scans
        names = tuple(dof.name for dof in self.dofs)
        if self.scan_mode == "product":
            return (ScanSpec(name="product", dofs=names),)
        return tuple(ScanSpec(name=name, dofs=(name,)) for name in names)

    def to_dict(self, *, resolve_paths: bool = False) -> dict[str, Any]:
        structure = self.structure.resolve() if resolve_paths else self.structure
        data: dict[str, Any] = {
            "config_schema_version": self.config_schema_version,
            "structure": str(structure),
            "dofs": [dof.to_dict() for dof in self.dofs],
            "calculator": self.calculator,
            "scan_mode": self.scan_mode,
            "geometry_mode": self.geometry_mode,
            "evaluate": self.evaluate,
            "failure_policy": self.failure_policy,
            "reference": dict(self.reference),
            "charge": self.charge,
            "spin": self.spin,
            "method": self.method,
            "basis": self.basis,
            "xc": self.xc,
            "multipole_force_step_angstrom": self.multipole_force_step_angstrom,
            "seed": self.seed,
        }
        if self.scans is not None:
            data["scans"] = [scan.to_dict() for scan in self.scans]
        if self.checkpoint is not None:
            path = self.checkpoint.resolve() if resolve_paths else self.checkpoint
            data["checkpoint"] = str(path)
        for name in ("calculator_config", "slako_dir", "workdir"):
            value = getattr(self, name)
            if value is not None:
                data[name] = str(value.resolve() if resolve_paths else value)
        if self.electric_field_au is not None:
            data["electric_field_au"] = list(self.electric_field_au)
        if self.executable is not None:
            data["executable"] = self.executable
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IcScanConfig:
        values = dict(data)
        values.pop("config_schema_version", None)
        values["structure"] = Path(values["structure"])
        values["dofs"] = tuple(
            DegreeOfFreedom.from_dict(item) for item in values["dofs"]
        )
        if values.get("scans") is not None:
            values["scans"] = tuple(ScanSpec.from_dict(item) for item in values["scans"])
        if values.get("checkpoint") is not None:
            values["checkpoint"] = Path(values["checkpoint"])
        for name in ("calculator_config", "slako_dir", "workdir"):
            if values.get(name) is not None:
                values[name] = Path(values[name])
        if values.get("electric_field_au") is not None:
            values["electric_field_au"] = tuple(values["electric_field_au"])
        values["reference"] = dict(values.get("reference") or {})
        # Drop unknown keys from forward-compatible YAML.
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        values = {key: value for key, value in values.items() if key in known}
        return cls(**values)
