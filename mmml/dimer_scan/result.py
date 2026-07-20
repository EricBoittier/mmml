"""Versioned dimer-scan records and self-describing artifact bundles."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import ase
from ase import Atoms
from ase.io import read, write

from .config import DimerScanConfig, RESULT_SCHEMA_VERSION


@dataclass(frozen=True)
class ScanRecord:
    """One requested scan point, whether successful or failed."""

    point_id: str
    index: int
    distance_angstrom: float
    min_contact_angstrom: float
    status: Literal["success", "failed"]
    energy_ev: float | None = None
    energy_kcal_mol: float | None = None
    total_energy_ev: float | None = None
    error_type: str | None = None
    error_message: str | None = None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_provenance(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {resolved}")
    if resolved.is_file():
        return {
            "path": str(resolved),
            "kind": "file",
            "size_bytes": resolved.stat().st_size,
            "sha256": sha256_file(resolved),
        }
    files = sorted(item for item in resolved.rglob("*") if item.is_file())
    digest = hashlib.sha256()
    for item in files:
        relative = item.relative_to(resolved).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(item)))
    return {
        "path": str(resolved),
        "kind": "directory",
        "file_count": len(files),
        "sha256": digest.hexdigest(),
    }


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


@dataclass(frozen=True)
class Provenance:
    """Runtime and scientific-input identity for a scan."""

    created_utc: str
    software: dict[str, str | None]
    platform: dict[str, str]
    git: dict[str, Any]
    checkpoint: dict[str, Any] | None
    calculator_inputs: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def capture(cls, config: DimerScanConfig) -> Provenance:
        calculator_inputs = {}
        for name in ("calculator_config", "slako_dir"):
            value = getattr(config, name)
            if value is not None:
                provenance = _checkpoint_provenance(value)
                if provenance is not None:
                    calculator_inputs[name] = provenance
        if config.executable:
            executable = shutil.which(config.executable)
            if executable is not None:
                provenance = _checkpoint_provenance(Path(executable))
                if provenance is not None:
                    calculator_inputs["executable"] = provenance
        return cls(
            created_utc=datetime.now(UTC).isoformat(),
            software={
                "mmml": _package_version("mmml"),
                "python": platform.python_version(),
                "ase": ase.__version__,
                "numpy": _package_version("numpy"),
                "jax": _package_version("jax"),
            },
            platform={
                "system": platform.system(),
                "machine": platform.machine(),
                "hostname": platform.node(),
            },
            git=_git_state(),
            checkpoint=_checkpoint_provenance(config.checkpoint),
            calculator_inputs=calculator_inputs,
        )


@dataclass
class ScanResult:
    """In-memory result with atomic bundle serialization."""

    config: DimerScanConfig
    records: list[ScanRecord]
    frames: list[Atoms]
    provenance: Provenance

    @property
    def has_failures(self) -> bool:
        return any(record.status == "failed" for record in self.records)

    def _manifest(self) -> dict[str, Any]:
        successes = sum(record.status == "success" for record in self.records)
        return {
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "orientation_schema_version": self.config.orientation_schema_version,
            "config": self.config.to_dict(resolve_paths=True),
            "provenance": asdict(self.provenance),
            "counts": {
                "requested": len(self.records),
                "successful": successes,
                "failed": len(self.records) - successes,
            },
            "units": {
                "distance": "angstrom",
                "energy": "eV",
                "force": "eV/angstrom",
            },
            "files": {
                "records": "data.csv",
                "trajectory": "trajectory.extxyz",
            },
        }

    def write(self, output_dir: str | Path, *, overwrite: bool = False) -> dict[str, Path]:
        """Write an artifact bundle atomically and return its principal paths."""

        target = Path(output_dir).expanduser().resolve()
        if target.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing result bundle: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
        try:
            config_path = temp / "resolved_config.json"
            config_path.write_text(
                json.dumps(self.config.to_dict(resolve_paths=True), indent=2) + "\n"
            )
            data_path = temp / "data.csv"
            field_names = [field.name for field in fields(ScanRecord)]
            with data_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=field_names)
                writer.writeheader()
                writer.writerows(asdict(record) for record in self.records)
            trajectory_path = temp / "trajectory.extxyz"
            write(trajectory_path, self.frames, format="extxyz")
            ase_trajectory_path = temp / "trajectory.traj"
            write(ase_trajectory_path, self.frames, format="traj")
            from .plotting import plot_energy

            plot_path = plot_energy(self, temp / "energy.png")
            manifest = self._manifest()
            manifest["files"].update(
                ase_trajectory="trajectory.traj",
                plot="energy.png",
            )
            manifest["output_sha256"] = {
                "resolved_config.json": sha256_file(config_path),
                "data.csv": sha256_file(data_path),
                "trajectory.extxyz": sha256_file(trajectory_path),
                "trajectory.traj": sha256_file(ase_trajectory_path),
                "energy.png": sha256_file(plot_path),
            }
            (temp / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            if target.exists():
                if not overwrite:
                    raise FileExistsError(target)
                shutil.rmtree(target)
            os.replace(temp, target)
        except Exception:
            shutil.rmtree(temp, ignore_errors=True)
            raise
        return {
            "manifest": target / "manifest.json",
            "data": target / "data.csv",
            "trajectory": target / "trajectory.extxyz",
            "ase_trajectory": target / "trajectory.traj",
            "plot": target / "energy.png",
        }

    @classmethod
    def read(cls, output_dir: str | Path) -> ScanResult:
        """Load and validate a result bundle written by :meth:`write`."""

        root = Path(output_dir).expanduser().resolve()
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("result_schema_version") != RESULT_SCHEMA_VERSION:
            raise ValueError(
                "unsupported dimer-scan result schema: "
                f"{manifest.get('result_schema_version')!r}"
            )
        for name, expected in manifest.get("output_sha256", {}).items():
            actual = sha256_file(root / name)
            if actual != expected:
                raise ValueError(f"artifact checksum mismatch: {name}")
        records: list[ScanRecord] = []
        with (root / "data.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                records.append(
                    ScanRecord(
                        point_id=row["point_id"],
                        index=int(row["index"]),
                        distance_angstrom=float(row["distance_angstrom"]),
                        min_contact_angstrom=float(row["min_contact_angstrom"]),
                        status=row["status"],  # type: ignore[arg-type]
                        energy_ev=float(row["energy_ev"]) if row["energy_ev"] else None,
                        energy_kcal_mol=(
                            float(row["energy_kcal_mol"])
                            if row["energy_kcal_mol"]
                            else None
                        ),
                        total_energy_ev=(
                            float(row["total_energy_ev"])
                            if row["total_energy_ev"]
                            else None
                        ),
                        error_type=row["error_type"] or None,
                        error_message=row["error_message"] or None,
                    )
                )
        frames = read(root / "trajectory.extxyz", index=":")
        if not isinstance(frames, list):
            frames = [frames]
        provenance = Provenance(**manifest["provenance"])
        return cls(
            config=DimerScanConfig.from_dict(manifest["config"]),
            records=records,
            frames=frames,
            provenance=provenance,
        )
