"""Structured results for mode checks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import RESULT_SCHEMA_VERSION


@dataclass
class ModeCheckResult:
    """JSON-serializable mode-check bundle."""

    schema_version: str = RESULT_SCHEMA_VERSION
    config: dict[str, Any] = field(default_factory=dict)
    setup: dict[str, Any] = field(default_factory=dict)
    energy_eV: float | None = None
    max_force_eVA: float | None = None
    r_bonds_A: list[float] = field(default_factory=list)
    bond_pairs: list[list[int]] = field(default_factory=list)
    fd: dict[str, float] | None = None
    bond_scans: dict[str, dict[str, Any]] = field(default_factory=dict)
    vibrations: dict[str, Any] | None = None
    kick: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: Path, *, overwrite: bool = True) -> Path:
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Drop bulky per-point rows from the summary JSON; scans are written separately.
        payload = self.to_dict()
        for key, scan in list(payload.get("bond_scans", {}).items()):
            if isinstance(scan, dict) and "rows" in scan:
                slim = {k: v for k, v in scan.items() if k != "rows"}
                payload["bond_scans"][key] = slim
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path
