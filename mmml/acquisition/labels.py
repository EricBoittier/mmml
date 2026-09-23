"""Reference-label cache and backends.

Expensive energies and forces are keyed by ``(structure_id, method_fingerprint)``
so overlapping selections reuse calculations.  Acquisition methods never call
this module; only the post-selection labeling stages do.

The mock backend is a deterministic Morse-like pair potential.  It is **not**
a scientific reference method — smoke and unit tests only.  Production must
set ``reference.backend`` to a real provider (PySCF / ORCA / Molpro / …).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from mmml.acquisition.ids import structure_id as make_sid
from mmml.acquisition.splits import StructureRecord


class ReferenceBackend(Protocol):
    name: str
    method_label: str

    def fingerprint(self) -> dict[str, Any]: ...

    def compute(self, record: StructureRecord) -> "LabelResult": ...


@dataclass
class LabelResult:
    structure_id: str
    energy: float
    forces: np.ndarray
    success: bool
    error: str | None = None
    backend: str = ""
    method_label: str = ""
    extra: dict[str, Any] | None = None


def method_hash(fingerprint: Mapping[str, Any]) -> str:
    blob = json.dumps(fingerprint, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


class LabelCache:
    """On-disk cache: ``root / structure_id / method_hash / labels.npz``."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        self.failures = 0

    def _dir(self, structure_id: str, mh: str) -> Path:
        return self.root / structure_id / mh

    def get(self, structure_id: str, fingerprint: Mapping[str, Any]) -> LabelResult | None:
        d = self._dir(structure_id, method_hash(fingerprint))
        path = d / "labels.npz"
        if not path.is_file():
            return None
        data = np.load(path, allow_pickle=True)
        self.hits += 1
        return LabelResult(
            structure_id=str(data["structure_id"]),
            energy=float(data["energy"]),
            forces=np.asarray(data["forces"], dtype=np.float64),
            success=bool(data["success"]),
            error=None if data["error"].shape == () and not str(data["error"]) else str(data["error"]),
            backend=str(data["backend"]),
            method_label=str(data["method_label"]),
        )

    def put(self, result: LabelResult, fingerprint: Mapping[str, Any]) -> Path:
        d = self._dir(result.structure_id, method_hash(fingerprint))
        d.mkdir(parents=True, exist_ok=True)
        path = d / "labels.npz"
        np.savez_compressed(
            path,
            structure_id=result.structure_id,
            energy=np.float64(result.energy),
            forces=np.asarray(result.forces, dtype=np.float64),
            success=np.bool_(result.success),
            error=np.asarray(result.error or ""),
            backend=np.asarray(result.backend),
            method_label=np.asarray(result.method_label),
        )
        meta = {
            "structure_id": result.structure_id,
            "method_fingerprint": dict(fingerprint),
            "success": result.success,
            "backend": result.backend,
            "method_label": result.method_label,
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
        return path

    def compute_or_cached(
        self,
        record: StructureRecord,
        backend: ReferenceBackend,
    ) -> LabelResult:
        fp = backend.fingerprint()
        cached = self.get(record.structure_id, fp)
        if cached is not None:
            return cached
        self.misses += 1
        result = backend.compute(record)
        if not result.success:
            self.failures += 1
        self.put(result, fp)
        return result


class MockMorseReference:
    """Deterministic mock expensive backend (tests and smoke only)."""

    name = "mock_morse"
    method_label = "mock-morse-v1"

    def __init__(self, *, de: float = 4.0, a: float = 1.4, re: float = 1.1, fail_ids: set[str] | None = None):
        self.de = float(de)
        self.a = float(a)
        self.re = float(re)
        self.fail_ids = set(fail_ids or ())

    def fingerprint(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "method_label": self.method_label,
            "de": self.de,
            "a": self.a,
            "re": self.re,
        }

    def compute(self, record: StructureRecord) -> LabelResult:
        if record.structure_id in self.fail_ids:
            return LabelResult(
                structure_id=record.structure_id,
                energy=float("nan"),
                forces=np.full((record.n_atoms, 3), np.nan),
                success=False,
                error="injected failure",
                backend=self.name,
                method_label=self.method_label,
            )
        R = np.asarray(record.positions, dtype=np.float64)
        Z = np.asarray(record.atomic_numbers, dtype=np.float64)
        n = int(record.n_atoms)
        energy = 0.0
        forces = np.zeros((n, 3), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                rij = R[i] - R[j]
                dist = float(np.linalg.norm(rij) + 1e-12)
                scale = np.sqrt(Z[i] * Z[j])
                de = self.de * (0.25 + 0.15 * scale)
                x = np.exp(-self.a * (dist - self.re))
                pair_e = de * (1.0 - x) ** 2
                energy += pair_e
                dE_dd = 2.0 * de * (1.0 - x) * (x * self.a)
                f_i = -dE_dd * rij / dist
                forces[i] += f_i
                forces[j] -= f_i
        return LabelResult(
            structure_id=record.structure_id,
            energy=float(energy),
            forces=forces,
            success=True,
            backend=self.name,
            method_label=self.method_label,
        )


class CallableReference:
    """Wrap a ``(record) -> LabelResult`` function (tests / injected QC)."""

    def __init__(self, fn: Callable[[StructureRecord], LabelResult], *, name: str, method_label: str, extra_fp: dict | None = None):
        self._fn = fn
        self.name = name
        self.method_label = method_label
        self._extra = extra_fp or {}

    def fingerprint(self) -> dict[str, Any]:
        return {"backend": self.name, "method_label": self.method_label, **self._extra}

    def compute(self, record: StructureRecord) -> LabelResult:
        return self._fn(record)


def provider_reference_backend(spec: Mapping[str, Any]) -> ReferenceBackend:
    """Build a real :class:`EnergyForcesProvider` backend from a config mapping.

    This is the production hook.  It is not used by the smoke workflow.
    """
    from mmml.interfaces.energy_forces.registry import build_provider, provider_from_dict

    provider = build_provider(provider_from_dict(dict(spec)))

    def _compute(record: StructureRecord) -> LabelResult:
        from ase import Atoms

        atoms = Atoms(
            numbers=np.asarray(record.atomic_numbers),
            positions=np.asarray(record.positions),
        )
        if record.cell is not None:
            c = np.asarray(record.cell)
            if c.size == 3:
                atoms.set_cell(np.diag(c))
                atoms.set_pbc(True)
            elif c.size == 9:
                atoms.set_cell(c.reshape(3, 3))
                atoms.set_pbc(True)
        try:
            out = provider.evaluate_batch(
                [atoms], properties=frozenset({"energy", "forces"})
            )
            e = float(np.asarray(out["energy"]).reshape(-1)[0])
            f = np.asarray(out["forces"][0], dtype=np.float64)
            return LabelResult(
                structure_id=record.structure_id,
                energy=e,
                forces=f,
                success=True,
                backend=provider.name,
                method_label=provider.method_label,
            )
        except Exception as exc:  # noqa: BLE001 — record and continue the campaign
            return LabelResult(
                structure_id=record.structure_id,
                energy=float("nan"),
                forces=np.full((record.n_atoms, 3), np.nan),
                success=False,
                error=repr(exc),
                backend=provider.name,
                method_label=getattr(provider, "method_label", provider.name),
            )

    return CallableReference(
        _compute,
        name=provider.name,
        method_label=getattr(provider, "method_label", provider.name),
        extra_fp={"provider_spec": dict(spec)},
    )


def build_backend(cfg: Mapping[str, Any]) -> ReferenceBackend:
    kind = str(cfg.get("backend") or cfg.get("name") or "mock_morse").lower()
    if kind in ("mock", "mock_morse", "smoke"):
        return MockMorseReference(
            de=float(cfg.get("de", 4.0)),
            a=float(cfg.get("a", 1.4)),
            re=float(cfg.get("re", 1.1)),
        )
    if kind in ("pyscf", "orca", "orca_qm", "molpro", "xtb", "ml", "physnet", "metatomic"):
        return provider_reference_backend(cfg)
    raise ValueError(
        f"Unknown reference backend {kind!r}. Smoke uses mock_morse; "
        "production must name a real provider and is otherwise unresolved."
    )


def labels_to_npz(records: list[StructureRecord], results: list[LabelResult]) -> dict[str, np.ndarray]:
    if len(records) != len(results):
        raise ValueError("records / results length mismatch")
    max_n = max((r.n_atoms for r in records), default=1)
    n = len(records)
    R = np.zeros((n, max_n, 3), dtype=np.float64)
    Z = np.zeros((n, max_n), dtype=np.int32)
    F = np.zeros((n, max_n, 3), dtype=np.float64)
    E = np.full(n, np.nan, dtype=np.float64)
    N = np.zeros(n, dtype=np.int32)
    ok = np.zeros(n, dtype=np.bool_)
    ids = []
    errors = []
    for i, (rec, lab) in enumerate(zip(records, results)):
        n_i = rec.n_atoms
        N[i] = n_i
        R[i, :n_i] = rec.positions
        Z[i, :n_i] = rec.atomic_numbers
        E[i] = lab.energy
        if lab.forces is not None and lab.success:
            f = np.asarray(lab.forces)
            F[i, : n_i] = f[:n_i]
        ok[i] = lab.success
        ids.append(rec.structure_id)
        errors.append(lab.error or "")
    return {
        "R": R,
        "Z": Z,
        "N": N,
        "E": E,
        "F": F,
        "structure_id": np.asarray(ids, dtype=object),
        "label_success": ok,
        "label_error": np.asarray(errors, dtype=object),
    }


# Imported by tests that construct IDs without going through splits.
make_structure_id = make_sid
