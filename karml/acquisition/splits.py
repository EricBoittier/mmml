"""Duplicate removal and trajectory-grouped candidate / validation / test splits.

Held-out validation and test trajectories never enter PCA fitting or
acquisition.  Neighboring frames of one trajectory stay in a single split.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from mmml.acquisition.ids import (
    composition_key,
    geometry_fingerprint,
    group_key,
    stratum_key,
    structure_id,
)


SPLIT_CANDIDATE = "candidate"
SPLIT_VALID = "valid"
SPLIT_TEST = "test"
SPLIT_SEED = "seed"
ALL_SPLITS = (SPLIT_CANDIDATE, SPLIT_VALID, SPLIT_TEST, SPLIT_SEED)


@dataclass
class StructureRecord:
    """One validated candidate or labeled structure."""

    index: int
    structure_id: str
    geometry_fingerprint: str
    composition: str
    stratum: str
    group: str
    n_atoms: int
    atomic_numbers: np.ndarray
    positions: np.ndarray
    cell: np.ndarray | None = None
    temperature: float | None = None
    pressure: float | None = None
    phase: str | None = None
    frame_index: int | None = None
    source: str | None = None
    split: str = SPLIT_CANDIDATE
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class PoolManifest:
    records: list[StructureRecord]
    duplicate_of: dict[str, str]
    """Map of dropped input index → kept structure_id."""
    n_input: int
    n_unique: int
    n_duplicates_removed: int

    def by_split(self, split: str) -> list[StructureRecord]:
        return [r for r in self.records if r.split == split]

    def ids(self, split: str | None = None) -> list[str]:
        recs = self.records if split is None else self.by_split(split)
        return [r.structure_id for r in recs]

    def index_map(self) -> dict[str, int]:
        return {r.structure_id: r.index for r in self.records}


def _row_meta(data: Mapping[str, Any], i: int) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    for key in (
        "group_id",
        "trajectory_id",
        "parent_id",
        "group_seed",
        "group_file",
        "group_frame",
        "group_step",
        "group_phase",
        "source_path",
        "source",
        "temperature",
        "T",
        "pressure",
        "P",
        "phase",
        "ensemble",
        "id",
        "frame_index",
    ):
        if key not in data:
            continue
        arr = data[key]
        try:
            meta[key] = _as_scalar(arr[i])
        except Exception:
            try:
                meta[key] = _as_scalar(arr)
            except Exception:
                continue
    return meta


def _as_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        if value.size == 1:
            return value.reshape(-1)[0].item() if hasattr(value.reshape(-1)[0], "item") else value.reshape(-1)[0]
        return value
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def records_from_npz(data: Mapping[str, Any]) -> list[StructureRecord]:
    """Build records from an NPZ-like mapping (``R``, ``Z``, ``N`` required)."""
    R = np.asarray(data["R"])
    Z = np.asarray(data["Z"])
    if R.ndim != 3 or R.shape[-1] != 3:
        raise ValueError(f"R must have shape (n, pad, 3), got {R.shape}")
    if Z.ndim == 1:
        Z = np.broadcast_to(Z, R.shape[:2]).copy()
    n_struct = int(R.shape[0])
    N = np.asarray(data.get("N", (Z > 0).sum(axis=1)), dtype=np.int32).reshape(-1)
    if len(N) != n_struct:
        raise ValueError("N length does not match R")
    cell_arr = data.get("cell")
    records: list[StructureRecord] = []
    for i in range(n_struct):
        n_i = int(N[i])
        z_i = np.asarray(Z[i, :n_i], dtype=np.int32)
        r_i = np.asarray(R[i, :n_i], dtype=np.float64)
        cell_i = None
        if cell_arr is not None:
            c = np.asarray(cell_arr)
            cell_i = np.asarray(c[i] if c.ndim >= 2 else c, dtype=np.float64)
        meta = _row_meta(data, i)
        rec = StructureRecord(
            index=i,
            structure_id=structure_id(r_i, z_i, n_i, cell=cell_i),
            geometry_fingerprint=geometry_fingerprint(r_i, z_i, n_i, cell=cell_i),
            composition=composition_key(z_i, n_i),
            stratum=stratum_key(z_i, n_i, meta),
            group=group_key(meta, i),
            n_atoms=n_i,
            atomic_numbers=z_i,
            positions=r_i,
            cell=cell_i,
            temperature=_maybe_float(meta.get("temperature", meta.get("T"))),
            pressure=_maybe_float(meta.get("pressure", meta.get("P"))),
            phase=(
                str(meta["phase"])
                if meta.get("phase") is not None
                else (str(meta["ensemble"]) if meta.get("ensemble") is not None else None)
            ),
            frame_index=_maybe_int(meta.get("frame_index", meta.get("group_frame", i))),
            source=str(meta.get("source_path", meta.get("source", ""))) or None,
            extra=meta,
        )
        records.append(rec)
    return records


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def deduplicate(records: Sequence[StructureRecord]) -> tuple[list[StructureRecord], dict[str, str]]:
    """Keep the first occurrence of each geometry fingerprint."""
    seen: dict[str, str] = {}
    unique: list[StructureRecord] = []
    duplicate_of: dict[str, str] = {}
    for rec in records:
        fp = rec.geometry_fingerprint
        if fp in seen:
            duplicate_of[str(rec.index)] = seen[fp]
            continue
        seen[fp] = rec.structure_id
        unique.append(rec)
    return unique, duplicate_of


def assign_grouped_splits(
    records: Sequence[StructureRecord],
    *,
    valid_fraction: float,
    test_fraction: float,
    seed: int,
    seed_groups: Sequence[str] | None = None,
    min_candidate_groups: int = 1,
) -> list[StructureRecord]:
    """Assign splits by shuffling *groups*, never individual frames.

    ``seed_groups`` (existing student-training trajectories) are tagged
    ``seed`` and excluded from candidate / valid / test.  They are a design
    prior for information-gain coverage, not evidence of calibrated
    uncertainty against the expensive reference method.
    """
    seed_set = set(seed_groups or ())
    groups: dict[str, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        groups[rec.group].append(i)

    free_groups = [g for g in groups if g not in seed_set]
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(free_groups))
    shuffled = [free_groups[i] for i in order]

    n_free = len(shuffled)
    n_test = int(round(float(test_fraction) * n_free))
    n_valid = int(round(float(valid_fraction) * n_free))
    if n_free >= 3:
        n_test = min(max(n_test, 1), n_free - 2)
        n_valid = min(max(n_valid, 1), n_free - n_test - 1)
    elif n_free == 2:
        n_test, n_valid = 1, 0
    else:
        n_test = n_valid = 0
    n_cand = n_free - n_test - n_valid
    if n_cand < min_candidate_groups and n_free > 0:
        # Prefer keeping at least one candidate group; shrink test then valid.
        deficit = min_candidate_groups - n_cand
        take_test = min(deficit, n_test)
        n_test -= take_test
        deficit -= take_test
        n_valid = max(0, n_valid - deficit)

    test_g = set(shuffled[:n_test])
    valid_g = set(shuffled[n_test : n_test + n_valid])
    assigned: list[StructureRecord] = []
    for rec in records:
        if rec.group in seed_set:
            split = SPLIT_SEED
        elif rec.group in test_g:
            split = SPLIT_TEST
        elif rec.group in valid_g:
            split = SPLIT_VALID
        else:
            split = SPLIT_CANDIDATE
        assigned.append(
            StructureRecord(**{**rec.__dict__, "split": split})
        )
    return assigned


def assert_split_isolation(records: Sequence[StructureRecord]) -> None:
    """Raise if a group or geometry fingerprint appears in more than one split."""
    group_splits: dict[str, set[str]] = defaultdict(set)
    fp_splits: dict[str, set[str]] = defaultdict(set)
    id_splits: dict[str, set[str]] = defaultdict(set)
    for rec in records:
        group_splits[rec.group].add(rec.split)
        fp_splits[rec.geometry_fingerprint].add(rec.split)
        id_splits[rec.structure_id].add(rec.split)
    leaks = []
    for name, mapping in (
        ("group", group_splits),
        ("fingerprint", fp_splits),
        ("structure_id", id_splits),
    ):
        for key, splits in mapping.items():
            if len(splits) > 1:
                leaks.append(f"{name} {key!r} in {sorted(splits)}")
    if leaks:
        raise ValueError("split isolation violated: " + "; ".join(leaks[:12]))


def build_pool(
    data: Mapping[str, Any],
    *,
    valid_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 0,
    seed_groups: Sequence[str] | None = None,
) -> PoolManifest:
    raw = records_from_npz(data)
    unique, duplicate_of = deduplicate(raw)
    assigned = assign_grouped_splits(
        unique,
        valid_fraction=valid_fraction,
        test_fraction=test_fraction,
        seed=seed,
        seed_groups=seed_groups,
    )
    assert_split_isolation(assigned)
    return PoolManifest(
        records=assigned,
        duplicate_of=duplicate_of,
        n_input=len(raw),
        n_unique=len(unique),
        n_duplicates_removed=len(duplicate_of),
    )


def subset_arrays(
    data: Mapping[str, np.ndarray],
    records: Sequence[StructureRecord],
) -> dict[str, np.ndarray]:
    """Slice NPZ arrays to the given records (original input indices)."""
    idx = np.asarray([r.index for r in records], dtype=np.int64)
    out: dict[str, np.ndarray] = {}
    n = int(np.asarray(data["R"]).shape[0])
    for key, value in data.items():
        arr = np.asarray(value)
        if arr.shape[:1] == (n,):
            out[key] = arr[idx]
        else:
            out[key] = arr
    out["structure_id"] = np.asarray([r.structure_id for r in records], dtype=object)
    out["split"] = np.asarray([r.split for r in records], dtype=object)
    out["group"] = np.asarray([r.group for r in records], dtype=object)
    out["stratum"] = np.asarray([r.stratum for r in records], dtype=object)
    out["composition"] = np.asarray([r.composition for r in records], dtype=object)
    return out
