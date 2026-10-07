"""Write PhysNet-train NPZ (eV / eV/Å) from labeled acetone geometries.

Provenance keys (``-1`` / ``""`` when unknown, e.g. the synthetic acetone
pool or the ``pdb_eq`` reference): ``group_seed`` (trajectory seed),
``group_file`` (input-file index), ``group_frame`` (frame index in the box
pool), ``group_step`` (MD step), ``group_phase`` (``fire``/``md``).

Splits: ``sample`` permutes individual samples (acetone pool default);
``seed`` draws whole trajectories (by ``group_seed``, else ``group_file``)
into valid, so every sample of a trajectory, including each dimer's AB/A/B
fragments, lands on one side. Ungrouped samples (the reference monomer)
stay in train.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from mmml.data.units import CALCULATOR_UNITS
from mmml.distill.acetone_pool import DIMER_ATOMS
from mmml.distill.teacher_label import LabeledSample

KIND_MONOMER = 0
KIND_DIMER = 1

SPLIT_SAMPLE = "sample"
SPLIT_SEED = "seed"
SPLIT_MODES = (SPLIT_SAMPLE, SPLIT_SEED)


def pad_sample(
    numbers: np.ndarray,
    positions: np.ndarray,
    forces: np.ndarray,
    *,
    pad_atoms: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Right-pad with Z=0 ghost atoms (PhysNet atom_mask)."""
    n = int(numbers.shape[0])
    pad = int(pad_atoms)
    if n > pad:
        raise ValueError(f"sample has {n} atoms > pad_atoms={pad}")
    z = np.zeros((pad,), dtype=np.int32)
    r = np.zeros((pad, 3), dtype=np.float64)
    f = np.zeros((pad, 3), dtype=np.float64)
    z[:n] = np.asarray(numbers, dtype=np.int32)
    r[:n] = np.asarray(positions, dtype=np.float64)
    f[:n] = np.asarray(forces, dtype=np.float64)
    return z, r, f, n


def split_train_valid(
    n: int,
    *,
    valid_fraction: float,
    seed: int,
) -> np.ndarray:
    """Return boolean mask, True = train. At least one valid when n>=2."""
    if n < 1:
        raise ValueError("empty split")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(n)
    n_valid = int(round(float(valid_fraction) * n))
    if n >= 2:
        n_valid = min(max(n_valid, 1), n - 1)
    else:
        n_valid = 0
    is_train = np.ones(n, dtype=bool)
    is_train[order[:n_valid]] = False
    return is_train


def split_group_key(geo) -> str | None:
    """Split group of a geometry: ``seed:<k>``, else ``file:<k>``, else None."""
    if getattr(geo, "group_seed", None) is not None:
        return f"seed:{int(geo.group_seed)}"
    if getattr(geo, "group_file", None) is not None:
        return f"file:{int(geo.group_file)}"
    return None


def split_train_valid_grouped(
    groups: list[str | None],
    *,
    valid_fraction: float,
    seed: int,
) -> tuple[np.ndarray, list[str]]:
    """Train mask plus the sorted valid group keys; the fraction applies to groups.

    Every sample of a group lands on the same side. ``None`` groups always
    train. Needs at least two groups (one valid, one train).
    """
    uniq = sorted({g for g in groups if g is not None})
    if len(uniq) < 2:
        raise ValueError(
            f"grouped split needs >= 2 trajectory groups, found {len(uniq)} ({uniq}); "
            "use split='sample' or add trajectories"
        )
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(uniq))
    n_valid = min(max(int(round(float(valid_fraction) * len(uniq))), 1), len(uniq) - 1)
    valid = sorted(uniq[i] for i in order[:n_valid])
    valid_set = set(valid)
    is_train = np.array([g not in valid_set for g in groups], dtype=bool)
    return is_train, valid


def samples_to_arrays(
    samples: list[LabeledSample],
    *,
    pad_atoms: int = DIMER_ATOMS,
    valid_fraction: float = 0.15,
    seed: int = 0,
    split: str = SPLIT_SAMPLE,
) -> dict[str, Any]:
    """Padded arrays + ``is_train``. ``_valid_groups`` lists valid groups (seed split)."""
    n = len(samples)
    if n == 0:
        raise ValueError("no labeled samples")
    pad = int(pad_atoms)
    R = np.zeros((n, pad, 3), dtype=np.float64)
    F = np.zeros((n, pad, 3), dtype=np.float64)
    Z = np.zeros((n, pad), dtype=np.int32)
    N = np.zeros((n,), dtype=np.int32)
    E = np.zeros((n,), dtype=np.float64)
    E_tot = np.zeros((n,), dtype=np.float64)
    E_int = np.full((n,), np.nan, dtype=np.float64)
    r_com = np.full((n,), np.nan, dtype=np.float64)
    kind = np.zeros((n,), dtype=np.int32)
    source = np.empty((n,), dtype=object)
    g_seed = np.full((n,), -1, dtype=np.int64)
    g_file = np.full((n,), -1, dtype=np.int64)
    g_frame = np.full((n,), -1, dtype=np.int64)
    g_step = np.full((n,), -1, dtype=np.int64)
    g_phase = np.array(
        [str(getattr(s.geometry, "group_phase", None) or "") for s in samples], dtype=str
    )
    mode = str(split).strip().lower()
    valid_groups: list[str] | None = None
    if mode == SPLIT_SAMPLE:
        is_train = split_train_valid(n, valid_fraction=valid_fraction, seed=seed)
    elif mode == SPLIT_SEED:
        is_train, valid_groups = split_train_valid_grouped(
            [split_group_key(s.geometry) for s in samples],
            valid_fraction=valid_fraction,
            seed=seed,
        )
    else:
        raise ValueError(f"split must be one of {SPLIT_MODES}, got {split!r}")
    for i, sample in enumerate(samples):
        geo = sample.geometry
        z, r, f, n_real = pad_sample(
            geo.numbers, geo.positions, sample.forces_ev_per_angstrom, pad_atoms=pad
        )
        Z[i], R[i], F[i], N[i] = z, r, f, n_real
        E[i] = float(sample.energy_eV)
        E_tot[i] = float(sample.energy_total_eV)
        if sample.energy_int_eV is not None:
            E_int[i] = float(sample.energy_int_eV)
        if geo.r_com_A is not None:
            r_com[i] = float(geo.r_com_A)
        kind[i] = KIND_DIMER if geo.kind == "dimer" else KIND_MONOMER
        source[i] = str(geo.source)
        for arr, attr in (
            (g_seed, "group_seed"),
            (g_file, "group_file"),
            (g_frame, "group_frame"),
            (g_step, "group_step"),
        ):
            value = getattr(geo, attr, None)
            if value is not None:
                arr[i] = int(value)
    units = {
        "R": "angstrom",
        "E": "ev",
        "F": "ev_angstrom",
        "E_total": "ev",
        "E_int": "ev",
        "r_com": "angstrom",
        "calculator": dict(CALCULATOR_UNITS),
    }
    return {
        "R": R,
        "Z": Z,
        "N": N,
        "E": E,
        "F": F,
        "E_total": E_tot,
        "E_int": E_int,
        "r_com": r_com,
        "kind": kind,
        "source": source,
        "group_seed": g_seed,
        "group_file": g_file,
        "group_frame": g_frame,
        "group_step": g_step,
        "group_phase": g_phase,
        "is_train": is_train.astype(np.int8),
        "_mmml_units": np.array(json.dumps(units)),
        "_split": mode,
        "_valid_groups": valid_groups,
    }


def write_distill_npz(
    samples: list[LabeledSample],
    out_dir: Path,
    *,
    pad_atoms: int = DIMER_ATOMS,
    valid_fraction: float = 0.15,
    seed: int = 0,
    split: str = SPLIT_SAMPLE,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write train/valid NPZ plus a JSON report. Returns output paths.

    ``split``: ``sample`` (per-sample permutation) or ``seed`` (whole
    trajectories; ``valid_fraction`` applies to trajectories). ``report.json``
    records ``split``, ``valid_groups`` and ``valid_seeds``.
    """
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    payload = samples_to_arrays(
        samples, pad_atoms=pad_atoms, valid_fraction=valid_fraction, seed=seed, split=split
    )
    split_mode = payload.pop("_split")
    valid_groups = payload.pop("_valid_groups")
    is_train = payload["is_train"].astype(bool)
    train_path = dest / "train.npz"
    valid_path = dest / "valid.npz"
    report_path = dest / "report.json"

    def _subset(mask: np.ndarray) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in payload.items():
            if key == "_mmml_units":
                out[key] = value
                continue
            arr = np.asarray(value)
            out[key] = arr[mask]
        return out

    np.savez_compressed(train_path, **_subset(is_train))
    np.savez_compressed(valid_path, **_subset(~is_train))
    report = {
        "n_samples": int(len(samples)),
        "n_train": int(is_train.sum()),
        "n_valid": int((~is_train).sum()),
        "n_monomers": int(np.sum(payload["kind"] == KIND_MONOMER)),
        "n_dimers": int(np.sum(payload["kind"] == KIND_DIMER)),
        "pad_atoms": int(pad_atoms),
        "energy_eV_min": float(np.min(payload["E"])),
        "energy_eV_max": float(np.max(payload["E"])),
        "force_max_abs_eV_A": float(np.max(np.abs(payload["F"]))),
        "train_npz": str(train_path.resolve()),
        "valid_npz": str(valid_path.resolve()),
        "split": split_mode,
        "split_seed": int(seed),
        "valid_fraction": float(valid_fraction),
    }
    if valid_groups is not None:
        report["valid_groups"] = valid_groups
        report["valid_seeds"] = [
            int(g.split(":", 1)[1]) for g in valid_groups if g.startswith("seed:")
        ]
        all_groups = {split_group_key(s.geometry) for s in samples} - {None}
        report["n_groups"] = len(all_groups)
        report["n_valid_groups"] = len(valid_groups)
    if metadata:
        report["metadata"] = metadata
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return {"train": train_path, "valid": valid_path, "report": report_path}
