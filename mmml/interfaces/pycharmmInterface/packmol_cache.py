"""Disk cache for Packmol cluster builds (monomer MM + Packmol + cluster MM)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

CACHE_VERSION = 3

# Prep / overlap gates forwarded from md-system; bust cache when tuning sweep variants.
PREP_GATE_CACHE_KEYS = (
    "max_grms_before_dyn",
    "mlpot_registration_max_grms",
    "geometry_packing_fire_bfgs_crossover_grms",
    "no_scale_max_grms",
    "allow_high_grms",
    "pre_mlpot_overlap_min_distance",
    "pre_mlpot_h_heavy_min_distance",
    "pre_mlpot_heavy_heavy_min_distance",
    "min_intermonomer_atom_distance",
)


def packmol_cache_root(
    *,
    output_dir: Path | None = None,
    override: Path | str | None = None,
) -> Path:
    """Root directory for Packmol cluster cache entries."""
    if override is not None:
        return Path(override).expanduser().resolve()
    env = os.environ.get("MMML_PACKMOL_CACHE", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve() / ".packmol_cache"
    return Path.home() / ".cache" / "mmml" / "packmol"


def packmol_prep_settings_from_mapping(data: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize prep gate / overlap settings that should bust Packmol cache."""
    out: dict[str, Any] = {}
    for key in PREP_GATE_CACHE_KEYS:
        if key not in data:
            continue
        val = data[key]
        if val is None:
            continue
        if isinstance(val, bool):
            out[key] = bool(val)
        else:
            out[key] = float(val)
    return out


def packmol_prep_settings_from_namespace(args: Any) -> dict[str, Any]:
    """Extract prep gate settings from an argparse / md-system namespace."""
    data = {key: getattr(args, key) for key in PREP_GATE_CACHE_KEYS if hasattr(args, key)}
    return packmol_prep_settings_from_mapping(data)


def packmol_cache_fingerprint(
    *,
    composition: list[tuple[str, int]],
    placement: str,
    center: tuple[float, float, float],
    cube_side: float | None = None,
    radius: float | None = None,
    tolerance: float,
    seed: int | None,
    charmm_sd_steps: int,
    charmm_abnr_steps: int,
    charmm_tolenr: float,
    charmm_tolgrd: float,
    packmol_padding_A: float | None = None,
    spacing: float | None = None,
    sim_cell_side: float | None = None,
    prep_gate_settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Canonical fingerprint for a Packmol cluster build (hashed into cache dir names)."""
    payload: dict[str, Any] = {
        "version": CACHE_VERSION,
        "composition": [[str(r).upper(), int(n)] for r, n in composition],
        "placement": str(placement),
        "center": [float(c) for c in center],
        "cube_side": None if cube_side is None else float(cube_side),
        "radius": None if radius is None else float(radius),
        "tolerance": float(tolerance),
        "seed": None if seed is None else int(seed),
        "charmm_sd_steps": int(charmm_sd_steps),
        "charmm_abnr_steps": int(charmm_abnr_steps),
        "charmm_tolenr": float(charmm_tolenr),
        "charmm_tolgrd": float(charmm_tolgrd),
        "packmol_padding_A": (
            None if packmol_padding_A is None else float(packmol_padding_A)
        ),
        "spacing": None if spacing is None else float(spacing),
        "sim_cell_side": None if sim_cell_side is None else float(sim_cell_side),
    }
    gates = packmol_prep_settings_from_mapping(prep_gate_settings or {})
    if gates:
        payload["prep_gate_settings"] = gates
    return payload


def packmol_cache_key(
    *,
    composition: list[tuple[str, int]],
    placement: str,
    center: tuple[float, float, float],
    cube_side: float | None = None,
    radius: float | None = None,
    tolerance: float,
    seed: int | None,
    charmm_sd_steps: int,
    charmm_abnr_steps: int,
    charmm_tolenr: float,
    charmm_tolgrd: float,
    packmol_padding_A: float | None = None,
    spacing: float | None = None,
    sim_cell_side: float | None = None,
    prep_gate_settings: Mapping[str, Any] | None = None,
) -> str:
    """Stable cache directory name from placement and CHARMM pre-relax parameters."""
    payload = packmol_cache_fingerprint(
        composition=composition,
        placement=placement,
        center=center,
        cube_side=cube_side,
        radius=radius,
        tolerance=tolerance,
        seed=seed,
        charmm_sd_steps=charmm_sd_steps,
        charmm_abnr_steps=charmm_abnr_steps,
        charmm_tolenr=charmm_tolenr,
        charmm_tolgrd=charmm_tolgrd,
        packmol_padding_A=packmol_padding_A,
        spacing=spacing,
        sim_cell_side=sim_cell_side,
        prep_gate_settings=prep_gate_settings,
    )
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def _entry_dir(root: Path, key: str) -> Path:
    return root / key


def save_monomer_geometries(
    entry_dir: Path,
    residue_geometries: dict[str, tuple[np.ndarray, list[str], np.ndarray]],
) -> None:
    """Cache per-residue minimized monomer (coords, names, Z) for PSF rebuild on cache hit."""
    for residue, (coords, names, mon_z) in residue_geometries.items():
        key = str(residue).upper()
        np.savez(
            entry_dir / f"monomer_{key}.npz",
            coords=np.asarray(coords, dtype=float),
            names=np.asarray(names, dtype=str),
            z=np.asarray(mon_z, dtype=np.int32),
        )


def load_monomer_geometries(
    entry_dir: Path,
    composition: list[tuple[str, int]],
) -> dict[str, tuple[np.ndarray, list[str], np.ndarray]] | None:
    """Load monomer caches if every composition residue type is present."""
    out: dict[str, tuple[np.ndarray, list[str], np.ndarray]] = {}
    for residue, _count in composition:
        key = str(residue).upper()
        if key in out:
            continue
        path = entry_dir / f"monomer_{key}.npz"
        if not path.is_file():
            return None
        data = np.load(path, allow_pickle=False)
        out[key] = (
            np.asarray(data["coords"], dtype=float),
            [str(x) for x in data["names"]],
            np.asarray(data["z"], dtype=int),
        )
    return out


def save_packmol_cluster_cache(
    entry_dir: Path,
    *,
    manifest: dict[str, Any],
    z: np.ndarray,
    positions: np.ndarray,
    atoms_per_list: list[int],
    residue_names: list[str],
    packmol_pdb: Path | None = None,
    residue_geometries: dict[str, tuple[np.ndarray, list[str], np.ndarray]] | None = None,
) -> None:
    """Write cluster geometry and manifest after a full Packmol build."""
    entry_dir.mkdir(parents=True, exist_ok=True)
    (entry_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez(
        entry_dir / "cluster.npz",
        z=np.asarray(z, dtype=np.int32),
        positions=np.asarray(positions, dtype=float),
        atoms_per_list=np.asarray(atoms_per_list, dtype=np.int32),
        residue_names=np.asarray(residue_names, dtype=str),
    )
    if packmol_pdb is not None and packmol_pdb.is_file():
        shutil.copy2(packmol_pdb, entry_dir / "init-packmol-sphere.pdb")
    if residue_geometries:
        save_monomer_geometries(entry_dir, residue_geometries)


def load_packmol_cluster_cache(
    entry_dir: Path,
    *,
    expected_fingerprint: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Load cached cluster if manifest version and cluster.npz match."""
    manifest_path = entry_dir / "manifest.json"
    npz_path = entry_dir / "cluster.npz"
    if not manifest_path.is_file() or not npz_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("version", 0)) != CACHE_VERSION:
            return None
        if expected_fingerprint is not None:
            stored = manifest.get("fingerprint")
            if stored != dict(expected_fingerprint):
                return None
        data = np.load(npz_path, allow_pickle=False)
        monomers = load_monomer_geometries(
            entry_dir,
            [(str(r), int(n)) for r, n in manifest.get("composition", [])],
        )
        return {
            "manifest": manifest,
            "z": np.asarray(data["z"], dtype=int),
            "positions": np.asarray(data["positions"], dtype=float),
            "atoms_per_list": [int(x) for x in data["atoms_per_list"]],
            "residue_names": [str(x) for x in data["residue_names"]],
            "packmol_pdb": entry_dir / "init-packmol-sphere.pdb",
            "residue_geometries": monomers,
        }
    except (OSError, json.JSONDecodeError, KeyError, ValueError):
        return None


def try_load_packmol_cluster_cache(
    *,
    composition: list[tuple[str, int]],
    placement: str,
    center: tuple[float, float, float],
    cube_side: float | None = None,
    radius: float | None = None,
    tolerance: float,
    seed: int | None,
    charmm_sd_steps: int,
    charmm_abnr_steps: int,
    charmm_tolenr: float,
    charmm_tolgrd: float,
    cache_root: Path,
    packmol_padding_A: float | None = None,
    spacing: float | None = None,
    sim_cell_side: float | None = None,
    prep_gate_settings: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    fingerprint = packmol_cache_fingerprint(
        composition=composition,
        placement=placement,
        center=center,
        cube_side=cube_side,
        radius=radius,
        tolerance=tolerance,
        seed=seed,
        charmm_sd_steps=charmm_sd_steps,
        charmm_abnr_steps=charmm_abnr_steps,
        charmm_tolenr=charmm_tolenr,
        charmm_tolgrd=charmm_tolgrd,
        packmol_padding_A=packmol_padding_A,
        spacing=spacing,
        sim_cell_side=sim_cell_side,
        prep_gate_settings=prep_gate_settings,
    )
    key = packmol_cache_key(
        composition=composition,
        placement=placement,
        center=center,
        cube_side=cube_side,
        radius=radius,
        tolerance=tolerance,
        seed=seed,
        charmm_sd_steps=charmm_sd_steps,
        charmm_abnr_steps=charmm_abnr_steps,
        charmm_tolenr=charmm_tolenr,
        charmm_tolgrd=charmm_tolgrd,
        packmol_padding_A=packmol_padding_A,
        spacing=spacing,
        sim_cell_side=sim_cell_side,
        prep_gate_settings=prep_gate_settings,
    )
    return load_packmol_cluster_cache(
        _entry_dir(cache_root, key),
        expected_fingerprint=fingerprint,
    )
