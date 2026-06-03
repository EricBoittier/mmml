"""Disk cache for Packmol sphere cluster builds (monomer MM + Packmol + cluster MM)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

CACHE_VERSION = 1


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


def packmol_cache_key(
    *,
    composition: list[tuple[str, int]],
    center: tuple[float, float, float],
    radius: float,
    tolerance: float,
    seed: int | None,
    charmm_sd_steps: int,
    charmm_abnr_steps: int,
    charmm_tolenr: float,
    charmm_tolgrd: float,
) -> str:
    """Stable cache directory name from placement and CHARMM pre-relax parameters."""
    payload: dict[str, Any] = {
        "version": CACHE_VERSION,
        "composition": [[str(r).upper(), int(n)] for r, n in composition],
        "center": [float(c) for c in center],
        "radius": float(radius),
        "tolerance": float(tolerance),
        "seed": None if seed is None else int(seed),
        "charmm_sd_steps": int(charmm_sd_steps),
        "charmm_abnr_steps": int(charmm_abnr_steps),
        "charmm_tolenr": float(charmm_tolenr),
        "charmm_tolgrd": float(charmm_tolgrd),
    }
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


def load_packmol_cluster_cache(entry_dir: Path) -> dict[str, Any] | None:
    """Load cached cluster if manifest version and cluster.npz match."""
    manifest_path = entry_dir / "manifest.json"
    npz_path = entry_dir / "cluster.npz"
    if not manifest_path.is_file() or not npz_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("version", 0)) != CACHE_VERSION:
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
    center: tuple[float, float, float],
    radius: float,
    tolerance: float,
    seed: int | None,
    charmm_sd_steps: int,
    charmm_abnr_steps: int,
    charmm_tolenr: float,
    charmm_tolgrd: float,
    cache_root: Path,
) -> dict[str, Any] | None:
    key = packmol_cache_key(
        composition=composition,
        center=center,
        radius=radius,
        tolerance=tolerance,
        seed=seed,
        charmm_sd_steps=charmm_sd_steps,
        charmm_abnr_steps=charmm_abnr_steps,
        charmm_tolenr=charmm_tolenr,
        charmm_tolgrd=charmm_tolgrd,
    )
    return load_packmol_cluster_cache(_entry_dir(cache_root, key))
