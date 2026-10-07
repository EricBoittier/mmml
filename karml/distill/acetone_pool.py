"""Acetone monomer/dimer geometry pool for PET → PhysNet distillation.

Mixes bundled dataset frames (DMC extxyz + ACO PDB) with synthetic
augmentations aimed at MD stability: Cartesian noise, bond stretches,
random relative orientations, and COM-distance coverage of the four
PES regions in ``docs/bayesian-pes-design.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from ase.io.extxyz import XYZError

from mmml.interfaces.pycharmmInterface.cutoffs import DEFAULT_MM_SWITCH_ON

ATOMS_PER_ACETONE = 10
DIMER_ATOMS = 2 * ATOMS_PER_ACETONE

# Carbonyl C and O in ``aco_monomer.pdb`` (0-based).
CARBONYL_O_INDEX = 0
CARBONYL_C_INDEX = 1

# Four intermolecular COM windows (Å). Acetone–acetone contact is ~4–5 Å.
COM_REPULSIVE_A = (3.6, 4.4)
COM_WELL_A = (4.4, 5.6)
COM_SHOULDER_A = (5.6, 7.0)
COM_LONG_A = (7.0, 10.0)

# Far-field replica: beyond the ML/MM handoff so teacher E_int should vanish.
FAR_FIELD_COM_A = float(DEFAULT_MM_SWITCH_ON) + 6.0

MIN_PAIR_DISTANCE_A = 0.75
CARTESIAN_NOISE_SIGMA_A = 0.05
LARGE_NOISE_SIGMA_A = 0.15
BOND_STRETCH_FRACTION = 0.08

_REPO = Path(__file__).resolve().parents[1]
ACO_PDB = _REPO / "generate" / "sample" / "pdb" / "aco_monomer.pdb"
ACO_DMC_EXTXYZ = _REPO / "generate" / "dmc" / "examples" / "acetone_dmc.extxyz"

PRESET_SMOKE = "smoke"
PRESET_MD = "md"
POOL_PRESETS = (PRESET_SMOKE, PRESET_MD)


@dataclass(frozen=True)
class AcetonePoolConfig:
    """Counts and scales for the acetone geometry mixture."""

    seed: int = 0
    n_monomer_noise: int = 8
    n_monomer_large_noise: int = 4
    n_bond_stretch: int = 4
    n_dimer_orientations: int = 4
    n_com_per_region: int = 2
    n_far_field: int = 2
    n_global_rotations_extra: int = 0
    cartesian_noise_sigma_A: float = CARTESIAN_NOISE_SIGMA_A
    large_noise_sigma_A: float = LARGE_NOISE_SIGMA_A
    bond_stretch_fraction: float = BOND_STRETCH_FRACTION
    min_pair_distance_A: float = MIN_PAIR_DISTANCE_A
    far_field_com_A: float = FAR_FIELD_COM_A
    com_repulsive_A: tuple[float, float] = COM_REPULSIVE_A
    com_well_A: tuple[float, float] = COM_WELL_A
    com_shoulder_A: tuple[float, float] = COM_SHOULDER_A
    com_long_A: tuple[float, float] = COM_LONG_A
    include_dataset: bool = True
    extra_extxyz: tuple[Path, ...] = field(default_factory=tuple)


@dataclass
class Geometry:
    """One unlabeled structure. Positions Å, no PBC."""

    numbers: np.ndarray
    positions: np.ndarray
    kind: str
    source: str
    r_com_A: float | None
    atoms_per_monomer: tuple[int, ...]
    # Provenance for structures cut from MD (``box_cluster_pool``); None for
    # the synthetic acetone pool. ``group_seed`` is the trajectory seed
    # (independent trajectory), ``group_file`` the input-file index (fallback
    # group when frames carry no seed), ``group_frame`` the frame index in
    # the pool's frame list, ``group_step``/``group_phase`` from frame info.
    group_seed: int | None = None
    group_file: int | None = None
    group_frame: int | None = None
    group_step: int | None = None
    group_phase: str | None = None


def pool_config_for_preset(preset: str, *, seed: int = 0) -> AcetonePoolConfig:
    name = str(preset).strip().lower()
    if name == PRESET_SMOKE:
        return AcetonePoolConfig(seed=int(seed))
    if name == PRESET_MD:
        return AcetonePoolConfig(
            seed=int(seed),
            n_monomer_noise=48,
            n_monomer_large_noise=24,
            n_bond_stretch=12,
            n_dimer_orientations=12,
            n_com_per_region=4,
            n_far_field=8,
            n_global_rotations_extra=1,
        )
    raise ValueError(f"unknown pool preset {preset!r}; use {POOL_PRESETS}")


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniform SO(3) via a unit quaternion."""
    q = rng.normal(size=4)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def min_pair_distance(positions: np.ndarray) -> float:
    pos = np.asarray(positions, dtype=np.float64)
    n = pos.shape[0]
    if n < 2:
        return float("inf")
    dmin = np.inf
    for i in range(n):
        delta = pos[i + 1 :] - pos[i]
        d = np.linalg.norm(delta, axis=1)
        if d.size:
            dmin = min(dmin, float(np.min(d)))
    return float(dmin)


def load_acetone_monomer(*, path: Path | None = None) -> Atoms:
    pdb = Path(path) if path is not None else ACO_PDB
    zs: list[int] = []
    pos: list[list[float]] = []
    for line in pdb.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        name = line[12:16].strip()
        zs.append(8 if name.startswith("O") else 6 if name.startswith("C") else 1)
        pos.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    atoms = Atoms(
        numbers=np.asarray(zs, dtype=np.int32),
        positions=np.asarray(pos, dtype=np.float64),
    )
    if len(atoms) != ATOMS_PER_ACETONE:
        raise ValueError(f"{pdb} has {len(atoms)} atoms, expected {ATOMS_PER_ACETONE}")
    return atoms


def _parse_element_xyz_blocks(text: str, *, n_atoms: int) -> list[Atoms]:
    """Parse '# comment' separated element-XYZ blocks (not ASE extxyz)."""
    frames: list[Atoms] = []
    zs: list[int] = []
    pos: list[list[float]] = []
    symbol_to_z = {"H": 1, "C": 6, "O": 8, "N": 7}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            if len(zs) == n_atoms:
                frames.append(
                    Atoms(
                        numbers=np.asarray(zs, dtype=np.int32),
                        positions=np.asarray(pos, dtype=np.float64),
                    )
                )
            zs, pos = [], []
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        z = symbol_to_z.get(parts[0])
        if z is None:
            continue
        zs.append(int(z))
        pos.append([float(parts[1]), float(parts[2]), float(parts[3])])
        if len(zs) == n_atoms:
            frames.append(
                Atoms(
                    numbers=np.asarray(zs, dtype=np.int32),
                    positions=np.asarray(pos, dtype=np.float64),
                )
            )
            zs, pos = [], []
    return frames


def load_dataset_dimers(*, path: Path | None = None) -> list[Atoms]:
    extxyz = Path(path) if path is not None else ACO_DMC_EXTXYZ
    try:
        frames = ase_read(str(extxyz), index=":")
        if not isinstance(frames, list):
            frames = [frames]
    except (XYZError, ValueError, OSError):
        frames = _parse_element_xyz_blocks(extxyz.read_text(), n_atoms=DIMER_ATOMS)
    out = [frame for frame in frames if len(frame) == DIMER_ATOMS]
    if not out:
        raise ValueError(f"no {DIMER_ATOMS}-atom dimer frames in {extxyz}")
    return out


def _center(positions: np.ndarray) -> np.ndarray:
    pos = np.asarray(positions, dtype=np.float64)
    return pos - pos.mean(axis=0)


def _geometry(
    numbers: np.ndarray,
    positions: np.ndarray,
    *,
    kind: str,
    source: str,
    r_com_A: float | None,
    atoms_per_monomer: tuple[int, ...],
    min_pair_distance_A: float,
) -> Geometry | None:
    if min_pair_distance(positions) < float(min_pair_distance_A):
        return None
    return Geometry(
        numbers=np.asarray(numbers, dtype=np.int32),
        positions=np.asarray(positions, dtype=np.float64),
        kind=kind,
        source=source,
        r_com_A=None if r_com_A is None else float(r_com_A),
        atoms_per_monomer=tuple(int(n) for n in atoms_per_monomer),
    )


def assemble_dimer(
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    *,
    com_distance_A: float,
    rotation_b: np.ndarray,
) -> np.ndarray:
    """Place A at the origin and B along +x at ``com_distance_A`` after rotating B."""
    a = _center(pos_a)
    b = _center(pos_b) @ np.asarray(rotation_b, dtype=np.float64).T
    b = b + np.array([float(com_distance_A), 0.0, 0.0], dtype=np.float64)
    return np.concatenate([a, b], axis=0)


def _stretch_carbonyl(pos: np.ndarray, scale: float) -> np.ndarray:
    out = np.asarray(pos, dtype=np.float64).copy()
    bond = out[CARBONYL_O_INDEX] - out[CARBONYL_C_INDEX]
    out[CARBONYL_O_INDEX] = out[CARBONYL_C_INDEX] + bond * float(scale)
    return out


def build_acetone_pool(config: AcetonePoolConfig | None = None) -> list[Geometry]:
    """Return unlabeled acetone monomers and dimers (dataset + synthetic)."""
    cfg = config if config is not None else AcetonePoolConfig()
    rng = np.random.default_rng(int(cfg.seed))
    mono = load_acetone_monomer()
    z_m = np.asarray(mono.get_atomic_numbers(), dtype=np.int32)
    r_eq = np.asarray(mono.get_positions(), dtype=np.float64)
    geos: list[Geometry] = []
    rejected = 0

    def _add(geo: Geometry | None) -> None:
        nonlocal rejected
        if geo is None:
            rejected += 1
            return
        geos.append(geo)

    _add(
        _geometry(
            z_m,
            r_eq,
            kind="monomer",
            source="pdb_eq",
            r_com_A=None,
            atoms_per_monomer=(ATOMS_PER_ACETONE,),
            min_pair_distance_A=cfg.min_pair_distance_A,
        )
    )

    def _noise_monomers(n: int, sigma: float, source: str) -> None:
        for _ in range(int(n)):
            rattled = r_eq + rng.normal(0.0, float(sigma), size=r_eq.shape)
            _add(
                _geometry(
                    z_m,
                    rattled,
                    kind="monomer",
                    source=source,
                    r_com_A=None,
                    atoms_per_monomer=(ATOMS_PER_ACETONE,),
                    min_pair_distance_A=cfg.min_pair_distance_A,
                )
            )

    _noise_monomers(cfg.n_monomer_noise, cfg.cartesian_noise_sigma_A, "noise")
    _noise_monomers(cfg.n_monomer_large_noise, cfg.large_noise_sigma_A, "noise_large")

    for _ in range(int(cfg.n_bond_stretch)):
        scale = 1.0 + float(cfg.bond_stretch_fraction) * float(rng.uniform(-1.0, 1.0))
        stretched = _stretch_carbonyl(r_eq, scale)
        _add(
            _geometry(
                z_m,
                stretched,
                kind="monomer",
                source="co_stretch",
                r_com_A=None,
                atoms_per_monomer=(ATOMS_PER_ACETONE,),
                min_pair_distance_A=cfg.min_pair_distance_A,
            )
        )

    z_d = np.concatenate([z_m, z_m])
    regions = (
        ("com_repulsive", cfg.com_repulsive_A),
        ("com_well", cfg.com_well_A),
        ("com_shoulder", cfg.com_shoulder_A),
        ("com_long", cfg.com_long_A),
    )
    for _ in range(int(cfg.n_dimer_orientations)):
        rot_b = random_rotation(rng)
        pos_a = r_eq + rng.normal(0.0, cfg.cartesian_noise_sigma_A, size=r_eq.shape)
        pos_b = r_eq + rng.normal(0.0, cfg.cartesian_noise_sigma_A, size=r_eq.shape)
        for source, (lo, hi) in regions:
            for _j in range(int(cfg.n_com_per_region)):
                com = float(rng.uniform(float(lo), float(hi)))
                dimer = assemble_dimer(pos_a, pos_b, com_distance_A=com, rotation_b=rot_b)
                _add(
                    _geometry(
                        z_d,
                        dimer,
                        kind="dimer",
                        source=source,
                        r_com_A=com,
                        atoms_per_monomer=(ATOMS_PER_ACETONE, ATOMS_PER_ACETONE),
                        min_pair_distance_A=cfg.min_pair_distance_A,
                    )
                )

    for _ in range(int(cfg.n_far_field)):
        rot_b = random_rotation(rng)
        dimer = assemble_dimer(
            r_eq, r_eq, com_distance_A=cfg.far_field_com_A, rotation_b=rot_b
        )
        _add(
            _geometry(
                z_d,
                dimer,
                kind="dimer",
                source="far_field",
                r_com_A=float(cfg.far_field_com_A),
                atoms_per_monomer=(ATOMS_PER_ACETONE, ATOMS_PER_ACETONE),
                min_pair_distance_A=cfg.min_pair_distance_A,
            )
        )

    if cfg.include_dataset:
        for frame in load_dataset_dimers():
            pos = np.asarray(frame.get_positions(), dtype=np.float64)
            com = float(
                np.linalg.norm(pos[:ATOMS_PER_ACETONE].mean(0) - pos[ATOMS_PER_ACETONE:].mean(0))
            )
            _add(
                _geometry(
                    np.asarray(frame.get_atomic_numbers(), dtype=np.int32),
                    pos,
                    kind="dimer",
                    source="dmc_extxyz",
                    r_com_A=com,
                    atoms_per_monomer=(ATOMS_PER_ACETONE, ATOMS_PER_ACETONE),
                    min_pair_distance_A=cfg.min_pair_distance_A,
                )
            )
            jitter = pos + rng.normal(0.0, cfg.cartesian_noise_sigma_A, size=pos.shape)
            _add(
                _geometry(
                    np.asarray(frame.get_atomic_numbers(), dtype=np.int32),
                    jitter,
                    kind="dimer",
                    source="dmc_extxyz_noise",
                    r_com_A=com,
                    atoms_per_monomer=(ATOMS_PER_ACETONE, ATOMS_PER_ACETONE),
                    min_pair_distance_A=cfg.min_pair_distance_A,
                )
            )

    for extra in cfg.extra_extxyz:
        frames = ase_read(str(extra), index=":")
        if not isinstance(frames, list):
            frames = [frames]
        for frame in frames:
            n = len(frame)
            if n == ATOMS_PER_ACETONE:
                kind = "monomer"
                per = (ATOMS_PER_ACETONE,)
                r_com = None
            elif n == DIMER_ATOMS:
                kind = "dimer"
                per = (ATOMS_PER_ACETONE, ATOMS_PER_ACETONE)
                pos = np.asarray(frame.get_positions(), dtype=np.float64)
                r_com = float(
                    np.linalg.norm(
                        pos[:ATOMS_PER_ACETONE].mean(0) - pos[ATOMS_PER_ACETONE:].mean(0)
                    )
                )
            else:
                continue
            _add(
                _geometry(
                    np.asarray(frame.get_atomic_numbers(), dtype=np.int32),
                    np.asarray(frame.get_positions(), dtype=np.float64),
                    kind=kind,
                    source=f"extra:{Path(extra).name}",
                    r_com_A=r_com,
                    atoms_per_monomer=per,
                    min_pair_distance_A=cfg.min_pair_distance_A,
                )
            )

    if int(cfg.n_global_rotations_extra) > 0:
        extras: list[Geometry] = []
        for geo in list(geos):
            for _ in range(int(cfg.n_global_rotations_extra)):
                rot = random_rotation(rng)
                extras.append(
                    Geometry(
                        numbers=geo.numbers.copy(),
                        positions=geo.positions @ rot.T,
                        kind=geo.kind,
                        source=geo.source + "+rot",
                        r_com_A=geo.r_com_A,
                        atoms_per_monomer=geo.atoms_per_monomer,
                    )
                )
        geos.extend(extras)

    if not geos:
        raise RuntimeError(
            f"acetone pool is empty after clash filter (rejected={rejected})"
        )
    return geos
