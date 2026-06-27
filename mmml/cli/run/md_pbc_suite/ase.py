#!/usr/bin/env python3
"""
10-mer MEOH cluster: MMML-only MD comparisons (PBC vs vacuum).

Runs (when --all):
  - NVE VelocityVerlet
  - NVT Nose–Hoover chain (symplectic NVT)
  - NVT Langevin

For each mode: periodic cubic box (MIC-safe) and non-periodic (no cell).

Performance / JIT
-----------------
The hybrid calculator JIT-compiles a large XLA program on the **first** energy/force
evaluation for a given (approx.) static configuration: atom count, PBC vs vacuum, and
cutoffs. That compile often dominates wall time for minutes; **BFGS and MD reuse the
same compiled code**, so later steps are much faster per evaluation.

Improving perceived performance:
  - Run one short ``--only`` job first to warm caches, then the full ``--all`` suite
    (separate processes still recompile unless you reuse one Python session).
  - Prefer a **single long run** over many restarts if the goal is total throughput.
  - Tighter ``--pre-min-steps`` / looser ``--pre-min-fmax`` cuts BFGS cost if acceptable.
  - ``ml_batch_size`` in ``setup_calculator`` (if exposed) trades memory vs compile size.
  - GPU: ensure persistent daemon (avoid re-JIT on every driver respawn); ``XLA_PYTHON_CLIENT_PREALLOCATE=false``
    can help memory fragmentation at the cost of some allocator overhead.

Stage timings (seconds) are recorded per run under ``timings_s`` and in ``suite_timing.json``.
Use ``PYTHONUNBUFFERED=1`` so timing lines appear promptly in ``nohup.log``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from ase import Atoms, units
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.optimize import BFGS
from ase.optimize.fire import FIRE
from ase.calculators.calculator import PropertyNotImplementedError

import mmml.interfaces.pycharmmInterface.import_pycharmm as pyci
from mmml.cli.base import resolve_checkpoint_paths
from mmml.interfaces.pycharmmInterface.import_pycharmm import reset_block, reset_block_no_internal
from mmml.interfaces.pycharmmInterface.cutoffs import (
    DEFAULT_MM_SWITCH_ON,
    DEFAULT_MM_SWITCH_WIDTH,
)
from mmml.interfaces.pycharmmInterface.mmml_calculator import CutoffParameters, setup_calculator
from mmml.interfaces.pycharmmInterface.mm_energy_forces import (
    DEFAULT_JAX_MD_SKIN_DISTANCE_A,
    _get_actual_psf_charges,
)
from mmml.utils.geometry_checks import assert_no_intermonomer_atom_overlap
from mmml.utils.jax_gpu_warmup import warmup_ase_mmml_energy_forces
import pycharmm.param as param
import pycharmm.psf as psf
import pycharmm.read as read
import pycharmm.settings as settings
import pycharmm.coor as coor
import pycharmm.generate as gen
import pycharmm.ic as ic
import pycharmm.minimize as charmm_minimize
from mmml.interfaces.pycharmmInterface.packmol_placement import (
    write_monomer_pdb_for_packmol as _write_monomer_pdb_for_packmol,
)
from mmml.interfaces.pycharmmInterface.utils import get_Z_from_psf
from mmml.cli.run.md_pbc_suite.cluster import _build_psf_ordered_cluster
from mmml.paths import default_meoh_template_pdb

pyci.read = read
pyci.settings = settings
pyci.psf = psf

reset_block()
reset_block_no_internal()


def _has_resolved_geometry(coords: np.ndarray, min_span: float = 1.0e-4) -> bool:
    """Return True if a residue has more than origin/identical coordinates."""
    if coords.size == 0 or not np.all(np.isfinite(coords)):
        return False
    return bool(np.max(np.ptp(coords, axis=0)) > min_span)


def _read_cgenff_toppar(*, enable_drude: bool = False) -> None:
    from mmml.interfaces.pycharmmInterface.nbonds_config import read_cgenff_toppar

    read_cgenff_toppar(enable_drude=enable_drude)


def _reset_pycharmm_system() -> None:
    pyci.pycharmm.lingo.charmm_script("DELETE ATOM SELE ALL END")
    reset_block()
    reset_block_no_internal()
    reset_block()


def _make_res_minimize(nbxmod: int, nstep: int = 1000) -> None:
    """Run the same nonbonded/minimization recipe used by make-res."""
    from mmml.interfaces.pycharmmInterface.nbonds_config import apply_vacuum_nbonds

    pyci.pycharmm_quiet()
    apply_vacuum_nbonds(nbxmod=nbxmod)
    charmm_minimize.run_abnr(nstep=nstep, tolenr=1e-3, tolgrd=1e-3)


def _generate_residue_with_make_res_recipe(
    residue: str,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Generate one residue in PyCHARMM using make-res style coordinate relaxation."""
    from mmml.interfaces.pycharmmInterface.mlpot.setup import prepare_charmm_vacuum

    _reset_pycharmm_system()
    prepare_charmm_vacuum()
    _read_cgenff_toppar()
    read.sequence_string(residue)
    gen.new_segment(seg_name="TMP", setup_ic=True)
    ic.prm_fill(replace_all=True)
    ic.build()

    initial = np.array(coor.get_positions().to_numpy(dtype=float), dtype=float, copy=True)
    atom_names = [str(x) for x in np.asarray(psf.get_atype(), dtype=str)]
    if initial.shape[0] != len(atom_names):
        raise RuntimeError(
            f"PSF atom-name count mismatch while generating {residue}: "
            f"{len(atom_names)} vs positions {initial.shape[0]}"
        )

    # make-res deliberately escapes incomplete IC tables by randomizing coordinates
    # before two CHARMM minimization passes.
    seed = sum((i + 1) * ord(ch) for i, ch in enumerate(residue.upper())) + 1729
    rng = np.random.default_rng(seed)
    xyz = pd.DataFrame(2.0 * rng.random(initial.shape), columns=["x", "y", "z"])
    coor.set_positions(xyz)
    _make_res_minimize(nbxmod=1)

    xyz = np.array(coor.get_positions().to_numpy(dtype=float), dtype=float, copy=True)
    xyz *= rng.random(xyz.shape)
    coor.set_positions(pd.DataFrame(xyz, columns=["x", "y", "z"]))
    _make_res_minimize(nbxmod=5)

    coords = np.array(coor.get_positions().to_numpy(dtype=float), dtype=float, copy=True)
    if not _has_resolved_geometry(coords):
        raise RuntimeError(f"PyCHARMM make-res coordinate generation failed for residue {residue!r}")
    z = np.asarray(get_Z_from_psf(), dtype=int)
    if int(z.shape[0]) != len(atom_names):
        raise RuntimeError(
            f"PSF Z count mismatch for {residue}: {z.shape[0]} vs {len(atom_names)} atom names"
        )
    return coords, atom_names, z


def _tmark() -> float:
    return time.perf_counter()


def _tlog(msg: str, log_lines: list[str] | None) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if log_lines is not None:
        log_lines.append(line)


def _cubic_box_length(positions: np.ndarray, ml_cutoff: float, pad: float = 10.0) -> float:
    """Side length (Å) so extent + 2*(cutoff+pad) fits; MIC-safe vs model cutoff."""
    r_cut = float(ml_cutoff) + 2.0
    d = np.ptp(positions, axis=0)
    span = float(np.max(d))
    return max(span + 2.0 * (r_cut + pad), 50.0)


def _enforce_min_com_separation(
    positions: np.ndarray,
    monomer_offsets: np.ndarray,
    min_com_distance: float,
    max_passes: int = 10,
) -> np.ndarray:
    """Push monomer COMs apart so all pair distances are >= min_com_distance."""
    n_molecules = int(len(monomer_offsets) - 1)
    if min_com_distance <= 0.0 or n_molecules <= 1:
        return positions
    pos = np.asarray(positions, dtype=float).copy()
    for _ in range(max_passes):
        coms = np.zeros((n_molecules, 3), dtype=float)
        for i in range(n_molecules):
            s = int(monomer_offsets[i])
            e = int(monomer_offsets[i + 1])
            coms[i] = pos[s:e].mean(axis=0)
        moved = False
        for i in range(n_molecules):
            for j in range(i + 1, n_molecules):
                dvec = coms[j] - coms[i]
                dist = float(np.linalg.norm(dvec))
                if dist < 1e-12:
                    dvec = np.array([1.0, 0.0, 0.0], dtype=float)
                    dist = 1e-12
                if dist < min_com_distance:
                    direction = dvec / dist
                    delta = 0.5 * (min_com_distance - dist) * direction
                    si, ei = int(monomer_offsets[i]), int(monomer_offsets[i + 1])
                    sj, ej = int(monomer_offsets[j]), int(monomer_offsets[j + 1])
                    pos[si:ei] -= delta
                    pos[sj:ej] += delta
                    coms[i] -= delta
                    coms[j] += delta
                    moved = True
        if not moved:
            break
    return pos


def _randomize_monomer_com_positions(
    positions: np.ndarray,
    monomer_offsets: np.ndarray,
    *,
    spacing: float,
    min_com_distance: float,
    seed: int,
) -> np.ndarray:
    """Place molecule centers at reproducible random 3D positions."""
    n_molecules = int(len(monomer_offsets) - 1)
    if n_molecules <= 1:
        return positions

    rng = np.random.default_rng(int(seed))
    min_dist = float(max(0.0, min_com_distance))
    side = max(float(spacing), min_dist, 1.0) * max(1.0, n_molecules ** (1.0 / 3.0))
    targets: list[np.ndarray] = []
    max_attempts = 5000

    while True:
        targets.clear()
        for _i in range(n_molecules):
            for _attempt in range(max_attempts):
                candidate = rng.uniform(0.0, side, size=3)
                if all(float(np.linalg.norm(candidate - prev)) >= min_dist for prev in targets):
                    targets.append(candidate)
                    break
            else:
                break
        if len(targets) == n_molecules:
            break
        side *= 1.25

    target_arr = np.asarray(targets, dtype=float)
    target_arr -= target_arr.mean(axis=0)
    randomized = np.asarray(positions, dtype=float).copy()
    for i, target in enumerate(target_arr):
        s = int(monomer_offsets[i])
        e = int(monomer_offsets[i + 1])
        com = randomized[s:e].mean(axis=0)
        randomized[s:e] += target - com
    return randomized


def _parse_composition(spec: str) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" in tok:
            residue, count_str = tok.split(":", 1)
            count = int(count_str)
        else:
            residue, count = tok, 1
        residue = residue.strip().upper()
        if not residue or count <= 0:
            raise ValueError(f"Invalid composition token: '{tok}'")
        out.append((residue, count))
    if not out:
        raise ValueError("Empty composition")
    return out


def _load_packmol_sphere_positions(
    pdb_path: str | Path,
    atoms_per_list: list[int],
    psf_atom_names: list[str],
) -> np.ndarray:
    from mmml.interfaces.pycharmmInterface.packmol_placement import (
        assign_packmol_pdb_to_psf_order,
    )

    return assign_packmol_pdb_to_psf_order(
        pdb_path,
        psf_atom_names,
        atoms_per_list,
    )


def _residue_geometries_for_composition(
    composition: list[tuple[str, int]],
) -> dict[str, tuple[np.ndarray, list[str], np.ndarray]]:
    """Relaxed monomer coords, atom names, and Z per residue type (SD-only cluster recipe)."""
    from mmml.cli.run.md_pbc_suite.cluster import relax_monomer_geometry_for_cluster

    residue_geometries: dict[str, tuple[np.ndarray, list[str], np.ndarray]] = {}
    for residue, _count in composition:
        if residue not in residue_geometries:
            residue_geometries[residue] = relax_monomer_geometry_for_cluster(residue)
    return residue_geometries


def _residue_geometries_for_packmol(
    composition: list[tuple[str, int]],
    *,
    charmm_sd_steps: int = 50,
    charmm_abnr_steps: int = 100,
    charmm_tolenr: float = 1e-3,
    charmm_tolgrd: float = 1e-3,
    verbose: bool = True,
) -> dict[str, tuple[np.ndarray, list[str], np.ndarray]]:
    """Minimized monomer geometries (PSF order) for Packmol input PDBs."""
    from mmml.cli.run.md_pbc_suite.cluster import build_minimized_monomer_for_packmol

    residue_geometries: dict[str, tuple[np.ndarray, list[str], np.ndarray]] = {}
    for residue, _count in composition:
        if residue not in residue_geometries:
            residue_geometries[residue] = build_minimized_monomer_for_packmol(
                residue,
                nstep_sd=charmm_sd_steps,
                nstep_abnr=charmm_abnr_steps,
                tolenr=charmm_tolenr,
                tolgrd=charmm_tolgrd,
                verbose=verbose,
            )
    return residue_geometries


def _build_cluster_psf_topology_only(
    composition: list[tuple[str, int]],
    *,
    expected_atoms: int,
    atoms_per_list: list[int],
    residue_labels: list[str],
) -> np.ndarray:
    """Build cluster PSF from composition without make-res monomer geometry work."""
    if int(sum(atoms_per_list)) != int(expected_atoms):
        raise RuntimeError(
            f"Handoff layout atom count ({int(sum(atoms_per_list))}) does not match "
            f"expected_atoms={int(expected_atoms)}"
        )
    if len(atoms_per_list) != len(residue_labels):
        raise RuntimeError(
            "Handoff residue label count does not match per-monomer atom layout"
        )

    sequence_items: list[str] = []
    for residue, count in composition:
        sequence_items.extend([residue] * int(count))
    sequence = " ".join(sequence_items)
    from mmml.interfaces.pycharmmInterface.mlpot.setup import prepare_charmm_vacuum

    _reset_pycharmm_system()
    prepare_charmm_vacuum()
    _read_cgenff_toppar()
    read.sequence_string(sequence)
    gen.new_segment(seg_name="CLST", setup_ic=True)
    ic.prm_fill(replace_all=True)
    ic.build()

    z = np.asarray(get_Z_from_psf(), dtype=int)
    if int(len(z)) != int(expected_atoms):
        raise RuntimeError(
            f"Composition-derived PSF atom count ({len(z)}) does not match handoff "
            f"geometry ({int(expected_atoms)}). Composition={composition}"
        )
    return z


def _build_cluster_psf_from_composition(
    composition: list[tuple[str, int]],
    *,
    residue_geometries: dict[str, tuple[np.ndarray, list[str], np.ndarray]] | None = None,
) -> tuple[np.ndarray, list[str], list[int], list[str]]:
    """Build CHARMM PSF for a mixed cluster; return Z, PSF atom names, atoms/monomer, residue order."""
    if residue_geometries is None:
        residue_geometries = _residue_geometries_for_composition(composition)
    else:
        for residue, _count in composition:
            if residue not in residue_geometries:
                from mmml.cli.run.md_pbc_suite.cluster import relax_monomer_geometry_for_cluster

                residue_geometries[residue] = relax_monomer_geometry_for_cluster(residue)

    sequence_items: list[str] = []
    for residue, count in composition:
        sequence_items.extend([residue] * int(count))
    sequence = " ".join(sequence_items)
    from mmml.interfaces.pycharmmInterface.mlpot.setup import prepare_charmm_vacuum

    _reset_pycharmm_system()
    prepare_charmm_vacuum()
    _read_cgenff_toppar()
    read.sequence_string(sequence)
    gen.new_segment(seg_name="CLST", setup_ic=True)
    ic.prm_fill(replace_all=True)
    ic.build()

    z = np.asarray(get_Z_from_psf(), dtype=int)
    atom_names = [str(x) for x in np.asarray(psf.get_atype(), dtype=str)]
    atoms_per_list: list[int] = []
    ordered_residue_names: list[str] = []
    for residue, count in composition:
        n_atoms_res = int(residue_geometries[residue][0].shape[0])
        for _ in range(int(count)):
            atoms_per_list.append(int(n_atoms_res))
            ordered_residue_names.append(residue)
    expected_atoms = int(np.sum(np.asarray(atoms_per_list, dtype=int)))
    if expected_atoms != int(len(atom_names)):
        raise RuntimeError(
            f"Composition-derived atom count ({expected_atoms}) does not match PSF "
            f"({len(atom_names)}). Composition={composition}"
        )

    offsets = np.zeros(len(atoms_per_list) + 1, dtype=int)
    offsets[1:] = np.cumsum(np.asarray(atoms_per_list, dtype=int))
    for i, residue in enumerate(ordered_residue_names):
        s = int(offsets[i])
        e = int(offsets[i + 1])
        residue_coords, residue_atom_names, _residue_z = residue_geometries[residue]
        local_atom_names = atom_names[s:e]
        if local_atom_names != residue_atom_names:
            raise RuntimeError(
                f"Atom order mismatch for {residue}: PSF has {local_atom_names}, "
                f"make-res produced {residue_atom_names}"
            )
        if int(residue_coords.shape[0]) != e - s:
            raise RuntimeError(f"Atom count mismatch for {residue} at monomer {i}")
    return z, atom_names, atoms_per_list, ordered_residue_names


def _build_cluster_from_composition_packmol(
    *,
    composition: list[tuple[str, int]],
    placement: str = "cube",
    center: tuple[float, float, float],
    cube_side: float | None = None,
    radius: float | None = None,
    tolerance: float = 2.0,
    seed: int | None = None,
    charmm_sd_steps: int = 50,
    charmm_abnr_steps: int = 100,
    charmm_tolenr: float = 1e-3,
    charmm_tolgrd: float = 1e-3,
    scratch_dir: str | Path | None = None,
    verbose: bool = True,
    reuse_packmol_cache: bool = True,
    packmol_cache_dir: str | Path | None = None,
    force_rebuild_packmol_cache: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[int], list[str]]:
    from mmml.cli.run.md_pbc_suite.cluster import build_packmol_composition_cluster

    return build_packmol_composition_cluster(
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
        scratch_dir=scratch_dir,
        verbose=verbose,
        reuse_packmol_cache=reuse_packmol_cache,
        packmol_cache_dir=packmol_cache_dir,
        force_rebuild_packmol_cache=force_rebuild_packmol_cache,
    )


def _build_cluster_from_composition_pyxtal(
    *,
    composition: list[tuple[str, int]],
    space_group: int = 14,
    dimension: int = 3,
    factor: float = 1.0,
    unit_stoichiometry: list[int] | None = None,
    supercell_reps: tuple[int, int, int] | None = None,
    seed: int | None = None,
    max_attempts: int = 20,
    charmm_sd_steps: int = 50,
    charmm_abnr_steps: int = 100,
    charmm_tolenr: float = 1e-3,
    charmm_tolgrd: float = 1e-3,
    scratch_dir: str | Path | None = None,
    verbose: bool = True,
    optimize_ase: bool = False,
    optimize_ase_emt: bool = False,
    trim_to_composition: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[int], list[str]]:
    from mmml.cli.run.md_pbc_suite.cluster import build_pyxtal_composition_cluster

    return build_pyxtal_composition_cluster(
        composition=composition,
        space_group=space_group,
        dimension=dimension,
        factor=factor,
        unit_stoichiometry=unit_stoichiometry,
        supercell_reps=supercell_reps,
        seed=seed,
        max_attempts=max_attempts,
        charmm_sd_steps=charmm_sd_steps,
        charmm_abnr_steps=charmm_abnr_steps,
        charmm_tolenr=charmm_tolenr,
        charmm_tolgrd=charmm_tolgrd,
        scratch_dir=scratch_dir,
        verbose=verbose,
        optimize_ase=optimize_ase,
        optimize_ase_emt=optimize_ase_emt,
        trim_to_composition=trim_to_composition,
    )


def resolve_cluster_packmol(args: argparse.Namespace) -> bool:
    from mmml.interfaces.pycharmmInterface.packmol_placement import resolve_packmol_use

    return resolve_packmol_use(
        composition=getattr(args, "composition", None),
        packmol=getattr(args, "packmol", None),
        pyxtal=getattr(args, "pyxtal", None),
    )


def resolve_cluster_pyxtal(args: argparse.Namespace) -> bool:
    from mmml.interfaces.pyxtal_placement import resolve_pyxtal_use

    return resolve_pyxtal_use(
        composition=getattr(args, "composition", None),
        pyxtal=getattr(args, "pyxtal", None),
    )


def resolve_cluster_packmol_sphere(args: argparse.Namespace) -> bool:
    """Backward-compatible alias: True when any Packmol placement is active."""
    return resolve_cluster_packmol(args)


def packmol_sphere_center_from_args(args: argparse.Namespace) -> tuple[float, float, float]:
    center = getattr(args, "packmol_center", None)
    if center is not None:
        if len(center) != 3:
            raise ValueError("--packmol-center requires three floats: CX CY CZ")
        return (float(center[0]), float(center[1]), float(center[2]))
    return (0.0, 0.0, 0.0)


def resolve_cluster_geometry(
    args: argparse.Namespace,
    handoff: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int], list[str], dict[str, int] | None]:
    """Build or load cluster geometry; skip Packmol when continuing from handoff."""
    if handoff is not None:
        from mmml.cli.run.md_handoff import cluster_geometry_from_handoff

        z, r0, atoms_per_list, residue_labels, composition_summary = (
            cluster_geometry_from_handoff(
                handoff,
                composition=getattr(args, "composition", None),
                n_molecules=int(getattr(args, "n_molecules", 1) or 1),
            )
        )
        if not getattr(args, "quiet", False):
            print(
                f"Continuing from handoff ({len(z)} atoms); skipped Packmol/cluster build",
                flush=True,
            )
        return z, r0, atoms_per_list, residue_labels, composition_summary
    return build_initial_cluster_from_args(args)


def build_initial_cluster_from_args(
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, list[int], list[str], dict[str, int] | None]:
    """Build cluster geometry; returns z, r0, atoms_per_list, residue_labels, composition_summary."""
    from mmml.interfaces.pycharmmInterface.packmol_placement import (
        resolve_packmol_cube_side_from_args,
        resolve_packmol_placement_mode,
        resolve_packmol_sphere_radius,
    )
    from mmml.interfaces.pyxtal_placement import parse_supercell_reps

    use_pyxtal = resolve_cluster_pyxtal(args)
    use_packmol = resolve_cluster_packmol(args)
    if use_pyxtal and not args.composition:
        raise ValueError("PyXtal placement requires --composition (e.g. MEOH:8).")
    if use_packmol and not args.composition:
        raise ValueError(
            "Packmol placement requires --composition (e.g. MEOH:30) "
            "and --box-size (Å) for cube packing."
        )

    if args.composition:
        composition = _parse_composition(args.composition)
        composition_summary = {res: int(cnt) for res, cnt in composition}
        if use_pyxtal:
            supercell_reps = None
            if getattr(args, "pyxtal_supercell", None):
                supercell_reps = parse_supercell_reps(str(args.pyxtal_supercell))
            z, r0, atoms_per_list, residue_labels = _build_cluster_from_composition_pyxtal(
                composition=composition,
                space_group=int(getattr(args, "pyxtal_spg", 14)),
                dimension=int(getattr(args, "pyxtal_dim", 3)),
                factor=float(getattr(args, "pyxtal_factor", 1.0)),
                unit_stoichiometry=getattr(args, "pyxtal_stoichiometry", None),
                supercell_reps=supercell_reps,
                seed=int(getattr(args, "seed", 0)),
                max_attempts=int(getattr(args, "pyxtal_attempts", 20)),
                charmm_sd_steps=int(getattr(args, "charmm_sd_steps", 50)),
                charmm_abnr_steps=int(getattr(args, "charmm_abnr_steps", 100)),
                charmm_tolenr=float(getattr(args, "charmm_tolenr", 1e-3)),
                charmm_tolgrd=float(getattr(args, "charmm_tolgrd", 1e-3)),
                scratch_dir=(
                    Path(args.output_dir) / "pyxtal_cluster"
                    if getattr(args, "output_dir", None) is not None
                    else None
                ),
                verbose=not getattr(args, "quiet", False),
                optimize_ase=bool(getattr(args, "optimize_pyxtal", False)),
                optimize_ase_emt=bool(getattr(args, "optimize_pyxtal_emt", False)),
                trim_to_composition=bool(getattr(args, "pyxtal_trim", True)),
            )
            print(
                f"Cluster built with PyXtal: spg={int(getattr(args, 'pyxtal_spg', 14))} "
                f"dim={int(getattr(args, 'pyxtal_dim', 3))}"
            )
        elif use_packmol:
            placement = resolve_packmol_placement_mode(
                packmol_placement=getattr(args, "packmol_placement", None),
                packmol_sphere=getattr(args, "packmol_sphere", None),
            )
            center = packmol_sphere_center_from_args(args)
            tolerance = float(getattr(args, "packmol_tolerance", 2.0))
            cube_side: float | None = None
            radius: float | None = None
            if placement == "sphere":
                radius = resolve_packmol_sphere_radius(
                    getattr(args, "packmol_radius", None),
                    getattr(args, "flat_bottom_radius", None),
                )
            else:
                cube_side = resolve_packmol_cube_side_from_args(args)
            z, r0, atoms_per_list, residue_labels = _build_cluster_from_composition_packmol(
                composition=composition,
                placement=placement,
                center=center,
                cube_side=cube_side,
                radius=radius,
                tolerance=tolerance,
                seed=int(getattr(args, "seed", 0)),
                charmm_sd_steps=int(getattr(args, "charmm_sd_steps", 50)),
                charmm_abnr_steps=int(getattr(args, "charmm_abnr_steps", 100)),
                charmm_tolenr=float(getattr(args, "charmm_tolenr", 1e-3)),
                charmm_tolgrd=float(getattr(args, "charmm_tolgrd", 1e-3)),
                scratch_dir=(
                    Path(args.output_dir) / "packmol_cluster"
                    if getattr(args, "output_dir", None) is not None
                    else None
                ),
                verbose=not getattr(args, "quiet", False),
                reuse_packmol_cache=bool(getattr(args, "reuse_packmol_cache", True)),
                packmol_cache_dir=getattr(args, "packmol_cache_dir", None),
                force_rebuild_packmol_cache=bool(
                    getattr(args, "rebuild_packmol", False)
                ),
            )
            if placement == "sphere":
                fb_r = getattr(args, "flat_bottom_radius", None)
                print(
                    f"Cluster built with Packmol sphere: center={center}, "
                    f"packmol_radius={radius:.3f} Å"
                    + (
                        f", flat_bottom_radius={float(fb_r):.3f} Å"
                        if fb_r is not None and float(fb_r) > 0
                        else ""
                    )
                )
            else:
                print(
                    f"Cluster built with Packmol cube: center={center}, "
                    f"side={cube_side:.3f} Å, tol={tolerance:.1f} Å"
                )
        else:
            z, r0, atoms_per_list, residue_labels = _build_cluster_from_composition(
                composition=composition,
                spacing=args.spacing,
            )
        return z, r0, atoms_per_list, residue_labels, composition_summary

    z, r0 = _build_psf_ordered_cluster(
        "MEOH",
        args.n_molecules,
        args.spacing,
        template_pdb=args.template_pdb.expanduser().resolve(),
    )
    n_atoms_tmp = len(z)
    atoms_per_uniform = n_atoms_tmp // args.n_molecules
    atoms_per_list = [int(atoms_per_uniform)] * int(args.n_molecules)
    residue_labels = ["MEOH"] * int(args.n_molecules)
    composition_summary = {"MEOH": int(args.n_molecules)}
    return z, r0, atoms_per_list, residue_labels, composition_summary


def _build_cluster_from_composition(
    *,
    composition: list[tuple[str, int]],
    spacing: float,
) -> tuple[np.ndarray, np.ndarray, list[int], list[str]]:
    # Build monomer templates once; do not call make-res again after the cluster PSF
    # (_generate_residue_with_make_res_recipe resets CHARMM and deletes the cluster).
    residue_geometries = _residue_geometries_for_composition(composition)
    z, atom_names, atoms_per_list, ordered_residue_names = _build_cluster_psf_from_composition(
        composition,
        residue_geometries=residue_geometries,
    )

    n_molecules = len(atoms_per_list)
    n_side = int(np.ceil(np.sqrt(n_molecules)))
    offsets = np.zeros(n_molecules + 1, dtype=int)
    offsets[1:] = np.cumsum(np.asarray(atoms_per_list, dtype=int))
    expected_atoms = int(offsets[-1])
    shifted = coor.get_positions().to_numpy(dtype=float).copy()
    if int(shifted.shape[0]) != expected_atoms:
        raise RuntimeError(
            f"CHARMM coordinate count ({shifted.shape[0]}) != cluster PSF atom count "
            f"({expected_atoms}). The cluster PSF may have been cleared after build."
        )
    for i in range(n_molecules):
        s = int(offsets[i])
        e = int(offsets[i + 1])
        residue = ordered_residue_names[i]
        residue_coords, _residue_atom_names, _residue_z = residue_geometries[residue]
        shifted[s:e] = residue_coords
        com = shifted[s:e].mean(axis=0)
        shift = np.array([(i % n_side) * spacing, (i // n_side) * spacing, 0.0], dtype=float)
        shifted[s:e] = shifted[s:e] - com + shift
    coor.set_positions(pd.DataFrame(shifted, columns=["x", "y", "z"]))
    return z, shifted, atoms_per_list, ordered_residue_names


def _factory_mmml(
    *,
    z: np.ndarray,
    r: np.ndarray,
    n_mol: int,
    atoms_per: int | list[int],
    base_ckpt_dir: Path,
    ml_cut: float,
    mm_sw: float,
    mm_cut: float,
    cell_scalar: float | None,
    verbose: bool,
    jax_md_capacity_multiplier: float,
    jax_md_capacity_growth_factor: float,
    jax_md_max_overflow_retries: int,
    jax_md_overflow_fallback_to_cell_list: bool,
    jax_md_update_interval: int,
    jax_md_skin_distance: float,
    max_pairs: int,
    do_ml: bool = True,
    do_ml_dimer: bool = True,
    do_mm: bool = True,
    timings: dict[str, float] | None = None,
    flat_bottom_radius: float | None = None,
    flat_bottom_force_const: float = 1.0,
    flat_bottom_mode: str = "system",
    min_com_restraint_distance: float | None = None,
    min_com_restraint_force_const: float = 1.0,
    defer_xla_gpu_warmup: bool = False,
    ml_batch_size: Optional[int] = None,
    ml_max_active_dimers: Optional[int] = None,
    ml_compute_dtype: Optional[str] = None,
    at_codes_override: np.ndarray | None = None,
):
    if at_codes_override is not None:
        at_codes = np.asarray(at_codes_override, dtype=int)
    else:
        at_codes = np.asarray(psf.get_iac(), dtype=int) - 1
    n_types = len(param.get_atc())
    cell_arg = float(cell_scalar) if cell_scalar is not None else False
    t0 = _tmark()
    if isinstance(atoms_per, int):
        max_atoms_per = int(atoms_per)
    else:
        max_atoms_per = int(max(atoms_per))
    factory = setup_calculator(
        ATOMS_PER_MONOMER=atoms_per,
        N_MONOMERS=n_mol,
        ml_switch_width=ml_cut,
        mm_switch_on=mm_sw,
        mm_switch_width=mm_cut,
        doML=do_ml,
        doMM=do_mm,
        doML_dimer=do_ml_dimer,
        debug=False,
        model_restart_path=base_ckpt_dir,
        MAX_ATOMS_PER_SYSTEM=max_atoms_per * 2,
        cell=cell_arg,
        ep_scale=np.ones(n_types),
        sig_scale=np.ones(n_types),
        at_codes_override=at_codes,
        verbose=verbose,
        max_pairs=max_pairs,
        jax_md_capacity_multiplier=jax_md_capacity_multiplier,
        jax_md_capacity_growth_factor=jax_md_capacity_growth_factor,
        jax_md_max_overflow_retries=jax_md_max_overflow_retries,
        jax_md_overflow_fallback_to_cell_list=jax_md_overflow_fallback_to_cell_list,
        jax_md_update_interval=jax_md_update_interval,
        jax_md_skin_distance=jax_md_skin_distance,
        flat_bottom_radius=flat_bottom_radius,
        flat_bottom_force_const=flat_bottom_force_const,
        flat_bottom_mode=flat_bottom_mode,
        min_com_restraint_distance=min_com_restraint_distance,
        min_com_restraint_force_const=min_com_restraint_force_const,
        defer_xla_gpu_warmup=defer_xla_gpu_warmup,
        ml_batch_size=ml_batch_size,
        ml_max_active_dimers=ml_max_active_dimers,
        ml_compute_dtype=ml_compute_dtype,
    )
    t1 = _tmark()
    cutoff = CutoffParameters(
        ml_switch_width=ml_cut,
        mm_switch_on=mm_sw,
        mm_switch_width=mm_cut,
    )
    calc_result = factory(
        atomic_numbers=z,
        atomic_positions=r,
        n_monomers=n_mol,
        cutoff_params=cutoff,
        doML=do_ml,
        doMM=do_mm,
        doML_dimer=do_ml_dimer,
        backprop=False,
        debug=False,
        energy_conversion_factor=1.0,
        force_conversion_factor=1.0,
        verbose=verbose,
    )
    t2 = _tmark()
    if timings is not None:
        timings["mmml_setup_calculator_s"] = t1 - t0
        timings["mmml_factory_call_s"] = t2 - t1
    if len(calc_result) == 3:
        mmml_calc, _, _ = calc_result
    else:
        mmml_calc, _ = calc_result
    return mmml_calc


def _validate_psf_charges(
    *,
    monomer_offsets: np.ndarray,
    residue_labels: list[str],
    total_atoms: int,
    log_lines: list[str] | None = None,
) -> dict:
    charges = _get_actual_psf_charges(total_atoms)[:total_atoms]
    atom_types = np.asarray(psf.get_atype(), dtype=str)[:total_atoms]
    if charges.shape[0] != total_atoms:
        raise RuntimeError(
            f"PSF charge count mismatch: expected {total_atoms}, got {charges.shape[0]}"
        )
    if atom_types.shape[0] != total_atoms:
        raise RuntimeError(
            f"PSF atom-type count mismatch: expected {total_atoms}, got {atom_types.shape[0]}"
        )
    if not np.all(np.isfinite(charges)):
        bad = np.where(~np.isfinite(charges))[0][:10].tolist()
        raise RuntimeError(f"PSF charges contain non-finite values at atom indices {bad}")

    residue_reference: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    residue_summaries: dict[str, dict] = {}
    for i, residue in enumerate(residue_labels):
        s = int(monomer_offsets[i])
        e = int(monomer_offsets[i + 1])
        local_types = atom_types[s:e]
        local_charges = charges[s:e]
        if residue in residue_reference:
            ref_types, ref_charges = residue_reference[residue]
            if list(local_types) != list(ref_types):
                raise RuntimeError(
                    f"PSF atom-type order differs between {residue} copies: "
                    f"{list(local_types)} != {list(ref_types)}"
                )
            if not np.allclose(local_charges, ref_charges, atol=1e-8, rtol=0.0):
                raise RuntimeError(
                    f"PSF charges differ between {residue} copies: "
                    f"{local_charges.tolist()} != {ref_charges.tolist()}"
                )
        else:
            residue_reference[residue] = (local_types.copy(), local_charges.copy())
            residue_summaries[residue] = {
                "n_atoms": int(e - s),
                "charge_sum_e": float(np.sum(local_charges)),
                "atom_types": [str(x) for x in local_types],
                "charges_e": [float(x) for x in local_charges],
            }

    total_charge = float(np.sum(charges))
    summary = {
        "total_charge_e": total_charge,
        "n_charged_atoms": int(np.count_nonzero(np.abs(charges) > 1e-12)),
        "min_charge_e": float(np.min(charges)) if charges.size else 0.0,
        "max_charge_e": float(np.max(charges)) if charges.size else 0.0,
        "residues": residue_summaries,
    }
    _tlog(
        "PSF charge validation: "
        f"total={total_charge:.6f} e, charged_atoms={summary['n_charged_atoms']}, "
        + ", ".join(
            f"{res}:{data['charge_sum_e']:.6f} e"
            for res, data in residue_summaries.items()
        ),
        log_lines,
    )
    return summary


def _numpy_wrap_monomers_primary_cell(
    positions: np.ndarray,
    monomer_offsets: np.ndarray,
    cell_matrix: np.ndarray,
) -> np.ndarray:
    from mmml.utils.geometry_checks import wrap_monomers_primary_cell

    return wrap_monomers_primary_cell(positions, monomer_offsets, cell_matrix)


def _run_charmm_minimize(
    atoms: Atoms,
    *,
    nstep_sd: int,
    nstep_abnr: int,
    tolenr: float,
    tolgrd: float,
    nbxmod: int = 5,
    timings: dict[str, float] | None = None,
    cubic_box_side_A: float | None = None,
    quiet: bool = True,
    show_energy: bool = False,
) -> None:
    """Run optional CHARMM minimization and write updated coords back to ASE atoms.

    When ``cubic_box_side_A`` is set, CHARMM crystal / IMAGE PBC is rebuilt for a
    cubic cell of that edge length (matches the MD periodic box for JAX-MD / ASE).

    By default CHARMM runs at PRNLev 0 (no per-step ENERGY / nonbond banners). Set
    ``show_energy=True`` for a compact energy table before and after minimization.
    """
    if nstep_sd <= 0 and nstep_abnr <= 0:
        return
    t0 = _tmark()
    reset_block()
    coor.set_positions(pd.DataFrame(atoms.get_positions(), columns=["x", "y", "z"]))
    try:
        if quiet:
            pyci.pycharmm_quiet()
        from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import apply_pbc_nbonds

        use_pbc_charmm = cubic_box_side_A is not None and float(cubic_box_side_A) > 0.0
        if use_pbc_charmm:
            from mmml.interfaces.pycharmmInterface.pycharmmCommands import pbcset
            from mmml.interfaces.pycharmmInterface.setupBox import _ensure_crystal_image_str

            _ensure_crystal_image_str()
            L = float(cubic_box_side_A)
            pyci.pycharmm.lingo.charmm_script(pbcset.format(SIDELENGTH=L))
            # IMAGE before NBONDS; CUTIM >= CUTNB or CHARMM errors (CUTNB larger than CUTIM).
            pyci.pycharmm.lingo.charmm_script(
                "open read unit 10 card name crystal_image.str\n"
                f"crystal defi cubic {L} {L} {L} 90. 90. 90.\n"
                "CRYSTAL READ UNIT 10 CARD\n"
                "image byres xcen 0.0 ycen 0.0 zcen 0.0 sele all end\n"
            )

        if use_pbc_charmm:
            cuts = apply_pbc_nbonds(nbxmod=nbxmod, cubic_box_side_A=L)
            if cuts.was_capped and not quiet:
                print(cuts.summary_line(), flush=True)
            nbond_kw = cuts.as_pbc_nbond_kwargs(nbxmod=nbxmod)
        else:
            from mmml.interfaces.pycharmmInterface.nbonds_config import (
                vacuum_nbond_kwargs,
            )

            nbond_kw = vacuum_nbond_kwargs(nbxmod=nbxmod)
        pyci.pycharmm.NonBondedScript(**nbond_kw).run()
        if show_energy:
            print("CHARMM energy before minimization:")
            pyci.pycharmm_quiet()
            pyci.safe_energy_show()
        if nstep_sd > 0:
            pyci.pycharmm_quiet()
            charmm_minimize.run_sd(nstep=nstep_sd, tolenr=tolenr, tolgrd=tolgrd)
        if nstep_abnr > 0:
            pyci.pycharmm_quiet()
            charmm_minimize.run_abnr(nstep=nstep_abnr, tolenr=tolenr, tolgrd=tolgrd)
        minimized_positions = coor.get_positions().to_numpy(dtype=float)
        coor.set_positions(pd.DataFrame(minimized_positions, columns=["x", "y", "z"]))
        if show_energy:
            print("CHARMM energy after minimization:")
            pyci.pycharmm_quiet()
            pyci.safe_energy_show()
    finally:
        pyci.pycharmm_quiet()
    atoms.set_positions(coor.get_positions().to_numpy(dtype=float))
    if timings is not None:
        timings["charmm_min_wall_s"] = _tmark() - t0


def _check_or_charmm_overlap_rescue(
    atoms: Atoms,
    monomer_offsets: np.ndarray,
    *,
    min_distance: float,
    context: str,
    nstep_sd: int,
    nstep_abnr: int,
    tolenr: float,
    tolgrd: float,
    nbxmod: int = 5,
    timings: dict[str, float] | None = None,
) -> float:
    """Check for inter-monomer overlaps, using CHARMM minimization as first rescue."""
    cell = atoms.cell.array if atoms.pbc.any() else None
    try:
        return assert_no_intermonomer_atom_overlap(
            atoms.get_positions(),
            monomer_offsets,
            min_distance=min_distance,
            cell=cell,
            context=context,
        )
    except RuntimeError as exc:
        print(f"{context}: overlap detected; attempting CHARMM SD/ABNR rescue before aborting.")
        if timings is not None:
            timings["overlap_rescue_attempted"] = float(timings.get("overlap_rescue_attempted", 0.0) + 1.0)
        _run_charmm_minimize(
            atoms,
            nstep_sd=nstep_sd,
            nstep_abnr=nstep_abnr,
            tolenr=tolenr,
            tolgrd=tolgrd,
            nbxmod=nbxmod,
            timings=timings,
            quiet=True,
        )
        try:
            return assert_no_intermonomer_atom_overlap(
                atoms.get_positions(),
                monomer_offsets,
                min_distance=min_distance,
                cell=cell,
                context=f"{context} after CHARMM overlap rescue",
            )
        except RuntimeError as rescue_exc:
            raise RuntimeError(f"{exc}; CHARMM overlap rescue failed: {rescue_exc}") from rescue_exc


def _save_cutoff_plot(
    *,
    out_dir: Path,
    ml_cutoff: float,
    mm_switch_on: float,
    mm_cutoff: float,
    log_lines: list[str] | None = None,
) -> Path:
    cutoff = CutoffParameters(
        ml_cutoff=ml_cutoff,
        mm_switch_on=mm_switch_on,
        mm_cutoff=mm_cutoff,
        complementary_handoff=True,
    )
    _ = cutoff.plot_cutoff_parameters(save_dir=out_dir)
    out_path = out_dir / (
        f"cutoffs_schematic_{float(ml_cutoff):.2f}_{float(mm_switch_on):.2f}_"
        f"{float(mm_cutoff):.2f}_complementary.png"
    )
    _tlog(f"cutoff plot: {out_path}", log_lines)
    return out_path


def run_md(
    *,
    name: str,
    atoms: Atoms,
    mode: str,
    dt_fs: float,
    nsteps: int,
    log_every: int,
    traj_every: int,
    traj_chunk_frames: int,
    out_dir: Path,
    nvt_temp_K: float,
    nve_temp_K: float,
    langevin_friction: float,
    seed: int,
    monomer_offsets: np.ndarray,
    min_intermonomer_atom_distance: float,
    overlap_rescue_charmm_sd_steps: int,
    overlap_rescue_charmm_abnr_steps: int,
    charmm_tolenr: float,
    charmm_tolgrd: float,
    charmm_nbxmod: int,
    path_prefix: Path | None = None,
    timings: dict[str, float] | None = None,
    log_lines: list[str] | None = None,
) -> dict:
    dt = dt_fs * units.fs
    traj_chunk_frames = int(max(0, traj_chunk_frames))
    traj_paths: list[Path] = []
    traj: Trajectory | None = None
    traj_chunk_idx = -1
    frames_in_chunk = 0
    total_frames_written = 0
    log_path = out_dir / f"{name}.log"
    summary_path = out_dir / f"{name}_run.json"
    rows: list[dict] = []
    rng = np.random.default_rng(seed)
    t_md_entry = _tmark()

    if mode == "nve":
        MaxwellBoltzmannDistribution(atoms, temperature_K=nve_temp_K, rng=rng)
        Stationary(atoms)
        ZeroRotation(atoms)
        dyn = VelocityVerlet(atoms, timestep=dt)
    elif mode == "nvt_nhc":
        MaxwellBoltzmannDistribution(atoms, temperature_K=nvt_temp_K, rng=rng)
        Stationary(atoms)
        ZeroRotation(atoms)
        tdamp = 100.0 * dt
        dyn = NoseHooverChainNVT(
            atoms,
            timestep=dt,
            temperature_K=nvt_temp_K,
            tdamp=tdamp,
            tchain=3,
            tloop=1,
        )
    elif mode == "nvt_langevin":
        MaxwellBoltzmannDistribution(atoms, temperature_K=nvt_temp_K, rng=rng)
        Stationary(atoms)
        ZeroRotation(atoms)
        dyn = Langevin(
            atoms,
            timestep=dt,
            temperature_K=nvt_temp_K,
            friction=langevin_friction,
            fixcm=False,
            rng=rng,
        )
    else:
        raise ValueError(mode)

    t_before_first_snapshot = _tmark()
    _tlog(f"{name}: MD initialization complete; writing initial frame and observables", log_lines)

    def snapshot(step: int) -> None:
        ep = float(atoms.get_potential_energy())
        ek = float(atoms.get_kinetic_energy())
        et = ep + ek
        temp = float(ek / (1.5 * len(atoms) * units.kB))
        fmax = float(np.abs(atoms.get_forces()).max())
        row = {
            "step": step,
            "time_ps": step * dt_fs * 1e-3,
            "Etot_eV": et,
            "Epot_eV": ep,
            "Ekin_eV": ek,
            "T_K": temp,
            "Fmax_eVA": fmax,
        }
        if mode == "nvt_nhc" and hasattr(dyn, "get_conserved_energy"):
            try:
                row["H_eV"] = float(dyn.get_conserved_energy())
            except (PropertyNotImplementedError, NotImplementedError):
                # Some ASE calculators (including this hybrid MMML calculator) do not
                # expose free_energy, which NoseHooverChainNVT may request internally.
                # Keep logging robust by skipping H in that case.
                pass
        rows.append(row)

    def _open_chunk() -> None:
        nonlocal traj, traj_chunk_idx, frames_in_chunk
        traj_chunk_idx += 1
        suffix = f".part{traj_chunk_idx:04d}.traj" if traj_chunk_frames > 0 else ".traj"
        path = out_dir / f"{name}{suffix}"
        traj_paths.append(path)
        traj = Trajectory(str(path), "w", atoms)
        frames_in_chunk = 0

    def _write_frame() -> None:
        nonlocal total_frames_written, frames_in_chunk
        if traj is None or (traj_chunk_frames > 0 and frames_in_chunk >= traj_chunk_frames):
            if traj is not None:
                traj.close()
            _open_chunk()
        assert traj is not None
        traj.write(atoms)
        frames_in_chunk += 1
        total_frames_written += 1

    _check_or_charmm_overlap_rescue(
        atoms,
        monomer_offsets,
        min_distance=min_intermonomer_atom_distance,
        context=f"{name}: initial MD frame",
        nstep_sd=overlap_rescue_charmm_sd_steps,
        nstep_abnr=overlap_rescue_charmm_abnr_steps,
        tolenr=charmm_tolenr,
        tolgrd=charmm_tolgrd,
        nbxmod=charmm_nbxmod,
        timings=timings,
    )
    _write_frame()  # initial frame
    snapshot(0)
    t_after_first_snapshot = _tmark()
    _tlog(
        f"{name}: initial MD snapshot {t_after_first_snapshot - t_before_first_snapshot:.3f} s; "
        f"running {nsteps} steps",
        log_lines,
    )
    dyn.attach(lambda: snapshot(dyn.get_number_of_steps()), interval=log_every)
    dyn.attach(
        lambda: _check_or_charmm_overlap_rescue(
            atoms,
            monomer_offsets,
            min_distance=min_intermonomer_atom_distance,
            context=f"{name}: MD step {dyn.get_number_of_steps()}",
            nstep_sd=overlap_rescue_charmm_sd_steps,
            nstep_abnr=overlap_rescue_charmm_abnr_steps,
            tolenr=charmm_tolenr,
            tolgrd=charmm_tolgrd,
            nbxmod=charmm_nbxmod,
            timings=timings,
        ),
        interval=max(1, traj_every),
    )
    dyn.attach(_write_frame, interval=max(1, traj_every))
    t_run0 = _tmark()
    dyn.run(nsteps)
    t_run1 = _tmark()
    _tlog(f"{name}: MD integrator loop {t_run1 - t_run0:.3f} s", log_lines)
    if traj is not None:
        traj.close()
    if timings is not None:
        timings["md_entry_to_integrator_ready_s"] = t_before_first_snapshot - t_md_entry
        timings["md_first_observable_snapshot_s"] = t_after_first_snapshot - t_before_first_snapshot
        timings["md_attach_overhead_s"] = max(0.0, t_run0 - t_after_first_snapshot)
        timings["md_integrator_loop_s"] = t_run1 - t_run0
        timings["md_per_step_mean_ms"] = (t_run1 - t_run0) / max(nsteps, 1) * 1000.0

    with log_path.open("w", encoding="utf-8") as f:
        keys = ["time_ps", "Etot_eV", "Epot_eV", "Ekin_eV", "T_K", "Fmax_eVA"]
        if mode == "nvt_nhc":
            keys.append("H_eV")
        f.write(" ".join(keys) + "\n")
        for r in rows:
            f.write(" ".join(str(r[k]) for k in keys if k in r) + "\n")

    et = np.array([r["Etot_eV"] for r in rows])
    tk = np.array([r["T_K"] for r in rows])
    if path_prefix is not None:
        traj_refs = [str(p.relative_to(path_prefix)) for p in traj_paths]
        log_ref = str(log_path.relative_to(path_prefix))
    else:
        traj_refs = [str(p) for p in traj_paths]
        log_ref = str(log_path)

    out = {
        "traj": traj_refs[0] if traj_refs else "",
        "traj_parts": traj_refs,
        "traj_part_count": len(traj_refs),
        "log": log_ref,
        "mode": mode,
        "frames_traj": int(total_frames_written),
        "log_samples": len(rows),
        "etot_start_eV": float(et[0]),
        "etot_end_eV": float(et[-1]),
        "etot_drift_eV": float(et[-1] - et[0]),
        "etot_span_eV": float(et.max() - et.min()),
        "temp_start_K": float(tk[0]),
        "temp_end_K": float(tk[-1]),
        "temp_mean_K": float(tk.mean()),
    }
    if mode == "nvt_nhc" and "H_eV" in rows[0]:
        h = np.array([r["H_eV"] for r in rows])
        out["H_drift_eV"] = float(h[-1] - h[0])
        out["H_span_eV"] = float(h.max() - h.min())
    if timings is not None:
        timings_payload: dict[str, float | int | str | bool] = {}
        for k, v in timings.items():
            if isinstance(v, (bool, int, float, str)):
                timings_payload[k] = v
            else:
                timings_payload[k] = str(v)
        out["timings_s"] = timings_payload
    summary_path.write_text(json.dumps(out, indent=2))
    return out


def main(argv: list[str] | None = None) -> int:
    from mmml.utils.jax_gpu_warmup import apply_xla_cuda_timer_log_filter

    apply_xla_cuda_timer_log_filter()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Portable .json or Orbax path (default: bundled manifest model with "
            "lowest validation force MAE, or $MMML_CKPT)."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/md_10mer_mmml_pbc_suite"))
    parser.add_argument("--template-pdb", type=Path, default=default_meoh_template_pdb())
    parser.add_argument("--n-molecules", type=int, default=10)
    parser.add_argument(
        "--composition",
        type=str,
        default=None,
        help="Residue composition as RES:count comma list (e.g. MEOH:5,TIP3:5). Overrides --n-molecules.",
    )
    parser.add_argument("--spacing", type=float, default=5.0, help="Target minimum random COM spacing in Angstrom.")
    parser.add_argument(
        "--min-com-start-distance",
        type=float,
        default=6.0,
        help="Minimum initial COM-COM distance (A) enforced before minimization.",
    )
    parser.add_argument(
        "--box-size",
        type=float,
        default=None,
        help="Override periodic cubic box side length in Angstrom (default: auto from initial geometry).",
    )
    parser.add_argument("--ps", type=float, default=1.0, help="Simulation length (ps)")
    parser.add_argument("--dt-fs", type=float, default=0.25)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--traj-every", type=int, default=1)
    parser.add_argument(
        "--traj-chunk-frames",
        type=int,
        default=0,
        help="Split trajectory into multiple files with at most this many frames each (0 = single file).",
    )
    parser.add_argument("--ml-cutoff", type=float, default=0.1)
    parser.add_argument("--mm-switch-on", type=float, default=DEFAULT_MM_SWITCH_ON)
    parser.add_argument(
        "--include-mm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include JAX MM LJ/Coulomb pairs; --no-include-mm = ML (PhysNet) only.",
    )
    parser.add_argument("--mm-cutoff", type=float, default=DEFAULT_MM_SWITCH_WIDTH)
    parser.add_argument("--pre-min-fmax", type=float, default=0.1)
    parser.add_argument("--pre-min-steps", type=int, default=50)
    parser.add_argument(
        "--zero-force-threshold",
        type=float,
        default=1e-8,
        help="Treat max|F| below this as near-zero in force sanity checks.",
    )
    parser.add_argument(
        "--high-energy-threshold",
        type=float,
        default=1e3,
        help="Treat |E| above this as suspicious when forces are near-zero.",
    )
    parser.add_argument(
        "--max-fmax-after-min",
        type=float,
        default=2.0,
        help="Abort before MD if post-minimization Fmax exceeds this threshold (eV/A).",
    )
    parser.add_argument("--bfgs-maxstep", type=float, default=0.05, help="ASE BFGS maxstep (A)")
    parser.add_argument(
        "--charmm-pre-minimize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run CHARMM SD/ABNR before ASE BFGS (default: enabled; use --no-charmm-pre-minimize to skip).",
    )
    parser.add_argument("--charmm-sd-steps", type=int, default=25, help="CHARMM SD steps before ABNR.")
    parser.add_argument("--charmm-abnr-steps", type=int, default=100, help="CHARMM ABNR steps before ASE BFGS.")
    parser.add_argument("--charmm-tolenr", type=float, default=1e-3, help="CHARMM minimization energy tolerance.")
    parser.add_argument("--charmm-tolgrd", type=float, default=1e-3, help="CHARMM minimization gradient tolerance.")
    parser.add_argument(
        "--charmm-nbxmod",
        type=int,
        default=5,
        help="CHARMM NBXMOD for SD/ABNR minimization (default 5, matching CGenFF).",
    )
    parser.add_argument(
        "--min-intermonomer-atom-distance",
        type=float,
        default=0.1,
        help="Abort if atoms from different monomers get closer than this distance in Angstrom (<=0 disables).",
    )
    parser.add_argument(
        "--rescue-minimize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If post-BFGS fmax is too high, run rescue minimization (CHARMM + FIRE) before aborting.",
    )
    parser.add_argument("--rescue-charmm-sd-steps", type=int, default=100, help="Rescue CHARMM SD steps.")
    parser.add_argument("--rescue-charmm-abnr-steps", type=int, default=300, help="Rescue CHARMM ABNR steps.")
    parser.add_argument("--rescue-fire-steps", type=int, default=300, help="Rescue ASE FIRE steps.")
    parser.add_argument("--rescue-fire-fmax", type=float, default=0.1, help="Rescue ASE FIRE fmax target (eV/A).")
    parser.add_argument("--rescue-fire-maxstep", type=float, default=0.02, help="Rescue ASE FIRE maxstep (A).")
    parser.add_argument("--nvt-temp-K", type=float, default=300.0)
    parser.add_argument("--nve-temp-K", type=float, default=10.0)
    parser.add_argument("--langevin-friction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--packmol",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Pack --composition with Packmol (default cube inside --box-size). "
            "Use --no-packmol for legacy grid placement."
        ),
    )
    parser.add_argument(
        "--packmol-placement",
        choices=("cube", "sphere"),
        default=None,
        help="Packmol constraint: cube (default) or sphere (--packmol-radius).",
    )
    parser.add_argument(
        "--packmol-sphere",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="packmol_sphere",
        help="Legacy alias for --packmol-placement sphere.",
    )
    parser.add_argument(
        "--packmol-radius",
        type=float,
        default=None,
        metavar="Å",
        dest="packmol_radius",
        help="Packmol inside-sphere radius in Angstrom (independent of --flat-bottom-radius).",
    )
    parser.add_argument(
        "--packmol-center",
        type=float,
        nargs=3,
        metavar=("CX", "CY", "CZ"),
        default=None,
        help="Packmol sphere center in Angstrom (default: 0 0 0; vacuum COM is re-centered later).",
    )
    parser.add_argument(
        "--packmol-tolerance",
        type=float,
        default=2.0,
        help="Packmol distance tolerance in Angstrom when using --packmol-sphere (default: 2.0).",
    )
    from mmml.interfaces.pyxtal_placement import add_pyxtal_cluster_args

    add_pyxtal_cluster_args(parser)
    from mmml.interfaces.pycharmmInterface.mlpot.box_sizing import add_box_sizing_args

    add_box_sizing_args(parser)
    parser.add_argument(
        "--flat-bottom-radius",
        type=float,
        default=None,
        metavar="Å",
        dest="flat_bottom_radius",
        help=(
            "Flat-bottom restraint on system COM: V=0 for |d|<=R, else V=k*(|d|-R)^2. "
            "Independent of --packmol-radius. If only this is set with --composition, Packmol "
            "still uses it as sphere radius (legacy); prefer --packmol-radius for packing."
        ),
    )
    parser.add_argument(
        "--flat-bottom-k",
        type=float,
        default=1.0,
        metavar="eV/Å²",
        dest="flat_bottom_k",
        help="Force constant k when COM is outside --flat-bottom-radius (default: 1.0).",
    )
    parser.add_argument(
        "--flat-bottom-mode",
        choices=["system", "monomer"],
        default="system",
        dest="flat_bottom_mode",
        help=(
            "system: harmonic on cluster COM; monomer: sum over monomer COMs (same R, k). "
            "Vacuum: anchor at origin; PBC: MIC to box center."
        ),
    )
    parser.add_argument(
        "--min-com-restraint-distance",
        type=float,
        default=None,
        metavar="Å",
        help=(
            "Pairwise inter-monomer COM lower wall. Adds 0.5*k*(r_min-r)^2 "
            "when COM distance r < r_min (default: disabled)."
        ),
    )
    parser.add_argument(
        "--min-com-restraint-k",
        type=float,
        default=1.0,
        metavar="eV/Å²",
        help="Force constant for --min-com-restraint-distance (default: 1.0).",
    )
    parser.add_argument("--verbose-calc", action="store_true")
    parser.add_argument(
        "--allow-ml-on-mixed",
        action="store_true",
        help="Deprecated no-op: ML terms are enabled for mixed compositions by default.",
    )
    parser.add_argument(
        "--mm-only-on-mixed",
        action="store_true",
        help="Disable ML terms for mixed compositions and use the MM-only fallback.",
    )
    parser.add_argument(
        "--jax-md-capacity-multiplier",
        type=float,
        default=1.75,
        help="Initial jax-md neighbor-list capacity multiplier (higher reduces overflow/re-JIT).",
    )
    parser.add_argument(
        "--jax-md-capacity-growth-factor",
        type=float,
        default=1.5,
        help="Capacity multiplier growth factor on overflow retries.",
    )
    parser.add_argument(
        "--jax-md-max-overflow-retries",
        type=int,
        default=4,
        help="Max overflow-triggered jax-md neighbor-list reallocations.",
    )
    parser.add_argument(
        "--jax-md-disable-fallback",
        action="store_true",
        help="Disable fallback to cell-list pair generation after persistent jax-md overflow.",
    )
    parser.add_argument(
        "--jax-md-update-interval",
        type=int,
        default=1,
        help="Update MM neighbor pairs every N calculator calls (reuse cached pairs in between).",
    )
    parser.add_argument(
        "--jax-md-skin-distance",
        type=float,
        default=DEFAULT_JAX_MD_SKIN_DISTANCE_A,
        help=(
            "Reuse cached MM neighbor pairs while max displacement since last update is below "
            f"this (Å). Default {DEFAULT_JAX_MD_SKIN_DISTANCE_A}: safe with jax-md dr_threshold=0.5 Å; "
            "use 0 only for debugging (rebuild every step, much slower)."
        ),
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=20_000,
        help="Upper bound for MM pair slots (lower this to reduce XLA/GPU memory pressure).",
    )
    parser.add_argument(
        "--ml-batch-size",
        type=int,
        default=None,
        metavar="N",
        help="Chunk PhysNet monomer/dimer batches (auto: 256 on GPU / 64 on CPU for n>=40)."
    )
    parser.add_argument(
        "--ml-max-active-dimers",
        type=int,
        default=None,
        metavar="N",
        help="Sparse ML dimer slot cap (PBC default max(1000, 6*n_monomers))."
    )
    from mmml.interfaces.pycharmmInterface.ml_dtypes import add_ml_compute_dtype_args
    add_ml_compute_dtype_args(parser)
    parser.add_argument("--all", action="store_true", help="Run all 6 combinations")
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Single run key, e.g. pbc_nve, vac_langevin",
    )
    parser.add_argument(
        "--skip-jit-warmup",
        action="store_true",
        help=(
            "Skip generic XLA GPU compile and pre-BFGS hybrid MMML energy/force warmup "
            "(may log XLA cuda_timer delay-kernel warnings on first GPU compile)."
        ),
    )
    parser.add_argument(
        "--quiet-bfgs",
        action="store_true",
        help="Hide ASE BFGS per-step output (default: print steps to stdout; large 10-mers can spend hours here before MD).",
    )
    parser.add_argument(
        "--handoff-pre-minimize",
        action="store_true",
        help="Run pre-minimization even when continuing from a handoff.",
    )
    parser.add_argument(
        "--continue-velocities",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use velocities from handoff when present (else re-thermalize).",
    )
    parser.add_argument(
        "--handoff-quality-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Evaluate initial MMML |F| on handoff and optionally pre-minimize.",
    )
    parser.add_argument(
        "--handoff-quality-fmax-eVA",
        type=float,
        default=1.0,
        help="|F| threshold (eV/Å) for --handoff-quality-gate.",
    )
    parser.add_argument(
        "--handoff-quality-action",
        choices=("minimize", "warn", "error"),
        default="minimize",
        help="Action when handoff quality gate threshold is exceeded.",
    )
    parser.add_argument(
        "--handoff-velocity-remove-drift",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove net P/L from handoff velocities before MD.",
    )
    parser.add_argument(
        "--handoff-require-cell",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require periodic cell in handoff for PBC continuation.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce console output.",
    )
    args = parser.parse_args(argv)
    if args.box_size is not None and args.box_size <= 0:
        raise ValueError("--box-size must be positive")

    t_suite0 = _tmark()
    out_dir = (Path.cwd() / args.output_dir.expanduser()).absolute()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.checkpoint is None:
        base_ckpt_dir, _ = resolve_checkpoint_paths(None)
    else:
        ckpt = args.checkpoint.expanduser().resolve()
        base_ckpt_dir, _ = resolve_checkpoint_paths(ckpt)

    timing_log: list[str] = []
    t_c0 = _tmark()

    from mmml.cli.run.md_handoff import (
        apply_handoff_geometry_to_atoms,
        ensure_psf_for_handoff_cluster,
        get_handoff_in,
        handoff_from_atoms,
        handoff_skip_pre_min,
        resolve_handoff_box,
        set_handoff_out,
    )

    handoff_in = get_handoff_in()
    skip_pre_min = handoff_skip_pre_min(
        handoff_in, handoff_pre_minimize=bool(getattr(args, "handoff_pre_minimize", False))
    )

    z, r0, atoms_per_list, residue_labels, composition_summary = resolve_cluster_geometry(
        args,
        handoff_in,
    )
    n_molecules = len(atoms_per_list)
    cluster_build_s = _tmark() - t_c0
    _tlog(f"cluster_build: {cluster_build_s:.3f} s", timing_log)
    n_atoms = len(z)
    monomer_offsets = np.zeros(n_molecules + 1, dtype=int)
    monomer_offsets[1:] = np.cumsum(np.asarray(atoms_per_list, dtype=int))
    if handoff_in is not None:
        if args.composition:
            handoff_composition = _parse_composition(args.composition)
        elif composition_summary:
            handoff_composition = [
                (str(res), int(cnt)) for res, cnt in composition_summary.items()
            ]
        else:
            handoff_composition = [(residue_labels[0], n_molecules)]
        ensure_psf_for_handoff_cluster(
            composition=handoff_composition,
            atomic_numbers=z,
            atoms_per_list=atoms_per_list,
            residue_labels=residue_labels,
            positions=r0,
            quiet=bool(getattr(args, "quiet", False)),
        )
    psf_charge_summary = _validate_psf_charges(
        monomer_offsets=monomer_offsets,
        residue_labels=residue_labels,
        total_atoms=n_atoms,
        log_lines=timing_log,
    )
    if not resolve_cluster_packmol_sphere(args) and handoff_in is None:
        r0 = _randomize_monomer_com_positions(
            r0,
            monomer_offsets,
            spacing=args.spacing,
            min_com_distance=max(float(args.spacing), float(args.min_com_start_distance)),
            seed=args.seed,
        )
        r0 = _enforce_min_com_separation(
            r0,
            monomer_offsets=monomer_offsets,
            min_com_distance=args.min_com_start_distance,
        )
    initial_atoms = Atoms(numbers=z, positions=r0)
    if handoff_in is None:
        _check_or_charmm_overlap_rescue(
            initial_atoms,
            monomer_offsets,
            min_distance=args.min_intermonomer_atom_distance,
            context="initial placement",
            nstep_sd=args.charmm_sd_steps,
            nstep_abnr=args.charmm_abnr_steps,
            tolenr=args.charmm_tolenr,
            tolgrd=args.charmm_tolgrd,
            nbxmod=args.charmm_nbxmod,
        )
        r0 = initial_atoms.get_positions()

    from mmml.interfaces.pycharmmInterface.mlpot.box_sizing import resolve_suite_auto_box_side

    auto_L, _auto_src = resolve_suite_auto_box_side(args, r0, ml_cutoff=float(args.ml_cutoff))
    L_resolved, box_source, box_warnings = resolve_handoff_box(
        handoff_in,
        yaml_box_size=args.box_size,
        free_space=False,
        auto_box_from_geometry=auto_L,
        require_cell=bool(getattr(args, "handoff_require_cell", False)),
    )
    L = float(L_resolved) if L_resolved is not None else auto_L
    for msg in box_warnings:
        _tlog(f"Handoff box: {msg}", timing_log)
    r_pbc = r0 - r0.mean(axis=0) + 0.5 * L
    from mmml.interfaces.pycharmmInterface.mlpot.mc_density import (
        apply_mc_density_equalization,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
        cubic_box_length_from_geometry,
    )

    r_pbc, L_after_mc, mc_density_summary = apply_mc_density_equalization(
        args,
        r_pbc,
        atoms_per_list=atoms_per_list,
        composition=composition_summary,
        box_side_A=L,
        use_pbc=True,
        handoff_present=handoff_in is not None,
        min_intermonomer_distance_A=float(args.min_intermonomer_atom_distance),
        min_box_side_A=cubic_box_length_from_geometry(r_pbc, ml_cutoff=float(args.ml_cutoff)),
    )
    if L_after_mc is not None:
        L = float(L_after_mc)
    if mc_density_summary.ran:
        _tlog(
            "mc_density_equalize: "
            f"L {mc_density_summary.initial_box_A:.3f} -> {mc_density_summary.final_box_A:.3f} Å; "
            f"rho {mc_density_summary.initial_density_g_cm3:.4f} -> "
            f"{mc_density_summary.final_density_g_cm3:.4f} g/cm^3",
            timing_log,
        )
    elif mc_density_summary.enabled:
        _tlog(f"mc_density_equalize: skipped ({mc_density_summary.reason})", timing_log)

    nsteps = int(round(args.ps * 1000.0 / args.dt_fs))
    if nsteps < 1:
        raise ValueError("nsteps < 1")

    suite_summary: dict = {
        "system": {
            "residue": "mixed" if args.composition else "MEOH",
            "composition": composition_summary,
            "residue_labels": residue_labels,
            "psf_charges": psf_charge_summary,
            "n_molecules": n_molecules,
            "n_atoms": n_atoms,
            "spacing_A": args.spacing,
            "placement": "random_3d",
            "placement_seed": int(args.seed),
        },
        "box_A": L,
        "mc_density_equalization": mc_density_summary.to_dict(),
        "md": {
            "ps": args.ps,
            "dt_fs": args.dt_fs,
            "nsteps": nsteps,
            "log_every": args.log_every,
            "traj_every": args.traj_every,
        },
        "cutoffs_A": {
            "ml_cutoff": args.ml_cutoff,
            "mm_switch_on": args.mm_switch_on,
            "mm_cutoff": args.mm_cutoff,
        },
        "charmm_minimization": {
            "nbxmod": int(args.charmm_nbxmod),
            "sd_steps": int(args.charmm_sd_steps),
            "abnr_steps": int(args.charmm_abnr_steps),
        },
        "runs": {},
        "timing": {
            "cluster_build_s": cluster_build_s,
            "skip_jit_warmup": bool(args.skip_jit_warmup),
            "runs": {},
        },
    }
    use_ml_terms = True
    use_ml_dimer_terms = True
    if args.composition and args.mm_only_on_mixed:
        use_ml_terms = False
        use_ml_dimer_terms = False
        _tlog(
            "mixed composition requested; using explicit MM-only fallback.",
            timing_log,
        )

    def do_one(key: str, use_pbc: bool, mode: str) -> None:
        r = r_pbc.copy() if use_pbc else r0.copy()
        atoms = Atoms(numbers=z, positions=r)
        if use_pbc:
            atoms.set_cell([L, L, L])
            atoms.set_pbc(True)
        else:
            atoms.set_cell(None)
            atoms.set_pbc(False)
        if handoff_in is not None:
            apply_handoff_geometry_to_atoms(
                atoms, handoff_in, monomer_offsets=monomer_offsets
            )

        run_timings: dict[str, float] = {}
        if handoff_in is None or not skip_pre_min:
            _check_or_charmm_overlap_rescue(
                atoms,
                monomer_offsets,
                min_distance=args.min_intermonomer_atom_distance,
                context=f"{key}: before minimization",
                nstep_sd=args.charmm_sd_steps,
                nstep_abnr=args.charmm_abnr_steps,
                tolenr=args.charmm_tolenr,
                tolgrd=args.charmm_tolgrd,
                nbxmod=args.charmm_nbxmod,
                timings=run_timings,
            )
        elif not getattr(args, "quiet", False):
            _tlog(
                f"{key}: skipping overlap check (continuing from equilibrated handoff)",
                timing_log,
            )
        if args.charmm_pre_minimize and not skip_pre_min:
            _tlog(
                f"{key}: CHARMM minimization starting (SD={args.charmm_sd_steps}, "
                f"ABNR={args.charmm_abnr_steps}, tolenr={args.charmm_tolenr:g}, tolgrd={args.charmm_tolgrd:g})",
                timing_log,
            )
            _run_charmm_minimize(
                atoms,
                nstep_sd=args.charmm_sd_steps,
                nstep_abnr=args.charmm_abnr_steps,
                tolenr=args.charmm_tolenr,
                tolgrd=args.charmm_tolgrd,
                nbxmod=args.charmm_nbxmod,
                timings=run_timings,
            )
            _check_or_charmm_overlap_rescue(
                atoms,
                monomer_offsets,
                min_distance=args.min_intermonomer_atom_distance,
                context=f"{key}: after CHARMM minimization",
                nstep_sd=args.charmm_sd_steps,
                nstep_abnr=args.charmm_abnr_steps,
                tolenr=args.charmm_tolenr,
                tolgrd=args.charmm_tolgrd,
                nbxmod=args.charmm_nbxmod,
                timings=run_timings,
            )
            _tlog(
                f"{key}: CHARMM minimization {run_timings.get('charmm_min_wall_s', 0.0):.3f} s",
                timing_log,
            )
        calc = _factory_mmml(
            z=z,
            r=atoms.get_positions(),
            n_mol=n_molecules,
            atoms_per=atoms_per_list,
            base_ckpt_dir=base_ckpt_dir,
            ml_cut=args.ml_cutoff,
            mm_sw=args.mm_switch_on,
            mm_cut=args.mm_cutoff,
            cell_scalar=L if use_pbc else None,
            verbose=args.verbose_calc,
            jax_md_capacity_multiplier=args.jax_md_capacity_multiplier,
            jax_md_capacity_growth_factor=args.jax_md_capacity_growth_factor,
            jax_md_max_overflow_retries=args.jax_md_max_overflow_retries,
            jax_md_overflow_fallback_to_cell_list=not args.jax_md_disable_fallback,
            jax_md_update_interval=args.jax_md_update_interval,
            jax_md_skin_distance=args.jax_md_skin_distance,
            max_pairs=args.max_pairs,
            do_ml=use_ml_terms,
            do_ml_dimer=use_ml_dimer_terms,
            do_mm=bool(getattr(args, "include_mm", True)),
            timings=run_timings,
            flat_bottom_radius=args.flat_bottom_radius,
            flat_bottom_force_const=args.flat_bottom_k,
            flat_bottom_mode=args.flat_bottom_mode,
            min_com_restraint_distance=args.min_com_restraint_distance,
            min_com_restraint_force_const=args.min_com_restraint_k,
            defer_xla_gpu_warmup=bool(args.skip_jit_warmup),
            ml_batch_size=getattr(args, "ml_batch_size", None),
            ml_max_active_dimers=getattr(args, "ml_max_active_dimers", None),
            ml_compute_dtype=getattr(args, "ml_compute_dtype", None),
        )
        atoms.calc = calc
        _save_cutoff_plot(
            out_dir=out_dir,
            ml_cutoff=args.ml_cutoff,
            mm_switch_on=args.mm_switch_on,
            mm_cutoff=args.mm_cutoff,
            log_lines=timing_log,
        )
        _tlog(
            f"{key}: mmml setup_calculator {run_timings.get('mmml_setup_calculator_s', 0):.3f} s, "
            f"factory_call {run_timings.get('mmml_factory_call_s', 0):.3f} s",
            timing_log,
        )

        if not args.skip_jit_warmup:
            t_w = _tmark()
            warmup_ase_mmml_energy_forces(atoms, include_forces=True)
            run_timings["jit_warmup_first_potential_s"] = _tmark() - t_w
            _tlog(
                f"{key}: JIT warmup (first potential energy) {run_timings['jit_warmup_first_potential_s']:.3f} s",
                timing_log,
            )
        else:
            run_timings["jit_warmup_first_potential_s"] = 0.0

        # Sanity check: large energy with near-zero forces indicates a broken force path.
        e0 = float(atoms.get_potential_energy())
        f0 = np.asarray(atoms.get_forces())
        f0_max = float(np.abs(f0).max())
        run_timings["pre_bfgs_energy_eV"] = e0
        run_timings["pre_bfgs_fmax_eVA"] = f0_max
        if abs(e0) > float(args.high_energy_threshold) and f0_max <= float(args.zero_force_threshold):
            raise RuntimeError(
                f"{key}: force sanity check failed before BFGS: "
                f"E={e0:.6f} eV, max|F|={f0_max:.3e} eV/A. "
                "This usually means forces are not being propagated for this setup "
                "(e.g., mixed composition path issue)."
            )

        run_local_skip_pre_min = skip_pre_min
        if (
            handoff_in is not None
            and not getattr(args, "handoff_pre_minimize", False)
            and getattr(args, "handoff_quality_gate", False)
        ):
            threshold = float(getattr(args, "handoff_quality_fmax_eVA", 1.0))
            if f0_max > threshold:
                action = str(getattr(args, "handoff_quality_action", "minimize")).lower()
                msg = (
                    f"{key}: handoff quality gate: max|F|={f0_max:.4f} eV/Å "
                    f"> threshold {threshold:.4f} eV/Å"
                )
                if action == "error":
                    raise RuntimeError(msg)
                if action == "warn":
                    _tlog(f"WARNING: {msg}", timing_log)
                elif action == "minimize":
                    _tlog(f"{msg}; enabling BFGS pre-min.", timing_log)
                    run_local_skip_pre_min = False

        t_b = _tmark()
        fmin = float(np.abs(atoms.get_forces()).max()) if run_local_skip_pre_min else None
        if not run_local_skip_pre_min:
            _tlog(
                f"{key}: BFGS starting (max {args.pre_min_steps} steps, fmax={args.pre_min_fmax}; "
                "this is pre-MD minimization, not dynamics yet)",
                timing_log,
            )
            min_traj_path = out_dir / f"{key}_min.traj"
            bfgs_log = None if args.quiet_bfgs else "-"
            opt = BFGS(atoms, logfile=bfgs_log, trajectory=str(min_traj_path), maxstep=args.bfgs_maxstep)
            opt.attach(
                lambda: _check_or_charmm_overlap_rescue(
                    atoms,
                    monomer_offsets,
                    min_distance=args.min_intermonomer_atom_distance,
                    context=f"{key}: ASE BFGS",
                    nstep_sd=args.charmm_sd_steps,
                    nstep_abnr=args.charmm_abnr_steps,
                    tolenr=args.charmm_tolenr,
                    tolgrd=args.charmm_tolgrd,
                    nbxmod=args.charmm_nbxmod,
                    timings=run_timings,
                ),
                interval=1,
            )
            opt.run(fmax=args.pre_min_fmax, steps=args.pre_min_steps)
            run_timings["bfgs_wall_s"] = _tmark() - t_b
            run_timings["bfgs_traj"] = str(min_traj_path.relative_to(out_dir))
            fmin = float(np.abs(atoms.get_forces()).max())
            n_bfgs = int(opt.get_number_of_steps())
            run_timings["bfgs_iterations"] = float(n_bfgs)
            _tlog(
                f"{key}: BFGS {run_timings['bfgs_wall_s']:.3f} s ({n_bfgs} iters)",
                timing_log,
            )
            _check_or_charmm_overlap_rescue(
                atoms,
                monomer_offsets,
                min_distance=args.min_intermonomer_atom_distance,
                context=f"{key}: after ASE BFGS",
                nstep_sd=args.charmm_sd_steps,
                nstep_abnr=args.charmm_abnr_steps,
                tolenr=args.charmm_tolenr,
                tolgrd=args.charmm_tolgrd,
                nbxmod=args.charmm_nbxmod,
                timings=run_timings,
            )
        if not run_local_skip_pre_min and fmin is not None and fmin > args.pre_min_fmax:
            if args.rescue_minimize:
                _tlog(
                    f"{key}: rescue minimization triggered (fmax={fmin:.6f} > {args.pre_min_fmax:.6f})",
                    timing_log,
                )
                _run_charmm_minimize(
                    atoms,
                    nstep_sd=args.rescue_charmm_sd_steps,
                    nstep_abnr=args.rescue_charmm_abnr_steps,
                    tolenr=args.charmm_tolenr,
                    tolgrd=args.charmm_tolgrd,
                    nbxmod=args.charmm_nbxmod,
                    timings=run_timings,
                )
                rescue_traj_path = out_dir / f"{key}_rescue_fire.traj"
                t_fire0 = _tmark()
                fire = FIRE(
                    atoms,
                    logfile=None if args.quiet_bfgs else "-",
                    trajectory=str(rescue_traj_path),
                    maxstep=args.rescue_fire_maxstep,
                )
                fire.attach(
                    lambda: _check_or_charmm_overlap_rescue(
                        atoms,
                        monomer_offsets,
                        min_distance=args.min_intermonomer_atom_distance,
                        context=f"{key}: ASE FIRE rescue",
                        nstep_sd=args.rescue_charmm_sd_steps,
                        nstep_abnr=args.rescue_charmm_abnr_steps,
                        tolenr=args.charmm_tolenr,
                        tolgrd=args.charmm_tolgrd,
                        nbxmod=args.charmm_nbxmod,
                        timings=run_timings,
                    ),
                    interval=1,
                )
                fire.run(fmax=args.rescue_fire_fmax, steps=args.rescue_fire_steps)
                run_timings["rescue_fire_wall_s"] = _tmark() - t_fire0
                run_timings["rescue_fire_traj"] = str(rescue_traj_path.relative_to(out_dir))
                fmin = float(np.abs(atoms.get_forces()).max())
                _tlog(f"{key}: rescue minimization done, fmax={fmin:.6f} eV/A", timing_log)
                _check_or_charmm_overlap_rescue(
                    atoms,
                    monomer_offsets,
                    min_distance=args.min_intermonomer_atom_distance,
                    context=f"{key}: after ASE FIRE rescue",
                    nstep_sd=args.rescue_charmm_sd_steps,
                    nstep_abnr=args.rescue_charmm_abnr_steps,
                    tolenr=args.charmm_tolenr,
                    tolgrd=args.charmm_tolgrd,
                    nbxmod=args.charmm_nbxmod,
                    timings=run_timings,
                )
            if fmin > args.max_fmax_after_min:
                raise RuntimeError(
                    f"{key}: post-minimization fmax={fmin:.6f} eV/A exceeds "
                    f"--max-fmax-after-min={args.max_fmax_after_min:.6f} even after rescue. "
                    "Increase minimization steps, tighten cutoffs, and/or reduce initial temperature."
                )
        elif run_local_skip_pre_min:
            fmin = float(np.abs(atoms.get_forces()).max())

        res = run_md(
            name=key,
            atoms=atoms,
            mode=mode,
            dt_fs=args.dt_fs,
            nsteps=nsteps,
            log_every=args.log_every,
            traj_every=args.traj_every,
            traj_chunk_frames=args.traj_chunk_frames,
            out_dir=out_dir,
            nvt_temp_K=args.nvt_temp_K,
            nve_temp_K=args.nve_temp_K,
            langevin_friction=args.langevin_friction,
            seed=args.seed,
            monomer_offsets=monomer_offsets,
            min_intermonomer_atom_distance=args.min_intermonomer_atom_distance,
            overlap_rescue_charmm_sd_steps=args.rescue_charmm_sd_steps,
            overlap_rescue_charmm_abnr_steps=args.rescue_charmm_abnr_steps,
            charmm_tolenr=args.charmm_tolenr,
            charmm_tolgrd=args.charmm_tolgrd,
            charmm_nbxmod=args.charmm_nbxmod,
            path_prefix=out_dir,
            timings=run_timings,
            log_lines=timing_log,
        )
        res["pbc"] = use_pbc
        res["box_A"] = L if use_pbc else None
        res["fmax_after_min_eVA"] = fmin
        if hasattr(calc, "get_mm_pair_update_stats"):
            stats = calc.get_mm_pair_update_stats()
            if stats:
                res["mm_pair_update_stats"] = stats
        suite_summary["runs"][key] = res
        suite_summary["timing"]["runs"][key] = dict(run_timings)
        set_handoff_out(
            handoff_from_atoms(
                atoms,
                temperature_K=float(args.nvt_temp_K if mode.startswith("nvt") else args.nve_temp_K),
                metadata={"backend": "ase", "mode": mode, "pbc": use_pbc},
            )
        )
        _tlog(
            f"{key}: MD integrator {run_timings.get('md_integrator_loop_s', 0):.3f} s "
            f"({run_timings.get('md_per_step_mean_ms', 0):.4f} ms/step mean)",
            timing_log,
        )

    if args.all:
        do_one("pbc_nve", True, "nve")
        do_one("pbc_nvt_nhc", True, "nvt_nhc")
        do_one("pbc_nvt_langevin", True, "nvt_langevin")
        do_one("vac_nve", False, "nve")
        do_one("vac_nvt_nhc", False, "nvt_nhc")
        do_one("vac_nvt_langevin", False, "nvt_langevin")
    elif args.only:
        mapping = {
            "pbc_nve": (True, "nve"),
            "pbc_nvt_nhc": (True, "nvt_nhc"),
            "pbc_nvt_langevin": (True, "nvt_langevin"),
            "vac_nve": (False, "nve"),
            "vac_nvt_nhc": (False, "nvt_nhc"),
            "vac_nvt_langevin": (False, "nvt_langevin"),
        }
        if args.only not in mapping:
            raise SystemExit(f"--only must be one of {sorted(mapping)}")
        use_pbc, mode = mapping[args.only]
        do_one(args.only, use_pbc, mode)
    else:
        raise SystemExit("Pass --all or --only <key>")

    suite_summary["timing"]["suite_total_wall_s"] = _tmark() - t_suite0
    timing_payload = {
        "timing_log": timing_log,
        "timing": suite_summary["timing"],
    }
    (out_dir / "suite_timing.json").write_text(json.dumps(timing_payload, indent=2))
    (out_dir / "suite_summary.json").write_text(json.dumps(suite_summary, indent=2))
    (out_dir / "timing_log.txt").write_text("\n".join(timing_log) + "\n", encoding="utf-8")
    print(json.dumps(suite_summary["runs"], indent=2))
    print(f"Wrote {out_dir / 'suite_summary.json'}")
    print(f"Wrote {out_dir / 'suite_timing.json'} and {out_dir / 'timing_log.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
