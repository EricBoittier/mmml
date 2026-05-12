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
import sys
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
from mmml.interfaces.pycharmmInterface.mmml_calculator import CutoffParameters, setup_calculator
from mmml.utils.geometry_checks import assert_no_intermonomer_atom_overlap
import pycharmm.param as param
import pycharmm.psf as psf
import pycharmm.read as read
import pycharmm.settings as settings
import pycharmm.coor as coor
import pycharmm.generate as gen
import pycharmm.ic as ic
import pycharmm.minimize as charmm_minimize
from mmml.interfaces.pycharmmInterface.utils import get_Z_from_psf

pyci.read = read
pyci.settings = settings
pyci.psf = psf

reset_block()
reset_block_no_internal()

_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))
import test_orbax_checkpoint_cluster as _toc  # noqa: E402

_build_psf_ordered_cluster = _toc._build_psf_ordered_cluster


def _has_resolved_geometry(coords: np.ndarray, min_span: float = 1.0e-4) -> bool:
    """Return True if a residue has more than origin/identical coordinates."""
    if coords.size == 0 or not np.all(np.isfinite(coords)):
        return False
    return bool(np.max(np.ptp(coords, axis=0)) > min_span)


def _read_cgenff_toppar() -> None:
    read.rtf(pyci.CGENFF_RTF)
    bl = settings.set_bomb_level(-2)
    wl = settings.set_warn_level(-2)
    read.prm(pyci.CGENFF_PRM)
    settings.set_bomb_level(bl)
    settings.set_warn_level(wl)
    pyci.pycharmm.lingo.charmm_script("bomlev 0")


def _reset_pycharmm_system() -> None:
    pyci.pycharmm.lingo.charmm_script("DELETE ATOM SELE ALL END")
    reset_block()
    reset_block_no_internal()
    reset_block()


def _make_res_minimize(nbxmod: int, nstep: int = 1000) -> None:
    """Run the same nonbonded/minimization recipe used by make-res."""
    pyci.pycharmm.NonBondedScript(
        cutnb=18.0,
        ctonnb=13.0,
        ctofnb=17.0,
        eps=1.0,
        cdie=True,
        atom=True,
        vatom=True,
        fswitch=True,
        vfswitch=True,
        nbxmod=nbxmod,
    ).run()
    charmm_minimize.run_abnr(nstep=nstep, tolenr=1e-3, tolgrd=1e-3)


def _generate_residue_with_make_res_recipe(residue: str) -> tuple[np.ndarray, list[str]]:
    """Generate one residue in PyCHARMM using make-res style coordinate relaxation."""
    _reset_pycharmm_system()
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
    return coords, atom_names


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


def _build_cluster_from_composition(
    *,
    composition: list[tuple[str, int]],
    spacing: float,
) -> tuple[np.ndarray, np.ndarray, list[int], list[str]]:
    # First generate each unique residue exactly as make-res does, in isolation.
    residue_geometries: dict[str, tuple[np.ndarray, list[str]]] = {}
    for residue, _count in composition:
        if residue not in residue_geometries:
            residue_geometries[residue] = _generate_residue_with_make_res_recipe(residue)

    # Rebuild the requested mixed system in one segment, then overwrite CHARMM's
    # placeholder IC coordinates with the PyCHARMM-relaxed residue geometries.
    sequence_items: list[str] = []
    for residue, count in composition:
        sequence_items.extend([residue] * int(count))
    sequence = " ".join(sequence_items)
    _reset_pycharmm_system()
    _read_cgenff_toppar()
    read.sequence_string(sequence)
    gen.new_segment(seg_name="CLST", setup_ic=True)
    ic.prm_fill(replace_all=True)
    ic.build()

    positions = coor.get_positions().to_numpy(dtype=float)
    z = np.asarray(get_Z_from_psf(), dtype=int)
    atom_names = np.asarray(psf.get_atype(), dtype=str)
    if len(atom_names) != int(positions.shape[0]):
        raise RuntimeError(f"PSF atom-name count mismatch: {len(atom_names)} vs positions {positions.shape[0]}")
    atoms_per_list: list[int] = []
    ordered_residue_names: list[str] = []
    for residue, count in composition:
        n_atoms_res = int(residue_geometries[residue][0].shape[0])
        for _ in range(int(count)):
            atoms_per_list.append(int(n_atoms_res))
            ordered_residue_names.append(residue)
    expected_atoms = int(np.sum(np.asarray(atoms_per_list, dtype=int)))
    if expected_atoms != int(positions.shape[0]):
        raise RuntimeError(
            f"Composition-derived atom count ({expected_atoms}) does not match built coordinates "
            f"({positions.shape[0]}). Composition={composition}"
        )

    n_molecules = len(atoms_per_list)
    n_side = int(np.ceil(np.sqrt(n_molecules)))
    shifted = positions.copy()
    offsets = np.zeros(n_molecules + 1, dtype=int)
    offsets[1:] = np.cumsum(np.asarray(atoms_per_list, dtype=int))
    for i in range(n_molecules):
        s = int(offsets[i])
        e = int(offsets[i + 1])
        residue = ordered_residue_names[i]
        residue_coords, residue_atom_names = residue_geometries[residue]
        local_atom_names = [str(x) for x in atom_names[s:e]]
        if local_atom_names != residue_atom_names:
            raise RuntimeError(
                f"Atom order mismatch for {residue}: final PSF has {local_atom_names}, "
                f"make-res generation produced {residue_atom_names}"
            )
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
    timings: dict[str, float] | None = None,
):
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
        ml_cutoff_distance=ml_cut,
        mm_switch_on=mm_sw,
        mm_cutoff=mm_cut,
        doML=do_ml,
        doMM=True,
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
    )
    t1 = _tmark()
    cutoff = CutoffParameters(ml_cutoff=ml_cut, mm_switch_on=mm_sw, mm_cutoff=mm_cut)
    calc_result = factory(
        atomic_numbers=z,
        atomic_positions=r,
        n_monomers=n_mol,
        cutoff_params=cutoff,
        doML=do_ml,
        doMM=True,
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
    charges = np.asarray(psf.get_charges(), dtype=float)[:total_atoms]
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
) -> None:
    """Run optional CHARMM minimization and write updated coords back to ASE atoms.

    When ``cubic_box_side_A`` is set, CHARMM crystal / IMAGE PBC is rebuilt for a
    cubic cell of that edge length (matches the MD periodic box for JAX-MD / ASE).
    """
    if nstep_sd <= 0 and nstep_abnr <= 0:
        return
    t0 = _tmark()
    reset_block()
    coor.set_positions(pd.DataFrame(atoms.get_positions(), columns=["x", "y", "z"]))
    try:
        use_pbc_charmm = cubic_box_side_A is not None and float(cubic_box_side_A) > 0.0
        if use_pbc_charmm:
            from mmml.interfaces.pycharmmInterface.pycharmmCommands import pbcset
            from mmml.interfaces.pycharmmInterface.setupBox import _ensure_crystal_image_str

            _ensure_crystal_image_str()
            L = float(cubic_box_side_A)
            pyci.pycharmm.lingo.charmm_script(pbcset.format(SIDELENGTH=L))
        pyci.pycharmm.NonBondedScript(
            cutnb=18.0,
            ctonnb=13.0,
            ctofnb=17.0,
            eps=1.0,
            cdie=True,
            atom=True,
            vatom=True,
            fswitch=True,
            vfswitch=True,
            nbxmod=nbxmod,
        ).run()
        if use_pbc_charmm:
            L = float(cubic_box_side_A)
            pyci.pycharmm.lingo.charmm_script(
                "open read unit 10 card name crystal_image.str\n"
                f"crystal defi cubic {L} {L} {L} 90. 90. 90.\n"
                "CRYSTAL READ UNIT 10 CARD\n"
                "image byres xcen 0.0 ycen 0.0 zcen 0.0 sele all end\n"
            )
        print("CHARMM energy before minimization:")
        pyci.pycharmm_loud()
        pyci.pycharmm.lingo.charmm_script("ENER")
        pyci.safe_energy_show()
        pyci.pycharmm_quiet()
        if nstep_sd > 0:
            charmm_minimize.run_sd(nstep=nstep_sd, tolenr=tolenr, tolgrd=tolgrd)
        if nstep_abnr > 0:
            charmm_minimize.run_abnr(nstep=nstep_abnr, tolenr=tolenr, tolgrd=tolgrd)
        minimized_positions = coor.get_positions().to_numpy(dtype=float)
        try:
            print("CHARMM energy after minimization:")
            pyci.pycharmm_loud()
            pyci.pycharmm.lingo.charmm_script("ENER")
            pyci.safe_energy_show()
        finally:
            coor.set_positions(pd.DataFrame(minimized_positions, columns=["x", "y", "z"]))
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


def main() -> int:
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
    parser.add_argument("--template-pdb", type=Path, default=Path("mmml/generate/sample/pdb/meoh.pdb"))
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
    parser.add_argument("--mm-switch-on", type=float, default=5.5)
    parser.add_argument("--mm-cutoff", type=float, default=2.0)
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
        default=1.25,
        help="Initial jax-md neighbor-list capacity multiplier.",
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
        default=10,
        help="Update MM neighbor pairs every N calculator calls (reuse cached pairs in between).",
    )
    parser.add_argument(
        "--jax-md-skin-distance",
        type=float,
        default=0.2,
        help="Reuse cached MM neighbor pairs while max displacement since last update is below this (A).",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=20_000,
        help="Upper bound for MM pair slots (lower this to reduce XLA/GPU memory pressure).",
    )
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
        help="Do not run an extra energy eval before BFGS (mixes compile time into first BFGS steps).",
    )
    parser.add_argument(
        "--quiet-bfgs",
        action="store_true",
        help="Hide ASE BFGS per-step output (default: print steps to stdout; large 10-mers can spend hours here before MD).",
    )
    args = parser.parse_args()
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
    if args.composition:
        composition = _parse_composition(args.composition)
        z, r0, atoms_per_list, residue_labels = _build_cluster_from_composition(
            composition=composition,
            spacing=args.spacing,
        )
        n_molecules = len(atoms_per_list)
        composition_summary = {res: int(cnt) for res, cnt in composition}
    else:
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
        n_molecules = int(args.n_molecules)
        composition_summary = {"MEOH": n_molecules}
    cluster_build_s = _tmark() - t_c0
    _tlog(f"cluster_build: {cluster_build_s:.3f} s", timing_log)
    n_atoms = len(z)
    monomer_offsets = np.zeros(n_molecules + 1, dtype=int)
    monomer_offsets[1:] = np.cumsum(np.asarray(atoms_per_list, dtype=int))
    psf_charge_summary = _validate_psf_charges(
        monomer_offsets=monomer_offsets,
        residue_labels=residue_labels,
        total_atoms=n_atoms,
        log_lines=timing_log,
    )
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

    L = float(args.box_size) if args.box_size is not None else _cubic_box_length(r0, args.ml_cutoff)
    r_pbc = r0 - r0.mean(axis=0) + 0.5 * L

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

        run_timings: dict[str, float] = {}
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
        if args.charmm_pre_minimize:
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
            timings=run_timings,
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
            _ = float(atoms.get_potential_energy())
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

        t_b = _tmark()
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
        if fmin > args.pre_min_fmax:
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
