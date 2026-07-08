#!/usr/bin/env python3
"""Scan central tri-alanine PHI/PSI and evaluate CHARMM plus ML ASE PES grids."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd
from ase import Atoms
from ase.constraints import FixInternals
from ase.io import write
from ase.io.trajectory import Trajectory
from ase.optimize.fire import FIRE
from ase.calculators.singlepoint import SinglePointCalculator

from mmml.interfaces.calculators.simple_inference import create_calculator_from_checkpoint
from mmml.interfaces.pycharmmInterface import setupRes
from mmml.interfaces.pycharmmInterface.cgenff_bonded_reference import set_charmm_positions
from mmml.interfaces.pycharmmInterface.import_pycharmm import (
    crystal_free_charmm_for_param_append,
    ensure_pycharmm_loaded,
    pycharmm,
    reset_block,
)
from mmml.interfaces.pycharmmInterface.nbonds_config import ic_prm_fill
from mmml.interfaces.pycharmmInterface.charmm_levels import capture_fortran_stdio
from mmml.interfaces.pycharmmInterface.trialanine_water_box import (
    TRIA_RESI_NAME,
    _load_cgenff_with_trialanine,
)
from mmml.interfaces.pycharmmInterface.utils import get_Z_from_psf
from mmml.utils.dcd_writer import save_trajectory_dcd


PEPTIDE_CKPT_PATH = "examples/params_aaa_long_2026-07-04_22-30-27.json"

PHI_CENTRAL = (14, 16, 18, 24)  # C1-N2-CA2-C2
PSI_CENTRAL = (16, 18, 24, 26)  # N2-CA2-C2-N3


def safe_grid_tag(phi_deg: float, psi_deg: float) -> str:
    """Stable filename tag for a PHI/PSI grid point."""
    return f"phi_{phi_deg:+07.2f}_psi_{psi_deg:+07.2f}".replace("+", "p").replace("-", "m").replace(".", "p")


def build_trialanine_peptide_in_charmm() -> tuple[np.ndarray, np.ndarray]:
    """Build PEPT/TRIA in CHARMM and return atomic numbers plus coordinates."""
    import pycharmm.coor as coor
    import pycharmm.generate as generate
    import pycharmm.ic as ic
    import pycharmm.read as read
    import pycharmm.settings as settings

    if not ensure_pycharmm_loaded():
        raise RuntimeError("PyCHARMM is not available; set CHARMM_LIB_DIR/libcharmm first.")

    crystal_free_charmm_for_param_append()
    pycharmm.lingo.charmm_script("DELETE ATOM SELE ALL END")
    reset_block()
    _load_cgenff_with_trialanine()
    settings.set_verbosity(0)

    read.sequence_string(TRIA_RESI_NAME)
    generate.new_segment(seg_name="PEPT", setup_ic=True)
    ic_prm_fill(replace_all=True)
    ic.build()

    positions = coor.get_positions()[["x", "y", "z"]].to_numpy(dtype=float)
    if np.any(np.abs(positions) > 9000.0) or float(np.std(positions)) < 0.05:
        setupRes.generate_coordinates(skip_energy_show=True, validate=True)
        positions = coor.get_positions()[["x", "y", "z"]].to_numpy(dtype=float)

    positions = positions - positions.mean(axis=0)
    set_charmm_positions(positions)
    atomic_numbers = np.asarray(get_Z_from_psf(), dtype=np.int32)
    return atomic_numbers, positions


def get_charmm_bond_pairs() -> list[tuple[int, int]]:
    """Return CHARMM PSF bond pairs as zero-based atom indices."""
    import pycharmm.psf as psf

    ib, jb = psf.get_ib_jb()
    pairs = []
    for i_atom, j_atom in zip(ib, jb):
        pairs.append((int(i_atom) - 1, int(j_atom) - 1))
    return pairs


def topology_excluded_pairs(n_atoms: int, bond_pairs: list[tuple[int, int]], max_bond_separation: int = 2) -> set[tuple[int, int]]:
    """Return atom pairs separated by at most max_bond_separation covalent bonds."""
    adjacency: list[set[int]] = [set() for _ in range(n_atoms)]
    for i_atom, j_atom in bond_pairs:
        if 0 <= i_atom < n_atoms and 0 <= j_atom < n_atoms:
            adjacency[i_atom].add(j_atom)
            adjacency[j_atom].add(i_atom)

    excluded: set[tuple[int, int]] = set()
    for start in range(n_atoms):
        frontier = {start}
        seen = {start}
        for _depth in range(max_bond_separation):
            next_frontier: set[int] = set()
            for atom in frontier:
                next_frontier.update(adjacency[atom])
            next_frontier -= seen
            for atom in next_frontier:
                excluded.add((min(start, atom), max(start, atom)))
            seen.update(next_frontier)
            frontier = next_frontier
    return excluded


def min_nonbonded_distance(positions: np.ndarray, excluded_pairs: set[tuple[int, int]]) -> float:
    """Minimum distance over non-topologically-near atom pairs."""
    pos = np.asarray(positions, dtype=np.float64)
    min_dist = np.inf
    for i_atom in range(pos.shape[0] - 1):
        delta = pos[i_atom + 1 :] - pos[i_atom]
        distances = np.linalg.norm(delta, axis=1)
        for offset, distance in enumerate(distances, start=1):
            j_atom = i_atom + offset
            if (i_atom, j_atom) in excluded_pairs:
                continue
            min_dist = min(min_dist, float(distance))
    return min_dist


def set_phi_psi(atoms: Atoms, phi_deg: float, psi_deg: float) -> Atoms:
    """Return a copy with central PHI/PSI set by ASE rotations."""
    scanned = atoms.copy()
    n_atoms = len(scanned)

    phi_mask = np.zeros(n_atoms, dtype=bool)
    phi_mask[PHI_CENTRAL[2] :] = True
    scanned.set_dihedral(*PHI_CENTRAL, phi_deg, mask=phi_mask)

    psi_mask = np.zeros(n_atoms, dtype=bool)
    psi_mask[PSI_CENTRAL[2] :] = True
    scanned.set_dihedral(*PSI_CENTRAL, psi_deg, mask=psi_mask)
    return scanned


def relax_with_fixed_phi_psi(
    atoms: Atoms,
    calc,
    phi_deg: float,
    psi_deg: float,
    *,
    fmax: float,
    steps: int,
    logfile: Path | None = None,
) -> Atoms:
    """Relax all other coordinates with PHI/PSI fixed."""
    relaxed = atoms.copy()
    relaxed.calc = calc
    relaxed.set_constraint(
        FixInternals(
            dihedrals_deg=[
                [float(phi_deg), list(PHI_CENTRAL)],
                [float(psi_deg), list(PSI_CENTRAL)],
            ]
        )
    )
    opt = FIRE(relaxed, logfile=str(logfile) if logfile is not None else None, maxstep=0.03)
    opt.run(fmax=fmax, steps=steps)
    relaxed.set_constraint()
    return relaxed


def minimize_charmm_with_phi_psi(
    positions: np.ndarray,
    phi_deg: float,
    psi_deg: float,
    *,
    force_kcal: float,
    sd_steps: int,
    abnr_steps: int,
    log_path: Path,
) -> tuple[np.ndarray, float]:
    """Run CHARMM MM minimization with PHI/PSI restrained to target values."""
    import pycharmm.coor as coor
    import pycharmm.energy as energy
    import pycharmm.settings as settings

    def bynum(atom_indices: tuple[int, int, int, int]) -> str:
        return " ".join(str(index + 1) for index in atom_indices)

    set_charmm_positions(np.asarray(positions, dtype=np.float64))
    commands = [
        "CONS CLDH",
        (
            f"CONS DIHE BYNUM {bynum(PHI_CENTRAL)} FORCE {force_kcal:.8g} "
            f"MIN {float(phi_deg):.8g} PERI 0"
        ),
        (
            f"CONS DIHE BYNUM {bynum(PSI_CENTRAL)} FORCE {force_kcal:.8g} "
            f"MIN {float(psi_deg):.8g} PERI 0"
        ),
    ]
    if sd_steps > 0:
        commands.append(f"MINI SD NSTEP {int(sd_steps)}")
    if abnr_steps > 0:
        commands.append(f"MINI ABNR NSTEP {int(abnr_steps)}")
    commands.append("ENER")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    old_prn = settings.set_verbosity(5)
    old_wrn = settings.set_warn_level(5)
    try:
        statuses: list[tuple[str, object]] = []
        with capture_fortran_stdio() as captured_path:
            for command in commands:
                statuses.append((command, pycharmm.lingo.charmm_script(command)))
            captured = Path(captured_path).read_text(errors="replace")
            os.unlink(captured_path)
        minimized = coor.get_positions()[["x", "y", "z"]].to_numpy(dtype=float)
        minimized = minimized - minimized.mean(axis=0)
        set_charmm_positions(minimized)
        e_total = float(energy.get_total())
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"# CHARMM constrained minimization log\n")
            log.write(f"phi_deg={float(phi_deg):.8f} psi_deg={float(psi_deg):.8f}\n")
            log.write(f"force_kcal_per_rad2={float(force_kcal):.8f}\n")
            log.write(f"sd_steps={int(sd_steps)} abnr_steps={int(abnr_steps)}\n")
            log.write("\n# Command status\n")
            for command, status in statuses:
                log.write(f"{status!r} :: {command}\n")
            log.write(f"\n# Final CHARMM energy kcal/mol\n{e_total:.12g}\n")
            log.write("\n# Captured CHARMM output\n")
            log.write(captured)
        return minimized, e_total
    finally:
        settings.set_verbosity(old_prn)
        settings.set_warn_level(old_wrn)
        pycharmm.lingo.charmm_script("CONS CLDH")


def charmm_energy_kcal(positions: np.ndarray) -> float:
    """Evaluate CHARMM MM energy at the supplied coordinates."""
    import pycharmm.energy as energy

    set_charmm_positions(np.asarray(positions, dtype=np.float64))
    pycharmm.lingo.charmm_script("ENER")
    return float(energy.get_total())


def write_charmm_vmd_files(
    out_dir: Path,
    prefix: str,
    atoms: Atoms,
    trajectory_positions: list[np.ndarray],
) -> tuple[Path, Path, Path]:
    """Write CHARMM PSF/CRD plus DCD for VMD from the active CHARMM peptide system."""
    import pycharmm.write as pywrite

    if not trajectory_positions:
        raise ValueError("cannot write VMD files without trajectory frames")

    psf_path = out_dir / f"{prefix}.psf"
    crd_path = out_dir / f"{prefix}.crd"
    dcd_path = out_dir / f"{prefix}.dcd"

    first_positions = np.asarray(trajectory_positions[0], dtype=np.float64)
    set_charmm_positions(first_positions)
    pywrite.psf_card(str(psf_path), title="trialanine PHI/PSI PES topology")
    pywrite.coor_card(str(crd_path), title="trialanine PHI/PSI PES first frame")
    save_trajectory_dcd(dcd_path, np.asarray(trajectory_positions, dtype=np.float32), atoms)
    return psf_path, crd_path, dcd_path


def parse_grid(values: str, count: int | None = None) -> np.ndarray:
    """Parse start:stop:step grid in degrees, including stop."""
    start, stop, step = (float(x) for x in values.split(":"))
    if count is not None:
        if count < 2:
            raise ValueError("grid count must be at least 2")
        return np.linspace(start, stop, int(count), dtype=float)
    if step <= 0:
        raise ValueError("grid step must be positive")
    return np.arange(start, stop + 0.5 * step, step, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=PEPTIDE_CKPT_PATH)
    parser.add_argument("--phi", default="-180:180:30", help="PHI grid as start:stop:step degrees")
    parser.add_argument("--psi", default="-180:180:30", help="PSI grid as start:stop:step degrees")
    parser.add_argument("--phi-count", type=int, default=None, help="Use exactly this many evenly spaced PHI grid points")
    parser.add_argument("--psi-count", type=int, default=None, help="Use exactly this many evenly spaced PSI grid points")
    parser.add_argument("--out", default="artifacts/trialanine_phi_psi_pes")
    parser.add_argument("--relax-ase", action="store_true", help="Compatibility alias; ML minimization is enabled by default when ML is loaded")
    parser.add_argument("--skip-ml", action="store_true", help="Only evaluate CHARMM energies")
    parser.add_argument("--no-mm-minimize", action="store_true", help="Skip per-grid CHARMM constrained minimization")
    parser.add_argument("--mm-sd-steps", type=int, default=100)
    parser.add_argument("--mm-abnr-steps", type=int, default=100)
    parser.add_argument("--mm-dihedral-force", type=float, default=500.0, help="CHARMM CONS DIHE force in kcal/mol/rad^2")
    parser.add_argument("--no-ml-minimize", action="store_true", help="Skip constrained ML minimization after MM minimization")
    parser.add_argument("--relax-steps", type=int, default=200)
    parser.add_argument("--relax-fmax", type=float, default=0.05)
    parser.add_argument("--min-nonbonded-distance", type=float, default=1.4, help="Reject final ML frame below this nonbonded distance in Å")
    parser.add_argument("--max-final-charmm-increase", type=float, default=100.0, help="Reject final ML frame if CHARMM energy rises this many kcal/mol above MM-min")
    parser.add_argument("--no-reject-ml-clashes", action="store_true", help="Keep ML-relaxed frames even when CHARMM detects clashes")
    parser.add_argument("--traj", default="phi_psi_pes.traj", help="ASE trajectory filename under --out")
    parser.add_argument("--no-vmd", action="store_true", help="Do not write CHARMM PSF/CRD/DCD VMD files")
    parser.add_argument("--vmd-prefix", default="trialanine_phi_psi_pes", help="Prefix for PSF/CRD/DCD files")
    parser.add_argument("--write-xyz", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    atomic_numbers, positions = build_trialanine_peptide_in_charmm()
    base_atoms = Atoms(numbers=atomic_numbers, positions=positions)
    bond_pairs = get_charmm_bond_pairs()
    excluded_pairs = topology_excluded_pairs(len(base_atoms), bond_pairs)
    charmm_log_dir = out_dir / "charmm_logs"
    ml_log_dir = out_dir / "ml_logs"
    charmm_log_dir.mkdir(parents=True, exist_ok=True)
    ml_log_dir.mkdir(parents=True, exist_ok=True)

    calc = None if args.skip_ml else create_calculator_from_checkpoint(args.checkpoint)
    phi_grid = parse_grid(args.phi, args.phi_count)
    psi_grid = parse_grid(args.psi, args.psi_count)

    charmm_kcal = np.full((len(phi_grid), len(psi_grid)), np.nan, dtype=float)
    charmm_mm_min_kcal = np.full_like(charmm_kcal, np.nan)
    ml_ev = np.full_like(charmm_kcal, np.nan)
    actual_phi = np.full_like(charmm_kcal, np.nan)
    actual_psi = np.full_like(charmm_kcal, np.nan)
    mm_min_nonbonded_A = np.full_like(charmm_kcal, np.nan)
    final_min_nonbonded_A = np.full_like(charmm_kcal, np.nan)
    ml_rejected = np.zeros_like(charmm_kcal, dtype=bool)
    positions_A = np.full((len(phi_grid), len(psi_grid), len(base_atoms), 3), np.nan, dtype=float)
    ml_forces_eVA = np.full_like(positions_A, np.nan)

    rows: list[dict[str, float]] = []
    trajectory_positions: list[np.ndarray] = []
    traj = Trajectory(out_dir / args.traj, "w")
    for i, phi in enumerate(phi_grid):
        for j, psi in enumerate(psi_grid):
            grid_tag = safe_grid_tag(phi, psi)
            atoms = set_phi_psi(base_atoms, phi, psi)
            if not args.no_mm_minimize:
                mm_log_path = charmm_log_dir / f"{grid_tag}_mm_min.log"
                minimized_positions, e_charmm_mm_min = minimize_charmm_with_phi_psi(
                    atoms.get_positions(),
                    phi,
                    psi,
                    force_kcal=args.mm_dihedral_force,
                    sd_steps=args.mm_sd_steps,
                    abnr_steps=args.mm_abnr_steps,
                    log_path=mm_log_path,
                )
                atoms.set_positions(minimized_positions)
            else:
                mm_log_path = charmm_log_dir / f"{grid_tag}_mm_min_skipped.log"
                e_charmm_mm_min = charmm_energy_kcal(atoms.get_positions())
                mm_log_path.write_text(
                    "CHARMM constrained minimization skipped by --no-mm-minimize\n"
                    f"single_point_charmm_energy_kcal_mol={e_charmm_mm_min:.12g}\n",
                    encoding="utf-8",
                )

            mm_positions = np.asarray(atoms.get_positions(), dtype=np.float64).copy()
            mm_min_dist = min_nonbonded_distance(mm_positions, excluded_pairs)
            mm_min_nonbonded_A[i, j] = mm_min_dist
            final_source = "mm_min"

            if calc is not None and not args.no_ml_minimize:
                atoms = relax_with_fixed_phi_psi(
                    atoms,
                    calc,
                    phi,
                    psi,
                    fmax=args.relax_fmax,
                    steps=args.relax_steps,
                    logfile=ml_log_dir / f"{grid_tag}_ml_fire.log",
                )
                final_source = "ml_min"
            e_ml = np.nan
            forces_ml = np.full((len(atoms), 3), np.nan, dtype=float)
            if calc is not None:
                atoms.calc = calc
                e_ml = float(atoms.get_potential_energy())
                forces_ml = np.asarray(atoms.get_forces(), dtype=float)
            e_charmm = charmm_energy_kcal(atoms.get_positions())
            final_min_dist = min_nonbonded_distance(atoms.get_positions(), excluded_pairs)

            reject_reasons: list[str] = []
            if final_min_dist < float(args.min_nonbonded_distance):
                reject_reasons.append(
                    f"final_nonbonded_distance_A={final_min_dist:.6g}<cutoff={float(args.min_nonbonded_distance):.6g}"
                )
            if e_charmm > e_charmm_mm_min + float(args.max_final_charmm_increase):
                reject_reasons.append(
                    f"final_charmm_energy_increase={e_charmm - e_charmm_mm_min:.6g}>cutoff={float(args.max_final_charmm_increase):.6g}"
                )
            reject_ml = final_source == "ml_min" and not args.no_reject_ml_clashes and bool(reject_reasons)
            if reject_ml:
                ml_rejected[i, j] = True
                atoms.set_positions(mm_positions)
                final_source = "mm_min_reused_after_ml_reject"
                if calc is not None:
                    atoms.calc = calc
                    e_ml = float(atoms.get_potential_energy())
                    forces_ml = np.asarray(atoms.get_forces(), dtype=float)
                e_charmm = charmm_energy_kcal(atoms.get_positions())
                final_min_dist = min_nonbonded_distance(atoms.get_positions(), excluded_pairs)
            reject_reason = ";".join(reject_reasons) if reject_reasons else ""
            mm_min_clashing = mm_min_dist < float(args.min_nonbonded_distance)

            final_min_nonbonded_A[i, j] = final_min_dist
            phi_actual = float(atoms.get_dihedral(*PHI_CENTRAL))
            psi_actual = float(atoms.get_dihedral(*PSI_CENTRAL))

            with mm_log_path.open("a", encoding="utf-8") as log:
                log.write("\n# Geometry diagnostics\n")
                log.write(f"mm_min_nonbonded_distance_A={mm_min_dist:.12g}\n")
                log.write(f"final_source={final_source}\n")
                log.write(f"final_charmm_energy_kcal_mol={e_charmm:.12g}\n")
                log.write(f"final_nonbonded_distance_A={final_min_dist:.12g}\n")
                log.write(f"ml_rejected={bool(reject_ml)}\n")
                log.write(f"ml_reject_reason={reject_reason}\n")
                log.write(f"mm_min_clashing={bool(mm_min_clashing)}\n")

            ml_ev[i, j] = e_ml
            charmm_mm_min_kcal[i, j] = e_charmm_mm_min
            charmm_kcal[i, j] = e_charmm
            actual_phi[i, j] = phi_actual
            actual_psi[i, j] = psi_actual
            positions_A[i, j] = atoms.get_positions()
            ml_forces_eVA[i, j] = forces_ml
            trajectory_positions.append(np.asarray(atoms.get_positions(), dtype=np.float64))
            rows.append(
                {
                    "phi_deg": float(phi),
                    "psi_deg": float(psi),
                    "actual_phi_deg": phi_actual,
                    "actual_psi_deg": psi_actual,
                    "charmm_mm_min_energy_kcal_mol": e_charmm_mm_min,
                    "ml_energy_eV": e_ml,
                    "charmm_energy_kcal_mol": e_charmm,
                    "mm_min_nonbonded_distance_A": mm_min_dist,
                    "final_nonbonded_distance_A": final_min_dist,
                    "ml_rejected": bool(reject_ml),
                    "ml_reject_reason": reject_reason,
                    "mm_min_clashing": bool(mm_min_clashing),
                    "final_source": final_source,
                }
            )

            frame = atoms.copy()
            frame.info["phi_deg"] = float(phi)
            frame.info["psi_deg"] = float(psi)
            frame.info["actual_phi_deg"] = phi_actual
            frame.info["actual_psi_deg"] = psi_actual
            frame.info["charmm_mm_min_energy_kcal_mol"] = e_charmm_mm_min
            frame.info["charmm_energy_kcal_mol"] = e_charmm
            frame.info["mm_min_nonbonded_distance_A"] = mm_min_dist
            frame.info["final_nonbonded_distance_A"] = final_min_dist
            frame.info["ml_rejected"] = bool(reject_ml)
            frame.info["ml_reject_reason"] = reject_reason
            frame.info["mm_min_clashing"] = bool(mm_min_clashing)
            frame.info["final_source"] = final_source
            if calc is not None:
                frame.calc = SinglePointCalculator(frame, energy=e_ml, forces=forces_ml)
            traj.write(frame)

            print(
                f"phi={phi:7.2f} psi={psi:7.2f} "
                f"MMmin={e_charmm_mm_min:14.6f} kcal/mol "
                f"ML={e_ml:14.6f} eV CHARMM(final)={e_charmm:14.6f} kcal/mol "
                f"min_nb={final_min_dist:6.3f} Å source={final_source}"
                f"{' REJECTED_ML' if reject_ml else ''}"
                f"{' MM_MIN_CLASH' if mm_min_clashing else ''}",
                flush=True,
            )

            if args.write_xyz:
                write(out_dir / f"phi_{phi:+07.2f}_psi_{psi:+07.2f}.xyz", atoms)

    traj.close()

    if not args.no_vmd:
        psf_path, crd_path, dcd_path = write_charmm_vmd_files(
            out_dir,
            args.vmd_prefix,
            base_atoms,
            trajectory_positions,
        )
    else:
        psf_path = crd_path = dcd_path = None

    np.savez(
        out_dir / "phi_psi_pes.npz",
        phi_grid_deg=phi_grid,
        psi_grid_deg=psi_grid,
        ml_energy_eV=ml_ev,
        charmm_mm_min_energy_kcal_mol=charmm_mm_min_kcal,
        charmm_energy_kcal_mol=charmm_kcal,
        actual_phi_deg=actual_phi,
        actual_psi_deg=actual_psi,
        mm_min_nonbonded_distance_A=mm_min_nonbonded_A,
        final_nonbonded_distance_A=final_min_nonbonded_A,
        ml_rejected=ml_rejected,
        positions_A=positions_A,
        ml_forces_eVA=ml_forces_eVA,
        phi_atoms=np.asarray(PHI_CENTRAL, dtype=np.int32),
        psi_atoms=np.asarray(PSI_CENTRAL, dtype=np.int32),
    )

    csv_path = out_dir / "phi_psi_pes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    pd.DataFrame(rows).to_json(out_dir / "phi_psi_pes.json", orient="records", indent=2)
    print(f"Wrote {out_dir / 'phi_psi_pes.npz'}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {out_dir / args.traj}")
    if psf_path is not None and crd_path is not None and dcd_path is not None:
        print(f"Wrote {psf_path}")
        print(f"Wrote {crd_path}")
        print(f"Wrote {dcd_path}")
        print(f"VMD: vmd {psf_path} {dcd_path}")


if __name__ == "__main__":
    main()
