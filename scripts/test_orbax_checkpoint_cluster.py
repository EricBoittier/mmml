#!/usr/bin/env python3
"""
Create a CHARMM/CGenFF-compatible 4-residue cluster and test two calculators.

What this script does:
1) Converts an Orbax checkpoint to portable JSON (cross-platform).
2) Builds a PSF/PDB cluster in PSF atom order using CHARMM residue generation.
3) Runs PhysNetJax calculator test.
4) Runs MMML hybrid calculator test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import ase
import numpy as np
import pandas as pd

from mmml.cli.base import resolve_checkpoint_paths
from mmml.interfaces.pycharmmInterface.import_pycharmm import (
    CGENFF_PRM,
    CGENFF_RTF,
    coor,
    pycharmm,
    reset_block,
)
from mmml.interfaces.pycharmmInterface.mmml_calculator import CutoffParameters, setup_calculator
from mmml.interfaces.pycharmmInterface.utils import get_Z_from_psf
from mmml.models.physnetjax.physnetjax.calc.helper_mlp import get_ase_calc
from mmml.models.physnetjax.physnetjax.restart.restart import get_params_model
from mmml.utils.model_checkpoint import orbax_to_json
from mmml.utils.hybrid_optimization import extract_lj_parameters_from_calculator

import pycharmm.psf as psf
import pycharmm.generate as gen
import pycharmm.ic as ic
import pycharmm.read as read
import pycharmm.settings as settings
import pycharmm.write as pywrite


def _latest_epoch_dir(ckpt_root: Path) -> Path:
    if (ckpt_root / "manifest.ocdbt").exists():
        return ckpt_root
    epoch_dirs = [d for d in ckpt_root.iterdir() if d.is_dir() and d.name.startswith("epoch-")]
    if not epoch_dirs:
        raise FileNotFoundError(f"No epoch-* directories found in: {ckpt_root}")
    return max(epoch_dirs, key=lambda d: int(d.name.split("epoch-")[-1]))


def _build_psf_ordered_cluster(residue: str, n_molecules: int, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    residue = residue.upper()
    sequence = " ".join([residue] * n_molecules)

    pycharmm.lingo.charmm_script("DELETE ATOM SELE ALL END")
    reset_block()
    read.rtf(CGENFF_RTF)
    bl = settings.set_bomb_level(-2)
    wl = settings.set_warn_level(-2)
    read.prm(CGENFF_PRM)
    settings.set_bomb_level(bl)
    settings.set_warn_level(wl)
    pycharmm.lingo.charmm_script("bomlev 0")

    read.sequence_string(sequence)
    gen.new_segment(seg_name="CLST", setup_ic=True)
    ic.prm_fill(replace_all=True)
    ic.build()

    pos_df = coor.get_positions()
    positions = pos_df.to_numpy(dtype=float)
    n_atoms = positions.shape[0]
    if n_atoms % n_molecules != 0:
        raise RuntimeError(
            f"Atom count {n_atoms} not divisible by n_molecules={n_molecules}; "
            "cannot form equal same-residue chunks."
        )
    atoms_per_res = n_atoms // n_molecules

    n_side = int(np.ceil(np.sqrt(n_molecules)))
    shifted = positions.copy()
    for i in range(n_molecules):
        start = i * atoms_per_res
        end = (i + 1) * atoms_per_res
        com = shifted[start:end].mean(axis=0)
        shift = np.array([(i % n_side) * spacing, (i // n_side) * spacing, 0.0], dtype=float)
        shifted[start:end] = shifted[start:end] - com + shift

    coor.set_positions(pd.DataFrame(shifted, columns=["x", "y", "z"]))
    z = np.asarray(get_Z_from_psf(), dtype=int)
    return z, shifted


def run(args: argparse.Namespace) -> int:
    ckpt_root = args.checkpoint.expanduser().resolve()
    epoch_dir = _latest_epoch_dir(ckpt_root)
    base_ckpt_dir, _ = resolve_checkpoint_paths(ckpt_root)
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    portable_json = out_dir / "params_portable.json"
    orbax_to_json(epoch_dir, portable_json)

    z, r = _build_psf_ordered_cluster(args.residue, args.n_molecules, args.spacing)
    n_atoms = len(z)
    atoms_per_monomer = n_atoms // args.n_molecules

    psf_path = out_dir / "cluster_4res.psf"
    pdb_path = out_dir / "cluster_4res.pdb"
    xyz_path = out_dir / "cluster_4res.xyz"
    pywrite.psf_card(str(psf_path))
    pywrite.coor_pdb(str(pdb_path))

    atoms = ase.Atoms(numbers=z, positions=r)
    ase.io.write(str(xyz_path), atoms)

    phys_params, phys_model = get_params_model(str(epoch_dir), natoms=n_atoms)
    phys_model.natoms = n_atoms
    phys_calc = get_ase_calc(phys_params, phys_model, atoms)
    atoms_phys = atoms.copy()
    atoms_phys.calc = phys_calc
    phys_energy = float(atoms_phys.get_potential_energy())
    phys_forces = atoms_phys.get_forces()

    lj = extract_lj_parameters_from_calculator(
        ATOMS_PER_MONOMER=atoms_per_monomer,
        N_MONOMERS=args.n_molecules,
    )
    n_types = len(lj["atc_epsilons"])
    ep_scale = np.ones(n_types, dtype=float)
    sig_scale = np.ones(n_types, dtype=float)

    factory = setup_calculator(
        ATOMS_PER_MONOMER=atoms_per_monomer,
        N_MONOMERS=args.n_molecules,
        ml_cutoff_distance=args.ml_cutoff,
        mm_switch_on=args.mm_switch_on,
        mm_cutoff=args.mm_cutoff,
        doML=True,
        doMM=True,
        doML_dimer=True,
        debug=False,
        model_restart_path=base_ckpt_dir,
        MAX_ATOMS_PER_SYSTEM=atoms_per_monomer * 2,
        cell=None,
        ep_scale=ep_scale,
        sig_scale=sig_scale,
        at_codes_override=lj["at_codes"],
    )
    cutoff = CutoffParameters(
        ml_cutoff=args.ml_cutoff,
        mm_switch_on=args.mm_switch_on,
        mm_cutoff=args.mm_cutoff,
    )
    mmml_calc, _ = factory(
        atomic_numbers=z,
        atomic_positions=r,
        n_monomers=args.n_molecules,
        cutoff_params=cutoff,
        doML=True,
        doMM=True,
        doML_dimer=True,
        backprop=True,
        debug=False,
        energy_conversion_factor=1.0,
        force_conversion_factor=1.0,
    )
    atoms_mmml = atoms.copy()
    atoms_mmml.calc = mmml_calc
    mmml_energy = float(atoms_mmml.get_potential_energy())
    mmml_forces = atoms_mmml.get_forces()

    summary = {
        "epoch_used": str(epoch_dir),
        "portable_checkpoint_json": str(portable_json),
        "cluster_psf": str(psf_path),
        "cluster_pdb": str(pdb_path),
        "cluster_xyz": str(xyz_path),
        "residue": args.residue.upper(),
        "n_molecules": args.n_molecules,
        "n_atoms": n_atoms,
        "physnetjax": {
            "energy_eV": phys_energy,
            "max_force_eVA": float(np.abs(phys_forces).max()),
        },
        "mmml": {
            "energy_eV": mmml_energy,
            "max_force_eVA": float(np.abs(mmml_forces).max()),
        },
    }
    summary_path = out_dir / "cluster_4res_test_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Portable checkpoint: {portable_json}")
    print(f"CHARMM outputs: {psf_path} | {pdb_path}")
    print(f"PSF-order XYZ: {xyz_path}")
    print(f"PhysNetJax energy (eV): {phys_energy:.8f}")
    print(f"PhysNetJax max |force| (eV/A): {np.abs(phys_forces).max():.8f}")
    print(f"MMML energy (eV): {mmml_energy:.8f}")
    print(f"MMML max |force| (eV/A): {np.abs(mmml_forces).max():.8f}")
    print(f"Summary: {summary_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create PSF-ordered 4-residue cluster and test PhysNetJax + MMML calculators."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Orbax checkpoint root or epoch-* directory")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/checkpoint_smoke"), help="Output directory")
    parser.add_argument("--residue", type=str, default="ACO", help="CHARMM/CGenFF residue name (e.g., ACO)")
    parser.add_argument("--n-molecules", type=int, default=4, help="Number of same residues in the cluster")
    parser.add_argument("--spacing", type=float, default=6.0, help="Residue COM grid spacing in Angstrom")
    parser.add_argument("--ml-cutoff", type=float, default=2.0, help="MMML ML cutoff (Angstrom)")
    parser.add_argument("--mm-switch-on", type=float, default=5.0, help="MMML switch-on (Angstrom)")
    parser.add_argument("--mm-cutoff", type=float, default=1.0, help="MMML switch width (Angstrom)")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
