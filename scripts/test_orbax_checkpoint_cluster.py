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
from ase.io import write

from mmml.cli.base import resolve_checkpoint_paths
import mmml.interfaces.pycharmmInterface.import_pycharmm as pyci
from mmml.interfaces.pycharmmInterface.import_pycharmm import (
    CGENFF_PRM,
    CGENFF_RTF,
    coor,
    pycharmm,
    reset_block,
    reset_block_no_internal,
)
reset_block()
reset_block_no_internal()
from mmml.interfaces.pycharmmInterface.mmml_calculator import CutoffParameters, setup_calculator
from mmml.interfaces.pycharmmInterface.utils import get_Z_from_psf
from mmml.models.physnetjax.physnetjax.calc.helper_mlp import get_ase_calc
from mmml.models.physnetjax.physnetjax.restart.restart import get_params_model
from mmml.utils.model_checkpoint import orbax_to_json
from orbax.checkpoint import PyTreeCheckpointer

import pycharmm.psf as psf
import pycharmm.param as param
import pycharmm.generate as gen
import pycharmm.ic as ic
import pycharmm.energy as energy
import pycharmm.minimize as minimize
import pycharmm.read as read
import pycharmm.settings as settings
import pycharmm.write as pywrite

# Some MMML modules import these names from import_pycharmm, but they are not
# always exported there. Inject them for compatibility in this utility script.
pyci.read = read
pyci.settings = settings
pyci.psf = psf


def _latest_epoch_dir(ckpt_root: Path) -> Path:
    if (ckpt_root / "manifest.ocdbt").exists():
        return ckpt_root
    epoch_dirs = [d for d in ckpt_root.iterdir() if d.is_dir() and d.name.startswith("epoch-")]
    if not epoch_dirs:
        raise FileNotFoundError(f"No epoch-* directories found in: {ckpt_root}")
    return max(epoch_dirs, key=lambda d: int(d.name.split("epoch-")[-1]))


def _load_template_pdb_coords(template_pdb: Path) -> dict[str, np.ndarray]:
    """Load atom-name keyed coordinates from a PDB template."""
    coords: dict[str, np.ndarray] = {}
    for line in template_pdb.read_text(encoding="utf-8").splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        atom_name = line[12:16].strip()
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        coords[atom_name] = np.array([x, y, z], dtype=float)
    if not coords:
        raise ValueError(f"No ATOM/HETATM coordinates found in template PDB: {template_pdb}")
    return coords


def _build_psf_ordered_cluster(
    residue: str,
    n_molecules: int,
    spacing: float,
    template_pdb: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
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
    atom_names = np.asarray(psf.get_atype())
    if len(atom_names) != n_atoms:
        raise RuntimeError(f"PSF atom-name count mismatch: {len(atom_names)} vs positions {n_atoms}")

    if template_pdb is not None:
        tmpl = _load_template_pdb_coords(template_pdb)
        for i in range(n_molecules):
            start = i * atoms_per_res
            end = (i + 1) * atoms_per_res
            local_names = atom_names[start:end]
            local_coords = []
            for nm in local_names:
                if nm not in tmpl:
                    raise KeyError(
                        f"Template PDB {template_pdb} missing atom name '{nm}' required by PSF order. "
                        f"Available: {sorted(tmpl.keys())}"
                    )
                local_coords.append(tmpl[nm])
            shifted[start:end] = np.asarray(local_coords, dtype=float)

    for i in range(n_molecules):
        start = i * atoms_per_res
        end = (i + 1) * atoms_per_res
        com = shifted[start:end].mean(axis=0)
        shift = np.array([(i % n_side) * spacing, (i // n_side) * spacing, 0.0], dtype=float)
        shifted[start:end] = shifted[start:end] - com + shift

    coor.set_positions(pd.DataFrame(shifted, columns=["x", "y", "z"]))
    z = np.asarray(get_Z_from_psf(), dtype=int)
    return z, shifted


def _collect_charmm_terms() -> dict[str, float]:
    """Return active CHARMM energy terms as a flat dict."""
    df = energy.get_energy()
    row = df.iloc[0].to_dict()
    terms: dict[str, float] = {}
    for key, value in row.items():
        if isinstance(value, (int, float, np.floating)):
            terms[str(key)] = float(value)
    return terms


def _to_float(value) -> float:
    arr = np.asarray(value)
    return float(arr.sum()) if arr.size > 1 else float(arr.reshape(()))


def run(args: argparse.Namespace) -> int:
    ckpt_root = args.checkpoint.expanduser().resolve()
    epoch_dir = _latest_epoch_dir(ckpt_root)
    base_ckpt_dir, _ = resolve_checkpoint_paths(ckpt_root)
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    portable_json = out_dir / "params_portable.json"
    orbax_to_json(epoch_dir, portable_json)

    restored = PyTreeCheckpointer().restore(str(epoch_dir))
    ckpt_natoms = int(restored["model_attributes"].get("natoms", 0))
    if ckpt_natoms <= 0:
        raise ValueError("Checkpoint model_attributes missing valid natoms")

    z, r = _build_psf_ordered_cluster(
        args.residue,
        args.n_molecules,
        args.spacing,
        template_pdb=args.template_pdb,
    )
    n_atoms = len(z)
    atoms_per_monomer = n_atoms // args.n_molecules
    if n_atoms > ckpt_natoms:
        raise ValueError(
            f"Cluster atom count ({n_atoms}) exceeds checkpoint natoms ({ckpt_natoms}). "
            f"Reduce --n-molecules or use a checkpoint trained with >= {n_atoms} atoms."
        )

    # Setup PyCHARMM non-bonded parameters
    reset_block()
    reset_block_no_internal()
    reset_block()
    nbonds = """!#########################################
    ! Bonded/Non-bonded Options & Constraints
    !#########################################

    ! Non-bonding parameters
    nbonds atom cutnb 14.0  ctofnb 12.0 ctonnb 10.0 -
    vswitch NBXMOD 3 -
    inbfrq -1 imgfrq -1
    """
    pycharmm.lingo.charmm_script(nbonds)
    charmm_pre = _collect_charmm_terms()
    minimize.run_abnr(nstep=args.minimize_steps, tolenr=args.tolenr, tolgrd=args.tolgrd)
    charmm_post = _collect_charmm_terms()
    r = coor.get_positions().to_numpy(dtype=float)

    psf_path = out_dir / "cluster_4res.psf"
    pdb_path = out_dir / "cluster_4res.pdb"
    xyz_path = out_dir / "cluster_4res.xyz"
    pywrite.psf_card(str(psf_path))
    pywrite.coor_pdb(str(pdb_path))

    atoms = ase.Atoms(numbers=z, positions=r)
    write(str(xyz_path), atoms)

    phys_params, phys_model = get_params_model(str(epoch_dir), natoms=n_atoms)
    phys_model.natoms = n_atoms
    phys_calc = get_ase_calc(phys_params, phys_model, atoms)
    atoms_phys = atoms.copy()
    atoms_phys.calc = phys_calc
    phys_energy = float(atoms_phys.get_potential_energy())
    phys_forces = atoms_phys.get_forces()

    # Use PSF IAC atom-type indices directly (CHARMM 1-indexed -> 0-indexed).
    at_codes = np.asarray(psf.get_iac(), dtype=int) - 1
    n_types = len(param.get_atc())
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
        at_codes_override=at_codes,
    )
    cutoff = CutoffParameters(
        ml_cutoff=args.ml_cutoff,
        mm_switch_on=args.mm_switch_on,
        mm_cutoff=args.mm_cutoff,
    )
    calc_result = factory(
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
        verbose=True,
    )
    if len(calc_result) == 3:
        mmml_calc, _, _ = calc_result
    else:
        mmml_calc, _ = calc_result
    atoms_mmml = atoms.copy()
    atoms_mmml.calc = mmml_calc
    mmml_energy = float(atoms_mmml.get_potential_energy())
    mmml_forces = atoms_mmml.get_forces()
    mmml_max_force = float(np.abs(mmml_forces).max())
    mmml_model = getattr(mmml_calc, "results", {})
    mmml_components = {
        "total_E_eV": _to_float(mmml_model.get("model_energy", mmml_energy)),
        "internal_E_eV": _to_float(mmml_model.get("model_internal_E", 0.0)),
        "ml_2b_E_eV": _to_float(mmml_model.get("model_ml_2b_E", 0.0)),
        "mm_E_eV": _to_float(mmml_model.get("model_mm_E", 0.0)),
        "internal_F_max_eVA": float(np.abs(np.asarray(mmml_model.get("model_internal_F", 0.0))).max()) if "model_internal_F" in mmml_model else 0.0,
        "ml_2b_F_max_eVA": float(np.abs(np.asarray(mmml_model.get("model_ml_2b_F", 0.0))).max()) if "model_ml_2b_F" in mmml_model else 0.0,
        "mm_F_max_eVA": float(np.abs(np.asarray(mmml_model.get("model_mm_F", 0.0))).max()) if "model_mm_F" in mmml_model else 0.0,
    }
    mmml_sanity_failed = (
        mmml_max_force <= args.mmml_zero_force_threshold
        and abs(mmml_energy) > args.mmml_high_energy_threshold
    )

    summary = {
        "epoch_used": str(epoch_dir),
        "portable_checkpoint_json": str(portable_json),
        "cluster_psf": str(psf_path),
        "cluster_pdb": str(pdb_path),
        "cluster_xyz": str(xyz_path),
        "residue": args.residue.upper(),
        "n_molecules": args.n_molecules,
        "n_atoms": n_atoms,
        "charmm": {
            "pre_minimization_terms": charmm_pre,
            "post_minimization_terms": charmm_post,
            "delta_ENER": float(charmm_post.get("ENER", 0.0) - charmm_pre.get("ENER", 0.0)),
        },
        "physnetjax": {
            "energy_eV": phys_energy,
            "max_force_eVA": float(np.abs(phys_forces).max()),
        },
        "mmml": {
            "energy_eV": mmml_energy,
            "max_force_eVA": mmml_max_force,
            "sanity_failed": bool(mmml_sanity_failed),
            "components": mmml_components,
        },
    }
    summary_path = out_dir / "cluster_4res_test_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Portable checkpoint: {portable_json}")
    print(f"CHARMM outputs: {psf_path} | {pdb_path}")
    print(f"PSF-order XYZ: {xyz_path}")
    print(f"CHARMM ENER pre/post (kcal/mol): {charmm_pre.get('ENER', float('nan')):.6f} -> {charmm_post.get('ENER', float('nan')):.6f}")
    print(f"PhysNetJax energy (eV): {phys_energy:.8f}")
    print(f"PhysNetJax max |force| (eV/A): {np.abs(phys_forces).max():.8f}")
    print(f"MMML energy (eV): {mmml_energy:.8f}")
    print(f"MMML max |force| (eV/A): {mmml_max_force:.8f}")
    print(
        "MMML components (eV): "
        f"internal={mmml_components['internal_E_eV']:.6f}, "
        f"ml_2b={mmml_components['ml_2b_E_eV']:.6f}, "
        f"mm={mmml_components['mm_E_eV']:.6f}"
    )
    print(
        "MMML component max|F| (eV/A): "
        f"internal={mmml_components['internal_F_max_eVA']:.6e}, "
        f"ml_2b={mmml_components['ml_2b_F_max_eVA']:.6e}, "
        f"mm={mmml_components['mm_F_max_eVA']:.6e}"
    )
    print(f"Summary: {summary_path}")
    if mmml_sanity_failed:
        msg = (
            "MMML sanity check failed: near-zero max force with high absolute energy. "
            f"(energy={mmml_energy:.6f} eV, max|F|={mmml_max_force:.3e} eV/A)"
        )
        if args.strict_mmml_sanity:
            raise RuntimeError(msg)
        print(f"WARNING: {msg}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create PSF-ordered 4-residue cluster and test PhysNetJax + MMML calculators."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Orbax checkpoint root or epoch-* directory")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/checkpoint_smoke"), help="Output directory")
    parser.add_argument("--residue", type=str, default="ACO", help="CHARMM/CGenFF residue name (e.g., ACO)")
    parser.add_argument(
        "--template-pdb",
        type=Path,
        default=Path("mmml/generate/sample/pdb/meoh.pdb"),
        help="Template PDB whose atom-name coordinates seed each residue (PSF order applied).",
    )
    parser.add_argument("--n-molecules", type=int, default=4, help="Number of same residues in the cluster")
    parser.add_argument("--spacing", type=float, default=6.0, help="Residue COM grid spacing in Angstrom")
    parser.add_argument("--ml-cutoff", type=float, default=5.0, help="MMML ML cutoff (Angstrom)")
    parser.add_argument("--mm-switch-on", type=float, default=5.0, help="MMML switch-on (Angstrom)")
    parser.add_argument("--mm-cutoff", type=float, default=3.0, help="MMML switch width (Angstrom)")
    parser.add_argument("--minimize-steps", type=int, default=500, help="PyCHARMM ABNR minimization steps")
    parser.add_argument("--tolenr", type=float, default=1e-3, help="ABNR energy tolerance")
    parser.add_argument("--tolgrd", type=float, default=1e-3, help="ABNR gradient tolerance")
    parser.add_argument("--mmml-zero-force-threshold", type=float, default=1e-8, help="Near-zero MMML max force threshold")
    parser.add_argument("--mmml-high-energy-threshold", type=float, default=1e3, help="High-energy threshold for MMML sanity check")
    parser.add_argument("--strict-mmml-sanity", action="store_true", help="Raise error when MMML sanity check fails")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
