#!/usr/bin/env python3
"""
Scan MEOH dimer COM distance and compare CHARMM / PhysNetJax / MMML energies and forces.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import ase
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mmml.interfaces.pycharmmInterface.import_pycharmm as pyci
from mmml.cli.base import load_physnet_params_and_ef_model, resolve_checkpoint_paths
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

import pycharmm.energy as energy
import pycharmm.generate as gen
import pycharmm.ic as ic
import pycharmm.param as param
import pycharmm.psf as psf
import pycharmm.read as read
import pycharmm.settings as settings
import pycharmm.write as pywrite

# Compatibility shims used by MMML internals.
pyci.read = read
pyci.settings = settings
pyci.psf = psf


def _latest_epoch_dir(ckpt_root: Path) -> Path:
    if (ckpt_root / "manifest.ocdbt").exists():
        return ckpt_root
    epochs = [p for p in ckpt_root.iterdir() if p.is_dir() and p.name.startswith("epoch-")]
    if not epochs:
        raise FileNotFoundError(f"No epoch-* directories in {ckpt_root}")
    return max(epochs, key=lambda p: int(p.name.split("epoch-")[-1]))


def _load_template_pdb_coords(template_pdb: Path) -> dict[str, np.ndarray]:
    coords: dict[str, np.ndarray] = {}
    for line in template_pdb.read_text(encoding="utf-8").splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        name = line[12:16].strip()
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        coords[name] = np.array([x, y, z], dtype=float)
    if not coords:
        raise ValueError(f"No ATOM records found in {template_pdb}")
    return coords


def _setup_charmm_meoh_dimer(template_pdb: Path, initial_sep: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    pycharmm.lingo.charmm_script("DELETE ATOM SELE ALL END")
    reset_block()
    read.rtf(CGENFF_RTF)
    bl = settings.set_bomb_level(-2)
    wl = settings.set_warn_level(-2)
    read.prm(CGENFF_PRM)
    settings.set_bomb_level(bl)
    settings.set_warn_level(wl)
    pycharmm.lingo.charmm_script("bomlev 0")
    read.sequence_string("MEOH MEOH")
    gen.new_segment(seg_name="DIMR", setup_ic=True)
    ic.prm_fill(replace_all=True)
    ic.build()

    positions = coor.get_positions().to_numpy(dtype=float)
    atom_names = np.asarray(psf.get_atype())
    z = np.asarray(get_Z_from_psf(), dtype=int)
    n_atoms = len(z)
    if n_atoms != 12:
        raise ValueError(f"Expected 12 atoms for MEOH dimer, got {n_atoms}")

    tmpl = _load_template_pdb_coords(template_pdb)
    out = positions.copy()
    monomer0 = []
    monomer1 = []
    for nm in atom_names[:6]:
        if nm not in tmpl:
            raise KeyError(f"Template missing atom name '{nm}'")
        monomer0.append(tmpl[nm])
    for nm in atom_names[6:12]:
        if nm not in tmpl:
            raise KeyError(f"Template missing atom name '{nm}'")
        monomer1.append(tmpl[nm])

    mono0 = np.asarray(monomer0, dtype=float)
    mono1 = np.asarray(monomer1, dtype=float)
    # Start from the same internal geometry but ensure a non-overlapping dimer seed.
    mono0 = mono0 - mono0.mean(axis=0)
    mono1 = mono1 - mono1.mean(axis=0) + np.array([initial_sep, 0.0, 0.0], dtype=float)
    out[:6] = mono0
    out[6:12] = mono1

    inter = np.linalg.norm(out[:6, None, :] - out[None, 6:12, :], axis=-1)
    min_inter = float(inter.min())
    if min_inter < 0.5:
        raise ValueError(
            f"Invalid dimer seed: min intermolecular distance {min_inter:.4f} A is too small."
        )

    return z, out


def _collect_charmm_terms() -> dict[str, float]:
    df = energy.get_energy()
    row = df.iloc[0].to_dict()
    return {str(k): float(v) for k, v in row.items() if isinstance(v, (int, float, np.floating))}


def _get_charmm_forces() -> np.ndarray:
    """Return CHARMM forces by temporarily using COOR FORCE."""
    pos = coor.get_positions()
    pycharmm.lingo.charmm_script("coor force sele all end")
    frc = coor.get_positions().to_numpy(dtype=float)
    coor.set_positions(pos)
    return frc


def _to_float_sum(value) -> float:
    arr = np.asarray(value)
    return float(arr.sum()) if arr.size > 1 else float(arr.reshape(()))


def main(args: argparse.Namespace) -> int:
    if args.checkpoint is None:
        ckpt_root, _ = resolve_checkpoint_paths(None)
    else:
        ckpt_root = args.checkpoint.expanduser().resolve()
    base_ckpt_dir, _ = resolve_checkpoint_paths(ckpt_root)

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    z, base_r = _setup_charmm_meoh_dimer(args.template_pdb.expanduser().resolve())
    com0 = base_r[:6].mean(axis=0)
    com1 = base_r[6:].mean(axis=0)
    unit = com1 - com0
    nrm = np.linalg.norm(unit)
    if nrm < 1e-8:
        unit = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        unit = unit / nrm

    if ckpt_root.is_file() and ckpt_root.suffix == ".json":
        phys_params, phys_model = load_physnet_params_and_ef_model(ckpt_root, natoms=len(z))
    else:
        epoch_dir = _latest_epoch_dir(ckpt_root)
        phys_params, phys_model = get_params_model(str(epoch_dir), natoms=len(z))
    phys_model.natoms = len(z)

    at_codes = np.asarray(psf.get_iac(), dtype=int) - 1
    n_types = len(param.get_atc())
    ep_scale = np.ones(n_types, dtype=float)
    sig_scale = np.ones(n_types, dtype=float)
    factory = setup_calculator(
        ATOMS_PER_MONOMER=6,
        N_MONOMERS=2,
        ml_cutoff_distance=args.ml_cutoff,
        mm_switch_on=args.mm_switch_on,
        mm_cutoff=args.mm_cutoff,
        doML=True,
        doMM=True,
        doML_dimer=True,
        debug=False,
        model_restart_path=base_ckpt_dir,
        MAX_ATOMS_PER_SYSTEM=12,
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

    rows: list[dict[str, float]] = []
    distances = np.linspace(args.dmin, args.dmax, args.npoints)
    for d in distances:
        r = base_r.copy()
        shift = d * unit
        r[:6] -= r[:6].mean(axis=0)
        r[6:] -= r[6:].mean(axis=0)
        r[6:] += shift

        coor.set_positions(pd.DataFrame(r, columns=["x", "y", "z"]))
        c_terms = _collect_charmm_terms()
        c_forces = _get_charmm_forces()
        c_fnorm = np.linalg.norm(c_forces, axis=1)

        atoms = ase.Atoms(numbers=z, positions=r)
        phys_calc = get_ase_calc(phys_params, phys_model, atoms)
        atoms.calc = phys_calc
        phys_e = float(atoms.get_potential_energy())
        phys_f = atoms.get_forces()
        phys_fnorm = np.linalg.norm(phys_f, axis=1)

        calc_result = factory(
            atomic_numbers=z,
            atomic_positions=r,
            n_monomers=2,
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
        mmml_calc = calc_result[0]
        atoms_mmml = atoms.copy()
        atoms_mmml.calc = mmml_calc
        mmml_e = float(atoms_mmml.get_potential_energy())
        mmml_f = atoms_mmml.get_forces()
        mmml_fnorm = np.linalg.norm(mmml_f, axis=1)
        res = mmml_calc.results

        rows.append(
            {
                "distance_A": float(d),
                "charmm_ENER_kcalmol": float(c_terms.get("ENER", np.nan)),
                "charmm_VDW_kcalmol": float(c_terms.get("VDW", np.nan)),
                "charmm_ELEC_kcalmol": float(c_terms.get("ELEC", np.nan)),
                "charmm_fnorm_mean": float(c_fnorm.mean()),
                "charmm_fnorm_max": float(c_fnorm.max()),
                "physnet_E_eV": phys_e,
                "physnet_fnorm_mean": float(phys_fnorm.mean()),
                "physnet_fnorm_max": float(phys_fnorm.max()),
                "mmml_E_eV": mmml_e,
                "mmml_internal_E_eV": _to_float_sum(res.get("model_internal_E", 0.0)),
                "mmml_ml2b_E_eV": _to_float_sum(res.get("model_ml_2b_E", 0.0)),
                "mmml_mm_E_eV": _to_float_sum(res.get("model_mm_E", 0.0)),
                "mmml_fnorm_mean": float(mmml_fnorm.mean()),
                "mmml_fnorm_max": float(mmml_fnorm.max()),
            }
        )

    csv_path = out_dir / "distance_scan.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = out_dir / "distance_scan.json"
    summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    x = np.array([r["distance_A"] for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, [r["physnet_E_eV"] for r in rows], label="PhysNet E (eV)")
    ax.plot(x, [r["mmml_E_eV"] for r in rows], label="MMML total E (eV)")
    ax.plot(x, [r["mmml_internal_E_eV"] for r in rows], "--", label="MMML internal E")
    ax.plot(x, [r["mmml_ml2b_E_eV"] for r in rows], "--", label="MMML ML-2B E")
    ax.plot(x, [r["mmml_mm_E_eV"] for r in rows], "--", label="MMML MM E")
    ax.set_xlabel("COM distance (A)")
    ax.set_ylabel("Energy")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "energies_vs_distance.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, [r["charmm_fnorm_max"] for r in rows], label="CHARMM max|F|")
    ax.plot(x, [r["physnet_fnorm_max"] for r in rows], label="PhysNet max|F|")
    ax.plot(x, [r["mmml_fnorm_max"] for r in rows], label="MMML max|F|")
    ax.plot(x, [r["charmm_fnorm_mean"] for r in rows], "--", label="CHARMM mean|F|")
    ax.plot(x, [r["physnet_fnorm_mean"] for r in rows], "--", label="PhysNet mean|F|")
    ax.plot(x, [r["mmml_fnorm_mean"] for r in rows], "--", label="MMML mean|F|")
    ax.set_xlabel("COM distance (A)")
    ax.set_ylabel("Force norm")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "force_norms_vs_distance.png", dpi=160)
    plt.close(fig)

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {out_dir / 'energies_vs_distance.png'}")
    print(f"Wrote: {out_dir / 'force_norms_vs_distance.png'}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MEOH dimer distance scan with CHARMM/PhysNet/MMML terms.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Orbax root, epoch-* dir, or portable .json "
            "(default: bundled manifest model with lowest validation force MAE, or $MMML_CKPT)."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/dimer_scan_meoh"))
    parser.add_argument("--template-pdb", type=Path, default=Path("mmml/generate/sample/pdb/meoh.pdb"))
    parser.add_argument("--dmin", type=float, default=3.0, help="Minimum COM distance (A)")
    parser.add_argument("--dmax", type=float, default=8.0, help="Maximum COM distance (A)")
    parser.add_argument("--npoints", type=int, default=11, help="Number of scan points")
    parser.add_argument("--ml-cutoff", type=float, default=5.0)
    parser.add_argument("--mm-switch-on", type=float, default=5.0)
    parser.add_argument("--mm-cutoff", type=float, default=3.0)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
