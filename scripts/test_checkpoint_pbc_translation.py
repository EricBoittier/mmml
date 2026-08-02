#!/usr/bin/env python3
"""Test periodic translation/image invariance of one hybrid ML/MM checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from ase.io import read


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", type=Path, required=True)
    parser.add_argument("--psf", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--box", type=float, default=28.0)
    parser.add_argument("--n-monomers", type=int, default=732)
    parser.add_argument("--ml-batch-size", type=int, default=64)
    parser.add_argument(
        "--mm-charge-mode",
        choices=("fixed", "q0", "latent_dynamic"),
        default="fixed",
        help="Charges used by the intermolecular MM Coulomb term.",
    )
    parser.add_argument(
        "--repeat-only",
        action="store_true",
        help="Evaluate base twice to isolate repeatability and force terms.",
    )
    args = parser.parse_args()
    output = args.output.expanduser().resolve()

    from mmml.interfaces.pycharmmInterface import import_pycharmm as ipy
    from mmml.interfaces.pycharmmInterface.cgenff_bonded_reference import (
        read_psf_card_file,
    )
    from mmml.interfaces.pycharmmInterface.nbonds_config import read_cgenff_toppar
    from mmml.cli.run.md_evaluate_npz import _attach_ase_mmml_calculator

    if not ipy.ensure_pycharmm_loaded():
        raise RuntimeError("PyCHARMM is unavailable")
    read_cgenff_toppar()
    read_psf_card_file(args.psf.resolve())

    atoms = read(args.pdb.resolve())
    z = np.asarray(atoms.numbers, dtype=np.int32)
    if len(atoms) != args.n_monomers * 3:
        raise ValueError(
            f"expected {args.n_monomers * 3} atoms, found {len(atoms)}"
        )
    atoms.set_cell([args.box] * 3)
    atoms.set_pbc(True)
    base = np.asarray(atoms.positions, dtype=np.float64)

    calc_args = SimpleNamespace(
        ml_switch_width=1.5,
        mm_switch_on=6.0,
        mm_switch_width=5.0,
        verbose_calc=False,
        jax_md_capacity_multiplier=1.75,
        jax_md_capacity_growth_factor=1.5,
        jax_md_max_overflow_retries=4,
        jax_md_disable_fallback=False,
        jax_md_update_interval=1,
        jax_md_skin_distance=0.5,
        max_pairs=20_000,
        flat_bottom_radius=None,
        flat_bottom_k=1.0,
        flat_bottom_mode="system",
        min_com_restraint_distance=None,
        min_com_restraint_k=1.0,
        ml_batch_size=args.ml_batch_size,
        ml_max_active_dimers=None,
        ml_compute_dtype="float64",
        mm_charge_mode=args.mm_charge_mode,
        mm_charge_correction=False,
        mm_latent_charge_template=None,
    )
    _attach_ase_mmml_calculator(
        calc_args,
        atoms=atoms,
        z=z,
        n_monomers=args.n_monomers,
        atoms_per_list=[3] * args.n_monomers,
        base_ckpt_dir=args.checkpoint.resolve(),
        use_pbc=True,
        L=args.box,
        at_codes_override=None,
    )

    arbitrary = np.array([3.7, -5.2, 7.1])
    arbitrary_shifted = base + arbitrary
    molecular_wrapped = arbitrary_shifted.copy()
    for start in range(0, len(molecular_wrapped), 3):
        oxygen = molecular_wrapped[start].copy()
        rel = molecular_wrapped[start + 1 : start + 3] - oxygen
        rel -= args.box * np.round(rel / args.box)
        oxygen_wrapped = np.mod(oxygen, args.box)
        molecular_wrapped[start] = oxygen_wrapped
        molecular_wrapped[start + 1 : start + 3] = oxygen_wrapped + rel
    cases = {
        "base": base,
        "lattice_shift_x": base + np.array([args.box, 0.0, 0.0]),
        "arbitrary_unwrapped": arbitrary_shifted,
        "arbitrary_atom_wrapped": np.mod(arbitrary_shifted, args.box),
        "arbitrary_molecule_wrapped": molecular_wrapped,
        "half_cell_atom_wrapped": np.mod(base + 0.5 * args.box, args.box),
        # Re-evaluate byte-identical input after every image variant. Any delta
        # here is statefulness/nondeterminism, not a PBC representation effect.
        "base_repeat": base.copy(),
    }
    if args.repeat_only:
        cases = {"base": base, "base_repeat": base.copy()}
    evaluated: dict[str, dict[str, object]] = {}
    for name, positions in cases.items():
        atoms.set_positions(positions)
        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=np.float64)
        model_output = atoms.calc.results.get("out")
        mm_charges = None
        if model_output is not None and hasattr(model_output, "mm_charges"):
            mm_charges = np.asarray(model_output.mm_charges, dtype=np.float64)
        force_terms = {}
        if model_output is not None:
            for key in ("internal_F", "ml_2b_F", "mm_F"):
                value = getattr(model_output, key, None)
                if value is not None:
                    force_terms[key] = np.asarray(value, dtype=np.float64)
        evaluated[name] = {
            "energy_eV": energy,
            "forces": forces,
            "mm_charges": mm_charges,
            "force_terms": force_terms,
        }

    reference = evaluated["base"]
    e0 = float(reference["energy_eV"])
    f0 = np.asarray(reference["forces"])
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "box_A": args.box,
        "n_atoms": len(atoms),
        "mm_charge_mode": args.mm_charge_mode,
        "cases": {},
    }
    arrays = {"Z": z, "R_base": base, "F_base": f0}
    for name, result in evaluated.items():
        force = np.asarray(result["forces"])
        charges = result["mm_charges"]
        delta = force - f0
        report["cases"][name] = {
            "energy_eV": float(result["energy_eV"]),
            "delta_energy_eV": float(result["energy_eV"]) - e0,
            "force_max_abs_delta_eV_A": float(np.max(np.abs(delta))),
            "force_rms_delta_eV_A": float(np.sqrt(np.mean(delta**2))),
            "force_max_eV_A": float(np.max(np.linalg.norm(force, axis=1))),
        }
        if charges is not None:
            charges = np.asarray(charges)
            monomer_sums = charges.reshape(args.n_monomers, 3).sum(axis=1)
            report["cases"][name]["mm_charge_min_e"] = float(charges.min())
            report["cases"][name]["mm_charge_max_e"] = float(charges.max())
            report["cases"][name]["mm_charge_total_e"] = float(charges.sum())
            report["cases"][name]["mm_charge_monomer_max_abs_sum_e"] = float(
                np.max(np.abs(monomer_sums))
            )
            arrays[f"Qmm_{name}"] = charges
        arrays[f"F_{name}"] = force
        for term_name, term_force in result["force_terms"].items():
            arrays[f"{term_name}_{name}"] = term_force
            reference_term = reference["force_terms"].get(term_name)
            if reference_term is not None:
                report["cases"][name][f"{term_name}_max_abs_delta_eV_A"] = float(
                    np.max(np.abs(term_force - reference_term))
                )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    np.savez(output.with_suffix(".npz"), **arrays)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
