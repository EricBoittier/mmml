#!/usr/bin/env python3
"""PBC translation / image invariance for the softwell on=5 DCM deploy ckpt.

DCM-aware (5 atoms/monomer). Uses campaign dense L=24 DCM:120 box by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from ase.io import read

ROOT = Path(__file__).resolve().parents[3]


def _wrap_molecules(positions: np.ndarray, box: float, atoms_per: int) -> np.ndarray:
    out = positions.copy()
    for start in range(0, len(out), atoms_per):
        anchor = out[start].copy()
        rel = out[start + 1 : start + atoms_per] - anchor
        rel -= box * np.round(rel / box)
        anchor_w = np.mod(anchor, box)
        out[start] = anchor_w
        out[start + 1 : start + atoms_per] = anchor_w + rel
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pdb",
        type=Path,
        default=ROOT / "artifacts/lj_scales/liquid_dense_L24/model.pdb",
    )
    p.add_argument(
        "--psf",
        type=Path,
        default=ROOT / "artifacts/lj_scales/liquid_dense_L24/model.psf",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT
        / "artifacts/lj_scales/ckpts/params_hybrid_mm_lever2_on5_softwell_2026-08-02_22-15-54.json",
    )
    p.add_argument(
        "--sidecar",
        type=Path,
        default=ROOT
        / "artifacts/lj_scales/ckpts/hybrid_mm_lever2_on5_softwell-657cb7db-74a1-4623-84a5-f772b8fe7928/hybrid_mm.json",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "docs/images/dense-dt-campaign/overbind_ablation/lever2_on5_softwell/pbc_translation.json",
    )
    p.add_argument("--box", type=float, default=24.0)
    p.add_argument("--n-monomers", type=int, default=120)
    p.add_argument("--atoms-per-monomer", type=int, default=5)
    p.add_argument("--ml-batch-size", type=int, default=32)
    p.add_argument("--mm-switch-on", type=float, default=None)
    p.add_argument("--ml-switch-width", type=float, default=None)
    p.add_argument("--mm-switch-width", type=float, default=None)
    args = p.parse_args()

    from mmml.cli.run.md_evaluate_npz import _attach_ase_mmml_calculator
    from mmml.interfaces.pycharmmInterface import import_pycharmm as ipy
    from mmml.interfaces.pycharmmInterface.cgenff_bonded_reference import (
        read_psf_card_file,
    )
    from mmml.interfaces.pycharmmInterface.nbonds_config import read_cgenff_toppar

    side = json.loads(Path(args.sidecar).read_text())
    mm_on = float(args.mm_switch_on if args.mm_switch_on is not None else side.get("mm_switch_on", 5.0))
    ml_w = float(
        args.ml_switch_width
        if args.ml_switch_width is not None
        else side.get("ml_switch_width", 1.5)
    )
    mm_w = float(
        args.mm_switch_width
        if args.mm_switch_width is not None
        else side.get("mm_switch_width", 5.0)
    )

    if not ipy.ensure_pycharmm_loaded():
        raise RuntimeError("PyCHARMM is unavailable")
    read_cgenff_toppar()
    read_psf_card_file(args.psf.resolve())

    atoms = read(args.pdb.resolve())
    z = np.asarray(atoms.numbers, dtype=np.int32)
    n_expect = args.n_monomers * args.atoms_per_monomer
    if len(atoms) != n_expect:
        raise ValueError(f"expected {n_expect} atoms, found {len(atoms)}")
    atoms.set_cell([args.box] * 3)
    atoms.set_pbc(True)
    # denser L24 boxes may have atoms slightly outside [0,L); wrap molecule-wise
    base = _wrap_molecules(np.asarray(atoms.positions, dtype=np.float64), args.box, args.atoms_per_monomer)
    atoms.set_positions(base)

    # Prefer portable JSON; calculator also reads hybrid_mm.json from sidecar path
    # via checkpoint directory discovery — copy/link sidecar next to portable if needed.
    ckpt = args.checkpoint.resolve()
    side_next_to = ckpt.parent / "hybrid_mm.json"
    if not side_next_to.is_file():
        # ephemeral symlink so loader finds scales + on=5
        try:
            side_next_to.symlink_to(args.sidecar.resolve())
        except FileExistsError:
            pass

    calc_args = SimpleNamespace(
        ml_switch_width=ml_w,
        mm_switch_on=mm_on,
        mm_switch_width=mm_w,
        verbose_calc=False,
        jax_md_capacity_multiplier=1.75,
        jax_md_capacity_growth_factor=1.5,
        jax_md_max_overflow_retries=4,
        jax_md_disable_fallback=False,
        jax_md_update_interval=1,
        jax_md_skin_distance=0.5,
        max_pairs=50_000,
        flat_bottom_radius=None,
        flat_bottom_k=1.0,
        flat_bottom_mode="system",
        min_com_restraint_distance=None,
        min_com_restraint_k=1.0,
        ml_batch_size=args.ml_batch_size,
        ml_max_active_dimers=None,
        ml_compute_dtype="float64",
        mm_charge_mode="fixed",
        mm_charge_correction=False,
        mm_latent_charge_template=None,
        hybrid_mm_json=str(args.sidecar.resolve()),
    )
    _attach_ase_mmml_calculator(
        calc_args,
        atoms=atoms,
        z=z,
        n_monomers=args.n_monomers,
        atoms_per_list=[args.atoms_per_monomer] * args.n_monomers,
        base_ckpt_dir=ckpt,
        use_pbc=True,
        L=args.box,
        at_codes_override=None,
    )

    arbitrary = np.array([3.7, -5.2, 7.1])
    arbitrary_shifted = base + arbitrary
    cases = {
        "base": base,
        "lattice_shift_x": base + np.array([args.box, 0.0, 0.0]),
        "lattice_shift_xyz": base + np.array([args.box, -args.box, args.box]),
        "arbitrary_molecule_wrapped": _wrap_molecules(
            arbitrary_shifted, args.box, args.atoms_per_monomer
        ),
        "half_cell_molecule_wrapped": _wrap_molecules(
            base + 0.5 * args.box, args.box, args.atoms_per_monomer
        ),
        "base_repeat": base.copy(),
    }

    evaluated: dict[str, dict[str, float]] = {}
    for name, positions in cases.items():
        atoms.set_positions(positions)
        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=np.float64)
        evaluated[name] = {"energy_eV": energy, "forces": forces}

    e0 = evaluated["base"]["energy_eV"]
    f0 = evaluated["base"]["forces"]
    report = {
        "checkpoint": str(ckpt),
        "sidecar": str(args.sidecar.resolve()),
        "pdb": str(args.pdb.resolve()),
        "box_A": args.box,
        "n_atoms": len(atoms),
        "n_monomers": args.n_monomers,
        "atoms_per_monomer": args.atoms_per_monomer,
        "mm_switch_on": mm_on,
        "ml_switch_width": ml_w,
        "mm_switch_width": mm_w,
        "cases": {},
        "pass_criteria": {
            "lattice_abs_delta_E_eV": 1e-4,
            "lattice_force_max_abs_delta_eV_A": 1e-3,
            "wrap_abs_delta_E_eV": 5e-4,
            "wrap_force_max_abs_delta_eV_A": 5e-3,
            "repeat_abs_delta_E_eV": 1e-6,
        },
    }
    for name, result in evaluated.items():
        force = np.asarray(result["forces"])
        delta = force - f0
        report["cases"][name] = {
            "energy_eV": float(result["energy_eV"]),
            "delta_energy_eV": float(result["energy_eV"] - e0),
            "force_max_abs_delta_eV_A": float(np.max(np.abs(delta))),
            "force_rms_delta_eV_A": float(np.sqrt(np.mean(delta**2))),
        }

    crit = report["pass_criteria"]
    checks = {
        "lattice_shift_x": (
            abs(report["cases"]["lattice_shift_x"]["delta_energy_eV"])
            <= crit["lattice_abs_delta_E_eV"]
            and report["cases"]["lattice_shift_x"]["force_max_abs_delta_eV_A"]
            <= crit["lattice_force_max_abs_delta_eV_A"]
        ),
        "lattice_shift_xyz": (
            abs(report["cases"]["lattice_shift_xyz"]["delta_energy_eV"])
            <= crit["lattice_abs_delta_E_eV"]
            and report["cases"]["lattice_shift_xyz"]["force_max_abs_delta_eV_A"]
            <= crit["lattice_force_max_abs_delta_eV_A"]
        ),
        "arbitrary_molecule_wrapped": (
            abs(report["cases"]["arbitrary_molecule_wrapped"]["delta_energy_eV"])
            <= crit["wrap_abs_delta_E_eV"]
            and report["cases"]["arbitrary_molecule_wrapped"]["force_max_abs_delta_eV_A"]
            <= crit["wrap_force_max_abs_delta_eV_A"]
        ),
        "half_cell_molecule_wrapped": (
            abs(report["cases"]["half_cell_molecule_wrapped"]["delta_energy_eV"])
            <= crit["wrap_abs_delta_E_eV"]
            and report["cases"]["half_cell_molecule_wrapped"]["force_max_abs_delta_eV_A"]
            <= crit["wrap_force_max_abs_delta_eV_A"]
        ),
        "base_repeat": (
            abs(report["cases"]["base_repeat"]["delta_energy_eV"])
            <= crit["repeat_abs_delta_E_eV"]
            and report["cases"]["base_repeat"]["force_max_abs_delta_eV_A"]
            <= crit["repeat_abs_delta_E_eV"]
        ),
    }
    report["checks"] = checks
    report["pbc_ok"] = bool(all(checks.values()))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("pbc_ok:", report["pbc_ok"])
    return 0 if report["pbc_ok"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
