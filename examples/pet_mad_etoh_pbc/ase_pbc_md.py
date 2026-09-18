#!/usr/bin/env python3
"""CHARMM-free PET-MAD periodic MD for liquid ethanol in a 32 Å cube.

Default: 338 ETOH (CGenFF count at 0.789 g/cm³), 300 K, 0.5 fs, PBC,
``--metatomic-eval-mode whole_system`` equivalent (one ASE eval per step).

Pass: finite energy/forces, N and density match the documented box, MD
steps complete, report.json written.

Example::

    export PET_MAD_CKPT=/tmp/mmml-metatomic-models/pet-mad-xs-v1.5.0.pt
    JAX_PLATFORMS=cpu MMML_METATOMIC_DEVICE=cpu \\
      uv run python examples/pet_mad_etoh_pbc/ase_pbc_md.py \\
        --checkpoint \"$PET_MAD_CKPT\" --n-steps 5 --ensemble nvt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from ase import Atoms, units
from ase.io import read as ase_read
from ase.io import write as ase_write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_MONOMER = EXAMPLE_DIR / "etoh.xyz"
DEFAULT_BOX_A = 32.0
DEFAULT_DT_FS = 0.5
DEFAULT_TEMPERATURE_K = 300.0
DEFAULT_N_STEPS = 5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_checkpoint() -> Path | None:
    env = os.environ.get("PET_MAD_CKPT", "").strip() or os.environ.get(
        "MMML_CKPT", ""
    ).strip()
    if env:
        return Path(env).expanduser()
    model_dir = os.environ.get("MMML_METATOMIC_MODEL_DIR", "").strip()
    if model_dir:
        candidate = Path(model_dir).expanduser() / "pet-mad-xs-v1.5.0.pt"
        if candidate.is_file():
            return candidate
    fallback = Path("/tmp/mmml-metatomic-models/pet-mad-xs-v1.5.0.pt")
    return fallback if fallback.is_file() else None


def _etoh_count(box_side_A: float, density_g_cm3: float) -> int:
    from mmml.interfaces.pycharmmInterface.mlpot.box_sizing import (
        n_molecules_for_target_density_in_fixed_box,
    )

    scaled = n_molecules_for_target_density_in_fixed_box(
        composition={"ETOH": 1},
        box_side_A=float(box_side_A),
        target_density_g_cm3=float(density_g_cm3),
    )
    return int(scaled["ETOH"])


def _mass_density_g_cm3(n_molecules: int, box_side_A: float) -> float:
    from mmml.interfaces.pycharmmInterface.mlpot.box_sizing import (
        total_mass_g_for_composition,
    )

    mass_g = total_mass_g_for_composition({"ETOH": int(n_molecules)})
    vol_cm3 = (float(box_side_A) * 1.0e-8) ** 3
    return float(mass_g / vol_cm3)


def build_etoh_box(
    *,
    monomer_xyz: Path,
    box_side_A: float,
    n_molecules: int,
    seed: int,
) -> Atoms:
    from mmml.utils.geometry_checks import tile_monomer_in_cubic_cell

    monomer = ase_read(str(monomer_xyz))
    z_mono = np.asarray(monomer.get_atomic_numbers(), dtype=int)
    r_mono = np.asarray(monomer.get_positions(), dtype=float)
    positions, _offsets = tile_monomer_in_cubic_cell(
        r_mono,
        int(n_molecules),
        float(box_side_A),
        seed=int(seed),
        random_rotations=True,
    )
    numbers = np.tile(z_mono, int(n_molecules))
    atoms = Atoms(
        numbers=numbers,
        positions=positions,
        cell=[float(box_side_A)] * 3,
        pbc=True,
    )
    atoms.wrap()
    return atoms


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=_default_checkpoint(),
        help="TorchScript AtomisticModel (.pt). Default: $PET_MAD_CKPT.",
    )
    parser.add_argument("--monomer-xyz", type=Path, default=DEFAULT_MONOMER)
    parser.add_argument("--box-size", type=float, default=DEFAULT_BOX_A)
    parser.add_argument(
        "--target-density-g-cm3",
        type=float,
        default=None,
        help="Bulk liquid density (default: SOLVENT_BULK_PROPS['ETOH']).",
    )
    parser.add_argument(
        "--n-molecules",
        type=int,
        default=None,
        help="Override molecule count (default: density → N in --box-size).",
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE_K)
    parser.add_argument("--dt-fs", type=float, default=DEFAULT_DT_FS)
    parser.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ensemble",
        choices=("nve", "nvt"),
        default="nvt",
        help="nve=VelocityVerlet; nvt=Langevin at --temperature.",
    )
    parser.add_argument(
        "--friction",
        type=float,
        default=0.01,
        help="ASE Langevin friction (1/fs) for --ensemble nvt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "scratch" / "pet_mad_etoh_pbc" / "ase_smoke",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Report path (default: <output-dir>/report.json).",
    )
    args = parser.parse_args(argv)

    from mmml.interfaces.calculators.metatomic import (
        have_metatomic,
        load_metatomic_calculator,
        metatomic_device_name,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.box_sizing import SOLVENT_BULK_PROPS

    rho = (
        float(args.target_density_g_cm3)
        if args.target_density_g_cm3 is not None
        else float(SOLVENT_BULK_PROPS["ETOH"]["rho_g_cm3"])
    )
    n_mol = (
        int(args.n_molecules)
        if args.n_molecules is not None
        else _etoh_count(args.box_size, rho)
    )
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    json_out = (args.json_out or (out / "report.json")).resolve()

    report: dict = {
        "ok": False,
        "box_side_A": float(args.box_size),
        "target_density_g_cm3": rho,
        "n_molecules": n_mol,
        "temperature_K": float(args.temperature),
        "dt_fs": float(args.dt_fs),
        "n_steps": int(args.n_steps),
        "ensemble": str(args.ensemble),
        "seed": int(args.seed),
        "device": metatomic_device_name(),
    }

    def _write() -> None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {json_out}")

    if args.checkpoint is None:
        print("FAIL: pass --checkpoint or set PET_MAD_CKPT", file=sys.stderr)
        _write()
        return 2
    ckpt = Path(args.checkpoint).expanduser().resolve()
    if not ckpt.is_file():
        print(f"FAIL: checkpoint not found: {ckpt}", file=sys.stderr)
        report["checkpoint"] = str(ckpt)
        _write()
        return 2
    if not have_metatomic():
        print("FAIL: uv sync --extra metatomic", file=sys.stderr)
        _write()
        return 2

    report["checkpoint"] = str(ckpt)
    report["checkpoint_sha256"] = _sha256(ckpt)

    t_pack = time.perf_counter()
    atoms = build_etoh_box(
        monomer_xyz=Path(args.monomer_xyz).expanduser().resolve(),
        box_side_A=float(args.box_size),
        n_molecules=n_mol,
        seed=int(args.seed),
    )
    report["pack_s"] = time.perf_counter() - t_pack
    report["n_atoms"] = int(len(atoms))
    report["atoms_per_monomer"] = int(len(atoms) // n_mol)
    report["density_g_cm3"] = _mass_density_g_cm3(n_mol, args.box_size)
    ase_write(str(out / "etoh_box_initial.xyz"), atoms)

    print(
        f"ETOH:{n_mol}  N={len(atoms)}  L={args.box_size:.1f} Å  "
        f"ρ={report['density_g_cm3']:.4f} g/cm³  T={args.temperature:.1f} K  "
        f"dt={args.dt_fs} fs  ensemble={args.ensemble}"
    )

    t_load = time.perf_counter()
    atoms.calc = load_metatomic_calculator(ckpt)
    report["load_s"] = time.perf_counter() - t_load

    rng = np.random.default_rng(int(args.seed))
    MaxwellBoltzmannDistribution(
        atoms,
        temperature_K=float(args.temperature),
        rng=rng,
    )
    Stationary(atoms)

    t0 = time.perf_counter()
    e0 = float(atoms.get_potential_energy())
    f0 = np.asarray(atoms.get_forces(), dtype=float)
    report["first_eval_s"] = time.perf_counter() - t0
    report["E0_eV"] = e0
    report["F0_max_eVA"] = float(np.max(np.linalg.norm(f0, axis=1)))
    report["T0_K"] = float(atoms.get_temperature())
    print(
        f"E0={e0:.6f} eV  |F|_max={report['F0_max_eVA']:.4f} eV/Å  "
        f"T0={report['T0_K']:.1f} K  first_eval={report['first_eval_s']:.2f} s"
    )
    if not np.isfinite(e0) or not np.all(np.isfinite(f0)):
        print("FAIL: non-finite energy or forces on the initial box", file=sys.stderr)
        _write()
        return 1

    if args.ensemble == "nvt":
        from ase.md.langevin import Langevin

        dyn = Langevin(
            atoms,
            timestep=float(args.dt_fs) * units.fs,
            temperature_K=float(args.temperature),
            friction=float(args.friction),
            rng=rng,
        )
    else:
        dyn = VelocityVerlet(atoms, timestep=float(args.dt_fs) * units.fs)

    t_md = time.perf_counter()
    dyn.run(int(args.n_steps))
    report["md_s"] = time.perf_counter() - t_md
    e1 = float(atoms.get_potential_energy())
    f1 = np.asarray(atoms.get_forces(), dtype=float)
    report["E1_eV"] = e1
    report["F1_max_eVA"] = float(np.max(np.linalg.norm(f1, axis=1)))
    report["T1_K"] = float(atoms.get_temperature())
    report["md_ps"] = float(args.n_steps) * float(args.dt_fs) * 1.0e-3
    ase_write(str(out / "etoh_box_final.xyz"), atoms)
    print(
        f"E1={e1:.6f} eV  |F|_max={report['F1_max_eVA']:.4f} eV/Å  "
        f"T1={report['T1_K']:.1f} K  md={report['md_s']:.2f} s  "
        f"({args.n_steps} × {args.dt_fs} fs)"
    )
    if not np.isfinite(e1) or not np.all(np.isfinite(f1)):
        print("FAIL: non-finite energy or forces after MD", file=sys.stderr)
        _write()
        return 1

    expected_n = 9 * n_mol
    if int(len(atoms)) != expected_n:
        print(
            f"FAIL: n_atoms={len(atoms)} expected {expected_n} (9 atoms/ETOH)",
            file=sys.stderr,
        )
        _write()
        return 1
    report["ok"] = True
    _write()
    print("PASS: PET-MAD ethanol PBC MD smoke")
    return 0


if __name__ == "__main__":
    sys.exit(main())
