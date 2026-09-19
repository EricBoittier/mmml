"""CHARMM-free metatomic ASE MD in a cubic liquid box.

Default recipe: 32 Å ethanol at 0.789 g/cm³ (ETOH:338), 300 K, 0.5 fs.
``--ensemble nve`` with FIRE mini is the conservation path.

Example::

    export PET_MAD_CKPT=/path/to/pet-mad-xs-v1.5.0.pt
    mmml metatomic-pbc-md --ensemble nve --minimize-steps 60 --n-steps 400
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# Argparse-only at import: ``mmml <cmd> --help`` and docs generation load this
# module. ASE / metatomic stay inside ``main``.

DEFAULT_BOX_A = 32.0
DEFAULT_DT_FS = 0.5
DEFAULT_TEMPERATURE_K = 300.0
DEFAULT_N_STEPS = 5
DEFAULT_RESIDUE = "ETOH"
DEFAULT_OUTPUT_DIR = Path("scratch") / "pet_mad_etoh_pbc" / "ase_smoke"
DEFAULT_MONOMER_XYZ = Path("examples") / "pet_mad_etoh_pbc" / "etoh.xyz"


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmml metatomic-pbc-md",
        description=(
            "CHARMM-free metatomic ASE MD in a cubic liquid box. Default: "
            "32 Å ethanol at experimental density, 300 K, 0.5 fs."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="TorchScript AtomisticModel (.pt). Default: $PET_MAD_CKPT.",
    )
    parser.add_argument(
        "--residue",
        type=str,
        default=DEFAULT_RESIDUE,
        help="CGenFF residue name for bulk-density count (default: ETOH).",
    )
    parser.add_argument(
        "--monomer-xyz",
        type=Path,
        default=None,
        help=f"Monomer xyz (default: {DEFAULT_MONOMER_XYZ.as_posix()}).",
    )
    parser.add_argument("--box-size", type=float, default=DEFAULT_BOX_A)
    parser.add_argument(
        "--target-density-g-cm3",
        type=float,
        default=None,
        help="Bulk liquid density (default: SOLVENT_BULK_PROPS[residue]).",
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
        "--minimize-steps",
        type=int,
        default=0,
        help="FIRE steps before assigning velocities (0 skips mini).",
    )
    parser.add_argument(
        "--minimize-fmax",
        type=float,
        default=0.2,
        help="FIRE force threshold (eV/Å).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Record PE/KE/Etot every N MD steps (always includes step 0).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR.as_posix()}).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Report path (default: <output-dir>/report.json).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import numpy as np
    from ase import units
    from ase.constraints import FixCom
    from ase.io import write as ase_write
    from ase.md.verlet import VelocityVerlet

    try:
        from ase.md.velocitydistribution import Stationary, thermalize_momenta
    except ImportError:  # ASE < 3.29
        from ase.md.velocitydistribution import (  # type: ignore[no-redef]
            MaxwellBoltzmannDistribution,
            Stationary,
        )

        def thermalize_momenta(atoms, temperature_K, *, rng=None, **_kwargs):
            MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=rng)

    from mmml.interfaces.calculators.metatomic import (
        have_metatomic,
        load_metatomic_calculator,
        metatomic_device_name,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.box_sizing import SOLVENT_BULK_PROPS
    from mmml.md.metatomic_pbc import (
        build_tiled_cubic_liquid,
        default_etoh_monomer_xyz,
        energy_snapshot,
        mass_density_g_cm3,
        n_molecules_for_residue_box,
        nve_conservation_stats,
        plot_nve_energy,
        write_energy_csv,
    )

    residue = str(args.residue).strip().upper()
    rho = (
        float(args.target_density_g_cm3)
        if args.target_density_g_cm3 is not None
        else float(SOLVENT_BULK_PROPS[residue]["rho_g_cm3"])
        if residue in SOLVENT_BULK_PROPS
        else None
    )
    if args.n_molecules is not None:
        n_mol = int(args.n_molecules)
        if rho is None:
            rho = mass_density_g_cm3(residue, n_mol, args.box_size)
    else:
        n_mol = n_molecules_for_residue_box(
            residue, box_side_A=args.box_size, density_g_cm3=rho
        )
        if rho is None:
            rho = mass_density_g_cm3(residue, n_mol, args.box_size)
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    json_out = (args.json_out or (out / "report.json")).resolve()
    log_every = max(int(args.log_every), 1)
    if args.monomer_xyz is None:
        monomer_xyz = default_etoh_monomer_xyz()
    else:
        monomer_xyz = Path(args.monomer_xyz).expanduser().resolve()
    ckpt_arg = args.checkpoint if args.checkpoint is not None else _default_checkpoint()

    report: dict = {
        "ok": False,
        "residue": residue,
        "box_side_A": float(args.box_size),
        "target_density_g_cm3": rho,
        "n_molecules": n_mol,
        "temperature_K": float(args.temperature),
        "dt_fs": float(args.dt_fs),
        "n_steps": int(args.n_steps),
        "ensemble": str(args.ensemble),
        "seed": int(args.seed),
        "device": metatomic_device_name(),
        "minimize_steps": int(args.minimize_steps),
        "minimize_fmax_eVA": float(args.minimize_fmax),
        "non_conservative": False,
        "monomer_xyz": str(monomer_xyz),
    }

    def _write() -> None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {json_out}")

    if ckpt_arg is None:
        print("FAIL: pass --checkpoint or set PET_MAD_CKPT", file=sys.stderr)
        _write()
        return 2
    ckpt = Path(ckpt_arg).expanduser().resolve()
    if not ckpt.is_file():
        print(f"FAIL: checkpoint not found: {ckpt}", file=sys.stderr)
        report["checkpoint"] = str(ckpt)
        _write()
        return 2
    if not have_metatomic():
        print("FAIL: uv sync --extra metatomic", file=sys.stderr)
        _write()
        return 2
    if not monomer_xyz.is_file():
        print(f"FAIL: monomer xyz not found: {monomer_xyz}", file=sys.stderr)
        _write()
        return 2

    report["checkpoint"] = str(ckpt)
    report["checkpoint_sha256"] = _sha256(ckpt)

    t_pack = time.perf_counter()
    atoms = build_tiled_cubic_liquid(
        monomer_xyz=monomer_xyz,
        box_side_A=float(args.box_size),
        n_molecules=n_mol,
        seed=int(args.seed),
    )
    atoms_per_monomer = int(len(atoms) // n_mol)
    report["pack_s"] = time.perf_counter() - t_pack
    report["n_atoms"] = int(len(atoms))
    report["atoms_per_monomer"] = atoms_per_monomer
    report["density_g_cm3"] = mass_density_g_cm3(residue, n_mol, args.box_size)
    ase_write(str(out / "box_initial.xyz"), atoms)

    print(
        f"{residue}:{n_mol}  N={len(atoms)}  L={args.box_size:.1f} Å  "
        f"ρ={report['density_g_cm3']:.4f} g/cm³  T={args.temperature:.1f} K  "
        f"dt={args.dt_fs} fs  ensemble={args.ensemble}"
    )

    t_load = time.perf_counter()
    atoms.calc = load_metatomic_calculator(
        ckpt,
        extra_kwargs={"non_conservative": False},
    )
    report["load_s"] = time.perf_counter() - t_load

    t0 = time.perf_counter()
    e0 = float(atoms.get_potential_energy())
    f0 = np.asarray(atoms.get_forces(), dtype=float)
    report["first_eval_s"] = time.perf_counter() - t0
    report["E0_eV"] = e0
    report["F0_max_eVA"] = float(np.max(np.linalg.norm(f0, axis=1)))
    print(
        f"E0={e0:.6f} eV  |F|_max={report['F0_max_eVA']:.4f} eV/Å  "
        f"first_eval={report['first_eval_s']:.2f} s"
    )
    if not np.isfinite(e0) or not np.all(np.isfinite(f0)):
        print("FAIL: non-finite energy or forces on the initial box", file=sys.stderr)
        _write()
        return 1

    if int(args.minimize_steps) > 0:
        from ase.optimize import FIRE

        atoms.set_momenta(np.zeros((len(atoms), 3)))
        t_mini = time.perf_counter()
        opt = FIRE(atoms, logfile=str(out / "fire.log"), maxstep=0.1)
        opt.run(fmax=float(args.minimize_fmax), steps=int(args.minimize_steps))
        report["minimize_s"] = time.perf_counter() - t_mini
        report["minimize_steps_run"] = int(opt.get_number_of_steps())
        e_mini = float(atoms.get_potential_energy())
        f_mini = np.asarray(atoms.get_forces(), dtype=float)
        report["E_mini_eV"] = e_mini
        report["F_mini_max_eVA"] = float(np.max(np.linalg.norm(f_mini, axis=1)))
        ase_write(str(out / "box_minimized.xyz"), atoms)
        print(
            f"FIRE {report['minimize_steps_run']} steps: "
            f"E={e_mini:.6f} eV  |F|_max={report['F_mini_max_eVA']:.4f} eV/Å  "
            f"({report['minimize_s']:.1f} s)"
        )
        if not np.isfinite(e_mini) or not np.all(np.isfinite(f_mini)):
            print("FAIL: non-finite energy or forces after FIRE", file=sys.stderr)
            _write()
            return 1

    rng = np.random.default_rng(int(args.seed))
    thermalize_momenta(
        atoms,
        temperature_K=float(args.temperature),
        rng=rng,
    )
    Stationary(atoms)

    if args.ensemble == "nvt":
        from ase.md.langevin import Langevin

        atoms.set_constraint(FixCom())
        dyn = Langevin(
            atoms,
            timestep=float(args.dt_fs) * units.fs,
            temperature_K=float(args.temperature),
            friction=float(args.friction),
            fixcm=False,
            rng=rng,
        )
    else:
        dyn = VelocityVerlet(atoms, timestep=float(args.dt_fs) * units.fs)

    rows: list[dict[str, float]] = [
        energy_snapshot(atoms, step=0, dt_fs=float(args.dt_fs))
    ]
    report["T0_K"] = rows[0]["T_K"]
    print(
        f"MD start: Etot={rows[0]['Etot_eV']:.6f} eV  "
        f"T={rows[0]['T_K']:.1f} K  |F|_max={rows[0]['Fmax_eVA']:.4f} eV/Å"
    )

    def _log() -> None:
        step = int(dyn.get_number_of_steps())
        if step % log_every != 0:
            return
        row = energy_snapshot(atoms, step=step, dt_fs=float(args.dt_fs))
        rows.append(row)
        if step == 0 or step % max(20, log_every) == 0:
            print(
                f"  step={step:5d}  t={row['time_fs']:.2f} fs  "
                f"Etot={row['Etot_eV']:.6f} eV  T={row['T_K']:.1f} K  "
                f"|F|_max={row['Fmax_eVA']:.4f} eV/Å"
            )

    dyn.attach(_log, interval=log_every)
    t_md = time.perf_counter()
    dyn.run(int(args.n_steps))
    report["md_s"] = time.perf_counter() - t_md
    if int(rows[-1]["step"]) != int(args.n_steps):
        rows.append(
            energy_snapshot(atoms, step=int(args.n_steps), dt_fs=float(args.dt_fs))
        )

    last = rows[-1]
    report["E1_eV"] = last["PE_eV"]
    report["F1_max_eVA"] = last["Fmax_eVA"]
    report["T1_K"] = last["T_K"]
    report["md_ps"] = float(args.n_steps) * float(args.dt_fs) * 1.0e-3
    energy_csv = out / "energy.csv"
    write_energy_csv(energy_csv, rows)
    report["energy_csv"] = str(energy_csv)
    ase_write(str(out / "box_final.xyz"), atoms)

    if str(args.ensemble) == "nve" and len(rows) >= 2:
        stats = nve_conservation_stats(
            np.array([row["time_fs"] * 1.0e-3 for row in rows], dtype=float),
            np.array([row["Etot_eV"] for row in rows], dtype=float),
            n_atoms=int(len(atoms)),
        )
        report.update(stats)
        plot_path = out / "nve_energy.png"
        try:
            plot_nve_energy(rows, plot_path)
            report["nve_plot"] = str(plot_path)
        except Exception as exc:  # plotting must not fail the MD
            report["nve_plot_error"] = str(exc)
        print(
            f"NVE conservation: drift={stats['etot_drift_eV']:.6f} eV "
            f"({stats['drift_eV_per_ps']:.4f} eV/ps, "
            f"{stats['drift_meV_per_atom_ps']:.4g} meV/atom/ps)  "
            f"span={stats['etot_span_eV']:.6f} eV  "
            f"std={stats['etot_std_eV']:.6f} eV  "
            f"over {stats['time_span_ps']:.4f} ps"
        )

    print(
        f"E1={last['PE_eV']:.6f} eV  |F|_max={last['Fmax_eVA']:.4f} eV/Å  "
        f"T1={last['T_K']:.1f} K  md={report['md_s']:.2f} s  "
        f"({args.n_steps} × {args.dt_fs} fs)"
    )
    if not np.isfinite(last["PE_eV"]) or not np.isfinite(last["Etot_eV"]):
        print("FAIL: non-finite energy after MD", file=sys.stderr)
        _write()
        return 1
    if last["T_K"] > 5000.0:
        print(f"FAIL: temperature exploded ({last['T_K']:.1f} K)", file=sys.stderr)
        _write()
        return 1

    expected_n = atoms_per_monomer * n_mol
    if int(len(atoms)) != expected_n:
        print(
            f"FAIL: n_atoms={len(atoms)} expected {expected_n} "
            f"({atoms_per_monomer} atoms/{residue})",
            file=sys.stderr,
        )
        _write()
        return 1
    report["ok"] = True
    _write()
    print(f"PASS: metatomic {residue} PBC MD")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
