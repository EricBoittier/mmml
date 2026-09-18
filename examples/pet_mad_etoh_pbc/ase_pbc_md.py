#!/usr/bin/env python3
"""CHARMM-free PET-MAD periodic MD for liquid ethanol in a 32 Å cube.

Default: 338 ETOH (CGenFF count at 0.789 g/cm³), 300 K, 0.5 fs, PBC,
``--metatomic-eval-mode whole_system`` equivalent (one ASE eval per step).

NVE conservation: FIRE mini, Maxwell–Boltzmann at 300 K, VelocityVerlet,
per-step PE/KE/Etot log. Forces are autograd of the energy
(``non_conservative=False``).

Example::

    export PET_MAD_CKPT=/tmp/mmml-metatomic-models/pet-mad-xs-v1.5.0.pt
    JAX_PLATFORMS=cpu MMML_METATOMIC_DEVICE=cpu \\
      uv run python examples/pet_mad_etoh_pbc/ase_pbc_md.py \\
        --checkpoint \"$PET_MAD_CKPT\" --ensemble nve \\
        --minimize-steps 60 --minimize-fmax 0.2 --n-steps 400
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from ase import Atoms, units
from ase.constraints import FixCom
from ase.io import read as ase_read
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

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_MONOMER = EXAMPLE_DIR / "etoh.xyz"
DEFAULT_BOX_A = 32.0
DEFAULT_DT_FS = 0.5
DEFAULT_TEMPERATURE_K = 300.0
DEFAULT_N_STEPS = 5
ENERGY_CSV_FIELDS = (
    "step",
    "time_fs",
    "PE_eV",
    "KE_eV",
    "Etot_eV",
    "T_K",
    "Fmax_eVA",
)


def nve_conservation_stats(
    time_ps: np.ndarray,
    etot_ev: np.ndarray,
    *,
    n_atoms: int,
) -> dict[str, float]:
    """Linear Etot drift and fluctuation from an NVE trace (eV, ps)."""
    t = np.asarray(time_ps, dtype=float).reshape(-1)
    e = np.asarray(etot_ev, dtype=float).reshape(-1)
    if t.size < 2 or t.size != e.size:
        raise ValueError("time_ps and etot_ev must be 1-D and length >= 2")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(e)):
        raise ValueError("time_ps and etot_ev must be finite")
    span = float(t[-1] - t[0])
    if span <= 0.0:
        raise ValueError("time_ps must increase")
    slope = float(np.polyfit(t, e, 1)[0])
    e0 = float(e[0])
    e_mean = float(np.mean(e))
    n = max(int(n_atoms), 1)
    return {
        "etot_start_eV": e0,
        "etot_end_eV": float(e[-1]),
        "etot_mean_eV": e_mean,
        "etot_std_eV": float(np.std(e, ddof=0)),
        "etot_span_eV": float(np.max(e) - np.min(e)),
        "etot_drift_eV": float(e[-1] - e0),
        "drift_eV_per_ps": slope,
        "drift_meV_per_atom_ps": 1.0e3 * slope / float(n),
        "rel_drift_per_ps": slope / abs(e_mean) if e_mean != 0.0 else 0.0,
        "time_span_ps": span,
        "n_samples": float(t.size),
    }


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


def _snapshot(atoms: Atoms, *, step: int, dt_fs: float) -> dict[str, float]:
    pe = float(atoms.get_potential_energy())
    ke = float(atoms.get_kinetic_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)
    return {
        "step": int(step),
        "time_fs": float(step) * float(dt_fs),
        "PE_eV": pe,
        "KE_eV": ke,
        "Etot_eV": pe + ke,
        "T_K": float(atoms.get_temperature()),
        "Fmax_eVA": float(np.max(np.linalg.norm(forces, axis=1))),
    }


def _write_energy_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ENERGY_CSV_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in ENERGY_CSV_FIELDS})


def _plot_nve_energy(rows: list[dict[str, float]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time_fs = [row["time_fs"] for row in rows]
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7.2, 8.0), constrained_layout=True)
    axes[0].plot(time_fs, [row["Etot_eV"] for row in rows], color="C0")
    axes[0].set_ylabel(r"$E_\mathrm{tot}$ / eV")
    axes[1].plot(time_fs, [row["PE_eV"] for row in rows], label="PE", color="C1")
    axes[1].plot(time_fs, [row["KE_eV"] for row in rows], label="KE", color="C2")
    axes[1].set_ylabel("E / eV")
    axes[1].legend(frameon=False)
    axes[2].plot(time_fs, [row["T_K"] for row in rows], color="C3")
    axes[2].set_ylabel("T / K")
    axes[2].set_xlabel("t / fs")
    fig.savefig(path, dpi=140)
    plt.close(fig)


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
    log_every = max(int(args.log_every), 1)

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
        "minimize_steps": int(args.minimize_steps),
        "minimize_fmax_eVA": float(args.minimize_fmax),
        "non_conservative": False,
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
        ase_write(str(out / "etoh_box_minimized.xyz"), atoms)
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

    rows: list[dict[str, float]] = [_snapshot(atoms, step=0, dt_fs=float(args.dt_fs))]
    report["T0_K"] = rows[0]["T_K"]
    print(
        f"MD start: Etot={rows[0]['Etot_eV']:.6f} eV  "
        f"T={rows[0]['T_K']:.1f} K  |F|_max={rows[0]['Fmax_eVA']:.4f} eV/Å"
    )

    def _log() -> None:
        step = int(dyn.get_number_of_steps())
        if step % log_every != 0:
            return
        row = _snapshot(atoms, step=step, dt_fs=float(args.dt_fs))
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
        rows.append(_snapshot(atoms, step=int(args.n_steps), dt_fs=float(args.dt_fs)))

    last = rows[-1]
    report["E1_eV"] = last["PE_eV"]
    report["F1_max_eVA"] = last["Fmax_eVA"]
    report["T1_K"] = last["T_K"]
    report["md_ps"] = float(args.n_steps) * float(args.dt_fs) * 1.0e-3
    energy_csv = out / "energy.csv"
    _write_energy_csv(energy_csv, rows)
    report["energy_csv"] = str(energy_csv)
    ase_write(str(out / "etoh_box_final.xyz"), atoms)

    if str(args.ensemble) == "nve" and len(rows) >= 2:
        stats = nve_conservation_stats(
            np.array([row["time_fs"] * 1.0e-3 for row in rows], dtype=float),
            np.array([row["Etot_eV"] for row in rows], dtype=float),
            n_atoms=int(len(atoms)),
        )
        report.update(stats)
        plot_path = out / "nve_energy.png"
        try:
            _plot_nve_energy(rows, plot_path)
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
    print("PASS: PET-MAD ethanol PBC MD")
    return 0


if __name__ == "__main__":
    sys.exit(main())
