"""NVE / NVT timestep sweep for PET (metatomic ASE) on the 32 Å ethanol box.

One FIRE-minimized ETOH:338 box is shared by every run; each (ensemble, dt)
starts from it with the same 300 K velocities, runs ``SWEEP_PS`` ps, and writes
``<ensemble>_dt<dt>fs/energy.csv``. ``summary.json`` + ``sweep_*.png`` go to
``SWEEP_OUT`` (and are copied to ``SWEEP_MEDIA`` when set).

Env::

    PET_MAD_CKPT   metatomic .pt (required)
    SWEEP_OUT      output dir (default scratch/pet_mad_etoh_pbc/ase_sweep)
    SWEEP_MEDIA    optional dir for a copy of the figures + summary
    SWEEP_DTS      comma list of fs (default 0.25,0.5,1.0,1.5,2.0)
    SWEEP_ENSEMBLES  default nve,nvt
    SWEEP_PS       simulated time per run (default 0.5)
    SWEEP_BOX_A    box side (default 32)
    MMML_METATOMIC_DEVICE  cuda / cpu
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from ase import units
from ase.constraints import FixCom
from ase.io import write as ase_write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import Stationary, thermalize_momenta
from ase.md.verlet import VelocityVerlet
from ase.optimize import FIRE

from mmml.interfaces.calculators.metatomic import load_metatomic_calculator
from mmml.md.metatomic_pbc import (
    build_tiled_cubic_liquid,
    default_etoh_monomer_xyz,
    energy_snapshot,
    mass_density_g_cm3,
    n_molecules_for_residue_box,
    nve_conservation_stats,
    write_energy_csv,
)

TEMPERATURE_K = 300.0
FRICTION_PER_FS = 0.01
SEED = 42


def _env_list(name: str, default: str) -> list[str]:
    return [x.strip() for x in os.environ.get(name, default).split(",") if x.strip()]


def _run_one(atoms0, calc, *, ensemble: str, dt_fs: float, n_steps: int, out: Path) -> dict:
    atoms = atoms0.copy()
    atoms.calc = calc
    rng = np.random.default_rng(SEED)
    thermalize_momenta(atoms, temperature_K=TEMPERATURE_K, rng=rng)
    Stationary(atoms)
    if ensemble == "nvt":
        atoms.set_constraint(FixCom())
        dyn = Langevin(
            atoms,
            timestep=dt_fs * units.fs,
            temperature_K=TEMPERATURE_K,
            friction=FRICTION_PER_FS,
            fixcm=False,
            rng=rng,
        )
    else:
        dyn = VelocityVerlet(atoms, timestep=dt_fs * units.fs)
    rows = [energy_snapshot(atoms, step=0, dt_fs=dt_fs)]

    def _log() -> None:
        step = int(dyn.get_number_of_steps())
        if step > 0:
            rows.append(energy_snapshot(atoms, step=step, dt_fs=dt_fs))

    dyn.attach(_log, interval=1)
    t0 = time.perf_counter()
    dyn.run(n_steps)
    wall = time.perf_counter() - t0
    out.mkdir(parents=True, exist_ok=True)
    write_energy_csv(out / "energy.csv", rows)
    ase_write(str(out / "final.xyz"), atoms)
    etot = np.array([r["Etot_eV"] for r in rows])
    temp = np.array([r["T_K"] for r in rows])
    rec = {
        "ensemble": ensemble,
        "dt_fs": dt_fs,
        "n_steps": n_steps,
        "n_atoms": len(atoms),
        "wall_s": wall,
        "ms_per_step": 1.0e3 * wall / max(n_steps, 1),
        "finite": bool(np.all(np.isfinite(etot))),
        "T_mean_K": float(np.mean(temp[len(temp) // 2 :])),
        "T_final_K": float(temp[-1]),
        "Fmax_final_eVA": float(rows[-1]["Fmax_eVA"]),
        "energy_csv": str(out / "energy.csv"),
    }
    if ensemble == "nve" and rec["finite"]:
        rec.update(
            nve_conservation_stats(
                np.array([r["time_fs"] for r in rows]) * 1.0e-3,
                etot,
                n_atoms=len(atoms),
            )
        )
    return rec


def _stable(rec: dict) -> bool:
    return bool(rec["finite"]) and rec["T_final_K"] < 3.0 * TEMPERATURE_K


def _plot(records: list[dict], out: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import csv

    import matplotlib.pyplot as plt

    paths = []
    for ens in sorted({r["ensemble"] for r in records}):
        recs = sorted((r for r in records if r["ensemble"] == ens), key=lambda r: r["dt_fs"])
        unstable = [f"{r['dt_fs']:g}" for r in recs if not _stable(r)]
        recs = [r for r in recs if _stable(r)]
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7.2, 6.4), constrained_layout=True)
        for i, r in enumerate(recs):
            with open(r["energy_csv"]) as fh:
                rows = list(csv.DictReader(fh))
            t = np.array([float(x["time_fs"]) for x in rows])
            et = np.array([float(x["Etot_eV"]) for x in rows])
            tk = np.array([float(x["T_K"]) for x in rows])
            label = f"{r['dt_fs']:g} fs"
            axes[0].plot(t, et - et[0], color=f"C{i}", label=label, lw=1)
            axes[1].plot(t, tk, color=f"C{i}", lw=1)
        axes[0].set_ylabel(r"$E_\mathrm{tot}-E_\mathrm{tot}(0)$ / eV")
        axes[1].set_ylabel("T / K")
        axes[1].set_xlabel("t / fs")
        axes[0].legend(frameon=False, ncol=3)
        title = f"PET ETOH 32 Å, {ens.upper()}"
        if unstable:
            title += f"  (unstable, not shown: dt = {', '.join(unstable)} fs)"
        axes[0].set_title(title)
        path = out / f"sweep_{ens}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)
    nve = sorted((r for r in records if r["ensemble"] == "nve" and _stable(r)),
                 key=lambda r: r["dt_fs"])
    if nve:
        fig, ax = plt.subplots(figsize=(5.0, 3.6), constrained_layout=True)
        dts = [r["dt_fs"] for r in nve]
        ax.plot(dts, [abs(r["drift_meV_per_atom_ps"]) for r in nve], "o-", label="|drift| (meV/atom/ps)")
        ax.plot(dts, [1.0e3 * r["etot_std_eV"] / r["n_atoms"] for r in nve], "s--",
                label="std(Etot)/N (meV/atom)")
        ax.set_yscale("log")
        ax.set_xlabel("dt / fs")
        ax.set_ylabel("meV/atom  (drift: meV/atom/ps)")
        ax.legend(frameon=False)
        path = out / "sweep_nve_drift.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)
    return paths


def main() -> int:
    ckpt = os.environ.get("PET_MAD_CKPT", "").strip()
    if not ckpt or not Path(ckpt).is_file():
        print("set PET_MAD_CKPT to a metatomic .pt", file=sys.stderr)
        return 2
    out = Path(os.environ.get("SWEEP_OUT", "scratch/pet_mad_etoh_pbc/ase_sweep")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    dts = [float(x) for x in _env_list("SWEEP_DTS", "0.25,0.5,1.0,1.5,2.0")]
    ensembles = _env_list("SWEEP_ENSEMBLES", "nve,nvt")
    total_ps = float(os.environ.get("SWEEP_PS", "0.5"))
    box = float(os.environ.get("SWEEP_BOX_A", "32"))

    n_mol = n_molecules_for_residue_box("ETOH", box_side_A=box)
    atoms = build_tiled_cubic_liquid(
        monomer_xyz=default_etoh_monomer_xyz(), box_side_A=box, n_molecules=n_mol, seed=SEED
    )
    calc = load_metatomic_calculator(ckpt, extra_kwargs={"non_conservative": False})
    atoms.calc = calc
    print(f"ETOH:{n_mol} N={len(atoms)} L={box} Å rho={mass_density_g_cm3('ETOH', n_mol, box):.4f}")
    t0 = time.perf_counter()
    FIRE(atoms, logfile=str(out / "fire.log"), maxstep=0.1).run(fmax=0.2, steps=100)
    print(f"FIRE done in {time.perf_counter() - t0:.1f} s  E={atoms.get_potential_energy():.4f} eV")
    ase_write(str(out / "box_minimized.xyz"), atoms)

    records = []
    for ens in ensembles:
        for dt in dts:
            n_steps = int(round(total_ps * 1.0e3 / dt))
            rec = _run_one(atoms, calc, ensemble=ens, dt_fs=dt, n_steps=n_steps,
                           out=out / f"{ens}_dt{dt:g}fs")
            records.append(rec)
            drift = rec.get("drift_meV_per_atom_ps")
            print(
                f"{ens} dt={dt:g} fs  steps={n_steps}  {rec['ms_per_step']:.1f} ms/step  "
                f"<T>={rec['T_mean_K']:.1f} K"
                + (f"  drift={drift:.4g} meV/atom/ps" if drift is not None else ""),
                flush=True,
            )
            (out / "summary.json").write_text(json.dumps(records, indent=2) + "\n")

    figs = _plot(records, out)
    media = os.environ.get("SWEEP_MEDIA", "").strip()
    if media:
        dest = Path(media)
        dest.mkdir(parents=True, exist_ok=True)
        for p in [*figs, out / "summary.json"]:
            shutil.copy2(p, dest / p.name)
    print(f"Wrote {out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
