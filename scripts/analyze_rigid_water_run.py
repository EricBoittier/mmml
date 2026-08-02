#!/usr/bin/env python3
"""NVE conservation gate and virial distributions for a rigid-water run.

Three questions, one pass over the trajectory:

1. Does E_tot drift?  With no thermostat, drift means the forces are not the
   gradient of the energy. This is the gate: no density number should be
   believed from a potential that fails it.

2. Do the constraints actually hold?  O-H and H-H over the whole run, against
   the values they were constrained to.

3. How does the virial split?  For a constrained system the atomic virial
   ``sum_i r_i . f_i`` and the molecular virial ``sum_mol R_com . F_mol`` differ
   by the internal/constraint contribution. Only the molecular form belongs in
   the pressure -- constraint forces do no work and must not appear in P. The
   NpT virial was previously measured as identically zero, so this reports the
   full distribution of each rather than a single number.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

KB_EV = 8.617333262e-5
EV_TO_KCAL = 23.060548
# eV/A^3 -> atm
EV_A3_TO_ATM = 1.602176634e-19 / 1e-30 / 101325.0


def _percentiles(x: np.ndarray) -> dict:
    q = np.percentile(x, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    return {
        "min": float(q[0]), "p01": float(q[1]), "p05": float(q[2]),
        "p25": float(q[3]), "median": float(q[4]), "p75": float(q[5]),
        "p95": float(q[6]), "p99": float(q[7]), "max": float(q[8]),
        "mean": float(x.mean()), "std": float(x.std()),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traj", type=Path, required=True)
    p.add_argument("--n-monomers", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--drift-threshold-meV-per-atom", type=float, default=1.0)
    args = p.parse_args()

    from ase.io import Trajectory

    traj = Trajectory(str(args.traj))
    n_frames = len(traj)
    if n_frames < 3:
        raise SystemExit(f"need >=3 frames to judge drift, got {n_frames}")

    a0 = traj[0]
    n_atoms = len(a0)
    n_mol = int(args.n_monomers)
    if n_atoms != 3 * n_mol:
        raise SystemExit(f"expected {3 * n_mol} atoms for {n_mol} waters, got {n_atoms}")
    cell = np.asarray(a0.cell.lengths())
    volume = float(np.prod(cell))

    e_pot, e_kin, temperature = [], [], []
    w_atomic, w_molecular, w_internal = [], [], []
    oh, hh = [], []

    masses = np.asarray(a0.get_masses()).reshape(n_mol, 3, 1)
    mass_tot = masses.sum(axis=1)

    for frame in traj:
        r = np.asarray(frame.get_positions()).reshape(n_mol, 3, 3)
        try:
            ep = float(frame.get_potential_energy())
            f = np.asarray(frame.get_forces()).reshape(n_mol, 3, 3)
        except Exception:
            continue
        ek = float(frame.get_kinetic_energy())
        e_pot.append(ep)
        e_kin.append(ek)
        temperature.append(2.0 * ek / (3.0 * n_atoms * KB_EV))

        # Virial split. Molecular form uses the COM and the net molecular force,
        # so anything that sums to zero within a molecule drops out of it.
        wa = float(np.sum(r * f))
        com = (r * masses).sum(axis=1) / mass_tot
        wm = float(np.sum(com * f.sum(axis=1)))
        w_atomic.append(wa)
        w_molecular.append(wm)
        w_internal.append(wa - wm)

        oh.append(np.linalg.norm(r[:, 1] - r[:, 0], axis=-1))
        oh.append(np.linalg.norm(r[:, 2] - r[:, 0], axis=-1))
        hh.append(np.linalg.norm(r[:, 2] - r[:, 1], axis=-1))

    e_pot = np.asarray(e_pot)
    e_kin = np.asarray(e_kin)
    e_tot = e_pot + e_kin
    temperature = np.asarray(temperature)
    oh = np.concatenate(oh)
    hh = np.concatenate(hh)
    w_atomic = np.asarray(w_atomic)
    w_molecular = np.asarray(w_molecular)
    w_internal = np.asarray(w_internal)

    drift_total_eV = float(e_tot[-1] - e_tot[0])
    drift_meV_per_atom = 1000.0 * drift_total_eV / n_atoms
    # Peak-to-peak is the honest number: a run can end where it started while
    # having swung wildly in between.
    ptp_meV_per_atom = 1000.0 * float(e_tot.max() - e_tot.min()) / n_atoms
    passed = abs(drift_meV_per_atom) < args.drift_threshold_meV_per_atom

    # Pressure from the molecular virial only.
    p_kin = 2.0 * e_kin.mean() / (3.0 * volume)
    p_vir = w_molecular.mean() / (3.0 * volume)
    pressure_atm = float((p_kin + p_vir) * EV_A3_TO_ATM)

    report = {
        "trajectory": str(args.traj),
        "n_frames_scored": int(len(e_tot)),
        "n_atoms": n_atoms,
        "n_molecules": n_mol,
        "box_A": cell.tolist(),
        "volume_A3": volume,
        "nve_gate": {
            "passed": bool(passed),
            "threshold_meV_per_atom": args.drift_threshold_meV_per_atom,
            "drift_meV_per_atom": drift_meV_per_atom,
            "peak_to_peak_meV_per_atom": ptp_meV_per_atom,
            "E_tot_first_eV": float(e_tot[0]),
            "E_tot_last_eV": float(e_tot[-1]),
            "E_pot_drop_kcal_per_water": float(
                (e_pot[-1] - e_pot[0]) / n_mol * EV_TO_KCAL
            ),
        },
        "temperature_K": _percentiles(temperature),
        "constraints": {
            "OH_A": _percentiles(oh),
            "HH_A": _percentiles(hh),
            "OH_max_abs_dev_A": float(np.abs(oh - 0.9572).max()),
        },
        "virial_eV": {
            "atomic": _percentiles(w_atomic),
            "molecular": _percentiles(w_molecular),
            "internal": _percentiles(w_internal),
            "internal_fraction_of_atomic": float(
                np.abs(w_internal).mean() / max(np.abs(w_atomic).mean(), 1e-30)
            ),
        },
        "pressure_from_molecular_virial_atm": pressure_atm,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    np.savez(
        args.output.with_suffix(".npz"),
        e_pot=e_pot, e_kin=e_kin, e_tot=e_tot, temperature=temperature,
        w_atomic=w_atomic, w_molecular=w_molecular, w_internal=w_internal,
        oh=oh, hh=hh,
    )

    g = report["nve_gate"]
    print(f"NVE gate: {'PASS' if passed else 'FAIL'}")
    print(f"  drift        {g['drift_meV_per_atom']:+10.4f} meV/atom "
          f"(threshold {args.drift_threshold_meV_per_atom})")
    print(f"  peak-to-peak {g['peak_to_peak_meV_per_atom']:10.4f} meV/atom")
    print(f"  E_pot change {g['E_pot_drop_kcal_per_water']:+10.2f} kcal/mol per water")
    print(f"\nT (K)   median {report['temperature_K']['median']:8.1f}  "
          f"min {report['temperature_K']['min']:.1f}  max {report['temperature_K']['max']:.1f}")
    c = report["constraints"]
    print(f"O-H (A) min {c['OH_A']['min']:.5f}  max {c['OH_A']['max']:.5f}  "
          f"max|dev| {c['OH_max_abs_dev_A']:.2e}")
    print(f"H-H (A) min {c['HH_A']['min']:.5f}  max {c['HH_A']['max']:.5f}")
    v = report["virial_eV"]
    print("\nvirial (eV)      median        p05        p95        std")
    for name in ("atomic", "molecular", "internal"):
        d = v[name]
        print(f"  {name:10s} {d['median']:11.3f} {d['p05']:10.3f} "
              f"{d['p95']:10.3f} {d['std']:10.3f}")
    print(f"  |internal| / |atomic| = {v['internal_fraction_of_atomic']:.4f}")
    print(f"\npressure (molecular virial) = {pressure_atm:.1f} atm")
    print(f"\nwrote {args.output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
