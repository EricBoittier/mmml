#!/usr/bin/env python3
"""Equilibrated liquid density from an NpT trajectory.

The point of the exercise: a Packmol box is built *at* a target density, so its
density is an input, not a measurement. Only an NpT run at 1 atm lets the box
find its own volume, and only then does the density test the potential. This
reads the ``density_g_cm3`` series ``mmml md-system`` records for NpT runs and
reports the equilibrated value with an honest error bar.

Three things this refuses to do quietly:

* **Report a mean over an unequilibrated trace.** The production window is
  checked for drift with a least-squares slope and its standard error; a slope
  that is significant at 2 sigma is reported as NOT EQUILIBRATED rather than
  averaged away.
* **Use the naive standard error.** NpT density is strongly autocorrelated;
  the naive SEM understates the uncertainty by roughly sqrt(2 tau / dt). Error
  bars come from block averaging.
* **Compare against a reference at the wrong temperature.** Density is strongly
  temperature dependent, so a reference quoted more than 5 K away from the run
  temperature is flagged.

Example::

    python scripts/analyze_npt_density.py --traj tip3_npt.h5 --species TIP3 \\
        --discard-frac 0.4 -o density_tip3.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Experimental liquid density (g/cm3) at the stated temperature (K).
# Ammonia is quoted at its normal boiling point: it is a gas at 298 K, so a
# 298 K "liquid" ammonia box is not a liquid and its density is meaningless.
REFERENCES: dict[str, tuple[float, float]] = {
    "TIP3": (0.99705, 298.15),
    "MEOH": (0.78660, 298.15),
    "AMM1": (0.68190, 239.82),
    "AR1": (1.37860, 90.0),  # NIST sat. liquid argon
}

# g/mol for density from volume when the traj lacks density_g_cm3 (npz).
_MOLAR_MASS: dict[str, float] = {
    "TIP3": 18.01528,
    "MEOH": 32.042,
    "AMM1": 17.031,
    "AR1": 39.948,
}
_AVOGADRO = 6.02214076e23


def block_average_sem(x: np.ndarray, n_blocks: int = 10) -> float:
    """Standard error from block means (autocorrelation-aware)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < n_blocks * 2:
        n_blocks = max(2, x.size // 2)
    trim = (x.size // n_blocks) * n_blocks
    blocks = x[:trim].reshape(n_blocks, -1).mean(axis=1)
    if blocks.size < 2:
        return float("nan")
    return float(np.std(blocks, ddof=1) / np.sqrt(blocks.size))


def drift(t: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares slope of ``y`` vs ``t`` and its standard error."""
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = t.size
    if n < 3:
        return float("nan"), float("nan")
    tm, ym = t.mean(), y.mean()
    stt = float(((t - tm) ** 2).sum())
    if stt <= 0:
        return float("nan"), float("nan")
    slope = float(((t - tm) * (y - ym)).sum() / stt)
    resid = y - (ym + slope * (t - tm))
    s2 = float((resid**2).sum() / (n - 2))
    return slope, float(np.sqrt(s2 / stt))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--traj",
        type=Path,
        required=True,
        help="NpT trajectory HDF5 or jaxmd-unified trajectory.npz",
    )
    ap.add_argument("--species", type=str, default=None, help="RESI for the reference")
    ap.add_argument("--reference", type=float, default=None,
                    help="experimental density g/cm3 (overrides --species)")
    ap.add_argument("--temperature", type=float, default=None,
                    help="run temperature K (default: read from the file attrs)")
    ap.add_argument("--discard-frac", type=float, default=0.4,
                    help="leading fraction treated as equilibration")
    ap.add_argument("--n-blocks", type=int, default=10)
    ap.add_argument("-o", "--output", type=Path, default=None)
    a = ap.parse_args(argv)

    attrs: dict = {}
    if a.traj.suffix == ".npz":
        z = np.load(a.traj, allow_pickle=True)
        if "volumes_A3" not in z.files:
            raise SystemExit(
                f"{a.traj} has no volumes_A3; cannot form a density series"
            )
        if not a.species or a.species.upper() not in _MOLAR_MASS:
            raise SystemExit(
                "npz density needs --species with a known molar mass "
                f"(one of {sorted(_MOLAR_MASS)})"
            )
        V = np.asarray(z["volumes_A3"], dtype=np.float64)
        atoms_per = {"AR1": 1, "TIP3": 3, "MEOH": 6, "AMM1": 4}
        sp = a.species.upper()
        n_atoms = int(np.asarray(z["Z"]).shape[0])
        apm = atoms_per[sp]
        if n_atoms % apm != 0:
            raise SystemExit(
                f"{a.traj}: {n_atoms} atoms not divisible by {apm} for {sp}"
            )
        n_mol = n_atoms // apm
        mass = _MOLAR_MASS[sp]
        # g/cm3 = n_mol * M / (N_A * V_cm3); V_A3 * 1e-24 = cm3
        rho = n_mol * mass / (_AVOGADRO * V * 1e-24)
        if "times_ps" in z.files:
            time_ps = np.asarray(z["times_ps"], dtype=np.float64)
        elif "time_ps" in z.files:
            time_ps = np.asarray(z["time_ps"], dtype=np.float64)
        else:
            time_ps = np.arange(rho.size, dtype=np.float64)
        if "target_temperatures_K" in z.files:
            attrs["temperature_target"] = float(
                np.asarray(z["target_temperatures_K"], dtype=np.float64).ravel()[0]
            )
    else:
        import h5py

        with h5py.File(a.traj, "r") as fh:
            if "density_g_cm3" not in fh:
                raise SystemExit(
                    f"{a.traj} has no 'density_g_cm3'. That series is only recorded "
                    f"for NpT runs - an NVT/NVE trajectory cannot measure density "
                    f"because its volume is fixed by construction. "
                    f"Datasets present: {sorted(fh.keys())[:10]}"
                )
            rho = np.asarray(fh["density_g_cm3"], dtype=np.float64)
            time_ps = (
                np.asarray(fh["time_ps"], dtype=np.float64)
                if "time_ps" in fh
                else np.arange(rho.size, dtype=np.float64)
            )
            attrs = dict(fh.attrs)

    if rho.size == 0:
        raise SystemExit(f"{a.traj} contains no frames")
    finite = np.isfinite(rho)
    if not finite.all():
        raise SystemExit(
            f"{a.traj}: {int((~finite).sum())} of {rho.size} density values are "
            "non-finite - the barostat diverged; do not average this."
        )

    temperature = a.temperature
    if temperature is None:
        temperature = float(attrs.get("temperature_target", float("nan")))

    start = int(round(a.discard_frac * rho.size))
    prod_rho, prod_t = rho[start:], time_ps[start:]
    if prod_rho.size < 20:
        raise SystemExit(
            f"only {prod_rho.size} frames after discarding {a.discard_frac:.0%} - "
            "run longer or lower --discard-frac"
        )

    mean = float(prod_rho.mean())
    sem = block_average_sem(prod_rho, a.n_blocks)
    slope, slope_se = drift(prod_t, prod_rho)
    equilibrated = bool(np.isfinite(slope) and abs(slope) <= 2.0 * slope_se)

    ref, ref_T = None, None
    if a.reference is not None:
        ref = float(a.reference)
    elif a.species and a.species.upper() in REFERENCES:
        ref, ref_T = REFERENCES[a.species.upper()]

    report = {
        "trajectory": str(a.traj),
        "frames_total": int(rho.size),
        "frames_production": int(prod_rho.size),
        "discard_frac": a.discard_frac,
        "temperature_K": temperature,
        "density_g_cm3": mean,
        "density_sem_g_cm3": sem,
        "drift_g_cm3_per_ps": slope,
        "drift_sem_g_cm3_per_ps": slope_se,
        "equilibrated": equilibrated,
        "reference_g_cm3": ref,
        "reference_temperature_K": ref_T,
    }
    if ref is not None:
        report["error_g_cm3"] = mean - ref
        report["error_percent"] = 100.0 * (mean - ref) / ref
        if ref_T is not None and np.isfinite(temperature) and abs(ref_T - temperature) > 5.0:
            report["warning"] = (
                f"reference is quoted at {ref_T} K but the run is at "
                f"{temperature} K; density is strongly temperature dependent"
            )

    print(f"density = {mean:.5f} +/- {sem:.5f} g/cm3  "
          f"({prod_rho.size} frames, T = {temperature:.1f} K)")
    print(f"  drift  = {slope:+.3e} +/- {slope_se:.3e} g/cm3/ps  "
          f"-> {'EQUILIBRATED' if equilibrated else 'NOT EQUILIBRATED (2 sigma)'}")
    if ref is not None:
        print(f"  reference = {ref:.5f} g/cm3  -> error {mean - ref:+.5f} "
              f"({100.0 * (mean - ref) / ref:+.2f}%)")
    if "warning" in report:
        print(f"  WARNING: {report['warning']}")
    if not equilibrated:
        print("  Do not quote this density: the production window is still drifting.")

    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
