#!/usr/bin/env python3
"""Enthalpy of vaporisation from a liquid-box MD trajectory plus a gas reference.

Why this and not density: the trainable LJ scales change the *intermolecular*
energy directly, and dH_vap is that quantity's most direct thermodynamic
observable. Density responds to it only through the equation of state, so it is
the weaker discriminator. Both sit outside the training loss, which is the point
(docs/des-lj-scales-handoff.md, "Validate on something outside the loss").

    dH_vap = <U>_gas  -  <U>_liquid / N_molecules  +  RT

The ``RT`` is the ideal-gas ``PV`` term; ``P V_liquid`` is ~0.1% of dH_vap at
ambient pressure and is dropped. This is the flexible-molecule form: both legs
carry their intramolecular energy and it cancels only to the extent the liquid
does not distort the monomer. Do not substitute an intermolecular-only liquid
energy here unless the gas leg is dropped to zero as well -- mixing the two
conventions double-counts the intramolecular term.

Statistical error uses **block averaging**, not the naive standard error: MD
energies are strongly autocorrelated and the naive SEM understates the
uncertainty by roughly sqrt(2 tau / dt), which for a liquid box is an order of
magnitude.

Trajectory format is the one ``mmml md-system`` writes (see
``scripts/analyze_jaxmd_nve_energy.py``): HDF5 with ``potential_energy`` in eV
and an ``n_atoms`` attribute.

Example::

    python scripts/analyze_enthalpy_vaporization.py \\
        --liquid artifacts/.../tip3_nvt.h5 --gas artifacts/.../tip3_gas.h5 \\
        --atoms-per-molecule 3 --temperature 298.0 --reference 10.51 \\
        -o artifacts/.../hvap_tip3.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

EV_TO_KCAL_MOL = 23.060547830619026
GAS_CONSTANT_KCAL_MOL_K = 1.987204258640832e-3

# Reference dH_vap (kcal/mol) at the stated temperature. Ammonia is quoted at
# its normal boiling point because it is a gas at 298 K -- a 298 K "liquid"
# ammonia box is not a liquid and its dH_vap is meaningless.
REFERENCES: dict[str, tuple[float, float]] = {
    "TIP3": (10.51, 298.15),
    "MEOH": (8.95, 298.15),
    "AMM1": (5.58, 239.82),
}


def read_potential_ev(path: Path, discard_frac: float) -> tuple[np.ndarray, int]:
    """Return (potential energies in eV after equilibration, n_atoms)."""
    import h5py

    with h5py.File(path, "r") as fh:
        if "potential_energy" not in fh:
            raise KeyError(
                f"{path} has no 'potential_energy' dataset; got {sorted(fh.keys())[:8]}"
            )
        pot = np.asarray(fh["potential_energy"], dtype=np.float64)
        n_atoms = int(fh.attrs.get("n_atoms", 0))

    if pot.size == 0:
        raise RuntimeError(f"{path} contains no frames")
    if not np.all(np.isfinite(pot)):
        raise RuntimeError(f"{path} contains non-finite potential energies")

    start = int(round(discard_frac * pot.size))
    kept = pot[start:]
    if kept.size < 10:
        raise RuntimeError(
            f"{path}: only {kept.size} frames after discarding {discard_frac:.0%} "
            "for equilibration -- run longer or lower --discard-frac"
        )
    return kept, n_atoms


def block_average_sem(x: np.ndarray, n_blocks: int = 10) -> float:
    """Standard error from block means.

    Autocorrelated samples make the naive SEM far too small. Splitting into
    blocks long compared with the correlation time and taking the SEM *of the
    block means* restores an honest error bar.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size < n_blocks * 2:
        n_blocks = max(2, x.size // 2)
    trim = (x.size // n_blocks) * n_blocks
    blocks = x[:trim].reshape(n_blocks, -1).mean(axis=1)
    if blocks.size < 2:
        return float("nan")
    return float(np.std(blocks, ddof=1) / np.sqrt(blocks.size))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--liquid", type=Path, required=True, help="liquid-box trajectory HDF5")
    ap.add_argument("--gas", type=Path, default=None,
                    help="isolated-monomer trajectory HDF5 (same T)")
    ap.add_argument("--gas-energy-kcal", type=float, default=None,
                    help="alternative to --gas: <U>_gas per molecule, kcal/mol")
    ap.add_argument("--atoms-per-molecule", type=int, required=True)
    ap.add_argument("--temperature", type=float, required=True, help="K")
    ap.add_argument("--discard-frac", type=float, default=0.3,
                    help="leading fraction of each trajectory treated as equilibration")
    ap.add_argument("--n-blocks", type=int, default=10)
    ap.add_argument("--species", type=str, default=None,
                    help="RESI name; looks up a built-in reference (TIP3/MEOH/AMM1)")
    ap.add_argument("--reference", type=float, default=None,
                    help="experimental dH_vap in kcal/mol (overrides --species)")
    ap.add_argument("-o", "--output", type=Path, default=None)
    a = ap.parse_args(argv)

    if a.gas is None and a.gas_energy_kcal is None:
        ap.error("supply --gas or --gas-energy-kcal for the gas-phase leg")

    liq_ev, n_atoms = read_potential_ev(a.liquid, a.discard_frac)
    if n_atoms <= 0:
        ap.error(f"{a.liquid} has no usable 'n_atoms' attribute")
    if n_atoms % a.atoms_per_molecule:
        ap.error(
            f"{n_atoms} atoms is not divisible by --atoms-per-molecule "
            f"{a.atoms_per_molecule}; wrong species or a mixed box?"
        )
    n_mol = n_atoms // a.atoms_per_molecule

    liq_kcal_per_mol = liq_ev * EV_TO_KCAL_MOL / n_mol
    u_liq = float(np.mean(liq_kcal_per_mol))
    u_liq_sem = block_average_sem(liq_kcal_per_mol, a.n_blocks)

    if a.gas is not None:
        gas_ev, gas_atoms = read_potential_ev(a.gas, a.discard_frac)
        if gas_atoms not in (0, a.atoms_per_molecule):
            ap.error(
                f"gas trajectory has {gas_atoms} atoms; expected a single "
                f"monomer of {a.atoms_per_molecule}"
            )
        gas_kcal = gas_ev * EV_TO_KCAL_MOL
        u_gas = float(np.mean(gas_kcal))
        u_gas_sem = block_average_sem(gas_kcal, a.n_blocks)
    else:
        u_gas = float(a.gas_energy_kcal)
        u_gas_sem = 0.0

    rt = GAS_CONSTANT_KCAL_MOL_K * a.temperature
    hvap = u_gas - u_liq + rt
    hvap_sem = float(np.hypot(u_gas_sem, u_liq_sem))

    ref, ref_T = (None, None)
    if a.reference is not None:
        ref = float(a.reference)
    elif a.species and a.species.upper() in REFERENCES:
        ref, ref_T = REFERENCES[a.species.upper()]

    report = {
        "liquid_trajectory": str(a.liquid),
        "gas_trajectory": str(a.gas) if a.gas else None,
        "n_atoms": n_atoms,
        "n_molecules": n_mol,
        "atoms_per_molecule": a.atoms_per_molecule,
        "temperature_K": a.temperature,
        "frames_used_liquid": int(liq_kcal_per_mol.size),
        "discard_frac": a.discard_frac,
        "u_liquid_per_molecule_kcal": u_liq,
        "u_liquid_sem_kcal": u_liq_sem,
        "u_gas_kcal": u_gas,
        "u_gas_sem_kcal": u_gas_sem,
        "RT_kcal": rt,
        "dH_vap_kcal_mol": hvap,
        "dH_vap_sem_kcal_mol": hvap_sem,
        "reference_kcal_mol": ref,
        "reference_temperature_K": ref_T,
    }
    if ref is not None:
        report["error_kcal_mol"] = hvap - ref
        report["error_percent"] = 100.0 * (hvap - ref) / ref
        if ref_T is not None and abs(ref_T - a.temperature) > 5.0:
            report["warning"] = (
                f"reference is quoted at {ref_T} K but the run is at "
                f"{a.temperature} K; dH_vap is strongly temperature dependent"
            )

    print(f"dH_vap = {hvap:.3f} +/- {hvap_sem:.3f} kcal/mol  "
          f"(T={a.temperature} K, N={n_mol} molecules)")
    print(f"  <U>_liquid/molecule = {u_liq:.3f} +/- {u_liq_sem:.3f} kcal/mol")
    print(f"  <U>_gas             = {u_gas:.3f} +/- {u_gas_sem:.3f} kcal/mol")
    print(f"  RT                  = {rt:.3f} kcal/mol")
    if ref is not None:
        print(f"  reference           = {ref:.3f} kcal/mol  "
              f"-> error {hvap - ref:+.3f} ({100.0 * (hvap - ref) / ref:+.1f}%)")
    if "warning" in report:
        print(f"  WARNING: {report['warning']}")

    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
