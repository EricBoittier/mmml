"""CHARMM-free metatomic ASE MD helpers (cubic liquid box + NVE stats).

Used by ``mmml metatomic-pbc-md`` and ``examples/pet_mad_etoh_pbc``. Does not
import torch or metatomic at module import time.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read as ase_read

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


def n_molecules_for_residue_box(
    residue: str,
    *,
    box_side_A: float,
    density_g_cm3: float | None = None,
) -> int:
    """Molecule count for ``residue`` in a fixed cube at the given (or bulk) density."""
    from mmml.interfaces.pycharmmInterface.mlpot.box_sizing import (
        SOLVENT_BULK_PROPS,
        n_molecules_for_target_density_in_fixed_box,
    )

    res = str(residue).strip().upper()
    rho = density_g_cm3
    if rho is None:
        if res not in SOLVENT_BULK_PROPS:
            raise ValueError(
                f"no bulk density for residue {res!r}; pass --target-density-g-cm3"
            )
        rho = float(SOLVENT_BULK_PROPS[res]["rho_g_cm3"])
    scaled = n_molecules_for_target_density_in_fixed_box(
        composition={res: 1},
        box_side_A=float(box_side_A),
        target_density_g_cm3=float(rho),
    )
    return int(scaled[res])


def mass_density_g_cm3(residue: str, n_molecules: int, box_side_A: float) -> float:
    from mmml.interfaces.pycharmmInterface.mlpot.box_sizing import (
        total_mass_g_for_composition,
    )

    mass_g = total_mass_g_for_composition({str(residue).strip().upper(): int(n_molecules)})
    vol_cm3 = (float(box_side_A) * 1.0e-8) ** 3
    return float(mass_g / vol_cm3)


def build_tiled_cubic_liquid(
    *,
    monomer_xyz: Path,
    box_side_A: float,
    n_molecules: int,
    seed: int,
) -> Atoms:
    """Grid-place ``n_molecules`` copies of ``monomer_xyz`` in a cubic PBC cell."""
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


def energy_snapshot(atoms: Atoms, *, step: int, dt_fs: float) -> dict[str, float]:
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


def write_energy_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ENERGY_CSV_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in ENERGY_CSV_FIELDS})


def plot_nve_energy(rows: list[dict[str, float]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mmml.utils.plotting.styles import apply_plot_style

    apply_plot_style("icml")
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


def default_etoh_monomer_xyz() -> Path:
    """Repo ``examples/pet_mad_etoh_pbc/etoh.xyz`` (``mmml/md/`` → parents[2] is root)."""
    return (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "pet_mad_etoh_pbc"
        / "etoh.xyz"
    )
