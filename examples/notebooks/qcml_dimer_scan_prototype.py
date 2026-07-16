# %% [markdown]
# # QCML dimer scan prototype
#
# Lightweight ASE path for testing shared dimer geometries before launching
# CHARMM/CGenFF, hybrid ML/MM, or long-range solver jobs.
#
# Backends covered here:
#
# - learned molecular multipole electrostatics (`q + μ`, fragment-level)
# - learned QCML MBD
# - xTB through the optional `xtb-python` ASE calculator
# - placeholder hooks for SpookyNet/SpookyPhysNet and CGenFF CSV merges
#
# The scan builder lives in `mmml.analysis.dimer_scans` so scripts and tests use
# the same geometry convention.

# %%
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase import Atoms

from mmml.analysis.dimer_scans import (
    DimerGeometry,
    distance_scan_geometries,
    evaluate_scan,
    make_xtb_calculator,
    molecule_pair_labels,
)
from mmml.models.mbd import QCMLMBDCalculator
from mmml.models.multipoles import LearnedMolecularMultipoleElectrostatics


# %% [markdown]
# ## Molecule registry
#
# Replace the approximate toy `Atoms` objects with optimized monomer fixtures
# once those are committed.  Residue IDs for this campaign:
#
# - DCM
# - ACE
# - BENZ
# - TIP3 using CGenFF params
# - MEOH

# %%
MOLECULES = {
    "DCM": {
        "residue_id": "DCM",
        "atoms": Atoms(
            "CCl2H2",
            positions=[
                [0.000, 0.000, 0.000],
                [1.760, 0.000, 0.000],
                [-1.760, 0.000, 0.000],
                [0.000, 0.950, 0.720],
                [0.000, -0.950, 0.720],
            ],
        ),
    },
    "ACE": {
        "residue_id": "ACE",
        "atoms": Atoms(
            "C3OH6",
            positions=[
                [0.000, 0.000, 0.000],
                [1.520, 0.000, 0.000],
                [-1.520, 0.000, 0.000],
                [0.000, 1.220, 0.000],
                [2.050, 0.900, 0.000],
                [2.050, -0.450, 0.780],
                [2.050, -0.450, -0.780],
                [-2.050, 0.900, 0.000],
                [-2.050, -0.450, 0.780],
                [-2.050, -0.450, -0.780],
            ],
        ),
    },
    "BENZ": {
        "residue_id": "BENZ",
        "atoms": Atoms(
            "C6H6",
            positions=[
                [1.397, 0.000, 0.000],
                [0.699, 1.210, 0.000],
                [-0.699, 1.210, 0.000],
                [-1.397, 0.000, 0.000],
                [-0.699, -1.210, 0.000],
                [0.699, -1.210, 0.000],
                [2.480, 0.000, 0.000],
                [1.240, 2.148, 0.000],
                [-1.240, 2.148, 0.000],
                [-2.480, 0.000, 0.000],
                [-1.240, -2.148, 0.000],
                [1.240, -2.148, 0.000],
            ],
        ),
    },
    "TIP3": {
        "residue_id": "TIP3",
        "atoms": Atoms(
            "OH2",
            positions=[
                [0.000000, 0.000000, 0.000000],
                [0.957200, 0.000000, 0.000000],
                [-0.239987, 0.926627, 0.000000],
            ],
        ),
    },
    "MEOH": {
        "residue_id": "MEOH",
        "atoms": Atoms(
            "COH4",
            positions=[
                [0.000, 0.000, 0.000],
                [1.430, 0.000, 0.000],
                [1.770, 0.910, 0.000],
                [-0.540, 0.900, 0.000],
                [-0.540, -0.450, 0.780],
                [-0.540, -0.450, -0.780],
            ],
        ),
    },
}

pd.DataFrame(
    [
        {"label": label, "residue_id": payload["residue_id"], "n_atoms": len(payload["atoms"])}
        for label, payload in MOLECULES.items()
    ]
)


# %% [markdown]
# ## Generate shared geometries

# %%
DISTANCES_ANGSTROM = np.linspace(3.0, 12.0, 19)
PAIR_LABELS = molecule_pair_labels(list(MOLECULES), include_homodimers=True)

PAIR_LABELS[:5], len(PAIR_LABELS)


# %%
def make_pair_scan(label_a: str, label_b: str) -> list[DimerGeometry]:
    return list(
        distance_scan_geometries(
            MOLECULES[label_a]["atoms"],
            MOLECULES[label_b]["atoms"],
            DISTANCES_ANGSTROM,
            pair=(label_a, label_b),
            axis=(1.0, 0.0, 0.0),
            center="centroid",
            mol_id_array="mol_id",
        )
    )


example_geometries = make_pair_scan("TIP3", "TIP3")
example_geometries[0]


# %% [markdown]
# ## Calculator factories
#
# Fill checkpoint paths on the cluster.  Each factory returns a fresh ASE
# calculator because ASE calculators cache per-structure state.

# %%
PATHS = {
    "multipole_checkpoint": Path("~/qcml_runs/multipoles_restart_20260711-100037/epoch-0100").expanduser(),
    "mbd_checkpoint": Path("~/qcml_runs/mbd_restart_20260711-100037/epoch-0100").expanduser(),
    "spooky_params": sorted(Path("examples").glob("sppoky-epoch-*_params.json")),
}


def multipole_calculator_factory():
    return LearnedMolecularMultipoleElectrostatics(
        checkpoint=PATHS["multipole_checkpoint"],
        fragments=None,
        mol_id_array="mol_id",
        charges=None,
        origin="nuclear_charge_centroid",
        softening_bohr=0.5,
    )


def mbd_calculator_factory():
    return QCMLMBDCalculator(checkpoint=PATHS["mbd_checkpoint"])


def xtb_calculator_factory():
    return make_xtb_calculator(method="GFN2-xTB")


BACKENDS = {
    "learned_multipole_qmu": multipole_calculator_factory,
    "learned_mbd": mbd_calculator_factory,
    "xtb_gfn2": xtb_calculator_factory,
}

PATHS


# %% [markdown]
# ## Run a scan
#
# Start with one pair and one backend, then loop over `PAIR_LABELS`.
#
# The learned multipole and MBD factories require real checkpoints.  xTB
# requires the optional `xtb-python` package and working xTB runtime.

# %%
def run_backend_scan(
    label_a: str,
    label_b: str,
    backend: str,
) -> pd.DataFrame:
    geometries = make_pair_scan(label_a, label_b)
    rows = evaluate_scan(geometries, BACKENDS[backend])
    output = pd.DataFrame(rows)
    output.insert(0, "backend", backend)
    return output


# Example:
# df_scan = run_backend_scan("TIP3", "TIP3", "xtb_gfn2")
# df_scan.head()


# %% [markdown]
# ## Plot a shared scan table

# %%
def plot_energy_scan(df: pd.DataFrame, *, reference_backend: str | None = None):
    fig, axes = plt.subplots(
        2 if reference_backend else 1,
        1,
        figsize=(7.0, 6.5 if reference_backend else 4.0),
        sharex=True,
        constrained_layout=True,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for backend, group in df.groupby("backend", sort=False):
        group = group.sort_values("distance_angstrom")
        axes[0].plot(
            group["distance_angstrom"],
            group["energy_kcal_mol"],
            marker="o",
            label=backend,
        )
    axes[0].set_ylabel("Energy / kcal mol$^{-1}$")
    axes[0].legend(frameon=False)

    if reference_backend:
        reference = (
            df[df["backend"] == reference_backend]
            .sort_values("distance_angstrom")
            .set_index("distance_angstrom")["energy_kcal_mol"]
        )
        for backend, group in df.groupby("backend", sort=False):
            if backend == reference_backend:
                continue
            group = group.sort_values("distance_angstrom").set_index("distance_angstrom")
            delta = group["energy_kcal_mol"] - reference
            axes[1].plot(delta.index, delta.values, marker="o", label=backend)
        axes[1].axhline(0.0, color="black", linewidth=0.8)
        axes[1].set_ylabel(f"ΔE vs {reference_backend} / kcal mol$^{{-1}}$")
        axes[1].legend(frameon=False)

    axes[-1].set_xlabel("Center distance / Å")
    return fig, axes


# Example:
# fig, axes = plot_energy_scan(df_scan)


# %% [markdown]
# ## Batch plan
#
# For the first production pass:
#
# 1. Generate scans for all 15 pairs from `PAIR_LABELS`.
# 2. Evaluate `learned_multipole_qmu`, `learned_mbd`, `xtb_gfn2`, and Spooky.
# 3. Run CHARMM/CGenFF and hybrid ML/MM through the PyCHARMM scan path.
# 4. Merge outputs on `(molecule_a, molecule_b, distance_angstrom, backend)`.
# 5. Plot raw energies, ΔE to `charmm_cgenff`, and long-range tails.

