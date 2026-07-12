# %% [markdown]
# # QCML / MMML dimer-scan campaign overview
#
# Target molecules:
#
# - dichloromethane
# - acetone
# - benzene
# - water
# - methanol
#
# Goal: build a reproducible dimer-scan workflow that compares learned
# molecular multipole electrostatics, learned MBD dispersion, xTB,
# SpookyNet/SpookyPhysNet, CHARMM/CGenFF, hybrid MM/ML, and long-range solver
# variants on the same geometries and units.

# %% [markdown]
# ## Current repository assets
#
# Existing pieces we can reuse directly:
#
# | Asset | Path | Status | Notes |
# |---|---|---:|---|
# | QCML multipole model + training/eval | `scripts/train_qcml_multipoles.py`, `scripts/analyze_qcml_multipoles.py` | usable | Unified and degree-specific model support exists. |
# | Learned molecular multipole electrostatics | `mmml/models/multipoles/electrostatics.py` | prototype | q+dipole only; units/sign tests are present. |
# | QCML MBD model + training/eval | `scripts/train_qcml_mbd.py`, `scripts/analyze_qcml_mbd.py` | usable | Package-level ASE calculator exists. |
# | Dimer scan helpers | `mmml/analysis/dimer_scans.py` | usable | Builds deterministic ASE rigid dimer scans and optional xTB calculators. |
# | QCML ASE diagnostics notebook | `notebooks/qcml_ase_calculators_diagnostics.py` | usable | Contains historical prototypes and multipole diagnostics. |
# | DCM/acetone dimer LR scan | `scripts/run_dcm_aco_dimer_lr_scans.sh` | usable for DCM/ACO | Wraps PyCHARMM MLpot scan and LR solver sweep. |
# | PyCHARMM dimer scan engine | `scripts/scan_mlpot_dimer_2d_pycharmm.py` | usable | Produces decomposed ML/MM, CHARMM, USER/VDW/ELEC terms. |
# | Dimer LR plotting | `scripts/plot_dimer_lr_scan_compare.py` | likely reusable | Existing plot entry point for current NPZ layout. |
# | Energy-provider abstraction | `mmml/interfaces/energy_forces` | reusable | Useful for normalizing ASE-style calculators. |

# %% [markdown]
# ## Molecule metadata to fill
#
# The campaign should keep one central table for all molecule-specific names.
# Fill in residue IDs, CHARMM topology labels, and any Packmol/build aliases here.

# %%
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class MoleculeSpec:
    label: str
    residue_id: str | None
    composition_tag: str | None
    smiles: str
    charge: int = 0
    multiplicity: int = 1
    notes: str = ""


MOLECULES = [
    MoleculeSpec("dichloromethane", residue_id="DCM", composition_tag="DCM", smiles="ClCCl"),
    MoleculeSpec("acetone", residue_id="ACE", composition_tag="ACE", smiles="CC(=O)C"),
    MoleculeSpec("benzene", residue_id="BENZ", composition_tag="BENZ", smiles="c1ccccc1"),
    MoleculeSpec("water", residue_id="TIP3", composition_tag="TIP3", smiles="O", notes="resname TIP3, CGenFF params"),
    MoleculeSpec("methanol", residue_id="MEOH", composition_tag="MEOH", smiles="CO"),
]

pd.DataFrame([spec.__dict__ for spec in MOLECULES])

# %% [markdown]
# ## Calculator matrix
#
# Each calculator should expose a common dimer-scan result:
#
# - geometry ID / molecule pair / scan coordinate
# - total energy in `kcal/mol` and `eV`
# - optional force components in `kcal/mol/Å` and `eV/Å`
# - decomposed terms when available
# - provenance: checkpoint, force field, CHARMM settings, LR solver, damping, units

# %%
CALCULATORS = pd.DataFrame(
    [
        {
            "name": "learned_multipole_qmu",
            "kind": "ASE",
            "existing": True,
            "entry_point": "mmml.models.multipoles.LearnedMolecularMultipoleElectrostatics",
            "outputs": "q+dipole electrostatic energy; field plots",
            "todo": "add l2/l3 after finite-difference sign tests; add dimer scan wrapper",
        },
        {
            "name": "learned_mbd",
            "kind": "ASE",
            "existing": True,
            "entry_point": "mmml.models.mbd.QCMLMBDCalculator",
            "outputs": "MBD energy, forces, polarizabilities, C6",
            "todo": "validate units vs QCML cache and add force finite-difference smoke test",
        },
        {
            "name": "xtb_gfn2",
            "kind": "ASE",
            "existing": "optional",
            "entry_point": "mmml.analysis.dimer_scans.make_xtb_calculator",
            "outputs": "xTB energy/forces through xtb-python ASE calculator",
            "todo": "install/validate xtb-python on target nodes; decide GFN1 vs GFN2 default",
        },
        {
            "name": "spookynet_or_spookyphysnet",
            "kind": "ASE/MLpot",
            "existing": True,
            "entry_point": "mmml.interfaces.energy_forces / CHARMM MLpot",
            "outputs": "ML energy/forces",
            "todo": "standardize checkpoint metadata and neutral singlet assumptions",
        },
        {
            "name": "charmm_cgenff",
            "kind": "PyCHARMM",
            "existing": True,
            "entry_point": "scripts/scan_mlpot_dimer_2d_pycharmm.py",
            "outputs": "ENER, VDW, ELEC, bonded/nonbond terms",
            "todo": "add all five molecules; verify residue/topology names",
        },
        {
            "name": "hybrid_mm_ml",
            "kind": "PyCHARMM MLpot",
            "existing": True,
            "entry_point": "scripts/scan_mlpot_dimer_2d_pycharmm.py",
            "outputs": "decomposed ML/MM hybrid terms",
            "todo": "ensure all target molecules are compatible with current MLpot path",
        },
        {
            "name": "long_range_solvers",
            "kind": "PyCHARMM/JAX PME/SCAFACOS",
            "existing": "partial",
            "entry_point": "scripts/run_dcm_aco_dimer_lr_scans.sh",
            "outputs": "MIC, JAX PME/Ewald/P3M, optional ScaFaCoS/NVAlChemiOps",
            "todo": "generalize from DCM/ACO to full molecule set",
        },
    ]
)

CALCULATORS

# %% [markdown]
# ## Scan geometries
#
# Start simple and reproducible. A useful staged design:
#
# 1. **1D COM separation scans**
#    - fixed monomer orientations
#    - distance range: e.g. `3.0–14.0 Å`
#    - enough points for minima and long-range tail, e.g. `40–80`
# 2. **orientation scans at selected separations**
#    - rotate monomer B around principal axes
#    - useful for dipole/quadrupole sign validation
# 3. **2D scans**
#    - current PyCHARMM script already supports a COM 2D grid for N≥3 and a 1D dimer mode
# 4. **relaxed scans**
#    - optional restrained minimization at each separation, after rigid scans are stable
#
# For learned molecular multipoles, rigid scans are more interpretable because
# the fragment origin and orientation are controlled.

# %%
SCAN_DEFAULTS = {
    "distance_min_angstrom": 3.0,
    "distance_max_angstrom": 14.0,
    "distance_steps": 56,
    "orientation_degrees": [0, 45, 90, 135, 180],
    "pair_scope": "homodimers_and_all_pairs",
    "box_size_angstrom": 36.0,
    "multipole_softening_bohr": 0.5,
    "mbd_softening": None,
}

SCAN_DEFAULTS

# %% [markdown]
# ## Proposed output layout
#
# Use a calculator-agnostic layout so plotting/aggregation does not depend on
# the backend.
#
# ```text
# artifacts/qcml_dimer_scans/
#   manifest.json
#   molecules.csv
#   <molecule_pair>/
#     geometries/
#       rigid_1d.xyz
#       scan_manifest.csv
#     learned_multipole_qmu/
#       scan.csv
#       field_summary_<point>.png
#       provenance.json
#     learned_mbd/
#       scan.csv
#       provenance.json
#     spookynet/
#       scan.csv
#       provenance.json
#     charmm_cgenff/
#       scan.csv
#       provenance.json
#     hybrid_mm_ml/
#       scan.csv
#       provenance.json
#     comparisons/
#       energy_components.png
#       delta_to_charmm.png
#       tail_scaling.png
# ```

# %% [markdown]
# ## Immediate implementation plan
#
# ### Phase 0 — metadata and fixtures
#
# - Fill `MoleculeSpec.residue_id` and `composition_tag` for all five molecules.
# - Add one monomer structure per molecule in a common source directory.
# - Add explicit fragment metadata (`mol_id`) to generated dimer structures.
# - Decide units for all reported tables: primary `kcal/mol`, secondary `eV`.
#
# ### Phase 1 — reusable calculators
#
# - Use package-level `mmml.models.mbd.QCMLMBDCalculator`.
# - Use `mmml.analysis.dimer_scans.make_xtb_calculator` for optional xTB scans.
# - Keep `LearnedMolecularMultipoleElectrostatics` q+dipole-only until l2/l3 are
#   separately validated.
# - Add tests:
#   - MBD ASE energy/force shapes and unit conversions.
#   - MBD force finite-difference smoke test on a tiny molecule.
#   - Multipole q/μ scan analytic parity on synthetic sources.
#
# ### Phase 2 — calculator-agnostic scan driver
#
# - Add an ASE-first dimer scan script for `learned_multipole_qmu`, `learned_mbd`,
#   `xtb_gfn2`, and standalone SpookyNet/SpookyPhysNet.
# - Reuse PyCHARMM scan for `charmm_cgenff`, `hybrid_mm_ml`, and LR solver sweeps.
# - Normalize all outputs to a common CSV schema.
#
# ### Phase 3 — comparison plots
#
# - Energy vs distance by backend.
# - Decomposed electrostatics/dispersion/ML/MM components where available.
# - Difference-to-reference plots.
# - Multipole field summary plots for selected geometries.
# - Long-range tail checks: fit leading powers (`1/R`, `1/R^3`, `1/R^6`).

# %%
TODO = pd.DataFrame(
    [
        ("metadata", "Confirm residue IDs/composition tags", "user", "done"),
        ("fixtures", "Collect or generate monomer structures", "agent/user", "blocking"),
        ("mbd", "Promote QCMLMBDCalculator to package module", "agent", "done"),
        ("mbd", "Add MBD calculator tests", "agent", "done"),
        ("xtb", "Add optional ASE xTB backend hook", "agent", "done"),
        ("multipoles", "Add q+dipole dimer scan wrapper", "agent", "high"),
        ("multipoles", "Add finite-difference q-Q / mu-Q convention tests before l2/l3", "agent", "medium"),
        ("spooky", "Use latest examples/sppoky-epoch-*.json param files", "agent", "high"),
        ("charmm", "Generalize DCM/ACO scan script molecule list", "agent", "medium"),
        ("plots", "Create unified comparison plotting script", "agent", "medium"),
        ("slurm", "Add scicore Slurm array for molecule/backend combinations", "agent", "medium"),
    ],
    columns=["area", "task", "owner", "priority"],
)

TODO

# %% [markdown]
# ## Information needed from you
#
# The main missing inputs are:
#
# 1. Confirm whether acetone should be `ACE` everywhere or mapped to legacy `ACO`
#    for scripts that already use `ACO`.
# 2. Confirm whether benzene has a committed monomer fixture or should be built
#    from SMILES/CGenFF.
# 3. Confirm which Spooky parameter JSON is the default comparison if multiple
#    `examples/sppoky-epoch-*.json` files are present.
# 4. Preferred reference for plotting deltas: CHARMM CGenFF, SpookyNet, or DFT/QC
#    if available.

# %%
REQUIRED_USER_INPUT = {
    "residue_ids": {
        "dichloromethane": "DCM",
        "acetone": "ACE",
        "benzene": "BENZ",
        "water": "TIP3",
        "methanol": "MEOH",
    },
    "water_model": "TIP3 resname with CGenFF params",
    "spooky_param_files": sorted(str(path) for path in Path("examples").glob("sppoky-epoch-*_params.json")),
    "scan_pairs": "homodimers_and_all_pairs",
    "delta_reference": "charmm_cgenff",
}

REQUIRED_USER_INPUT

# %% [markdown]
# ## First concrete next step
#
# I would do this next:
#
# 1. Add an ASE dimer scan script that supports:
#    - `--backend learned_multipole_qmu`
#    - `--backend learned_mbd`
#    - `--backend xtb_gfn2`
#    - `--backend spookynet`
# 2. Keep PyCHARMM scans separate initially, then merge outputs by CSV schema.
#
# This avoids coupling the notebook prototype to PyCHARMM early, and gives us a
# fast CPU/GPU path for testing the learned multipole + MBD components before
# launching cluster-wide CHARMM jobs.
