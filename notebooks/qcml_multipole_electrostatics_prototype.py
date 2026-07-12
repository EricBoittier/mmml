# %% [markdown]
# # Learned QCML molecular multipole electrostatics prototype
#
# This notebook-style script demonstrates:
# - loading a learned QCML molecular multipole checkpoint,
# - assigning fragments by `atoms.arrays["mol_id"]` or explicit index lists,
# - computing q+dipole intermolecular electrostatics,
# - plotting orthogonal electric-field/potential slices.
#
# Units:
# - model positions: Bohr
# - learned l0: assumed electron charge units
# - learned l1: assumed electron Bohr
# - electrostatic energy: Hartree internally, eV for ASE `energy`
# - electric field: atomic units internally, V/Angstrom in plots

# %%
from pathlib import Path

import numpy as np
from ase.io import read

from mmml.models.multipoles import (
    LearnedMolecularMultipoleElectrostatics,
    plot_field_summary,
)

# %%
# Edit these paths for an actual run.
CHECKPOINT = Path("~/qcml_runs/multipoles_restart_YYYYMMDD-HHMMSS/epoch-XXXX").expanduser()
XYZ = Path("~/some_dimer_or_cluster.xyz").expanduser()

# %%
# The preferred fragment interface is `atoms.arrays["mol_id"]`.
# If absent, pass explicit fragments to the calculator:
# fragments = [list(range(0, n_a)), list(range(n_a, n_a + n_b))]
atoms = read(XYZ)

# Example explicit fallback:
fragments = None
if "mol_id" not in atoms.arrays:
    n_half = len(atoms) // 2
    fragments = [np.arange(0, n_half), np.arange(n_half, len(atoms))]

# %%
calc = LearnedMolecularMultipoleElectrostatics(
    CHECKPOINT,
    fragments=fragments,
    charges=None,  # defaults to neutral fragments; pass e.g. [0, 0] or [+1, -1]
    multiplicities=None,  # defaults to singlets
    origin="nuclear_charge_centroid",
    softening_bohr=0.5,
)

atoms.calc = calc
energy_ev = atoms.get_potential_energy()
print(f"Interfragment q+dipole electrostatic energy: {energy_ev:.8f} eV")
print(f"Energy: {calc.results['energy_hartree']:.8f} Hartree")
print("Pair energies (i, j, Hartree, eV):")
for row in calc.results["pair_energies"]:
    print(row)

# %%
print("Fragment origins [Angstrom]:")
print(calc.results["origins_angstrom"])

print("Predicted packed multipoles:")
print(calc.results["multipoles"])

# %%
fig = plot_field_summary(
    calc.results["origins_bohr"],
    calc.results["charges"],
    calc.results["dipoles_bohr"],
    planes=("xy", "xz", "yz"),
    extent_angstrom=10.0,
    n_grid=141,
    softening_bohr=0.5,
    output=Path("~/qcml_runs/electrostatic_field_summary.png").expanduser(),
)
fig
