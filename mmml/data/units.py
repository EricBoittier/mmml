"""
Central unit conversion constants for MMML.

All conversion factors are defined here to avoid magic numbers and ensure
consistency across train_joint, fix_and_split, DCMNet, PhysNet, and calculators.

Reference: CODATA 2018 / NIST
"""

# -----------------------------------------------------------------------------
# Length
# -----------------------------------------------------------------------------
# 1 Bohr (a₀) = 0.529177 × 10⁻¹⁰ m (CODATA)
# 1 Angstrom = 10⁻¹⁰ m
BOHR_TO_ANGSTROM = 0.529177
ANGSTROM_TO_BOHR = 1.88973  # 1 / BOHR_TO_ANGSTROM

# -----------------------------------------------------------------------------
# Energy
# -----------------------------------------------------------------------------
# 1 Hartree (E_h) = 27.211386 eV (CODATA)
HARTREE_TO_EV = 27.211386
EV_TO_HARTREE = 1.0 / HARTREE_TO_EV

# 1 eV = 23.060549 kcal/mol (for display)
EV_TO_KCAL_MOL = 23.060549
# 1 Hartree = 627.509474 kcal/mol
HARTREE_TO_KCAL_MOL = 627.509474
KCAL_MOL_TO_HARTREE = 1.0 / HARTREE_TO_KCAL_MOL

# -----------------------------------------------------------------------------
# Forces
# -----------------------------------------------------------------------------
# Hartree/Bohr → eV/Angstrom
# dE/dr: 1 Ha/Bohr × (27.211 eV/Ha) × (0.529 Å/Bohr) = 51.422065 eV/Å
# Or: 1 Ha/Bohr = HARTREE_TO_EV / BOHR_TO_ANGSTROM
HARTREE_BOHR_TO_EV_ANGSTROM = HARTREE_TO_EV / BOHR_TO_ANGSTROM  # ≈ 51.422065

# -----------------------------------------------------------------------------
# Dipole moment
# -----------------------------------------------------------------------------
# 1 Debye (D) = 10⁻¹⁸ esu·cm = 0.208194 e·Å (elementary charge × Angstrom)
# 1 e·Å = 4.80320425 D
DEBYE_TO_EANGSTROM = 0.208194
EANGSTROM_TO_DEBYE = 1.0 / DEBYE_TO_EANGSTROM  # ≈ 4.803204

# -----------------------------------------------------------------------------
# ESP (electrostatic potential)
# -----------------------------------------------------------------------------
# V = q/r in atomic units: r in Bohr, V in Hartree/e
# When positions are in Angstrom: r_bohr = r_angstrom * ANGSTROM_TO_BOHR
# So: V [Ha/e] = q / (r_angstrom * ANGSTROM_TO_BOHR)

# -----------------------------------------------------------------------------
# Coulomb energy
# -----------------------------------------------------------------------------
# E_coul = (1/2) Σᵢⱼ qᵢqⱼ/rᵢⱼ  in Hartree when r is in Bohr
# With r in Angstrom: r_bohr = r_angstrom * ANGSTROM_TO_BOHR
# E_coul [Ha] = (1/2) Σ qᵢqⱼ / (r_ij_angstrom * ANGSTROM_TO_BOHR)
# E_coul [eV] = E_coul [Ha] * HARTREE_TO_EV
