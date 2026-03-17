"""
Regression test: Na+ Cl- at 10 Å vs point-charge Coulomb and dipole.

Compares:
1. PySCF (HF) energy and dipole for Na+ Cl- at 10 Å separation
2. Point-charge Coulomb energy: E = q1*q2/r [Hartree], r in Bohr
3. DCMNet calc_esp and Coulomb energy from train_joint formula
4. Dipole: point charges vs PySCF

Reference: Two point charges +1 and -1 at 10 Å
- r = 10 * 1.88973 = 18.897 Bohr
- E_coul = -1/18.897 = -0.05293 Ha = -1.44 eV
- Dipole (COM at midpoint): μ = 1*(-5) + (-1)*(5) = -10 e·Å (x-component)

Run with:
  JAX_PLATFORMS=cpu pytest tests/unit/test_na_cl_coulomb_regression.py -v

Or add to pytest.ini:
  [env]
  JAX_PLATFORMS = cpu
"""

import os
import numpy as np
import pytest

# Force JAX CPU to avoid CUDA init failures when no GPU
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Skip entire module if pyscf not available
pyscf = pytest.importorskip("pyscf")

from mmml.data.units import (
    ANGSTROM_TO_BOHR,
    HARTREE_TO_EV,
    DEBYE_TO_EANGSTROM,
    EANGSTROM_TO_DEBYE,
)
from mmml.models.dcmnet.dcmnet.electrostatics import calc_esp


# Geometry: Na+ at origin, Cl- at (10, 0, 0) Angstrom
R_NA = np.array([[0.0, 0.0, 0.0]])
R_CL = np.array([[10.0, 0.0, 0.0]])
R_PAIR = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])

SEPARATION_ANGSTROM = 10.0
R_BOHR = SEPARATION_ANGSTROM * ANGSTROM_TO_BOHR


def point_charge_coulomb_energy_hartree(q1: float, q2: float, r_angstrom: float) -> float:
    """E = q1*q2/r [Hartree] with r in Bohr."""
    r_bohr = r_angstrom * ANGSTROM_TO_BOHR
    return (q1 * q2) / (r_bohr + 1e-12)


def point_charge_dipole_eangstrom(charges: np.ndarray, positions_angstrom: np.ndarray) -> np.ndarray:
    """μ = Σ q_i * (r_i - r_COM) [e·Å]. Use geometric center as origin for simplicity."""
    com = np.mean(positions_angstrom, axis=0)
    r_rel = positions_angstrom - com
    return np.sum(charges[:, None] * r_rel, axis=0)


def train_joint_coulomb_energy(charges: np.ndarray, positions_angstrom: np.ndarray) -> float:
    """E = (1/2) Σᵢⱼ qᵢqⱼ/rᵢⱼ [Hartree], matches train_joint formula."""
    diff = positions_angstrom[:, None, :] - positions_angstrom[None, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    distances = np.where(distances < 1e-6, 1e10, distances)
    r_bohr = distances * ANGSTROM_TO_BOHR
    pairwise = charges[:, None] * charges[None, :] / (r_bohr + 1e-10)
    return 0.5 * np.sum(pairwise)


def run_pyscf_na_cl(separation_angstrom: float = 10.0):
    """Run PySCF HF on Na+ Cl- at given separation. Returns (E_Ha, dipole_Debye)."""
    import pyscf.gto
    import pyscf.scf

    r_bohr = separation_angstrom * ANGSTROM_TO_BOHR
    mol = pyscf.gto.M(
        atom=[
            ["Na", (0.0, 0.0, 0.0)],
            ["Cl", (r_bohr, 0.0, 0.0)],
        ],
        basis="sto3g",
        charge=0,
        spin=0,
        unit="Bohr",
    )
    mf = pyscf.scf.RHF(mol)
    mf.kernel()
    e_tot = mf.e_tot
    dipole_debye = mf.dip_moment(unit="Debye")
    return e_tot, dipole_debye


def run_pyscf_isolated_ions():
    """Run PySCF on isolated Na+ and Cl-. Returns (E_Na, E_Cl)."""
    import pyscf.gto
    import pyscf.scf

    mol_na = pyscf.gto.M(atom="Na 0 0 0", basis="sto3g", charge=1, spin=0)
    mf_na = pyscf.scf.RHF(mol_na)
    mf_na.kernel()

    mol_cl = pyscf.gto.M(atom="Cl 0 0 0", basis="sto3g", charge=-1, spin=0)
    mf_cl = pyscf.scf.RHF(mol_cl)
    mf_cl.kernel()

    return mf_na.e_tot, mf_cl.e_tot


class TestNaClCoulombRegression:
    """Regression tests for Na+ Cl- vs point-charge Coulomb/dipole."""

    def test_point_charge_coulomb_energy(self):
        """Reference: E = -1/r_bohr for +1 and -1 at 10 Å."""
        e_ref = point_charge_coulomb_energy_hartree(1.0, -1.0, SEPARATION_ANGSTROM)
        e_expected = -1.0 / R_BOHR
        assert np.isclose(e_ref, e_expected, rtol=1e-10)
        assert np.isclose(e_ref, -0.05293, rtol=1e-3)
        e_ev = e_ref * HARTREE_TO_EV
        assert np.isclose(e_ev, -1.44, rtol=1e-2)

    def test_train_joint_coulomb_matches_point_charge(self):
        """train_joint formula for 2 charges: E = 0.5 * 2 * (q1*q2/r) = q1*q2/r."""
        charges = np.array([1.0, -1.0])
        positions = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        e_train = train_joint_coulomb_energy(charges, positions)
        e_ref = point_charge_coulomb_energy_hartree(1.0, -1.0, 10.0)
        assert np.isclose(e_train, e_ref, rtol=1e-10)

    def test_dcmnet_esp_at_midpoint(self):
        """ESP at midpoint (5,0,0) from two charges: V = 1/5 + (-1)/5 = 0 (in Bohr: 1/9.45 - 1/9.45)."""
        import jax.numpy as jnp

        charge_positions = jnp.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        charge_values = jnp.array([1.0, -1.0])
        grid_point = jnp.array([[5.0, 0.0, 0.0]])  # Midpoint
        esp = calc_esp(charge_positions, charge_values, grid_point)
        # V = q1/r1 + q2/r2, r1=r2=5 Å at midpoint
        # V = 1/(5*1.88973) + (-1)/(5*1.88973) = 0
        assert np.isclose(float(esp[0]), 0.0, atol=1e-10)

    def test_dcmnet_esp_near_na(self):
        """ESP at 1 Å from Na+ (between ions): V ≈ +1/1 - 1/9 [Ha/e]."""
        import jax.numpy as jnp

        charge_positions = jnp.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        charge_values = jnp.array([1.0, -1.0])
        grid_point = jnp.array([[1.0, 0.0, 0.0]])  # 1 Å from Na+
        esp = calc_esp(charge_positions, charge_values, grid_point)
        # r1=1 Å, r2=9 Å
        v_expected = 1.0 / (1.0 * ANGSTROM_TO_BOHR) + (-1.0) / (9.0 * ANGSTROM_TO_BOHR)
        # Relax rtol for float32 precision in JAX
        assert np.isclose(float(esp[0]), v_expected, rtol=1e-5)

    def test_point_charge_dipole(self):
        """Dipole for +1 at 0 and -1 at 10: μ = -10 e·Å (pointing toward Cl-)."""
        charges = np.array([1.0, -1.0])
        positions = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        mu = point_charge_dipole_eangstrom(charges, positions)
        # COM at 5. Na+ at -5, Cl- at +5. μ = 1*(-5) + (-1)*(5) = -10
        assert np.isclose(mu[0], -10.0, rtol=1e-10)
        assert np.isclose(mu[1], 0.0, atol=1e-10)
        assert np.isclose(mu[2], 0.0, atol=1e-10)

    def test_pyscf_vs_point_charge_energy(self):
        """PySCF interaction energy should be close to point-charge at 10 Å (large separation)."""
        e_pair, dipole_pyscf = run_pyscf_na_cl(SEPARATION_ANGSTROM)
        e_na, e_cl = run_pyscf_isolated_ions()
        e_interaction_pyscf = e_pair - e_na - e_cl
        e_point_charge = point_charge_coulomb_energy_hartree(1.0, -1.0, SEPARATION_ANGSTROM)

        # At 10 Å, polarization is small; interaction should be close to point-charge
        # Allow 10% tolerance for basis set / correlation effects
        assert np.isclose(e_interaction_pyscf, e_point_charge, rtol=0.15), (
            f"PySCF interaction {e_interaction_pyscf:.6f} Ha vs point-charge {e_point_charge:.6f} Ha"
        )

    def test_pyscf_vs_point_charge_dipole(self):
        """PySCF dipole (in Debye) vs point-charge dipole (10 e·Å = 48 D)."""
        _, dipole_pyscf_debye = run_pyscf_na_cl(SEPARATION_ANGSTROM)
        dipole_pyscf_eangstrom = dipole_pyscf_debye * DEBYE_TO_EANGSTROM

        charges = np.array([1.0, -1.0])
        positions = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        dipole_point = point_charge_dipole_eangstrom(charges, positions)
        dipole_point_mag = np.linalg.norm(dipole_point)

        # PySCF includes electronic screening; at 10 Å dipole should be close to point-charge
        # Point-charge: 10 e·Å. PySCF typically 8-10 e·Å for ion pair
        assert np.isclose(np.linalg.norm(dipole_pyscf_eangstrom), dipole_point_mag, rtol=0.2), (
            f"PySCF dipole {np.linalg.norm(dipole_pyscf_eangstrom):.2f} e·Å vs "
            f"point-charge {dipole_point_mag:.2f} e·Å"
        )

    def test_units_consistency(self):
        """Sanity check: 10 e·Å = 48 D."""
        assert np.isclose(10.0 * EANGSTROM_TO_DEBYE, 48.03, rtol=1e-2)
