"""
Electric-field–dependent DFT: SCF in a uniform field.

- **Lightweight:** :func:`scf_efield_energy_forces_dipole` — energy, forces, dipole (no Hessian/IR).
- **Full:** :func:`maxwell_eval_ir_freq_intensity` — adds polarizability, Hessian, IR frequencies/intensities.

E-field *responsive* scans reuse the density as the next SCF guess; optional finite-difference dμ/dE.
"""

from __future__ import annotations

from functools import reduce
from typing import Any, List, Tuple

import ase.atoms
import cupy
import numpy as np
from pyscf.data import elements, nist
from pyscf.hessian import thermo
from scipy.constants import physical_constants

from gpu4pyscf.dft import rks
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.properties import polarizability

LINDEP_THRESHOLD = 1e-7


def _dm_kernel_guess(dm_init_guess: Any) -> Any:
    if dm_init_guess is None:
        return None
    if isinstance(dm_init_guess, (list, tuple)) and len(dm_init_guess) > 0:
        return dm_init_guess[0]
    return dm_init_guess


def parse_efield_points(spec: str) -> np.ndarray:
    """
    Parse semicolon-separated field points "Ex,Ey,Ez;Ex,Ey,Ez" (atomic units).

    Example: "0,0,0;0,0,0.001;0,0,-0.001"
    """
    spec = spec.strip()
    if not spec:
        return np.zeros((1, 3), dtype=np.float64)
    rows = []
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        nums = [float(x) for x in part.split(",")]
        if len(nums) != 3:
            raise ValueError(f"Expected three components Ex,Ey,Ez, got {part!r}")
        rows.append(nums)
    if not rows:
        return np.zeros((1, 3), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def run_scf_uniform_efield(
    E: np.ndarray,
    mol,
    dm_init_guess: Any = None,
    *,
    xc: str = "PBE0",
) -> Tuple[Any, float]:
    """
    Build RKS with hcore = T + Vnuc + E·μ and run SCF.

    Returns
    -------
    mf, e_tot
        Converged mean-field object and total energy (Hartree).
    """
    E = np.asarray(E, dtype=np.float64).reshape(3)
    mol.set_common_orig([0, 0, 0])
    h_core = mol.intor("cint1e_kin_sph") + mol.intor("cint1e_nuc_sph")
    h_core = h_core + np.einsum("x,xij->ij", E, mol.intor("cint1e_r_sph", comp=3))
    mf = rks.RKS(mol, xc=xc).density_fit()
    mf.get_hcore = lambda *args, hc=h_core: hc
    dm0_in = _dm_kernel_guess(dm_init_guess)
    e_tot = float(mf.kernel(dm0_in))
    return mf, e_tot


def _grad_to_numpy(F: Any) -> np.ndarray:
    if F is None:
        raise ValueError("gradient is None")
    return np.asarray(F.get() if hasattr(F, "get") else F, dtype=np.float64)


def scf_efield_energy_forces_dipole(
    E: np.ndarray,
    mol,
    dm_init_guess: Any = None,
    *,
    xc: str = "PBE0",
    dipole_unit: str = "AU",
    forces: bool = True,
) -> Tuple[dict, Any]:
    """
    SCF in a uniform electric field **E** (atomic units); return energy, dipole, and optionally forces.

    No Hessian, IR, or polarizability — suitable for cheap E-field scans.

    Parameters
    ----------
    dipole_unit
        Passed to ``mf.dip_moment`` (e.g. ``"AU"``, ``"DEBYE"``).
    forces
        If False, skip nuclear gradient (only energy + dipole).

    Returns
    -------
    summary, dm
        ``summary`` keys: ``Ef``, ``E`` (energy), ``D`` (dipole in ``dipole_unit``), ``D_au`` (always),
        ``F`` (Hartree/Bohr, numpy) if ``forces``, ``R`` (Å), ``Z``.
    """
    mf, e_tot = run_scf_uniform_efield(E, mol, dm_init_guess, xc=xc)
    Evec = np.asarray(E, dtype=np.float64).reshape(3)
    dip_primary = mf.dip_moment(unit=dipole_unit)
    dip_au = np.asarray(mf.dip_moment(unit="AU"), dtype=np.float64).ravel()[:3]

    out: dict = {
        "Ef": Evec.copy(),
        "E": e_tot,
        "D": dip_primary,
        "D_au": dip_au,
    }
    if forces:
        g = mf.nuc_grad_method()
        out["F"] = _grad_to_numpy(-g.kernel())

    atoms = ase.atoms.Atoms(mf.mol.elements, mf.mol.atom_coords(unit="AU"))
    out["R"] = atoms.get_positions()
    out["Z"] = atoms.get_atomic_numbers()
    dm = mf.make_rdm1()
    return out, dm


def efield_scf_scan(
    mol,
    efields: np.ndarray,
    *,
    xc: str = "PBE0",
    dm0_start: Any = None,
    dipole_unit: str = "AU",
    forces: bool = True,
) -> dict:
    """
    Like :func:`efield_ir_scan` but only SCF properties (energy, dipole, forces) per field point.

    Reuses density between points. Populates the same keys as :func:`efield_ir_scan` where
    applicable: ``Ef``, ``energy``, ``D_au``, and optionally ``forces`` stacked as ``F``.
    """
    efields = np.asarray(efields, dtype=np.float64).reshape(-1, 3)
    summaries: List[dict] = []
    dm_guess: Any = dm0_start

    for row in efields:
        s, dm_guess = scf_efield_energy_forces_dipole(
            row,
            mol,
            dm_init_guess=[dm_guess],
            xc=xc,
            dipole_unit=dipole_unit,
            forces=forces,
        )
        summaries.append(s)

    energies = np.array([float(s["E"]) for s in summaries], dtype=np.float64)
    D_au = np.stack([np.asarray(s["D_au"], dtype=np.float64).ravel()[:3] for s in summaries])
    out: dict = {
        "summaries": summaries,
        "Ef": efields.copy(),
        "energy": energies,
        "D_au": D_au,
    }
    if forces:
        out["F"] = np.stack([s["F"] for s in summaries], axis=0)
    return out


def maxwell_eval_ir_freq_intensity(
    E: np.ndarray,
    mol,
    dm_init_guess: Any = None,
    *,
    xc: str = "PBE0",
) -> Tuple[dict, Any]:
    """Calculate IR frequencies and intensities in a static uniform electric field E (a.u.).

    Args:
        E: shape (3,) field in atomic units.
        mol: PySCF molecular object (same molecule object used for intor / GPU mf).
        dm_init_guess: optional density matrix (or [dm]) to warm-start SCF.
        xc: DFT functional for the GPU RKS calculation.

    Returns:
        (summary_dict, dm) where summary contains Ef, D, D_au, E, H, A, F, freq, intensity, dDdR, R, Z.
    """
    mf, e_tot = run_scf_uniform_efield(E, mol, dm_init_guess, xc=xc)
    E = np.asarray(E, dtype=np.float64).reshape(3)

    dip_m = mf.dip_moment(unit="AU")
    dip_au = np.asarray(dip_m, dtype=np.float64).ravel()[:3]
    polar = polarizability.eval_polarizability(mf)
    summary: dict = {
        "Ef": E.copy(),
        "D": dip_m,
        "D_au": dip_au,
        "E": e_tot,
        "A": polar,
    }

    g = mf.nuc_grad_method()
    summary["F"] = -g.kernel()

    hessian_obj = mf.Hessian()
    hessian_obj.auxbasis_response = 2
    mf.cphf_grids.atom_grid = (50, 194)
    hessian = hessian_obj.kernel()
    summary["H"] = hessian

    log = logger.new_logger(hessian_obj, mf.mol.verbose)

    hartree_kj = nist.HARTREE2J * 1e3
    unit2cm = (
        (hartree_kj * nist.AVOGADRO) ** 0.5
        / (nist.BOHR * 1e-10)
        / (2 * np.pi * nist.LIGHT_SPEED_SI)
        * 1e-2
    )
    natm = mf.mol.natm
    nao = mf.mol.nao
    dm0 = mf.make_rdm1()

    atom_charges = mf.mol.atom_charges()
    mass = cupy.array([elements.MASSES[atom_charges[i]] for i in range(natm)])
    hessian_mass = contract("ijkl,i->ijkl", cupy.array(hessian), 1 / cupy.sqrt(mass))
    hessian_mass = contract("ijkl,j->ijkl", hessian_mass, 1 / cupy.sqrt(mass))

    TR = thermo._get_TR(mass.get(), mf.mol.atom_coords())
    TRspace: List = [TR[:3]]

    rot_const = thermo.rotation_const(mass.get(), mf.mol.atom_coords())
    rotor_type = thermo._get_rotor_type(rot_const)
    if rotor_type == "ATOM":
        pass
    elif rotor_type == "LINEAR":
        TRspace.append(TR[3:5])
    else:
        TRspace.append(TR[3:])

    TRspace = cupy.vstack(TRspace)
    q, r = cupy.linalg.qr(TRspace.T)
    P = cupy.eye(natm * 3) - q.dot(q.T)
    w, v = cupy.linalg.eigh(P)
    bvec = v[:, w > LINDEP_THRESHOLD]
    h_int = reduce(
        cupy.dot,
        (
            bvec.T,
            hessian_mass.transpose(0, 2, 1, 3).reshape(3 * natm, 3 * natm),
            bvec,
        ),
    )
    evals, mode = cupy.linalg.eigh(h_int)
    mode = bvec.dot(mode)

    c = contract("ixn,i->ixn", mode.reshape(natm, 3, -1), 1 / np.sqrt(mass)).reshape(
        3 * natm, -1
    )
    freq = cupy.sign(evals) * cupy.sqrt(cupy.abs(evals)) * unit2cm

    mo_coeff = cupy.array(mf.mo_coeff)
    mo_occ = cupy.array(mf.mo_occ)
    mo_energy = cupy.array(mf.mo_energy)
    mocc = mo_coeff[:, mo_occ > 0]

    atmlst = range(natm)
    h1ao = hessian_obj.make_h1(mo_coeff, mo_occ, None, atmlst)
    fx = hessian_obj.gen_vind(mo_coeff, mo_occ)
    mo1, mo_e1 = hessian_obj.solve_mo1(
        mo_energy, mo_coeff, mo_occ, h1ao, fx, atmlst, hessian_obj.max_memory, log
    )
    mo1 = cupy.asarray(mo1)
    mo_e1 = cupy.asarray(mo_e1)

    tmp = cupy.empty((3, 3, natm))
    aoslices = mf.mol.aoslice_by_atom()
    with mf.mol.with_common_orig((0, 0, 0)):
        hmuao = cupy.array(mf.mol.intor("int1e_r"))
        hmuao11 = -cupy.array(mf.mol.intor("int1e_irp").reshape(3, 3, nao, nao))

    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        h11ao = cupy.zeros((3, 3, nao, nao))
        h11ao[:, :, :, p0:p1] += hmuao11[:, :, :, p0:p1]
        h11ao[:, :, p0:p1] += hmuao11[:, :, :, p0:p1].transpose(0, 1, 3, 2)

        tmp0 = contract("ypi,vi->ypv", mo1[ia], mocc)
        dm1 = contract("ypv,up->yuv", tmp0, mo_coeff)
        tmp[:, :, ia] = -contract("xuv,yuv->xy", hmuao, dm1) * 4
        tmp[:, :, ia] -= contract("xyuv,vu->xy", h11ao, dm0)
        tmp[:, :, ia] += mf.mol.atom_charge(ia) * cupy.eye(3)

    alpha_fs = physical_constants["fine-structure constant"][0]
    amu = physical_constants["atomic mass constant"][0]
    m_e = physical_constants["electron mass"][0]
    N_A = physical_constants["Avogadro constant"][0]
    a_0 = physical_constants["Bohr radius"][0]
    unit_kmmol = alpha_fs**2 * (1e-3 / amu) * m_e * N_A * np.pi * a_0 / 3

    intensity = contract("xym,myn->xn", tmp, c.reshape(natm, 3, -1))
    intensity = contract("xn,xn->n", intensity, intensity) * unit_kmmol

    atoms = ase.atoms.Atoms(mol.elements, mol.atom_coords(unit="AU"))
    summary.update(
        {
            "freq": freq,
            "intensity": intensity,
            "dDdR": tmp,
            "R": atoms.get_positions(),
            "Z": atoms.get_atomic_numbers(),
        }
    )
    return summary, dm0


def efield_ir_scan(
    mol,
    efields: np.ndarray,
    *,
    xc: str = "PBE0",
    dm0_start: Any = None,
) -> dict:
    """
    Run :func:`maxwell_eval_ir_freq_intensity` for each row of ``efields`` (n, 3), reusing
    the previous density as SCF guess (E-field–responsive / continuation).

    Returns a dict with stacked scalars/vectors and the per-field summaries.
    """
    efields = np.asarray(efields, dtype=np.float64).reshape(-1, 3)
    summaries: List[dict] = []
    dm_guess: Any = dm0_start

    for row in efields:
        s, dm_guess = maxwell_eval_ir_freq_intensity(
            row, mol, dm_init_guess=[dm_guess], xc=xc
        )
        summaries.append(s)

    energies = np.array([float(s["E"]) for s in summaries], dtype=np.float64)
    D_rows = []
    for s in summaries:
        if "D_au" in s:
            d = np.asarray(s["D_au"], dtype=np.float64).ravel()
        else:
            d = np.asarray(s["D"], dtype=np.float64).ravel()
        D_rows.append(d[:3] if d.size >= 3 else np.pad(d, (0, max(0, 3 - d.size))))
    D_au = np.stack(D_rows, axis=0)

    pol = []
    for s in summaries:
        a = s["A"]
        pol.append(np.asarray(a.get() if hasattr(a, "get") else a, dtype=np.float64))
    polar_stack = np.stack(pol, axis=0) if pol else None

    return {
        "summaries": summaries,
        "Ef": efields.copy(),
        "energy": energies,
        "D_au": D_au,
        "polarizability": polar_stack,
    }


def efield_response_finite_difference(
    scan: dict,
    *,
    axis: int = 2,
) -> dict:
    """
    Estimate ∂μ/∂E_axis and ∂²E/∂E_axis² from a collinear scan in ``scan['Ef']``.

    Uses central differences when possible; falls back to forward difference for endpoints.
    Assumes field points are ordered along one Cartesian direction (default z).
    """
    Ef = np.asarray(scan["Ef"], dtype=np.float64)
    D = np.asarray(scan["D_au"], dtype=np.float64)
    en = np.asarray(scan["energy"], dtype=np.float64)
    n = Ef.shape[0]
    if n < 2:
        return {"d_mu_dE": None, "d2_E_dE2": None, "note": "need at least 2 field points"}

    # Field component along chosen axis
    h = Ef[:, axis]
    if np.allclose(np.ptp(h), 0.0):
        return {
            "d_mu_dE": None,
            "d2_E_dE2": None,
            "note": f"no variation in E along axis {axis}",
        }

    dmu_dE = np.full((n, 3), np.nan, dtype=np.float64)
    d2E_dE2 = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        if i > 0 and i < n - 1:
            dh = h[i + 1] - h[i - 1]
            if abs(dh) < 1e-14:
                continue
            dmu_dE[i] = (D[i + 1] - D[i - 1]) / dh
            x0, x1, x2 = h[i - 1], h[i], h[i + 1]
            f0, f1, f2 = en[i - 1], en[i], en[i + 1]
            d2E_dE2[i] = 2.0 * (
                (f2 - f1) / (x2 - x1 + 1e-30) - (f1 - f0) / (x1 - x0 + 1e-30)
            ) / (x2 - x0 + 1e-30)
        elif i == 0 and n > 1:
            dh = h[1] - h[0]
            if abs(dh) > 1e-14:
                dmu_dE[i] = (D[1] - D[0]) / dh
        elif i == n - 1 and n > 1:
            dh = h[-1] - h[-2]
            if abs(dh) > 1e-14:
                dmu_dE[i] = (D[-1] - D[-2]) / dh

    return {
        "d_mu_dE_au": dmu_dE,
        "d2_E_dE2_au": d2E_dE2,
        "axis": axis,
        "E_component": h,
    }
