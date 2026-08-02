"""PSF/CGenFF angle restraints restore tetrahedral geometry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
PSF = ROOT / "artifacts/lj_scales/liquid_nvt/mini.psf"
PDB = ROOT / "artifacts/lj_scales/prod_20ps_dual/nvt20_gpu0/pbc_nvt_jaxmd_minimized.pdb"


@pytest.mark.skipif(not PSF.is_file() or not PDB.is_file(), reason="DCM liquid artifacts missing")
def test_psf_angle_restraint_opposes_angle_distortion():
    from ase.io import read

    from mmml.md.restraints.psf_angles import build_psf_angle_restraint_fns

    atoms = read(str(PDB))
    pos = np.asarray(atoms.get_positions(), dtype=np.float64)
    # Distort H–C–H on the first DCM (atoms 0=C? check symbols)
    symbols = atoms.get_chemical_symbols()
    # First residue is 5 atoms: typically Cl, Cl, C, H, H or C, H, H, Cl, Cl — use PSF order.
    # Pull two H atoms of residue 0 apart in angle by moving one H.
    h_idx = [i for i, s in enumerate(symbols[:5]) if s == "H"]
    c_idx = next(i for i, s in enumerate(symbols[:5]) if s == "C")
    assert len(h_idx) >= 2
    pos_bad = pos.copy()
    # Move H away from the bisector to shrink/open the angle.
    v = pos_bad[h_idx[0]] - pos_bad[c_idx]
    pos_bad[h_idx[0]] = pos_bad[c_idx] + 1.6 * v

    e_fn, f_fn, info = build_psf_angle_restraint_fns(
        PSF, pos, box_A=30.0, scale=1.0, include_urey=True
    )
    assert info.n_angles == 720
    e0 = float(e_fn(pos))
    e1 = float(e_fn(pos_bad))
    assert e1 > e0 + 0.01  # distorted geometry costs more

    f_bad = np.asarray(f_fn(pos_bad))
    # Force on the moved H should have a component pulling back toward C–H equilibrium angle.
    assert np.linalg.norm(f_bad[h_idx[0]]) > 1e-6
