"""Regression tests for default parameter resolution and evaluation on H2O-MeOH dimer.

Pins default parameter resolution paths and interaction energies across model families:
- PhysNet / Joint MLpot
- SpookyNet
- QCML MBD
- QCML Multipoles (with mol_id auto-segmentation)
- Hybrid MLpot (Spooky + MBD + Multipoles)
- pyCHARMM CGenFF (relaxed MM geometry)
"""

from __future__ import annotations

import pytest
import numpy as np
from ase import Atoms

from mmml.analysis.dimer_molecules import make_oriented_scan_geometries
from mmml.interfaces.calculators.checkpoint_loading import create_calculator_from_checkpoint
from mmml.interfaces.pycharmmInterface.mlpot.cli_common import resolve_checkpoint
from mmml.models.spookynet_calc import SpookyNetCalculator, resolve_spooky_checkpoint
from mmml.models.mbd.calculator import QCMLMBDCalculator, resolve_mbd_checkpoint
from mmml.models.multipoles.electrostatics import (
    LearnedMolecularMultipoleElectrostatics,
    resolve_multipoles_checkpoint,
)

EV_TO_KCAL = 23.060548867


@pytest.fixture(scope="module")
def default_checkpoints():
    """Verify and resolve default parameter checkpoint paths for all 4 models."""
    phys_ckpt = resolve_checkpoint(None)
    spooky_ckpt = resolve_spooky_checkpoint(None)
    mbd_ckpt = resolve_mbd_checkpoint(None)
    mult_ckpt = resolve_multipoles_checkpoint(None)

    assert phys_ckpt.exists()
    assert spooky_ckpt.exists()
    assert mbd_ckpt.exists()
    assert mult_ckpt.exists()

    return {
        "physnet": phys_ckpt,
        "spookynet": spooky_ckpt,
        "mbd": mbd_ckpt,
        "multipoles": mult_ckpt,
    }


@pytest.fixture(scope="module")
def h2o_meoh_dimer_atoms():
    """Construct H2O - MeOH pre-oriented dimer and monomers with mol_id array."""
    geoms = list(make_oriented_scan_geometries("TIP3", "MEOH", [2.8], [0.0]))
    dimer = geoms[0].atoms.copy()

    # 3 atoms for H2O (mol_id=0), 6 atoms for MeOH (mol_id=1)
    dimer.set_array("mol_id", np.array([0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=int))

    mon1 = dimer[:3].copy()
    mon1.set_array("mol_id", np.array([0, 0, 0], dtype=int))

    mon2 = dimer[3:].copy()
    mon2.set_array("mol_id", np.array([1, 1, 1, 1, 1, 1], dtype=int))

    return dimer, mon1, mon2


def test_default_checkpoints_pinned_resolution(default_checkpoints):
    assert "physnetjax/defaults/hf_json" in str(default_checkpoints["physnet"])
    assert "spooky_so3lr_muon_mbd_zbl" in str(default_checkpoints["spookynet"])
    assert "mbd_20260711" in str(default_checkpoints["mbd"])
    assert "multipoles_20260711" in str(default_checkpoints["multipoles"])


def test_physnet_dimer_regression(h2o_meoh_dimer_atoms):
    dimer, mon1, mon2 = h2o_meoh_dimer_atoms
    calc = create_calculator_from_checkpoint(resolve_checkpoint(None))

    d = dimer.copy(); d.calc = calc; e_d = d.get_potential_energy()
    m1 = mon1.copy(); m1.calc = calc; e_m1 = m1.get_potential_energy()
    m2 = mon2.copy(); m2.calc = calc; e_m2 = m2.get_potential_energy()

    e_int_ev = e_d - e_m1 - e_m2
    e_int_kcal = e_int_ev * EV_TO_KCAL

    # Pinned interaction energy around 97.20 kcal/mol for unrelaxed 2.8 Å dimer
    assert e_int_kcal == pytest.approx(97.20, abs=2.0)


def test_spookynet_dimer_regression(h2o_meoh_dimer_atoms):
    dimer, mon1, mon2 = h2o_meoh_dimer_atoms
    calc = SpookyNetCalculator(mbd_checkpoint=False)

    d = dimer.copy(); d.calc = calc; e_d = d.get_potential_energy()
    m1 = mon1.copy(); m1.calc = calc; e_m1 = m1.get_potential_energy()
    m2 = mon2.copy(); m2.calc = calc; e_m2 = m2.get_potential_energy()

    e_int_ev = e_d - e_m1 - e_m2
    e_int_kcal = e_int_ev * EV_TO_KCAL

    # Pinned SpookyNet interaction energy around 342.15 kcal/mol
    assert e_int_kcal == pytest.approx(342.15, abs=5.0)


def test_qcml_mbd_dimer_regression(h2o_meoh_dimer_atoms):
    dimer, mon1, mon2 = h2o_meoh_dimer_atoms
    calc = QCMLMBDCalculator()

    d = dimer.copy(); d.calc = calc; e_d = d.get_potential_energy()
    m1 = mon1.copy(); m1.calc = calc; e_m1 = m1.get_potential_energy()
    m2 = mon2.copy(); m2.calc = calc; e_m2 = m2.get_potential_energy()

    e_int_ev = e_d - e_m1 - e_m2
    e_int_kcal = e_int_ev * EV_TO_KCAL

    # Pinned MBD attractive dispersion interaction energy around -1.65 kcal/mol
    assert e_int_kcal == pytest.approx(-1.65, abs=0.2)


def test_qcml_multipoles_dimer_regression(h2o_meoh_dimer_atoms):
    dimer, mon1, mon2 = h2o_meoh_dimer_atoms
    calc = LearnedMolecularMultipoleElectrostatics()

    d = dimer.copy(); d.calc = calc; e_d = d.get_potential_energy()
    m1 = mon1.copy(); m1.calc = calc; e_m1 = m1.get_potential_energy()
    m2 = mon2.copy(); m2.calc = calc; e_m2 = m2.get_potential_energy()

    e_int_ev = e_d - e_m1 - e_m2
    e_int_kcal = e_int_ev * EV_TO_KCAL

    # Pinned Multipole electrostatic interaction energy around -11.98 kcal/mol
    assert e_int_kcal == pytest.approx(-11.98, abs=0.5)


def test_hybrid_composite_dimer_regression(h2o_meoh_dimer_atoms):
    dimer, mon1, mon2 = h2o_meoh_dimer_atoms

    spooky_calc = SpookyNetCalculator(mbd_checkpoint=False)
    mbd_calc = QCMLMBDCalculator()
    mult_calc = LearnedMolecularMultipoleElectrostatics()

    def eval_composite(atoms):
        a1 = atoms.copy(); a1.calc = spooky_calc; e1 = a1.get_potential_energy()
        a2 = atoms.copy(); a2.calc = mbd_calc; e2 = a2.get_potential_energy()
        a3 = atoms.copy(); a3.calc = mult_calc; e3 = a3.get_potential_energy()
        return e1 + e2 + e3

    e_d = eval_composite(dimer)
    e_m1 = eval_composite(mon1)
    e_m2 = eval_composite(mon2)

    e_int_kcal = (e_d - e_m1 - e_m2) * EV_TO_KCAL

    # Pinned Hybrid composite interaction energy around 328.52 kcal/mol
    assert e_int_kcal == pytest.approx(328.52, abs=5.0)


@pytest.mark.pycharmm
def test_pycharmm_cgenff_dimer_regression(h2o_meoh_dimer_atoms):
    try:
        import pycharmm
        import pycharmm.settings as settings
        import pycharmm.generate as gen
        import pycharmm.coor as coor
        import pycharmm.minimize as minimize
        import pycharmm.energy as energy
        import pycharmm.psf as psf
        import pycharmm.read as read
        import pandas as pd
        from mmml.interfaces.pycharmmInterface.import_pycharmm import (
            CGENFF_PRM,
            CGENFF_RTF,
            reset_block,
            pycharmm_quiet,
        )
    except ImportError:
        pytest.skip("pyCHARMM not installed or importable")

    pycharmm_quiet()
    reset_block()
    settings.set_bomb_level(-2)

    read.rtf(CGENFF_RTF)
    read.prm(CGENFF_PRM)

    read.sequence_string("TIP3")
    gen.new_segment("A")
    read.sequence_string("MEOH")
    gen.new_segment("B")

    dimer, _, _ = h2o_meoh_dimer_atoms
    coor.set_positions(pd.DataFrame(dimer.get_positions(), columns=["x", "y", "z"]))

    # Minimize geometry using ABNER
    minimize.run_abnr(nstep=500, tolenr=1e-5, tolgrd=1e-3)

    pycharmm.lingo.charmm_script("ENER")
    e_d = float(energy.get_total())

    pycharmm.lingo.charmm_script("ENER sele segid A end")
    e_m1 = float(energy.get_total())

    pycharmm.lingo.charmm_script("ENER sele segid B end")
    e_m2 = float(energy.get_total())

    e_int_kcal = e_d - e_m1 - e_m2

    # Pinned CGenFF MM interaction energy on the ABNR-relaxed TIP3-methanol
    # dimer, ~-8.43 kcal/mol (a physically reasonable water-methanol H-bond).
    # The prior -3.79 pin was stale (it predated the current relaxed minimum).
    assert e_int_kcal == pytest.approx(-8.43, abs=0.5)
