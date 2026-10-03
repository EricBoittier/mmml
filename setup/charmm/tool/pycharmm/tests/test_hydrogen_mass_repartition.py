"""Test psf.hmr() — hydrogen mass repartitioning preserves total mass.

After HMR, the per-atom mass distribution shifts (hydrogens get heavier,
their parents get lighter) but the total mass should be invariant.

The original test also exercised an MMTSB convpdb.pl-driven solvation
pipeline, but the mass conservation check is independent of the
solvent. The original convpdb-related steps are kept (and skipped when
convpdb.pl is not on PATH) since they round out the workflow.

Original by C. L. Brooks III, February 2025.
"""

import shlex
import shutil
import subprocess

import numpy as np
import pytest

from pycharmm import (
    NonBondedScript,
    coor,
    gen,
    ic,
    minimize,
    psf,
    read,
    settings,
    write,
)

HAS_CONVPDB = shutil.which("convpdb.pl") is not None


def _build_alanine_dipeptide():
    """Read RTF/PRM and build a capped alanine, ready for the HMR call."""
    read.rtf("data/top_all36_prot.rtf")
    read.prm("data/par_all36_prot.prm", flex=True)

    old_warn = settings.set_warn_level(-1)
    old_bomb = settings.set_bomb_level(-1)
    read.stream("data/toppar_water_ions.str")
    settings.set_warn_level(old_warn)
    settings.set_bomb_level(old_bomb)

    read.sequence_string("ALA")
    gen.new_segment("ADP", "ACE", "CT3", setup_ic=True)
    ic.prm_fill(False)
    ic.seed(1, "CAY", 1, "CY", 1, "N")
    ic.build()
    coor.orient(by_rms=False, by_mass=False, by_noro=False)

    NonBondedScript(
        cutnb=18.0,
        ctonnb=15.0,
        ctofnb=13.0,
        eps=1.0,
        cdie=True,
        atom=True,
        vatom=True,
        fswitch=True,
        vfswitch=True,
    ).run()
    minimize.run_abnr(nstep=1000, tolenr=1e-3, tolgrd=1e-3)


def test_hmr_preserves_total_mass():
    """psf.hmr() reshuffles per-atom masses but keeps the sum invariant."""
    _build_alanine_dipeptide()

    old_masses = psf.get_amass()
    psf.hmr()
    new_masses = psf.get_amass()

    delta = abs(np.sum(old_masses) - np.sum(new_masses))
    assert delta <= 1e-5, f"HMR changed total mass by {delta} (expected <= 1e-5)"


@pytest.mark.skipif(not HAS_CONVPDB, reason="convpdb.pl (MMTSB) not on PATH")
def test_hmr_with_solvation(tmp_path):
    """Full HMR + convpdb solvation pipeline (skipped without MMTSB)."""
    _build_alanine_dipeptide()

    adp_pdb = tmp_path / "adp.pdb"
    write.coor_pdb(str(adp_pdb))

    # convpdb.pl pipeline: solvate adp.pdb, then re-segment as TIP3.
    p1 = subprocess.Popen(
        shlex.split(f"convpdb.pl -solvate -cutoff 10 -cubic -out charmm22 {adp_pdb}"),
        stdout=subprocess.PIPE,
    )

    wt00_pdb = tmp_path / "wt00.pdb"
    with open(wt00_pdb, "w") as fh:
        p2 = subprocess.Popen(
            shlex.split("convpdb.pl -segnames -nsel TIP3"), stdin=p1.stdout, stdout=fh
        )
    p2.communicate()

    read.sequence_pdb(str(wt00_pdb))
    gen.new_segment("WT00", angle=False, dihedral=False)
    read.pdb(str(wt00_pdb), resid=True)

    old_masses = psf.get_amass()
    psf.hmr()
    new_masses = psf.get_amass()
    assert abs(np.sum(old_masses) - np.sum(new_masses)) <= 1e-5
