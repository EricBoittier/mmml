"""Test crystal.define_cubic and image.setup_segment/residue.

Builds an alanine peptide solvated in 350 TIP3 waters, sets up a cubic
crystal image, and verifies that the energy is finite both with and
without Ewald summation. This is essentially the test_dynamics setup
without the dynamics step.

Original by C. L. Brooks III, November 2020.
"""

import math

import pytest

from pycharmm import (
    NonBondedScript,
    coor,
    crystal,
    energy,
    gen,
    image,
    read,
)
from pycharmm.lingo import charmm_script


# Leaves CUTIM=11 in CHARMM, which trips `CUTNB > CUTIM` in the next
# nbonds list build elsewhere in the sweep. Run with `pytest -m stateful`.
@pytest.mark.stateful
def test_alanine_in_waterbox_pme_and_ewald():
    """Cubic box + PME and Ewald nonbonded both produce a finite energy."""
    read.rtf("data/top_all36_prot.rtf")
    read.prm("data/par_all36_prot.prm", flex=True)
    charmm_script("stream data/toppar_water_ions.str")

    read.sequence_string("ALA")
    gen.new_segment(seg_name="PRO0", first_patch="ACE", last_patch="CT3", setup_ic=True)
    charmm_script("read sequ tip3 350")
    gen.new_segment(seg_name="WT00", angle=False, dihedral=False)
    read.pdb("data/ws0.pdb", resid=True)

    nbonds = dict(
        elec=True,
        atom=True,
        cdie=True,
        eps=1,
        switch=True,
        pmewald=True,
        kappa=0.32,
        fftx=24,
        ffty=24,
        fftz=24,
        order=4,
        vdw=True,
        vatom=True,
        vswitch=True,
        cutnb=11,
        ctofnb=10,
        ctonnb=10,
    )
    nbonds["cutim"] = nbonds["cutnb"]
    NonBondedScript(**nbonds).run()
    e_pme_no_image = energy.get_total()
    assert math.isfinite(e_pme_no_image)

    stats = coor.stat()
    size = (
        (stats["xmax"] - stats["xmin"])
        + (stats["ymax"] - stats["ymin"])
        + (stats["zmax"] - stats["zmin"])
    ) / 3
    offset = size / 2
    xyz = coor.get_positions()
    xyz += offset
    coor.set_positions(xyz)

    crystal.define_cubic(length=size)
    crystal.build(cutoff=nbonds["cutim"])
    image.setup_segment(offset, offset, offset, "PRO0")
    image.setup_residue(offset, offset, offset, "TIP3")

    e_pme_with_image = energy.get_total()
    assert math.isfinite(e_pme_with_image)

    nbonds["ewald"] = True
    NonBondedScript(**nbonds).run()
    e_ewald = energy.get_total()
    assert math.isfinite(e_ewald)


# Same solvated cubic PME box built both ways; leaves crystal/image state
# behind like the test above, so run with `pytest -m stateful`.
@pytest.mark.stateful
def test_build_matches_native_command():
    """`crystal.build()` must reproduce the native `CRYSTAL BUILD` command.

    Regression test for issue #24. The crystal.build() wrapper crosses
    into Fortran through ``crystal_build`` in ``source/api/api_crystal.F90``,
    passing the symmetry-operation count ``nops`` by value; the binding
    must declare that argument ``value`` to match. When it did not, the
    C-interop read a garbage operation count and produced a wrong (often
    still finite) energy, while ``charmm_script('crystal build ...')``
    gave the correct result -- exactly the "gives garbage" behaviour in
    the report.

    Building the identical box both ways and requiring the total energies
    to agree catches that class of bug; the finiteness check in the test
    above does not, since garbage energies can still be finite.
    """
    read.rtf("data/top_all36_prot.rtf")
    read.prm("data/par_all36_prot.prm", flex=True)
    charmm_script("stream data/toppar_water_ions.str")

    read.sequence_string("ALA")
    gen.new_segment(seg_name="PRO0", first_patch="ACE", last_patch="CT3", setup_ic=True)
    charmm_script("read sequ tip3 350")
    gen.new_segment(seg_name="WT00", angle=False, dihedral=False)
    read.pdb("data/ws0.pdb", resid=True)

    nbonds = dict(
        elec=True,
        atom=True,
        cdie=True,
        eps=1,
        switch=True,
        pmewald=True,
        kappa=0.32,
        fftx=24,
        ffty=24,
        fftz=24,
        order=4,
        vdw=True,
        vatom=True,
        vswitch=True,
        cutnb=11,
        ctofnb=10,
        ctonnb=10,
    )
    nbonds["cutim"] = nbonds["cutnb"]
    NonBondedScript(**nbonds).run()

    stats = coor.stat()
    size = (
        (stats["xmax"] - stats["xmin"])
        + (stats["ymax"] - stats["ymin"])
        + (stats["zmax"] - stats["zmin"])
    ) / 3
    offset = size / 2
    xyz = coor.get_positions()
    xyz += offset
    coor.set_positions(xyz)

    def build_and_energy(use_api):
        crystal.define_cubic(length=size)
        if use_api:
            crystal.build(cutoff=nbonds["cutim"])
        else:
            # NOPER 0 == no extra symmetry operations, i.e. identity only,
            # matching crystal.build(cutoff) with no sym_ops.
            charmm_script(f"crystal build cutoff {nbonds['cutim']:f} noper 0")
        image.setup_segment(offset, offset, offset, "PRO0")
        image.setup_residue(offset, offset, offset, "TIP3")
        energy.show()
        return energy.get_total(), image.get_ntrans()

    e_api, ntrans_api = build_and_energy(use_api=True)
    crystal.free()
    e_script, ntrans_script = build_and_energy(use_api=False)

    assert math.isfinite(e_api)
    assert ntrans_api == ntrans_script, (
        f"crystal.build() built {ntrans_api} image transformations but the "
        f"native command built {ntrans_script}"
    )
    assert e_api == pytest.approx(e_script, rel=1e-6, abs=1e-4), (
        f"crystal.build() energy {e_api} disagrees with the native "
        f"'crystal build' command energy {e_script}"
    )
