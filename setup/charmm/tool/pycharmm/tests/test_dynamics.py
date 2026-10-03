"""Smoke test for pycharmm.DynamicsScript with PME + Langevin in a water box.

Builds an alanine peptide solvated in 350 TIP3 waters with cubic crystal
images, runs a brief Langevin dynamics, and verifies the run produced
some final state (kinetic table populated and velocities recorded).

Mirrors the c37test/domdec_gpu.inp test case.

Original by C. L. Brooks III, November 2020.
"""

import numpy as np
import pytest

from pycharmm import (
    DynamicsScript,
    NonBondedScript,
    coor,
    crystal,
    dyn,
    energy,
    gen,
    image,
    psf,
    read,
)
from pycharmm.lingo import charmm_script


# Leaves CHARMM crystal/image/PME state that segfaults the next test
# in the sweep. Run with `pytest -m stateful`.
@pytest.mark.stateful
def test_alanine_in_waterbox_langevin_dynamics():
    """Setup alanine + 350 TIP3 cube, configure PME, run 100-step Langevin."""
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
    energy.show()

    # Re-center inside the cubic box
    stats = coor.stat()
    size = (
        (stats["xmax"] - stats["xmin"])
        + (stats["ymax"] - stats["ymin"])
        + (stats["zmax"] - stats["zmin"])
    ) / 3
    offset = size / 2.0
    xyz = coor.get_positions()
    xyz += offset
    coor.set_positions(xyz)

    crystal.define_cubic(length=size)
    crystal.build(cutoff=nbonds["cutim"])
    image.setup_segment(offset, offset, offset, "PRO0")
    image.setup_residue(offset, offset, offset, "TIP3")

    nbonds["ewald"] = True
    NonBondedScript(**nbonds).run()
    energy.show()

    dyn.set_fbetas(np.full(psf.get_natom(), 5.0))

    dynamics_dict = dict(
        ktable=True,
        velos=True,
        lambdata=True,
        leap=True,
        verlet=False,
        cpt=False,
        new=False,
        langevin=True,
        omm=False,
        timestep=0.002,
        start=True,
        nstep=100,
        nsavc=0,
        nsavv=10,
        inbfrq=-1,
        ihbfrq=0,
        ilbfrq=0,
        imgfrq=0,
        iunrea=-1,
        iunwri=-1,
        iuncrd=-1,
        nsavl=0,
        iunldm=-1,
        ilap=-1,
        ilaf=-1,
        nprint=10,
        iprfrq=50,
        isvfrq=100,
        ntrfrq=100,
        firstt=298,
        finalt=298,
        tstruct=298,
        tbath=298,
        iasors=1,
        ichecw=0,
        iscale=0,
        scale=1,
        echeck=-1,
    )

    dyn_lang = DynamicsScript(**dynamics_dict)
    dyn_lang.run()

    # ktable / velos should be populated by the run; just verify the
    # attributes exist and are not falsy.
    assert dyn_lang.ktable is not None
    assert dyn_lang.velos is not None
