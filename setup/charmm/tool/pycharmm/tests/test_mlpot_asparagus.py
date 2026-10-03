"""End-to-end MLpot: an Asparagus model driving ammonia in TIP3P water.

Companion to test_mlpot_energy_terms.py. That one drives the MLPO/MLEL
term plumbing with stub callbacks and needs nothing installed; this one
runs the real thing -- a trained PaiNN model of ammonia, in water,
through minimization and dynamics.

Not a chemically useful setup: the model cannot describe proton
transfer from water to ammonia. It exercises the interface.

Skipped unless the optional pieces are present:

  * ASE, for reading atomic numbers out of the PDB;
  * the Asparagus package (https://github.com/MMunibas/Asparagus);
  * the trained model under data/model_nh3.

CI has none of those, so this skips there.

Marked ``slow`` -- two 1 ps runs at a 0.25 fs timestep is 8000 steps,
each one a neural-network evaluation. Marked ``stateful`` because it
reads a parameter set, builds images and sets nonbonded options, all of
which a later test in the same process would inherit.

Original script by Kai Toepfer (kai.toepfer@fu-berlin.de); restructured
here as a pytest test. It still runs standalone -- see the bottom of the
file.
"""

import math
from pathlib import Path

import pytest

ase_io = pytest.importorskip("ase.io", reason="MLpot end-to-end test needs ASE")
_asparagus = pytest.importorskip(
    "asparagus", reason="MLpot end-to-end test needs the Asparagus package"
)
Asparagus = _asparagus.Asparagus

import pycharmm  # noqa: E402  (after the optional-dependency gate)
import pycharmm.cons_fix as cons_fix  # noqa: E402
import pycharmm.crystal as crystal  # noqa: E402
import pycharmm.energy as energy  # noqa: E402
import pycharmm.lingo as stream  # noqa: E402
import pycharmm.minimize as minimize  # noqa: E402
import pycharmm.read as read  # noqa: E402
import pycharmm.settings as settings  # noqa: E402

MODEL_CONFIG = Path("data/model_nh3/config.json")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.stateful,
    pytest.mark.skipif(
        not MODEL_CONFIG.is_file(),
        reason=f"trained MLpot model not present at {MODEL_CONFIG}",
    ),
]

# 0.25 fs steps, 1 ps per leg. Short for a simulation, long for a test;
# this is why the module is marked slow.
TIMESTEP = 0.00025
STEPS_PER_PS = int(1.0 / TIMESTEP)
TEMPERATURE = 300.0


def _build_solvated_ammonia():
    """Ammonia in a 30 A cube of TIP3P, set up for an ML/MM run."""
    settings.set_bomb_level(-1)
    settings.set_warn_level(-1)

    read.rtf("data/top_all36_cgenff.rtf")
    read.prm("data/par_all36_cgenff.prm", flex=True)
    stream.charmm_script("stream data/toppar_water_ions.str")

    settings.set_bomb_level(-2)

    stream.charmm_script(
        """
        open read card unit 10 name data/ammonia.pdb
        read sequence pdb unit 10
        generate AMM1 setup warn first none last none

        open read card unit 10 name data/ammonia.pdb
        read coor pdb  unit 10 resid
        """
    )

    read.psf_card("data/water_cube.psf", append=True)
    read.coor_card("data/water_cube.crd", append=True)

    stream.charmm_script(
        """
        delete atom select ( .byres. ( (segid AMM1 .around. 2.0 ) -
            .and. (segid TIP3 .and. type OH2 ))) end
        """
    )

    # lrc=False is required, not a preference. With every atom in the ML
    # selection the long-range correction cannot cope with atoms whose
    # bonds are broken and which are fully excluded from the nonbonded
    # list, and the energy comes back as NaN or Infinity. See
    # doc/mlpot.info.
    pycharmm.NonBondedScript(
        atom=True,
        vdw=True,
        vswitch=True,
        cutnb=14,
        ctofnb=12,
        ctonnb=10,
        cutim=14,
        lrc=False,
        inbfrq=-1,
        imgfrq=-1,
    ).run()

    crystal.define_cubic(length=30.0)
    crystal.build(cutoff=14.0)
    stream.charmm_script(
        "image byres xcen 0.0 ycen 0.0 zcen 0.0 sele all end"
    )
    stream.charmm_script("shake bonh para sele resname TIP3 end")


def _attach_mlpot():
    """Hand the ML segment over to the Asparagus model."""
    ml_Z = ase_io.read(
        "data/ammonia.pdb", format="proteindatabank"
    ).get_atomic_numbers()
    return pycharmm.MLpot(
        Asparagus(config=str(MODEL_CONFIG)),
        ml_Z,
        pycharmm.SelectAtoms(seg_id="AMM1"),
        ml_charge=0,
        ml_fq=True,
    )


def _minimize_with_ml_atoms_fixed():
    """Relax the solvent around the ML atoms, then relax everything."""
    settings.set_bomb_level(-2)
    cons_fix.setup(pycharmm.SelectAtoms(seg_id="AMM1"))
    minimize.run_sd(nstep=100, nprint=10, tolenr=1e-5, tolgrd=1e-5)
    cons_fix.turn_off()
    settings.set_bomb_level(-1)
    minimize.run_sd(nstep=100, nprint=10, tolenr=1e-5, tolgrd=1e-5)


def _run_dynamics(scratch, *, restart_from=None, name, nstep):
    """One leg of dynamics. Heating when restart_from is None, else NVE."""
    units = []
    read_unit = None
    if restart_from is not None:
        read_file = pycharmm.CharmmFile(
            file_name=str(restart_from), file_unit=3,
            formatted=True, read_only=False,
        )
        units.append(read_file)
        read_unit = read_file.file_unit

    res_file = pycharmm.CharmmFile(
        file_name=str(scratch / f"{name}.res"), file_unit=2,
        formatted=True, read_only=False,
    )
    dcd_file = pycharmm.CharmmFile(
        file_name=str(scratch / f"{name}.dcd"), file_unit=1,
        formatted=False, read_only=False,
    )
    units += [res_file, dcd_file]

    options = dict(
        verlet=True,
        timestep=TIMESTEP,
        nstep=nstep,
        nsavc=int(0.1 / TIMESTEP),
        inbfrq=-1,
        ihbfrq=50,
        ilbfrq=50,
        imgfrq=50,
        ixtfrq=1000,
        iunwri=res_file.file_unit,
        iuncrd=dcd_file.file_unit,
        nprint=100,
        iprfrq=500,
        isvfrq=1000,
        echeck=-1,
    )
    if restart_from is None:
        options.update(
            new=True, start=True,
            ntrfrq=1000, ihtfrq=200, ieqfrq=1000,
            firstt=TEMPERATURE / 2.0, finalt=TEMPERATURE, tbath=TEMPERATURE,
        )
    else:
        options.update(
            new=False, start=False, restart=True,
            iunrea=read_unit,
            ntrfrq=0, ihtfrq=0, ieqfrq=0,
        )

    try:
        pycharmm.DynamicsScript(**options).run()
    finally:
        for handle in units:
            handle.close()

    return scratch / f"{name}.res"


def test_mlpot_drives_ammonia_in_water(scratch_dir):
    """The model contributes a finite MLPO energy and the run stays finite.

    The checks, in order:

      * MLPO and MLEL read zero before a model is attached;
      * attaching one gives a non-zero, finite MLPO;
      * the total energy is finite -- this is the lrc=False contract,
        and the thing that regressed into NaN when it was not honoured;
      * minimization lowers the energy;
      * NVE dynamics conserves total energy to within a loose bound.
    """
    _build_solvated_ammonia()

    energy.show()
    assert energy.get_eterm(energy.TERM_MLPO) == pytest.approx(0.0)
    assert energy.get_eterm(energy.TERM_MLEL) == pytest.approx(0.0)

    _attach_mlpot()

    energy.show()
    e_ml = energy.get_eterm(energy.TERM_MLPO)
    assert math.isfinite(e_ml), "MLpot returned a non-finite internal energy"
    assert e_ml != 0.0, "the ML model contributed nothing to the energy"

    e_before = energy.get_eprop(energy.EPROP_EPOT)
    assert math.isfinite(e_before), (
        "total energy is not finite; with an all-ML selection this is what "
        "lrc=True produces -- see doc/mlpot.info"
    )

    _minimize_with_ml_atoms_fixed()

    energy.show()
    e_after = energy.get_eprop(energy.EPROP_EPOT)
    assert math.isfinite(e_after)
    assert e_after < e_before, "minimization did not lower the energy"

    restart = _run_dynamics(
        scratch_dir, name="heat", nstep=STEPS_PER_PS, restart_from=None
    )
    e_heated = energy.get_eprop(energy.EPROP_TOTE)
    assert math.isfinite(e_heated)

    _run_dynamics(
        scratch_dir, name="nve", nstep=STEPS_PER_PS, restart_from=restart
    )
    e_nve = energy.get_eprop(energy.EPROP_TOTE)
    assert math.isfinite(e_nve)

    # Deliberately loose. NVE here is checking that the ML forces are the
    # gradient of the ML energy -- a sign error or a missing term shows up
    # as drift of tens of percent, not tenths. Tighten once this has been
    # run enough times to know the real figure.
    drift = abs(e_nve - e_heated) / abs(e_heated)
    assert drift < 0.05, f"NVE total energy drifted by {drift:.1%}"


if __name__ == "__main__":
    # Standalone use, as the original script was run:  python test_mlpot_asparagus.py
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        test_mlpot_drives_ammonia_in_water(Path(tmp))
    print("MLpot end-to-end run finished")
