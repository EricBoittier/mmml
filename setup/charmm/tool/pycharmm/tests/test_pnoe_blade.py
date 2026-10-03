"""
Test PNOE (Point NOE) functionality in BLaDE.

A dummy atom is placed at the origin and progressively moved
along the x axis. NOE energies and forces from CPU and BLaDE are compared.

NOE is set up such that no potential is acted on the dummy atom until it is
moved outside a radius.

Original author: Thanh Lai, October 8, 2025
"""

import os

import numpy as np
import pytest

import pycharmm
import pycharmm.coor as coor
import pycharmm.crystal as crystal
import pycharmm.energy as energy
import pycharmm.generate as gen
import pycharmm.image as image
import pycharmm.read as read
from pycharmm.lingo import charmm_script


def _has_slurm_gpu_allocation():
    """BLaDE tests require a real GPU allocation, not just a local CPU env."""
    return bool(os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_STEP_ID"))


@pytest.mark.skipif(
    not _has_slurm_gpu_allocation(),
    reason="Requires a GPU allocation for BLaDE; run under srun -p Super --gres=gpu:1",
)
def test_pnoe_blade_energy_force():
    """Test that PNOE restraints on BLaDE match CPU NOE implementation."""
    radius = 2.0  # harmonic NOE activates past this radius
    restraint_reference = [0.0, 0.0, 0.0]
    dxs = np.arange(0, 3.5, 0.5)
    cpu_energies = []
    blade_energies = []
    cpu_forces = []
    blade_forces = []

    # Setup topology and parameters for dummy atom
    charmm_script("""
                  read rtf card
                   * Single atom topology file
                   *
                      20    1
                   MASS     -1 X     12.0

                   auto angles dihe patch

                   RESI DUMB       0.0
                   GROUP
                   ATOM A    X     0.0
                   PATC  FIRS NONE LAST NONE
                   END

                  read param card flex
                   * dummy parameters for testing
                   *
                   atom
                   MASS     -1 X     12.0

                   !
                   NONBONDED   ATOM CDIEL SWITCH VATOM VDISTANCE VSWITCH -
                        CUTNB 8.0  CTOFNB 7.5  CTONNB 6.5  EPS 1.0  E14FAC 1.0  WMIN 1.5
                   !
                   X        0.0000    -0.0       0.0000

                   END
                  """)

    read.sequence_string("DUMB")
    gen.new_segment(seg_name="DUMB")
    positions = coor.get_positions()
    positions.iloc[0] = np.array([0.0, 0.0, 0.0])
    coor.set_positions(positions)

    # Setup PBC for BLaDE
    box_size = 200
    origin = 0.0

    xyz = coor.get_positions()
    xyz["x"] += origin
    xyz["y"] += origin
    xyz["z"] += origin
    coor.set_positions(xyz)

    crystal.define_cubic(box_size)
    crystal.build(box_size / 2.0)

    image.setup_residue(origin, origin, origin, "DUMB")

    # Setup nonbonds
    nbonds_dict = {
        "atom": True,
        "vatom": True,
        "cdie": True,
        "eps": 1.0,
        "inbfrq": -1,
        "imgfrq": -1,
        "vfswitch": True,
        "fswitch": True,
    }
    nbonds = pycharmm.NonBondedScript(**nbonds_dict)
    nbonds.run()

    # Setup NOE restraint and compute energies on CPU
    charmm_script(f"""
        NOE
        assign -
        kmin 0 rmin 0.0 kmax 5 rmax {radius} fmax 9999 -
        cnox {restraint_reference[0]} cnoy {restraint_reference[1]} cnoz {restraint_reference[2]} -
        sele segid DUMB .and. resid 1 .and. type A end
        print analysis
        END""")

    energy.show()

    for dx in dxs:
        positions = coor.get_positions()
        positions.iloc[0] = np.array([dx, 0.0, 0.0])
        coor.set_positions(positions)

        energy.show()

        cpu_energies.append(energy.get_total())
        cpu_forces.append(np.array(coor.get_forces().iloc[0]))

    # Reset position and NOE, compute energies on BLaDE (requires GPU)
    positions = coor.get_positions()
    positions.iloc[0] = np.array([0.0, 0.0, 0.0])
    coor.set_positions(positions)

    charmm_script("""
                  NOE
                    RESET
                  END""")

    charmm_script(f"""
        NOE
        assign -
        kmin 0 rmin 0.0 kmax 5 rmax {radius} fmax 9999 -
        cnox {restraint_reference[0]} cnoy {restraint_reference[1]} cnoz {restraint_reference[2]} -
        sele segid DUMB .and. resid 1 .and. type A end
        print analysis
        END""")

    charmm_script("energy blade")

    for dx in dxs:
        positions = coor.get_positions()
        positions.iloc[0] = np.array([dx, 0.0, 0.0])
        coor.set_positions(positions)

        charmm_script("energy blade")

        blade_energies.append(energy.get_total())
        blade_forces.append(np.array(coor.get_forces().loc[0]))

    cpu_energies = np.array(cpu_energies)
    blade_energies = np.array(blade_energies)
    cpu_forces = np.array(cpu_forces)
    blade_forces = np.array(blade_forces)

    print("Displacements:", dxs, flush=True)
    print("CPU energies:", cpu_energies, flush=True)
    print("BLaDE energies:", blade_energies, flush=True)
    print("Energy difference:", cpu_energies - blade_energies, flush=True)
    print("CPU forces:", cpu_forces, flush=True)
    print("BLaDE forces:", blade_forces, flush=True)
    print("Force difference:", cpu_forces - blade_forces, flush=True)

    # Assert energy match
    np.testing.assert_allclose(
        cpu_energies,
        blade_energies,
        rtol=1e-5,
        err_msg="PNOE energies from BLaDE do not match CPU implementation",
    )

    # Assert force match (skip first point where atom is at reference - force undefined)
    np.testing.assert_allclose(
        cpu_forces[1:],
        blade_forces[1:],
        rtol=1e-5,
        err_msg="PNOE forces from BLaDE do not match CPU implementation",
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
