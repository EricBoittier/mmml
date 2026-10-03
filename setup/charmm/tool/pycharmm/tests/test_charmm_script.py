"""Build an alanine dipeptide via the pyCHARMM API and write a DCD.

A smoke test for the basic pyCHARMM workflow:
  - read RTF/PRM
  - generate a single-residue alanine peptide
  - seed and build internal coordinates
  - orient
  - configure non-bonded interactions
  - write a trajectory file via lingo.charmm_script

Original by C. L. Brooks III, April 2019.
"""

from pycharmm import coor
from pycharmm.lingo import charmm_script


def test_build_alanine_dipeptide_and_write_dcd(alanine_dipeptide_with_nbonds, tmp_path):
    """Build alanine dipeptide and emit a DCD trajectory + crd."""
    coor.show()

    dcd_path = tmp_path / "traj.dcd"
    crd_path = tmp_path / "test.crd"
    charmm_script(f"""
        open write file unit 80 name {dcd_path}
        traj IWRITE 80 NWRITE 1 NFILE 360 SKIP 1
        traj write
        close unit 80
        write coor card name {crd_path}
    """)

    assert dcd_path.is_file(), f"DCD not written: {dcd_path}"
    assert crd_path.is_file(), f"CRD not written: {crd_path}"
    assert dcd_path.stat().st_size > 0, "DCD is empty"
    assert crd_path.stat().st_size > 0, "CRD is empty"
