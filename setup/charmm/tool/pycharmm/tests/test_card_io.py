"""Round-trip CHARMM coordinate-card I/O.

Replicates test/c31test/ioformat.inp: build a single TIP3 water, write
it to a .crd via write.coor_card, modify the structure, then read the
.crd back in and verify the readback ran cleanly.

Original by C. L. Brooks III, April 2019.
"""

import os

import pytest

from pycharmm import coor, lingo, read, write


def _find_toph19(data_dir):
    """The CHARMM 19 topology lives under different names depending on
    which toppar tree is on disk: 'toph19.inp' (modern toppar/) or
    'toph19.rtf' (test/data/). Return the first that exists."""
    for name in ("toph19.inp", "toph19.rtf"):
        path = os.path.join(data_dir, name)
        if os.path.isfile(path):
            return path
    return None


def _find_param19(data_dir):
    """Sibling helper for the CHARMM 19 parameter file."""
    for name in ("param19.inp", "param19.prm"):
        path = os.path.join(data_dir, name)
        if os.path.isfile(path):
            return path
    return None


def test_card_io_round_trip(tmp_path):
    """Write a coordinate card and read it back without error."""
    toph = _find_toph19("data")
    par = _find_param19("data")
    if toph is None or par is None:
        pytest.skip("toph19/param19 not found under tests/data")
    read.rtf(toph)
    read.prm(par)

    lingo.charmm_script("""
        read sequence tip3 1
        gene wat nodihe noangle
        coor set xdir 0.0 ydir 0.0 zdir 0.0
        print coor""")

    crd_path = tmp_path / "ioform1.crd"
    write.coor_card(str(crd_path), title="one water molecule, standard format")
    assert crd_path.is_file(), f"write.coor_card did not produce {crd_path}"

    lingo.charmm_script("""
        rename segid bigwater sele segid wat end
        q 1 2
        print coor
        q 1 2
        coor trans xdir 1.0
    """)

    read.coor_card(str(crd_path))
    coor.show()  # would raise if the readback put coordinates in a bad state
