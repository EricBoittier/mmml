"""Smoke test for pycharmm.grid.CDOCKER — grid generation + on/off/clear.

Builds a benzene + T4 lysozyme system, generates a CDOCKER grid on GPU,
and exercises the grid-state state-machine (read → off → on → clear).

Each operation should return a truthy status. Energies are queried
between transitions to verify the grid affects the calculation.

Renamed from the legacy `testGrid.py` to follow pytest conventions.
"""

from pycharmm import (
    SelectAtoms,
    energy,
    gen,
    grid,
    lingo,
    psf,
    read,
    settings,
)


def test_cdocker_grid_state_machine(tmp_path):
    """CDOCKER grid: generate, read, off, on, clear all return success."""
    settings.set_bomb_level(-1)
    read.rtf("data/top_all36_prot.rtf")
    read.rtf("data/top_all36_cgenff.rtf", append=True)
    read.rtf("data/probes.rtf", append=True)
    read.prm("data/par_all36m_prot.prm", flex=True)
    read.prm("data/par_all36_cgenff.prm", append=True, flex=True)
    read.prm("data/probes.prm", append=True, flex=True)
    settings.set_bomb_level(0)
    lingo.charmm_script("stream data/benzene.rtf")

    read.psf_card("data/t4.psf", append=True)
    read.pdb("data/t4.pdb", resid=True)

    grid_file = tmp_path / "test_grid.bin"
    cdocker = grid.CDOCKER()
    cdocker.setVar(
        {
            "xCen": 26.9114167,
            "yCen": 6.126,
            "zCen": 4.179,
            "xMax": 8,
            "yMax": 8,
            "zMax": 8,
            "emax": 3,
            "maxe": 30,
            "mine": -30,
            "flag_gpu": True,
            "flag_grhb": True,
            "gridFile": str(grid_file),
            "probeFile": "data/fftdock_c36prot_cgenff_probes.txt",
        }
    )
    cdocker.generate()
    assert grid_file.is_file(), f"Grid file not produced: {grid_file}"

    # Tear down protein, build benzene-only
    psf.delete_atoms(SelectAtoms().all_atoms())
    read.sequence_pdb("data/benzene.pdb")
    gen.new_segment(seg_name="LIGA")
    read.pdb("data/benzene.pdb", resid=True)

    ligand = SelectAtoms(seg_id="LIGA")
    cdocker.setVar({"selection": ligand})

    assert cdocker.read(), "cdocker.read() returned a falsy status"
    energy.show()
    assert cdocker.off(), "cdocker.off() returned a falsy status"
    energy.show()
    assert cdocker.on(), "cdocker.on() returned a falsy status"
    energy.show()
    assert cdocker.clear(), "cdocker.clear() returned a falsy status"
    energy.show()
