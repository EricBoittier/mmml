"""Tests for pycharmm.cdocker — Rigid_CDOCKER and Flexible_CDOCKER.

These exercise the FFTDOCK pipeline against a benzene/T4-lysozyme system.
The shared CHARMM topology and parameters are loaded once per test module
via the `docking_setup` fixture; each test gets its own scratch tmpdir
and cleans up automatically.

Skips:
  - Both tests carry ``@pytest.mark.requires_feature("FFTDOCK")``; the
    conftest auto-skip hook drops them on builds without FFTDOCK
    compiled in.  Without that marker, ``Rigid_CDOCKER`` /
    ``Flexible_CDOCKER`` reach the CHARMM ``api_grid`` path, which
    calls ``wrndie(-1)`` -> ``_gfortran_exit`` and takes the whole
    pytest process down silently, dropping every alphabetically-later
    test in the collected order.  (See test_grid_ommd.py for the same
    pattern.)
  - Rigid CDOCKER additionally skips when MMTSB's ``obrotamer`` and
    ``convpdb.pl`` aren't on PATH (they pre-generate ligand rotamer
    conformers).  Today these MMTSB skipif markers happen to fire
    first on the install-sccdftb pipeline-dev config and inadvertently
    hide the FFTDOCK issue, but they shouldn't be relied on for that.

Pass criterion: top-1 docked ligand center is within 2 Å of the
co-crystal pose. Center-of-ligand is used (rather than full atom-pair
RMSD) because benzene is symmetric and atom labelling drifts.
"""

import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

import pycharmm.coor as coor
import pycharmm.generate as gen
import pycharmm.lingo as lingo
import pycharmm.psf as psf
import pycharmm.read as read
import pycharmm.settings as settings
from pycharmm.cdocker import Flexible_CDOCKER, Rigid_CDOCKER

# Grid box (T4 lysozyme L99A binding pocket)
GRID_CENTER = (26.9114167, 6.126, 4.179)
GRID_HALF_LEN = 15
RMSD_TOLERANCE_A = 2.0


def _have(tool):
    """Return True if `tool` is callable on PATH."""
    return shutil.which(tool) is not None


@pytest.fixture(scope="module")
def docking_setup():
    """Load topology and parameters for protein + benzene ligand.

    Module-scoped: runs once for the whole file. Individual tests
    invalidate the PSF as needed via `psf.delete_atoms()` but the
    loaded RTF/PRM remain.
    """
    settings.set_bomb_level(-1)
    read.rtf("data/top_all36_prot.rtf")
    read.rtf("data/top_all36_cgenff.rtf", append=True)
    read.prm("data/par_all36m_prot.prm", flex=True)
    read.prm("data/par_all36_cgenff.prm", append=True, flex=True)
    settings.set_bomb_level(0)
    lingo.charmm_script("stream data/benzene.rtf")


@pytest.fixture
def ligand_pdb(tmp_path):
    """Stage benzene.pdb as `<tmp>/scratch/ligand.pdb` and return the path."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    dest = scratch / "ligand.pdb"
    shutil.copy("data/benzene.pdb", dest)
    return dest


def _benzene_center():
    """Reload benzene.pdb into a fresh segment and return its centroid."""
    psf.delete_atoms()
    read.sequence_pdb("data/benzene.pdb")
    settings.set_bomb_level(-1)
    gen.new_segment(seg_name="LIGA")
    settings.set_bomb_level(0)
    read.pdb("data/benzene.pdb", resid=True)
    return coor.get_positions().to_numpy().mean(axis=0)


def _docked_center(docked_pdb):
    """Read a docked-pose PDB into the existing segment and return centroid."""
    read.pdb(str(docked_pdb), resid=True)
    return coor.get_positions().to_numpy().mean(axis=0)


def _ligand_center_rmsd(docked_pdb):
    """Center-of-ligand RMSD between benzene.pdb and a docked pose PDB.

    Center-of-ligand (rather than per-atom RMSD) is used because benzene
    has 6-fold symmetry; atom labelling between docked poses and the
    crystal reference is not stable.
    """
    crystal = _benzene_center()
    docked = _docked_center(docked_pdb)
    return float(np.linalg.norm(crystal - docked))


@pytest.mark.requires_feature("FFTDOCK")
@pytest.mark.skipif(not _have("obrotamer"), reason="MMTSB tool 'obrotamer' not on PATH")
@pytest.mark.skipif(not _have("convpdb.pl"), reason="MMTSB tool 'convpdb.pl' not on PATH")
def test_rigid_cdocker_recovers_crystal_pose(docking_setup, ligand_pdb, tmp_path):
    """Rigid CDOCKER docks benzene back into T4 within RMSD_TOLERANCE_A."""
    # Pre-generate three rotamer conformations via MMTSB
    confdir = tmp_path / "conformer"
    confdir.mkdir()
    for rotamer in range(1, 4):
        out = confdir / f"{rotamer}.pdb"
        subprocess.run(
            f"obrotamer data/benzene.pdb | convpdb.pl -segnames -setseg LIGA > {out}",
            shell=True,
            check=True,
        )

    save_dir = tmp_path / "result"
    placement_dir = tmp_path / "placement"

    xcen, ycen, zcen = GRID_CENTER
    Rigid_CDOCKER(
        xcen=xcen,
        ycen=ycen,
        zcen=zcen,
        maxlen=GRID_HALF_LEN,
        numPlace=10,
        confDir=str(confdir) + "/",
        receptorPDB="data/t4.pdb",
        receptorPSF="data/t4.psf",
        ligPDB=str(ligand_pdb),
        saveDir=str(save_dir) + "/",
        placementDir=str(placement_dir) + "/",
        probeFile="./data/fftdock_c36prot_cgenff_probes.txt",
    )

    top1 = save_dir / "cluster" / "top_1.pdb"
    assert top1.is_file(), f"Expected top-1 pose at {top1}"
    rmsd = _ligand_center_rmsd(top1)
    assert rmsd <= RMSD_TOLERANCE_A, (
        f"Rigid CDOCKER top-1 ligand-center RMSD = {rmsd:.3f} Å, expected <= {RMSD_TOLERANCE_A} Å"
    )


@pytest.mark.requires_feature("FFTDOCK")
@pytest.mark.slow
def test_flexible_cdocker_recovers_crystal_pose(docking_setup, ligand_pdb, tmp_path):
    """Flexible CDOCKER (with side-chain flexibility) docks within tolerance."""
    flexchain = pd.DataFrame(
        {
            "res_id": [84, 87, 99, 111, 118],
            "seg_id": ["PROT"] * 5,
        }
    )
    psf.delete_atoms()

    save_dir = tmp_path / "result"
    xcen, ycen, zcen = GRID_CENTER

    Flexible_CDOCKER(
        xcen=xcen,
        ycen=ycen,
        zcen=zcen,
        maxlen=GRID_HALF_LEN,
        flexchain=flexchain,
        probeFile="data/fftdock_c36prot_cgenff_probes.txt",
        receptorPDB="data/t4.pdb",
        receptorPSF="data/t4.psf",
        ligPDB=str(ligand_pdb),
        saveLig=str(tmp_path / "ligand") + "/",
        saveProt=str(tmp_path / "protein") + "/",
        crossoverLig=str(tmp_path / "crossover_ligand") + "/",
        crossoverProt=str(tmp_path / "crossover_protein") + "/",
        saveLigFinal=str(tmp_path / "ligand_final") + "/",
        saveProtFinal=str(tmp_path / "protein_final") + "/",
        placementDir=str(tmp_path / "placement") + "/",
        num=10,
        copy=10,
        saveDir=str(save_dir) + "/",
    )

    top1 = save_dir / "cluster" / "ligand" / "top_1.pdb"
    assert top1.is_file(), f"Expected top-1 pose at {top1}"
    rmsd = _ligand_center_rmsd(top1)
    assert rmsd <= RMSD_TOLERANCE_A, (
        f"Flexible CDOCKER top-1 ligand-center RMSD = {rmsd:.3f} Å, "
        f"expected <= {RMSD_TOLERANCE_A} Å"
    )
