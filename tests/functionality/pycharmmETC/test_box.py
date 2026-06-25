from pathlib import Path

import pytest


def _can_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _can_import("pycharmm"), reason="pycharmm not available")
def test_setup_box_generic_smoke(pycharmm_workdir: Path):
    from mmml.interfaces.pycharmmInterface import setupBox

    from tests.functionality.pycharmmETC._paths import workdir_pdb

    pdb_path = workdir_pdb("init-packmol.pdb")
    if not (pycharmm_workdir / pdb_path).is_file():
        pytest.skip(f"Missing input pdb seed in workdir: {pdb_path}")

    atoms = setupBox.setup_box_generic(
        str(pdb_path), side_length=10.0, tag="tip3", skip_energy_show=True
    )
    assert atoms is not None
    assert (pycharmm_workdir / "psf").exists()