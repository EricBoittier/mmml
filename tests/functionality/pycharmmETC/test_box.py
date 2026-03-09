from pathlib import Path

import pytest


def _can_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _can_import("pycharmm"), reason="pycharmm not available")
def test_setup_box_generic_smoke():
    from mmml.interfaces.pycharmmInterface import setupBox

    pdb_path = Path("pdb/init-packmol.pdb")
    if not pdb_path.exists():
        pytest.skip(f"Missing input pdb: {pdb_path}")

    atoms = setupBox.setup_box_generic(str(pdb_path), side_length=10.0, tag="tip3")
    assert atoms is not None