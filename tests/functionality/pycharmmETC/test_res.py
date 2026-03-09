import pytest


def _can_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _can_import("pycharmm"), reason="pycharmm not available")
def test_setup_res_smoke():
    from mmml.interfaces.pycharmmInterface import setupRes
    from mmml.interfaces.pycharmmInterface.import_pycharmm import (
        reset_block,
        reset_block_no_internal,
    )

    atoms = setupRes.main("TIP3")
    atoms = setupRes.generate_coordinates()
    positions = setupRes.coor.get_positions()
    atoms.set_positions(positions)
    reset_block()
    reset_block_no_internal()
    reset_block()

    atoms = setupRes.generate_coordinates()
    positions = setupRes.coor.get_positions()
    atoms.set_positions(positions)
    reset_block()
    reset_block_no_internal()
    reset_block()

    assert atoms is not None