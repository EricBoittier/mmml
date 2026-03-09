import pytest


def _can_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _can_import("pycharmm"), reason="pycharmm not available")
def test_run_packmol_smoke():
    from mmml.interfaces.pycharmmInterface import setupBox

    result = setupBox.run_packmol(10, 10)
    assert result is None or result is not False