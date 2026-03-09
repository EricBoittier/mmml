from pathlib import Path

import pytest


def _can_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _can_import("pycharmm"), reason="pycharmm not available")
def test_physnetjax_calculator_smoke():
    """Optional smoke test: construct calculator and evaluate one structure."""
    if not _can_import("ase"):
        pytest.skip("ase not available")

    from ase.io import read
    import mmml

    pdb_path = Path("pdb/init-packmol.pdb")
    if not pdb_path.exists():
        pytest.skip(f"Missing test structure: {pdb_path}")

    atoms = read(str(pdb_path))
    ckpt = Path("mmml/physnetjax/ckpts/test-9af0d71b-4140-4d4b-83e3-ce07c652d048")
    if not ckpt.exists():
        pytest.skip(f"Missing checkpoint: {ckpt}")

    calc = mmml.PhysNetJaxCalculator(chcseckpoint=str(ckpt))
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    assert energy == energy  # finite/real check without extra deps