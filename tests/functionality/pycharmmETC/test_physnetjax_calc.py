from pathlib import Path
import os

import pytest


def _can_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def _resolve_ckpt_path() -> Path | None:
    candidates = []
    ckpt_env = Path(_v) if (_v := os.environ.get("MMML_CKPT")) else None
    if ckpt_env is not None:
        candidates.append(ckpt_env)
    candidates.extend(
        [
            Path("ckpts_json/DES"),
            Path("ckpts_json"),
            Path("mmml/physnetjax/ckpts/test-9af0d71b-4140-4d4b-83e3-ce07c652d048"),
            Path("mmml/physnetjax/ckpts"),
        ]
    )
    for ckpt in candidates:
        if ckpt.exists():
            return ckpt
    return None


@pytest.mark.skipif(not _can_import("pycharmm"), reason="pycharmm not available")
def test_physnetjax_calculator_smoke():
    """Optional smoke test: construct calculator and evaluate one structure."""
    if not _can_import("ase"):
        pytest.skip("ase not available")

    from ase.io import read
    import mmml

    if not hasattr(mmml, "PhysNetJaxCalculator"):
        pytest.skip("mmml.PhysNetJaxCalculator is not exposed in this build")

    pdb_path = Path("pdb/init-packmol.pdb")
    if not pdb_path.exists():
        pytest.skip(f"Missing test structure: {pdb_path}")

    atoms = read(str(pdb_path))
    ckpt = _resolve_ckpt_path()
    if ckpt is None:
        pytest.skip("Missing checkpoint: set MMML_CKPT or add ckpts_json/DES")

    calc = mmml.PhysNetJaxCalculator(chcseckpoint=str(ckpt))
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    assert energy == energy  # finite/real check without extra deps