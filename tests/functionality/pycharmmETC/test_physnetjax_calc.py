from pathlib import Path
import os

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _can_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def _can_import_e3x_nn() -> bool:
    try:
        __import__("e3x.nn.modules", fromlist=["initializers"])
        return True
    except Exception:
        return False


def _resolve_ckpt_path() -> Path | None:
    candidates = []
    if ckpt_env := os.environ.get("MMML_CKPT"):
        env_path = Path(ckpt_env)
        candidates.append(env_path if env_path.is_absolute() else PROJECT_ROOT / env_path)
    candidates.extend(
        [
            PROJECT_ROOT / "examples/ckpts_json/DESdimers_params.json",
            PROJECT_ROOT / "examples/ckpts_json/DES",
            PROJECT_ROOT / "examples/ckpts_json",
            PROJECT_ROOT / "ckpts_json/DESdimers_params.json",
            PROJECT_ROOT / "ckpts_json",
            PROJECT_ROOT / "mmml/models/physnetjax/ckpts/DESdimers",
        ]
    )
    for ckpt in candidates:
        if ckpt.exists():
            return ckpt.resolve()
    return None


@pytest.mark.skipif(not _can_import("pycharmm"), reason="pycharmm not available")
def test_physnetjax_calculator_smoke():
    if not _can_import("ase"):
        pytest.skip("ase not available")
    if not _can_import("jax"):
        pytest.skip("jax not available")
    if not _can_import_e3x_nn():
        pytest.skip("e3x.nn not available")

    ckpt = _resolve_ckpt_path()
    if ckpt is None:
        pytest.skip("No PhysNet checkpoint found")

    pdb_path = PROJECT_ROOT / "tests/pdb/init-packmol.pdb"
    if not pdb_path.exists():
        pytest.skip(f"Missing input pdb: {pdb_path}")

    import ase
    from ase.io import read
    from mmml.interfaces.pycharmmInterface.mmml_calculator import setup_calculator
    from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters

    atoms_in = read(str(pdb_path))
    R = np.asarray(atoms_in.get_positions()[:20])
    Z = np.asarray(atoms_in.get_atomic_numbers()[:20], dtype=np.int32)

    factory = setup_calculator(
        ATOMS_PER_MONOMER=10,
        N_MONOMERS=2,
        doML=True,
        doMM=False,
        model_restart_path=ckpt,
        MAX_ATOMS_PER_SYSTEM=20,
    )
    calc, _ = factory(
        atomic_numbers=Z,
        atomic_positions=R,
        n_monomers=2,
        cutoff_params=CutoffParameters(),
    )
    atoms = ase.Atoms(Z, R)
    atoms.calc = calc
    energy = float(atoms.get_potential_energy())
    assert np.isfinite(energy)
