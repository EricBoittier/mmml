from pathlib import Path
import os
import json

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


def _resolve_spooky_ckpt_path() -> Path | None:
    candidates = []
    if ckpt_env := os.environ.get("MMML_CKPT_SPOOKY"):
        env_path = Path(ckpt_env)
        candidates.append(env_path if env_path.is_absolute() else PROJECT_ROOT / env_path)
    spooky_json_dir = PROJECT_ROOT / "examples/ckpts_json"
    if spooky_json_dir.exists():
        candidates.extend(sorted(spooky_json_dir.glob("spooky*.json")))
    candidates.extend(
        [
            PROJECT_ROOT / "examples/ckpts_json/spooky_epoch-0004-chunk-000040-step-00584747.json",
        ]
    )
    for ckpt in candidates:
        if ckpt.exists():
            return ckpt.resolve()
    return None


@pytest.mark.skipif(not _can_import("pycharmm"), reason="pycharmm not available")
def test_spookynetjax_calculator_smoke(tmp_path: Path):
    if not _can_import("ase"):
        pytest.skip("ase not available")
    if not _can_import("jax"):
        pytest.skip("jax not available")
    if not _can_import_e3x_nn():
        pytest.skip("e3x.nn not available")

    ckpt = _resolve_spooky_ckpt_path()
    if ckpt is None:
        pytest.skip("No SpookyNet checkpoint found")

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

    ckpt_for_test = ckpt
    if ckpt.is_file() and ckpt.suffix == ".json":
        with open(ckpt) as f:
            payload = json.load(f)
        if "config" not in payload:
            from mmml.models.physnetjax.physnetjax.models.spooky_model import EF as SpookyEF

            default_cfg = SpookyEF().return_attributes()
            payload["config"] = default_cfg

            patched_json = tmp_path / "spooky_with_default_config.json"
            with open(patched_json, "w") as f:
                json.dump(payload, f)
            ckpt_for_test = patched_json

    factory = setup_calculator(
        ATOMS_PER_MONOMER=10,
        N_MONOMERS=2,
        doML=True,
        doMM=False,
        model_restart_path=ckpt_for_test,
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
