from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _can_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


JSON_CKPT = Path("ckpts_json/DESdimers_params.json")
ORBAX_EPOCH_1985 = Path("mmml/models/physnetjax/ckpts/DESdimers/epoch-1985")


@pytest.mark.skipif(not _can_import("pycharmm"), reason="pycharmm not available")
def test_json_checkpoint_matches_epoch1985_energy_and_forces(tmp_path: Path):
    """
    Compare model outputs from:
    - existing JSON checkpoint (ckpts_json/DESdimers_params.json)
    - orbax checkpoint at mmml/models/physnetjax/ckpts/DESdimers/epoch-1985

    The orbax checkpoint is converted to a temporary JSON first so both paths
    go through the same JSON-loading codepath before evaluating energy/forces.
    """
    if not _can_import("jax"):
        pytest.skip("jax not available in this environment")
    if not _can_import("e3x"):
        pytest.skip("e3x not available in this environment")
    if not _can_import("ase"):
        pytest.skip("ase not available in this environment")
    if not _can_import("orbax"):
        pytest.skip("orbax not available in this environment")
    if not JSON_CKPT.exists():
        pytest.skip(f"Missing JSON checkpoint: {JSON_CKPT}")
    if not ORBAX_EPOCH_1985.exists():
        pytest.skip(f"Missing orbax checkpoint: {ORBAX_EPOCH_1985}")

    import ase
    from mmml.interfaces.pycharmmInterface.mmml_calculator import setup_calculator
    from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters
    from mmml.utils.model_checkpoint import orbax_to_json

    converted_json = tmp_path / "epoch1985_params.json"
    orbax_to_json(
        orbax_checkpoint_dir=ORBAX_EPOCH_1985,
        output_path=converted_json,
    )

    rng = np.random.default_rng(0)
    Z = np.array([6] * 20, dtype=np.int32)
    R = rng.uniform(2.0, 10.0, size=(20, 3)).astype(np.float64)

    def _energy_forces(ckpt_path: Path) -> tuple[float, np.ndarray]:
        factory = setup_calculator(
            ATOMS_PER_MONOMER=10,
            N_MONOMERS=2,
            doML=True,
            doMM=False,
            model_restart_path=ckpt_path,
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
        return float(atoms.get_potential_energy()), np.asarray(atoms.get_forces())

    e_json, f_json = _energy_forces(JSON_CKPT)
    e_orbax, f_orbax = _energy_forces(converted_json)

    assert np.isfinite(e_json) and np.isfinite(e_orbax)
    np.testing.assert_allclose(e_json, e_orbax, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(f_json, f_orbax, rtol=1e-5, atol=1e-5)

