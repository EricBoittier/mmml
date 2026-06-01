"""Pytest: MLpot energy/forces match ASE on ACO dimer cluster (script 03 logic)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _can_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def _resolve_ckpt() -> Path | None:
    ckpt_env = os.environ.get("MMML_CKPT")
    candidates: list[Path] = []
    if ckpt_env:
        candidates.append(Path(ckpt_env))
    candidates.extend(
        [
            PROJECT_ROOT / "examples/ckpts_json/DESdimers_params.json",
            PROJECT_ROOT / "mmml/models/physnetjax/ckpts/DESdimers",
        ]
    )
    for p in candidates:
        if p.exists():
            return p.resolve()
    return None


@pytest.mark.skipif(not _can_import("pycharmm"), reason="pycharmm not available")
@pytest.mark.skipif(not _can_import("jax"), reason="jax not available")
def test_mlpot_energy_matches_ase():
    ckpt = _resolve_ckpt()
    if ckpt is None:
        pytest.skip("No PhysNet checkpoint for MLpot test")

    import ase
    import e3x
    import numpy as np

    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm.energy as energy

    import sys

    mlpot_dir = Path(__file__).resolve().parent
    if str(mlpot_dir) not in sys.path:
        sys.path.insert(0, str(mlpot_dir))
    from _common import build_acetone_dimer_cluster
    from mmml.interfaces.pycharmmInterface.mmml_calculator import ev2kcalmol
    from mmml.interfaces.pycharmmInterface.mlpot import (
        load_physnet_mlpot_bundle,
        register_mlpot,
        select_all_atoms,
        setup_default_nbonds,
    )
    from mmml.models.physnetjax.physnetjax.calc.helper_mlp import get_ase_calc

    z, r = build_acetone_dimer_cluster(4.0)
    n_atoms = len(z)
    setup_default_nbonds()

    atoms = ase.Atoms(numbers=z, positions=r)
    params, model, pyCModel = load_physnet_mlpot_bundle(ckpt, n_atoms, atoms)
    model.natoms = n_atoms

    ase_calc = get_ase_calc(
        params, model, atoms, conversion={"energy": ev2kcalmol, "forces": ev2kcalmol}
    )
    atoms.calc = ase_calc
    atoms.get_potential_energy()
    e_ase = float(ase_calc.results["energy"])
    f_ase = np.asarray(ase_calc.results["forces"], dtype=float)

    ctx = register_mlpot(pyCModel, z, select_all_atoms())
    try:
        energy.show()
        df = energy.get_energy()
        e_user = float(df.iloc[0].get("USER", float("nan")))
        e_ener = float(df.iloc[0].get("ENER", float("nan")))
    finally:
        ctx.unset()

    pyc = pyCModel.get_pycharmm_calculator()
    dst, src = e3x.ops.sparse_pairwise_indices(n_atoms)
    dx = np.zeros(n_atoms)
    e_cb = pyc.calculate_charmm(
        Natom=n_atoms,
        Ntrans=0,
        Natim=n_atoms,
        idxp=np.arange(n_atoms),
        x=r[:, 0],
        y=r[:, 1],
        z=r[:, 2],
        dx=dx,
        dy=np.zeros(n_atoms),
        dz=np.zeros(n_atoms),
        Nmlp=len(dst),
        Nmlmmp=0,
        idxi=dst,
        idxj=src,
        idxjp=np.arange(n_atoms)[: len(dst)],
        idxu=[],
        idxv=[],
        idxup=[],
        idxvp=[],
    )
    f_cb = np.asarray(pyc.results["forces"], dtype=float)

    rtol = 0.02
    scale = max(abs(e_ase), 1e-6)
    assert abs(e_ase - e_cb) <= rtol * scale
    assert abs(e_ase - e_user) <= rtol * scale
    assert abs(e_ase - e_ener) <= rtol * scale
    assert np.abs(f_ase - f_cb).max() <= 5.0
