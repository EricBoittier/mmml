"""Live CHARMM: the MLpot pair lists must match an independently derived oracle.

``mlpot_update`` in ``setup/charmm/source/api/api_func.F90`` builds the ML-ML and
ML-MM index lists that every MLpot energy call consumes, then shifts them to
0-based for Python. Until now nothing checked the lists themselves — only that
the resulting *energy* agreed with ASE, which is a weak test of the indexing: a
symmetric error in the pair list can cancel out.

CHARMM already exposes the lists (``mlpot_get_pair_counts``,
``mlpot_export_mlml_pairs``, ``mlpot_export_mlmm_pairs``) and nothing in mmml
called them. This uses them as a direct read-back.

The oracle is exact rather than approximate. With every atom selected as ML and
no image cells, ``mlpot_update`` must produce the complete directed graph over
the ML atoms: ``Nmlp == Nml * (Nml - 1)`` and the pair set equals every ordered
``(i, j)`` with ``i != j``. That is precisely
``e3x.ops.sparse_pairwise_indices``, which the rest of the stack already treats
as the reference ordering. ``Nmlmmp`` must be 0, because ``mlpot_update`` skips
the ML-MM walk when no atom is MM.

This also pins the 0-based conversion: an off-by-one shows up as an index of -1
or ``Nml``, which the bounds assertion catches.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _can_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:  # noqa: BLE001
        return False


def _resolve_ckpt() -> Path | None:
    candidates: list[Path] = []
    ckpt_env = os.environ.get("MMML_CKPT")
    if ckpt_env:
        candidates.append(Path(ckpt_env))
    candidates.extend(
        [
            PROJECT_ROOT / "examples/ckpts_json/DESdimers_params.json",
            PROJECT_ROOT / "mmml/models/physnetjax/physnetjax/ckpts/DESdimers",
            PROJECT_ROOT / "mmml/models/physnetjax/ckpts/DESdimers",
        ]
    )
    for p in candidates:
        if p.exists():
            return p.resolve()
    return None


@pytest.mark.pycharmm
@pytest.mark.skipif(not _can_import("pycharmm"), reason="pycharmm not available")
@pytest.mark.skipif(not _can_import("jax"), reason="jax not available")
@pytest.mark.skipif(not _can_import("e3x"), reason="e3x not available")
def test_mlpot_ml_ml_pairs_are_the_complete_graph():
    ckpt = _resolve_ckpt()
    if ckpt is None:
        pytest.skip("No PhysNet checkpoint for MLpot test")

    import ase
    import e3x
    import numpy as np

    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm.energy as energy

    mlpot_dir = Path(__file__).resolve().parent
    if str(mlpot_dir) not in sys.path:
        sys.path.insert(0, str(mlpot_dir))
    from _common import build_acetone_dimer_cluster

    from mmml.interfaces.pycharmmInterface.mlpot import (
        load_physnet_mlpot_bundle,
        register_mlpot,
        select_all_atoms,
        setup_default_nbonds,
    )

    try:
        from pycharmm.energy_mlpot import (
            export_mlpot_mlml_pairs,
            export_mlpot_mlmm_pairs,
            get_mlpot_pair_counts,
        )
    except ImportError:  # pragma: no cover - older pycharmm
        pytest.skip("pycharmm.energy_mlpot pair-export helpers not available")

    z, r = build_acetone_dimer_cluster(4.0)
    n_atoms = len(z)
    setup_default_nbonds()

    atoms = ase.Atoms(numbers=z, positions=r)
    _params, model, pyCModel = load_physnet_mlpot_bundle(ckpt, n_atoms, atoms)
    model.natoms = n_atoms

    ctx = register_mlpot(pyCModel, z, select_all_atoms())
    try:
        # Forces mlpot_update to run and populate the lists.
        energy.show()
        counts = get_mlpot_pair_counts()
        if counts is None:
            pytest.skip("this libcharmm does not export MLpot pair counts")
        n_mlml, n_mlmm = counts
        exported = export_mlpot_mlml_pairs()
        exported_mm = export_mlpot_mlmm_pairs()
    finally:
        ctx.unset()

    assert n_mlml == n_atoms * (n_atoms - 1), (
        f"ML-ML pair count is {n_mlml}, expected the complete directed graph over "
        f"{n_atoms} ML atoms ({n_atoms * (n_atoms - 1)})"
    )
    assert n_mlmm == 0, (
        f"ML-MM pair count is {n_mlmm}, expected 0 when every atom is ML "
        "(mlpot_update skips the ML-MM walk under `Nml < natom`)"
    )
    if exported_mm is not None:
        assert exported_mm == ([], []) or len(exported_mm[0]) == 0

    assert exported is not None, "mlpot_export_mlml_pairs returned nothing"
    idxi, idxj = (np.asarray(a, dtype=int) for a in exported)
    assert idxi.size == n_mlml and idxj.size == n_mlml

    # 0-based and in range: an off-by-one in the Fortran->Python shift lands here.
    assert idxi.min() >= 0 and idxj.min() >= 0, (
        f"negative atom index in the exported pairs (i>={idxi.min()}, "
        f"j>={idxj.min()}); the Fortran 1-based lists were shifted twice"
    )
    assert idxi.max() < n_atoms and idxj.max() < n_atoms, (
        f"atom index out of range (i<={idxi.max()}, j<={idxj.max()}, "
        f"natom={n_atoms}); the lists were not shifted to 0-based"
    )
    assert not np.any(idxi == idxj), "self-pair (i == j) in the ML-ML list"

    dst, src = e3x.ops.sparse_pairwise_indices(n_atoms)
    expected = {(int(a), int(b)) for a, b in zip(np.asarray(dst), np.asarray(src))}
    got = {(int(a), int(b)) for a, b in zip(idxi, idxj)}
    assert got == expected, (
        "CHARMM's ML-ML pair set does not match the complete directed graph: "
        f"{len(expected - got)} missing, {len(got - expected)} unexpected"
    )
