#!/usr/bin/env python3
"""Tier 2 spatial MPI smoke: validate env + hybrid MLpot callback under mpirun.

Does **not** replace a full ``md-system --ml-spatial-mpi`` mini on a production cluster,
but confirms the MLpot callback uses spatial batch indices and MPI allreduce.

**Without CHARMM** (callback-only, recommended first):

```bash
MMML_MPI_NP=2 MMML_MLPOT_SPATIAL_MPI=1 \\
  ./scripts/mmml-charmm-mpirun.sh python \\
  tests/functionality/mlpot/06_spatial_mpi_tier2_smoke.py
```

**With CHARMM registration** (optional second step):

```bash
MMML_MPI_NP=2 MMML_MLPOT_SPATIAL_MPI=1 \\
  ./scripts/mmml-charmm-mpirun.sh python \\
  tests/functionality/mlpot/06_spatial_mpi_tier2_smoke.py --charmm-ener \\
  --residue ACO --n-molecules 4
```

Pass: exit 0; callback mode reports matching allreduced energy on all ranks;
``--charmm-ener`` completes ``energy.show()`` without NaN.
"""

from __future__ import annotations

import argparse
import sys
from unittest import mock

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--charmm-ener",
        action="store_true",
        help="Also register MLpot in CHARMM and run energy.show() (requires PyCHARMM).",
    )
    parser.add_argument("--residue", default="ACO")
    parser.add_argument("--n-molecules", type=int, default=4)
    parser.add_argument("--spacing", type=float, default=5.0)
    parser.add_argument("--box-side", type=float, default=32.0)
    parser.add_argument("--checkpoint", default=None)
    return parser.parse_args()


def _mpi_info() -> tuple[int, int]:
    from mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge import mpi_rank_size

    return mpi_rank_size()


def _tier2_validate() -> int:
    from mmml.interfaces.pycharmmInterface.mlpot.spatial_mpi_validate import (
        render_tier2_report,
        validate_tier2_spatial_mpi_env,
    )

    report = validate_tier2_spatial_mpi_env(strict=False)
    rank, _size = _mpi_info()
    if rank == 0:
        print(render_tier2_report(report))
    return 0 if report.ok else 1


def _hybrid_callback_smoke(box_side: float) -> int:
    """Exercise DecomposedMlpotCalculator spatial path without CHARMM ENER."""
    import jax.numpy as jnp

    from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters
    from mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot import DecomposedMlpotCalculator
    from mmml.interfaces.pycharmmInterface.mlpot.medium_pbc_validation import (
        lattice_positions_cubic_pbc,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge import mpi_rank_size

    rank, size = mpi_rank_size()
    n_monomers = 8
    atoms_per = 10
    pos = lattice_positions_cubic_pbc(
        n_monomers, atoms_per, box_side, spacing_A=5.0, seed=11
    )
    z = np.tile(np.array([6, 1, 1, 1, 8, 1, 1, 1, 1, 1], dtype=int), n_monomers)

    calc = DecomposedMlpotCalculator(
        mock.MagicMock(),
        CutoffParameters(),
        n_monomers,
        z,
        cell=box_side,
        spatial_mpi=True,
    )
    n = len(z)
    captured: dict = {}

    def _fake_forward_fn(*, n_atoms, atomic_numbers_jax, box_jax):
        def _eval(
            positions_jax,
            mm_pair_idx,
            mm_pair_mask,
            use_mm_pairs,
            spatial_monomer_indices,
            spatial_dimer_indices,
            use_spatial,
        ):
            captured["use_spatial"] = bool(use_spatial)
            captured["mono"] = int(spatial_monomer_indices.shape[0])
            val = float(rank + 10)
            return jnp.array(val), jnp.full((n_atoms, 3), val)

        return _eval

    calc._get_spherical_forward_fn = mock.MagicMock(side_effect=_fake_forward_fn)
    x, y, zc = pos[:, 0], pos[:, 1], pos[:, 2]
    dx = dy = dz = np.zeros(n, dtype=np.float64)

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.resolve_mlpot_mic_box_side_A",
        return_value=(box_side, "smoke"),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.jax_device_policy.mlpot_jax_device_context",
        return_value=mock.MagicMock(__enter__=mock.MagicMock(), __exit__=mock.MagicMock()),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.recover_mpi_for_charmm_after_jax",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot.as_ml_array",
        side_effect=lambda arr, dtype=None: jnp.asarray(arr),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot.resolve_ml_compute_dtype",
        return_value=jnp.float32,
    ):
        energy = calc.calculate_charmm(
            n,
            0,
            0,
            None,
            x,
            y,
            zc,
            dx,
            dy,
            dz,
            0,
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    if size > 1:
        expected_e = sum(float(r + 10) for r in range(size))
        if abs(float(energy) - expected_e) > 1e-6:
            print(
                f"FAIL rank {rank}: energy {energy} != allreduce sum {expected_e}",
                file=sys.stderr,
            )
            return 1
        if not captured.get("use_spatial"):
            print(f"FAIL rank {rank}: use_spatial=False at np>1", file=sys.stderr)
            return 1
    if rank == 0:
        print(
            f"PASS hybrid callback smoke: size={size} energy={float(energy):.6f} "
            f"use_spatial={captured.get('use_spatial')} owned_mono={captured.get('mono')}"
        )
    return 0


def _charmm_ener_smoke(args: argparse.Namespace) -> int:
    from _common import (
        add_cluster_args,
        all_atom_selection,
        build_ase_cluster,
        check_mlpot_symbols,
        load_physnet_for_cluster,
        resolve_checkpoint,
        setup_charmm_nbonds,
    )

    rank, size = _mpi_info()
    if rank == 0:
        print("CHARMM MLpot ENER smoke (spatial MPI env must be set externally)")
    missing = check_mlpot_symbols()
    if missing:
        print(f"FAIL: missing symbols: {missing}", file=sys.stderr)
        return 1

    ckpt = resolve_checkpoint(args.checkpoint)
    z, r = build_ase_cluster(args.residue, args.n_molecules, args.spacing)
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm
    import pycharmm.energy as energy

    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import setup_charmm_environment

    setup_charmm_environment(use_pbc=True, cubic_box_side_A=float(args.box_side))
    setup_charmm_nbonds()
    params, model = load_physnet_for_cluster(ckpt, len(z))
    model.natoms = len(z)
    import ase

    atoms = ase.Atoms(numbers=z, positions=r)
    from mmml.models.physnetjax.physnetjax.calc.helper_mlp import get_pyc

    pyCModel = get_pyc(params, model, atoms)
    mlpot = pycharmm.MLpot(
        ml_model=pyCModel,
        ml_Z=list(z),
        ml_selection=all_atom_selection(),
        ml_charge=0,
        ml_fq=True,
    )
    energy.show()
    terms = energy.get_total()
    if not np.isfinite(float(terms)):
        print(f"FAIL rank {rank}: non-finite CHARMM energy {terms}", file=sys.stderr)
        mlpot.unset_mlpot()
        return 1
    if rank == 0:
        print(f"PASS CHARMM ENER: TOTE={float(terms):.6f} kcal/mol (mpi_size={size})")
    mlpot.unset_mlpot()
    return 0


def main() -> int:
    from mmml.interfaces.pycharmmInterface.charmm_mpi import prepare_serial_charmm_mpi_env

    prepare_serial_charmm_mpi_env()
    args = _parse_args()
    rank, size = _mpi_info()
    if rank == 0:
        print(f"06_spatial_mpi_tier2_smoke: rank {rank}/{size}")

    code = _tier2_validate()
    if code != 0:
        return code

    code = _hybrid_callback_smoke(args.box_side)
    if code != 0:
        return code

    if args.charmm_ener:
        return _charmm_ener_smoke(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
