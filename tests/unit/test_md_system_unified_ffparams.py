"""Isolated CHARMM smoke: packmol+PSF must produce FFParams.

Kept in its own module so CI's stateful smoke runner can give it a fresh
interpreter. ``test_md_system_unified``'s other end-to-end cases already load
CGenFF; re-reading ``prm`` in the same process segfaults inside
``pycharmm.read.prm`` (see ``scripts/ci/run_pycharmm_smoke_pytest.sh``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
CKPT = REPO / "examples" / "sppoky-epoch-0010_params.json"


def _args(**overrides):
    base = dict(
        setup="pbc_nve",
        dt_fs=1.0,
        ps=0.01,
        temperature=300.0,
        pressure=1.0,
        composition="TIP3:4",
        n_molecules=None,
        box_size=15.0,
        builder=None,
        template_pdb=None,
        continue_from=None,
        seed=1,
        checkpoint=str(CKPT),
        output_dir=None,
        sampler="md",
        ff=None,
        mbd_checkpoint=None,
        mbd_weight=1.0,
        multipole_checkpoint=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _pycharmm_or_skip():
    try:
        import pycharmm  # noqa: F401  (triggers libcharmm load)
    except OSError:
        pytest.skip("libcharmm not available")
    if not CKPT.exists():
        pytest.skip(f"checkpoint {CKPT.name} not present")


@pytest.mark.pycharmm
def test_end_to_end_builds_ffparams():
    """The packmol+PSF helper must produce FFParams, or mm_nonbonded can't run."""
    _pycharmm_or_skip()
    # The monomer geometry gate runs armed here. It used to be disabled because
    # every build except pc-studix distorted monomers during the cluster ABNR;
    # that was the sticky READ PARAM APPEND in api_read.F90 zeroing the VDW
    # table, fixed at the source. See docs/packmol-monomer-geometry-gate.md.
    from mmml.cli.run.md_system_unified import build_packmol_system_with_ffparams
    from mmml.md.lowering import runconfig_from_md_system_args

    run_config = runconfig_from_md_system_args(_args(setup="pbc_nve", seed=23))
    system = build_packmol_system_with_ffparams(run_config.system)
    assert system.ff_params is not None
    assert system.n_atoms == 12  # 4 TIP3 waters
    # TIP3 charges: O=-0.834, H=+0.417
    assert np.allclose(
        sorted(system.ff_params.charges), sorted([-0.834, 0.417, 0.417] * 4)
    )
