"""Live PyCHARMM smoke under mpirun (CHARMM CI job)."""

from __future__ import annotations

import pytest

from tests.conftest import can_import_pycharmm


pytestmark = pytest.mark.skipif(
    not can_import_pycharmm(),
    reason="pycharmm / libcharmm not available",
)


def test_mpi_rank_size_under_mpirun():
    from mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge import mpi_rank_size

    rank, size = mpi_rank_size()
    assert 0 <= rank < max(1, size)
    assert size >= 1


def test_mpi_check_cli_under_mpirun():
    from mmml.cli.run.mpi_check import main

    assert main(["--json"]) == 0


def test_tip3_energy_finite_under_mpirun(tip3_charmm_ff):
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import charmm_grms_after_ener_force

    grms = charmm_grms_after_ener_force(silent=True)
    assert grms >= 0.0
