"""Live PyCHARMM smoke under mpirun (CHARMM CI job)."""

from __future__ import annotations

import pytest

from tests.conftest import can_import_pycharmm


def _under_mpirun() -> bool:
    try:
        from mmml.interfaces.pycharmmInterface.charmm_mpi import _under_mpirun as under

        return bool(under())
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(
        not can_import_pycharmm(),
        reason="pycharmm / libcharmm not available",
    ),
    pytest.mark.skipif(
        not _under_mpirun(),
        reason=(
            "requires mpirun; use "
            "MMML_MPI_NP=1 ./scripts/mmml-charmm-mpirun.sh pytest tests/charmm_mpi/test_mpi_live_energy.py"
        ),
    ),
]


def test_mpi_rank_size_under_mpirun():
    from mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge import mpi_rank_size

    rank, size = mpi_rank_size()
    assert 0 <= rank < max(1, size)
    assert size >= 1


def test_mpi_check_cli_under_mpirun():
    from mmml.cli.run.mpi_check import main, run_mpi_check

    report = run_mpi_check(prelaunch=True)
    if not report.ok:
        detail = "\n".join(
            [*(f"error: {e}" for e in report.errors), *(f"warning: {w}" for w in report.warnings)]
        )
        pytest.fail(f"mpi-check failed under mpirun:\n{detail}")
    assert main(["--json", "--prelaunch"]) == 0


def test_tip3_energy_finite_under_mpirun(tip3_charmm_ff):
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import charmm_grms_after_ener_force

    grms = charmm_grms_after_ener_force(silent=True)
    assert grms >= 0.0
