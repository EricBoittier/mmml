"""Regression checks for process isolation in the live PyCHARMM CI runner."""

from pathlib import Path


def test_stateful_charmm_smokes_run_separately_from_aggregate_selection() -> None:
    root = Path(__file__).resolve().parents[2]
    script = (root / "scripts/ci/run_pycharmm_smoke_pytest.sh").read_text()

    isolated_variables = (
        "PYCHARMM_RES_SMOKE",
        "MPI_LIVE_ENERGY_SMOKE",
        "COMP_VELOCITIES_SMOKE",
    )
    assert "for smoke_path in" in script
    assert '"$smoke_path" "$@"' in script
    for variable in isolated_variables:
        assert f'"${variable}"' in script.split("for smoke_path in", 1)[1].split("do", 1)[0]
        assert f'--ignore="${variable}"' in script
