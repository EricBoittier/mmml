"""Regression checks for process isolation in the live PyCHARMM CI runner."""

import re
from pathlib import Path


def test_stateful_charmm_smokes_run_separately_from_aggregate_selection() -> None:
    root = Path(__file__).resolve().parents[2]
    script = (root / "scripts/ci/run_pycharmm_smoke_pytest.sh").read_text()

    # Every stateful CHARMM module must be (a) collected in the isolation array,
    # (b) run in its own process by iterating that array, and (c) excluded from
    # the aggregate remainder run via --ignore. CHARMM owns process-global state,
    # so dropping any of these back into the shared run reintroduces cross-test
    # pollution / native aborts.
    isolated_variables = (
        "PYCHARMM_RES_SMOKE",
        "MPI_LIVE_ENERGY_SMOKE",
        "COMP_VELOCITIES_SMOKE",
        "CG_JAXMD_SMOKE",
        "DIMER_MODELS_SMOKE",
        "MD_SYSTEM_SMOKE",
    )

    # The isolation set is declared as a STATEFUL_SMOKE_PATHS=( ... ) array.
    array_block = re.search(r"STATEFUL_SMOKE_PATHS=\((.*?)\)", script, re.DOTALL)
    assert array_block is not None, "expected a STATEFUL_SMOKE_PATHS=( ... ) array"
    array_body = array_block.group(1)

    # The array is iterated so each stateful module runs in its own process,
    # and the aggregate remainder run --ignores every isolated module.
    assert 'for smoke_path in "${STATEFUL_SMOKE_PATHS[@]}"' in script
    assert '"$smoke_path" "$@"' in script
    assert "--ignore=$smoke_path" in script
    assert '"${ignore_args[@]}"' in script

    for variable in isolated_variables:
        assert f'{variable}="' in script, f"{variable} must be defined"
        assert f'"${variable}"' in array_body, f"{variable} must be in the isolation array"
