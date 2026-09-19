"""End-to-end fail-closed check of the MLpot callback, called as CHARMM calls it.

Each case runs in a subprocess with ``MMML_MLPOT_CALLBACK_ABORT`` and
``PYTEST_CURRENT_TEST`` unset, so the child uses the production default. pytest
may still be imported there: CuPy's testing shim imports it in production too. The
calculator is called through a ``ctypes.CFUNCTYPE``. A failure must end the
process with a nonzero status before the "next MD step" line and the success
marker. Without the guard, ctypes prints the exception and returns 0 or garbage,
and the next step runs.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

_SCRIPT = textwrap.dedent(
    """
    import ctypes, sys
    from pathlib import Path
    from unittest.mock import MagicMock, patch
    import numpy as np

    from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters
    from mmml.interfaces.pycharmmInterface.mlpot.callback_failstop import set_mlpot_dynamics_armed
    from mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot import DecomposedMlpotCalculator

    failure, armed, marker = sys.argv[1], sys.argv[2] == "armed", Path(sys.argv[3])
    calc = DecomposedMlpotCalculator(
        MagicMock(), CutoffParameters(), 2, np.zeros(8, dtype=int), do_mm=True,
        get_update_fn=MagicMock(return_value=MagicMock(
            return_value=(np.zeros((1, 2), dtype=np.int32), np.zeros((1,), dtype=bool)))),
        mm_pair_source="charmm_callback",
    )
    if failure == "extent":
        def _raise(*a, **k):
            raise ValueError("Molecule extent 2.636 A (max atom-to-centroid distance) exceeds "
                             "the 2.597 A assumed for the MM pair list radius (12.94 A)")
        calc._resolve_mm_pairs_from_callback = _raise
        nmlmmp = 4
    else:  # zero ML/MM pairs with JAX MM on -> _CallbackPairListUnavailable
        nmlmmp = 0

    n = 8
    x, y, z, dx, dy, dz = (np.zeros(n) for _ in range(6))
    cb = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_int)(lambda natom: calc.calculate_charmm(
        natom, 0, 0, None, x, y, z, dx, dy, dz, 0, nmlmmp, None, None, None, None, None, None, None))

    set_mlpot_dynamics_armed(armed)
    with patch("mmml.interfaces.pycharmmInterface.jax_device_policy.mlpot_jax_device_context",
               return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())):
        user = cb(n)
    print("NEXT_MD_STEP_REACHED user=%r error=%r" % (user, getattr(calc, "_last_callback_error", None)), flush=True)
    marker.write_text("success\\n")
    """
)


def _run(failure: str, mode: str, tmp_path: Path):
    marker = tmp_path / f"{failure}_{mode}.ok"
    env = {k: v for k, v in os.environ.items() if k not in ("MMML_MLPOT_CALLBACK_ABORT", "PYTEST_CURRENT_TEST")}
    env.update(PYTHONPATH=str(REPO), JAX_PLATFORMS="cpu")
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT, failure, mode, str(marker)],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=300,
    )
    return proc, marker


@pytest.mark.parametrize("failure,mode", [("extent", "armed"), ("extent", "disarmed"), ("pairlist", "armed")])
def test_callback_failure_ends_process_nonzero_before_next_step(failure, mode, tmp_path):
    proc, marker = _run(failure, mode, tmp_path)
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, out
    assert "NEXT_MD_STEP_REACHED" not in proc.stdout, out
    assert not marker.exists(), out
    assert "FATAL: MLpot CHARMM callback failed" in proc.stderr, out
    expected = "Molecule extent" if failure == "extent" else "returned zero ML/MM pairs"
    assert expected in proc.stderr, out


def test_pairlist_unavailable_before_dynamics_returns_zero_for_recovery(tmp_path):
    """Disarmed (setup): assert_mlpot_user_active's recovery ladder needs USER = 0."""
    proc, marker = _run("pairlist", "disarmed", tmp_path)
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    assert "NEXT_MD_STEP_REACHED user=0.0" in proc.stdout, out
    assert "returned zero ML/MM pairs" in proc.stdout, out
    assert marker.exists(), out


def test_production_default_abort_mode_is_exit_not_charmm_stop():
    """CHARMM STOP ends the process with status 0, so it must not be the default."""
    env = {k: v for k, v in os.environ.items() if k not in ("MMML_MLPOT_CALLBACK_ABORT", "PYTEST_CURRENT_TEST")}
    env.update(PYTHONPATH=str(REPO))
    code = (
        "import sys; from mmml.interfaces.pycharmmInterface.mlpot.callback_failstop "
        "import resolve_mlpot_callback_abort_mode as r; import pytest; print(r())"  # pytest loaded, as with CuPy
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True, timeout=120)
    assert out.stdout.strip() == "exit", out.stdout + out.stderr
