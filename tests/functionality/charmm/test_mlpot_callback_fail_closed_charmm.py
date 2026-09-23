"""Live CHARMM: an exception in the MLpot energy callback ends the run (exit 86).

CHARMM calls ``calculate_charmm`` through a ctypes ``CFUNCTYPE``. Before the
fail-closed guard, an exception there was printed by ctypes and CHARMM kept
integrating with an undefined USER energy; the job then wrote its restart,
stage summary and ``next_run`` advice and exited 0 (a 7 ps CGenFF validation
NVE was lost this way). See ``mlpot/callback_failstop.py``.

Each case runs ``mmml md-system`` (3 ethanol, vacuum, CPU) in a child process:

* ``extent``: from the 6th energy call inside dynamics, the MM pair update
  raises the molecule-extent ``ValueError`` (the #215 failure) inside the
  callback.
* ``pairs``: from the 4th energy call inside dynamics, the callback sees
  ``Nmlmmp = 0`` with MM on for a multi-monomer system, so the real
  ``_resolve_mm_pairs_from_callback`` raises ``_CallbackPairListUnavailable``.
* ``pairs_genuine``: ``--mm-pair-source charmm_callback`` on an all-ML system,
  whose Fortran ML/MM list is empty. The first ENER probe is still disarmed,
  so USER = 0 and recovery runs; dynamics is refused. Not exit 86.
* ``control``: no fault; proves the harness reaches dynamics and that the
  artifacts checked for absence are written by a successful run.

For each failure the child must exit with exactly 86, make no energy call and
print no ``DYNA>`` line after the failing one, and leave no final restart,
trajectory, stage summary, job manifest or ``next_run`` advice behind.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import can_import_pycharmm

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CKPT = _REPO_ROOT / "examples" / "ckpts_json" / "DESdimers_params.json"
EXIT_CODE = 86
BANNER = "MMML MLPOT CALLBACK FAILURE"

pytestmark = [
    pytest.mark.pycharmm,
    pytest.mark.mlpot,
    pytest.mark.charmm_serial,
    pytest.mark.skipif(
        not can_import_pycharmm(),
        reason="pycharmm / libcharmm not available",
    ),
    pytest.mark.skipif(not _CKPT.is_file(), reason=f"missing checkpoint {_CKPT}"),
]

# Child driver: patch DecomposedMlpotCalculator.calculate_charmm to log every
# CHARMM energy call (and whether it happens inside dynamics), inject the fault,
# then run the real `mmml md-system` CLI in this process.
_DRIVER = r"""
import sys
from pathlib import Path

mode, fail_at, log_path = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
md_argv = sys.argv[4:]

from mmml.interfaces.pycharmmInterface.mlpot import hybrid_mlpot as hm

_orig = hm.DecomposedMlpotCalculator.calculate_charmm
_DYN_FRAMES = {"run_dynamics", "_run_dynamics_via_c_api"}
_state = {"dyn": 0, "all": 0}


def _in_dynamics():
    f = sys._getframe(1)
    while f is not None:
        if f.f_code.co_name in _DYN_FRAMES:
            return True
        f = f.f_back
    return False


def _log(line):
    with log_path.open("a") as fh:
        fh.write(line + "\n")


def _patched(self, *args):
    dyn = _in_dynamics()
    _state["all"] += 1
    if dyn:
        _state["dyn"] += 1
    _log(f"call {_state['all']} {'dyn' if dyn else 'setup'} {_state['dyn']}")
    if mode in ("extent", "pairs") and dyn and _state["dyn"] >= fail_at:
        _log(f"inject {mode} at dyn call {_state['dyn']}")
        if mode == "extent":
            def _raise_extent(pos, box):
                raise ValueError(
                    "Molecule extent 9.999 A (max atom-to-centroid distance) exceeds "
                    "the 2.597 A assumed for the MM pair list radius (12.94 A): atom "
                    "pairs of switched-on dimers may be missing. (injected by test)"
                )
            self._mm_pair_source = "jax"
            self._resolve_mm_pairs = _raise_extent
        else:
            # Stale Fortran list at step N: callback pair source with Nmlmmp = 0.
            self._mm_pair_source = "charmm_callback"
            args = list(args)
            args[11] = 0
    return _orig(self, *args)


hm.DecomposedMlpotCalculator.calculate_charmm = _patched

from mmml.cli.__main__ import main

sys.argv = ["mmml", *md_argv]
rc = main()
_log(f"main returned {rc}")
raise SystemExit(rc)
"""

# Files a finished NVE stage leaves behind (checked in the control run).
_DONE_ARTIFACTS = (
    "out/nve.res",
    "out/nve.dcd",
    "out/stage_summary.json",
    "out/run_manifest.json",
    "out/next_run_advice.json",
    "out/next_run.sh",
    "out/next_run.yaml",
    "out/handoff/final.res",
)


def _run_case(tmp_path: Path, mode: str, fail_at: int, *extra: str):
    env = dict(os.environ)
    for key in (
        "MMML_MLPOT_CALLBACK_FAIL_EXIT_CODE",
        "MMML_MLPOT_ALLOW_MISSING_CALLBACK_PAIRS",
        "MMML_MLPOT_ALLOW_PERIODIC_COULOMB_FAILURE",
        "MMML_MM_PAIR_SOURCE",
    ):
        env.pop(key, None)
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(p for p in (str(_REPO_ROOT), env.get("PYTHONPATH", "")) if p),
            "JAX_PLATFORMS": "cpu",
            "CUDA_VISIBLE_DEVICES": "",
            "MMML_MLPOT_DEVICE": "cpu",
            "MMML_NO_CHARMM_MPI": "1",
            "MMML_NO_MPI_RERUN": "1",
        }
    )
    calls = tmp_path / "calls.log"
    argv = [
        "md-system",
        "--setup",
        "free_nve",
        "--backend",
        "pycharmm",
        "--composition",
        "ETOH:3",
        "--checkpoint",
        str(_CKPT),
        "--ps",
        "0.01",
        "--dt-fs",
        "0.5",
        "--mini-nstep",
        "5",
        "--output-dir",
        str(tmp_path / "out"),
        *extra,
    ]
    proc = subprocess.run(
        [sys.executable, "-c", _DRIVER, mode, str(fail_at), str(calls), *argv],
        cwd=str(tmp_path),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=900,
    )
    lines = calls.read_text().splitlines() if calls.is_file() else []
    return proc, lines


def _diag(proc, lines) -> str:
    return f"rc={proc.returncode}\ncalls tail: {lines[-5:]}\noutput tail:\n{proc.stdout[-4000:]}"


def _assert_failed_closed(tmp_path: Path, proc, lines, *, inject_line: str | None):
    out = proc.stdout
    assert proc.returncode == EXIT_CODE, _diag(proc, lines)
    assert BANNER in out, _diag(proc, lines)
    assert "Traceback (most recent call last)" in out
    assert "Exception ignored on calling ctypes callback" not in out
    assert "main returned" not in "\n".join(lines)
    call_lines = [ln for ln in lines if ln.startswith("call ")]
    if inject_line is not None:
        # The injected call is the last CHARMM energy call: no later step ran.
        assert lines[-1] == inject_line, _diag(proc, lines)
    # Nothing CHARMM prints for a later dynamics step appears after the banner.
    after = out[out.index(BANNER) :]
    assert "DYNA>" not in after, _diag(proc, lines)
    assert "NVE complete" not in out
    for rel in _DONE_ARTIFACTS:
        assert not (tmp_path / rel).exists(), f"{rel} written after failure\n" + _diag(proc, lines)
    jobs = tmp_path / "artifacts" / "md_system" / "jobs"
    assert not jobs.exists() or not any(jobs.iterdir()), "job manifest written"
    return call_lines


def test_extent_failure_inside_dynamics_exits_86(tmp_path: Path) -> None:
    proc, lines = _run_case(tmp_path, "extent", 6)
    call_lines = _assert_failed_closed(tmp_path, proc, lines, inject_line="inject extent at dyn call 6")
    assert call_lines[-1].endswith("dyn 6")
    assert "Molecule extent 9.999 A" in proc.stdout
    # Dynamics had started (step 0 was printed) and did not get further.
    dyna_steps = [int(ln.split()[1]) for ln in proc.stdout.splitlines() if ln.startswith("DYNA>")]
    assert dyna_steps == [0], _diag(proc, lines)


def test_stale_pair_list_inside_dynamics_exits_86(tmp_path: Path) -> None:
    proc, lines = _run_case(tmp_path, "pairs", 4)
    call_lines = _assert_failed_closed(tmp_path, proc, lines, inject_line="inject pairs at dyn call 4")
    assert call_lines[-1].endswith("dyn 4")
    assert "_CallbackPairListUnavailable" in proc.stdout
    assert "returned zero ML/MM pairs" in proc.stdout
    # The old path returned 0.0 with a WARN and kept integrating.
    assert "WARN: Decomposed MLpot: charmm_callback returned zero" not in proc.stdout


def test_zero_callback_pairs_without_injection_refuses_before_dynamics(
    tmp_path: Path,
) -> None:
    """All-ML + Fortran pair source: first ENER probe is disarmed (USER = 0).

    Recovery cannot invent pairs; ``assert_mlpot_user_active`` refuses dynamics.
    Exit 86 is for an empty list after USER was already verified.
    """
    proc, lines = _run_case(tmp_path, "none", 0, "--mm-pair-source", "charmm_callback")
    out = proc.stdout
    assert proc.returncode != 0, _diag(proc, lines)
    assert proc.returncode != EXIT_CODE, _diag(proc, lines)
    assert BANNER not in out, _diag(proc, lines)
    assert "DYNA>" not in out, _diag(proc, lines)
    assert "NVE complete" not in out
    assert "returned zero ML/MM pairs" in out or "USER term" in out
    for rel in _DONE_ARTIFACTS:
        assert not (tmp_path / rel).exists(), f"{rel} written after refuse\n" + _diag(
            proc, lines
        )


def test_control_run_completes_and_writes_done_artifacts(tmp_path: Path) -> None:
    proc, lines = _run_case(tmp_path, "none", 0)
    assert proc.returncode == 0, _diag(proc, lines)
    assert BANNER not in proc.stdout
    assert any(ln.endswith("dyn 6") for ln in lines), "harness never reached dynamics"
    for rel in ("out/nve.res", "out/stage_summary.json", "out/run_manifest.json"):
        assert (tmp_path / rel).is_file(), f"control run did not write {rel}"
