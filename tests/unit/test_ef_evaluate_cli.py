from __future__ import annotations

import subprocess
import sys


def test_ef_evaluate_help_lists_rot_and_test_npz_flags() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "mmml.models.EF.evaluate", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    out = proc.stdout
    assert "--rot-augment" in out
    assert "--rot-perturbation" in out
    assert "--test-npz" in out
    assert "--save-output-npz" in out
