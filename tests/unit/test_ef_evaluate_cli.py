from __future__ import annotations

import subprocess
import sys

from mmml.models.EF.eval_paths import resolve_evaluation_output_dir


def test_resolve_evaluation_output_dir_rotaug_subdirs() -> None:
    base = resolve_evaluation_output_dir("/tmp/ef_eval/run0", rot_augment=False)
    assert base.name == "run0"
    aug05 = resolve_evaluation_output_dir("/tmp/ef_eval/run0", rot_augment=True, rot_perturbation=0.5)
    aug10 = resolve_evaluation_output_dir("/tmp/ef_eval/run0", rot_augment=True, rot_perturbation=1.0)
    assert aug05 == base / "rotaug_pert0p5"
    assert aug10 == base / "rotaug_pert1"
    assert aug05 != aug10


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
