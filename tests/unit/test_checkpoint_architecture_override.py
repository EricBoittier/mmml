"""Warm start must report the architecture it overrode.

Every key in ``CHECKPOINT_ARCH_KEYS`` changes the parameter tree, so a warm
start has to adopt the checkpoint's values. That is correct but silent: job
19360535 asked for ``use_energy_bias=True, features=64, zbl=False`` and trained
``use_energy_bias=False, features=32, zbl=True`` with nothing in the log saying
so, which cost real debugging time.
"""

from __future__ import annotations

import argparse

from mmml.models.physnetjax.checkpoint_utils import (
    CHECKPOINT_ARCH_KEYS,
    apply_checkpoint_architecture,
)


def _args(**kw):
    return argparse.Namespace(**kw)


def test_reports_only_the_keys_that_actually_changed():
    args = _args(features=64, cutoff=6.0, zbl=False)
    changed = apply_checkpoint_architecture(
        args, {"features": 32, "cutoff": 6.0, "zbl": True}, verbose=False
    )

    assert changed == {"features": (64, 32), "zbl": (False, True)}
    assert "cutoff" not in changed, "unchanged keys must not be reported"


def test_values_are_still_applied():
    args = _args(features=64, zbl=False)
    apply_checkpoint_architecture(args, {"features": 32, "zbl": True}, verbose=False)
    assert args.features == 32
    assert args.zbl is True


def test_regression_job_19360535_energy_bias_flip_is_surfaced():
    """The specific silent flip that made the DES warm start hard to diagnose."""
    args = _args(
        features=64, max_degree=0, n_res=2, zbl=False,
        max_atomic_number=28, use_energy_bias=True,
    )
    changed = apply_checkpoint_architecture(
        args,
        {
            "features": 32, "max_degree": 1, "n_res": 3, "zbl": True,
            "max_atomic_number": 118, "use_energy_bias": False,
        },
        verbose=False,
    )

    assert changed["use_energy_bias"] == (True, False)
    assert len(changed) == 6
    assert args.use_energy_bias is False


def test_prints_each_override(capsys):
    args = _args(use_energy_bias=True)
    apply_checkpoint_architecture(args, {"use_energy_bias": False}, verbose=True)

    out = capsys.readouterr().out
    assert "use_energy_bias" in out
    assert "requested=True" in out
    assert "applied=False" in out


def test_silent_when_nothing_changes(capsys):
    args = _args(features=32)
    changed = apply_checkpoint_architecture(args, {"features": 32}, verbose=True)

    assert changed == {}
    assert capsys.readouterr().out == "", "no warm-start noise when nothing is overridden"


def test_ignores_keys_absent_from_args_or_config():
    args = _args(features=64)
    changed = apply_checkpoint_architecture(
        args, {"features": 32, "not_an_arg": 1}, verbose=False
    )
    assert changed == {"features": (64, 32)}
    assert not hasattr(args, "not_an_arg")


def test_energy_bias_is_a_tracked_architecture_key():
    # It allocates the `energy_bias` param, so it must stay in the override set.
    assert "use_energy_bias" in CHECKPOINT_ARCH_KEYS
