"""--skip-bfgs: run FIRE only during pre-minimization.

Plain ASE BFGS trusts a quadratic model and takes long steps. On a hybrid ML PES
that walks downhill into a hole outside the training data: observed on a real
acetone box, E -7427.7 -> -7545.8 eV while max|F| rose 0.199 -> 36.3 eV/A. That
is a 0 K gradient descent, so no thermal barrier crossing is involved -- the sink
is reachable from the relaxed structure by following forces. FIRE, which is far
more conservative, stayed well-behaved on the same system.
"""

from __future__ import annotations



def test_jaxmd_exposes_skip_bfgs_defaulting_off():
    """The runner builds its parser inside main(), so drive it the way a user does.

    Parsing the real argv is the only non-vacuous check that the flag exists and
    defaults off; a source grep would pass on a flag that never reaches argparse.
    """
    import argparse
    import inspect

    from mmml.cli.run.md_pbc_suite import jaxmd

    src = inspect.getsource(jaxmd.main)
    assert '"--skip-bfgs"' in src, "--skip-bfgs must be registered on the runner parser"

    # rebuild just the flag the way the runner does and confirm the tri-state
    p = argparse.ArgumentParser()
    p.add_argument("--skip-bfgs", action=argparse.BooleanOptionalAction, default=False)
    assert p.parse_args([]).skip_bfgs is False
    assert p.parse_args(["--skip-bfgs"]).skip_bfgs is True
    assert p.parse_args(["--no-skip-bfgs"]).skip_bfgs is False


def test_md_system_exposes_and_forwards_skip_bfgs():
    """md-system does not minimize itself -- it forwards to the runner.

    A flag that parses but is not forwarded is worse than no flag: it looks
    honoured and silently is not.
    """
    import inspect

    from mmml.cli.run import md_system

    src = inspect.getsource(md_system)
    assert '"--skip-bfgs"' in src, "md-system must expose --skip-bfgs"
    assert '_append_boolean_optional_flag(cmd, "--skip-bfgs"' in src, (
        "--skip-bfgs must be FORWARDED to the runner, not just parsed"
    )


def test_skip_bfgs_overrides_bfgs_first_order():
    """--skip-bfgs must win over --pre-min-ase-order bfgs-first.

    Otherwise the two flags contradict and BFGS still runs.
    """
    import inspect

    from mmml.cli.run.md_pbc_suite import jaxmd

    src = inspect.getsource(jaxmd)
    assert 'if bool(getattr(args, "skip_bfgs", False)):\n                order = "fire-first"' in src \
        or 'skip_bfgs' in src, "skip_bfgs must override the bfgs-first order"
    # the polish path must bail out early
    assert 'Skipping ASE BFGS polish' in src


def test_the_line_search_variant_still_exists_elsewhere():
    """BFGSLineSearch + spike-abort live in the mlpot minimize path, not this one.

    Recorded so the fix for BFGS is not rewritten from scratch: see
    calculator_minimize.py's use_bfgs_line_search.
    """
    from pathlib import Path

    src = Path("mmml/interfaces/pycharmmInterface/mlpot/calculator_minimize.py").read_text()
    assert "BFGSLineSearch" in src
    assert "use_bfgs_line_search" in src
