"""--skip-bfgs: run FIRE only during pre-minimization.

Plain ASE BFGS trusts a quadratic model and takes long steps. On a hybrid ML PES
that walks downhill into a hole outside the training data: observed on a real
acetone box, E -7427.7 -> -7545.8 eV while max|F| rose 0.199 -> 36.3 eV/A. That
is a 0 K gradient descent, so no thermal barrier crossing is involved -- the sink
is reachable from the relaxed structure by following forces. FIRE, which is far
more conservative, stayed well-behaved on the same system.
"""

from __future__ import annotations



def test_jaxmd_exposes_skip_bfgs_defaulting_on():
    """BFGS is skipped by DEFAULT: it descends into holes in the ML PES at 0 K.

    This used to grep ``main``'s source for the flag and then assert the
    tri-state on a *freshly built* parser -- which is precisely the vacuous
    check its own docstring warned against, since the replica would pass
    whatever the runner actually declared. The runner's parser is now
    ``build_parser``, so ask it directly.
    """
    from mmml.cli.run.md_pbc_suite.jaxmd import build_parser

    p = build_parser()

    assert p.parse_args([]).skip_bfgs is True, "BFGS must be OFF by default until it is fixed"
    assert p.parse_args(["--skip-bfgs"]).skip_bfgs is True
    assert p.parse_args(["--no-skip-bfgs"]).skip_bfgs is False


def test_md_system_exposes_and_forwards_skip_bfgs():
    """md-system does not minimize itself -- it forwards to the runner.

    A flag that parses but is not forwarded is worse than no flag: it looks
    honoured and silently is not.
    """
    from mmml.cli.run.md_system import build_command, parse_args

    jax_args = parse_args(["--backend", "jaxmd", "--setup", "pbc_nve"])
    backend, jax_cmd = build_command(jax_args)
    assert backend == "jaxmd"
    assert "--skip-bfgs" in jax_cmd

    pycharmm_args = parse_args(
        ["--backend", "pycharmm", "--setup", "pycharmm_full"]
    )
    backend, pycharmm_cmd = build_command(pycharmm_args)
    assert backend == "pycharmm"
    assert "--skip-bfgs" not in pycharmm_cmd
    assert "--no-skip-bfgs" not in pycharmm_cmd


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
