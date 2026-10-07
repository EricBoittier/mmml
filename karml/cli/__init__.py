"""KARML CLI commands."""

from karml.cli.make import make_res, make_box


def run(*args, **kwargs):
    """Run simulation entrypoint exposed at package level."""
    from karml.cli.run.run_sim import run as _run
    return _run(*args, **kwargs)

__all__ = ["make_res", "make_box", "run"]
