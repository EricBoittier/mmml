"""Unit tests for compact ASE BFGS/FIRE progress logging."""

from __future__ import annotations

import argparse
from io import StringIO

from mmml.cli.run.ase_minimize_log import (
    CompactAseOptimizerLog,
    attach_compact_ase_optimizer_log,
    resolve_ase_log_every,
    resolve_ase_optimizer_logfile,
)


def test_resolve_logfile_default_is_silent_for_ase_table():
    args = argparse.Namespace(quiet_bfgs=False, verbose_bfgs=False)
    assert resolve_ase_optimizer_logfile(args) is None


def test_resolve_logfile_quiet_and_verbose():
    assert resolve_ase_optimizer_logfile(argparse.Namespace(quiet_bfgs=True)) is None
    assert (
        resolve_ase_optimizer_logfile(argparse.Namespace(quiet_bfgs=False, verbose_bfgs=True))
        == "-"
    )


def test_resolve_log_every_defaults_to_about_ten_lines():
    args = argparse.Namespace(bfgs_log_every=None)
    assert resolve_ase_log_every(args, n_steps=50) == 5
    assert resolve_ase_log_every(args, n_steps=100) == 10
    assert resolve_ase_log_every(argparse.Namespace(bfgs_log_every=3), n_steps=50) == 3


def test_compact_log_prints_sparse_steps():
    class _FakeAtoms:
        def get_potential_energy(self):
            return -10.0

        def get_forces(self):
            import numpy as np

            return np.array([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]])

    class _FakeOpt:
        def __init__(self):
            self.atoms = _FakeAtoms()
            self.nsteps = 0
            self.observers = []

        def attach(self, fn, interval=1):
            self.observers.append((fn, interval))

        def call(self):
            for fn, interval in self.observers:
                if self.nsteps % interval == 0:
                    fn()

    stream = StringIO()
    opt = _FakeOpt()
    CompactAseOptimizerLog("ASE BFGS", every=5, max_steps=50, stream=stream).attach(opt)
    for n in range(0, 51):
        opt.nsteps = n
        opt.call()
    lines = [ln for ln in stream.getvalue().splitlines() if ln.strip()]
    # step 0, 5, 10, ..., 50 → 11 lines
    assert len(lines) == 11
    assert lines[0].startswith("ASE BFGS: step    0/50")
    assert "fmax=" in lines[0]
    assert lines[-1].startswith("ASE BFGS: step   50/50")


def test_attach_respects_quiet_and_verbose():
    class _Opt:
        def __init__(self):
            self.attached = 0

        def attach(self, *_a, **_k):
            self.attached += 1

    quiet = _Opt()
    attach_compact_ase_optimizer_log(
        quiet, argparse.Namespace(quiet_bfgs=True, verbose_bfgs=False), label="x"
    )
    assert quiet.attached == 0

    verbose = _Opt()
    attach_compact_ase_optimizer_log(
        verbose, argparse.Namespace(quiet_bfgs=False, verbose_bfgs=True), label="x"
    )
    assert verbose.attached == 0

    compact = _Opt()
    attach_compact_ase_optimizer_log(
        compact,
        argparse.Namespace(quiet_bfgs=False, verbose_bfgs=False, bfgs_log_every=None),
        label="x",
        max_steps=50,
    )
    assert compact.attached == 1
