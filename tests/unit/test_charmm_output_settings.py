"""Unit tests for CHARMM print / heating cadence resolution (no CHARMM runtime)."""

from __future__ import annotations

import argparse

from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
    resolve_dynamics_print_kwargs,
    resolve_heat_ihtfrq,
)
from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
    apply_heat_ramp_frequencies,
    apply_heat_ramp_overlap_chunk,
    finalize_heat_dynamics_frequencies,
    heat_ramp_bath_target_K,
)
from mmml.interfaces.pycharmmInterface.mlpot.staged_workflow import _build_stage_dynamics_kw


def test_resolve_heat_ihtfrq_defaults_to_dyn_nprint():
    args = argparse.Namespace(heat_ihtfrq=0, dyn_nprint=500, quiet=False)
    assert resolve_heat_ihtfrq(args, nstep=2000) == 500


def test_resolve_heat_ihtfrq_explicit_override():
    args = argparse.Namespace(heat_ihtfrq=40, dyn_nprint=500, quiet=False)
    assert resolve_heat_ihtfrq(args, nstep=2000) == 40


def test_resolve_heat_ihtfrq_quiet_uses_full_stage():
    args = argparse.Namespace(heat_ihtfrq=0, dyn_nprint=500, quiet=True)
    assert resolve_heat_ihtfrq(args, nstep=123) == 123


def test_build_stage_heat_ihtfrq_matches_dyn_nprint():
    args = argparse.Namespace(heat_ihtfrq=0, dyn_nprint=250, quiet=False)
    dyn_print = resolve_dynamics_print_kwargs(args, nstep=5000)
    kw = _build_stage_dynamics_kw(
        "heat",
        args=args,
        timestep_ps=0.0005,
        nstep=5000,
        save_interval_ps=0.1,
        temp=300.0,
        echeck=100.0,
        dyn_print=dyn_print,
        restart=False,
        use_pbc=False,
    )
    assert kw["ihtfrq"] == 250
    assert kw["nprint"] == 250


def test_apply_heat_ramp_frequencies_recomputes_teminc():
    kw = {"firstt": 60.0, "finalt": 300.0}
    apply_heat_ramp_frequencies(kw, nstep=4000, ihtfrq=40)
    assert kw["ihtfrq"] == 40
    assert kw["TEMINC"] == (300.0 - 60.0) / (4000 // 40)


def test_heat_ramp_bath_target_at_step():
    assert heat_ramp_bath_target_K(
        firstt=0.0,
        finalt=240.0,
        teminc=0.12,
        ihtfrq=100,
        step=2000,
    ) == 2.4
    assert heat_ramp_bath_target_K(
        firstt=0.0,
        finalt=240.0,
        teminc=0.12,
        ihtfrq=100,
        step=250000,
    ) == 240.0


def test_apply_heat_ramp_overlap_chunk_continues_ramp():
    chunk_kw: dict = {}
    spec = {"firstt": 0.0, "finalt": 240.0, "teminc": 0.12, "ihtfrq": 100}
    apply_heat_ramp_overlap_chunk(
        chunk_kw,
        chunk_index=2,
        steps_done=2000,
        ramp_spec=spec,
    )
    assert chunk_kw["firstt"] == 2.4
    assert chunk_kw["finalt"] == 240.0
    assert chunk_kw["TEMINC"] == 0.12
    assert chunk_kw["iasors"] == 0
    assert chunk_kw["iasvel"] == 0


def test_finalize_heat_dynamics_frequencies_harmonizes_ihtfrq_and_teminc():
    kw = {
        "nstep": 2000,
        "firstt": 0.0,
        "finalt": 240.0,
        "ihtfrq": 100,
        "iprfrq": 2000,
        "nprint": 500,
        "isvfrq": 2000,
        "ntrfrq": 1000,
    }
    apply_heat_ramp_frequencies(kw, nstep=2000, ihtfrq=100)
    changes = finalize_heat_dynamics_frequencies(kw)
    assert kw["ihtfrq"] == 100
    assert kw["TEMINC"] == 240.0 / (2000 // 100)
    assert "ihtfrq" not in changes


def test_free_space_equi_restart_disables_ihtfrq():
    args = argparse.Namespace(heat_ihtfrq=0, dyn_nprint=500, quiet=False)
    dyn_print = resolve_dynamics_print_kwargs(args, nstep=500)
    kw = _build_stage_dynamics_kw(
        "equi",
        args=args,
        timestep_ps=0.0001,
        nstep=500,
        save_interval_ps=0.04,
        temp=300.0,
        echeck=50000.0,
        dyn_print=dyn_print,
        restart=True,
        use_pbc=False,
    )
    assert kw["ihtfrq"] == 0
    assert kw["TEMINC"] == 0.0
