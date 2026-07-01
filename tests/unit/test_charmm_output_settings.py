"""Unit tests for CHARMM print / heating cadence resolution (no CHARMM runtime)."""

from __future__ import annotations

import argparse

import pytest

from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
    resolve_dynamics_freq_cadence,
    resolve_dynamics_print_kwargs,
    resolve_heat_ihtfrq,
)
from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
    apply_heat_ramp_frequencies,
    apply_heat_ramp_overlap_chunk,
    apply_heat_segment_ramp_kwargs,
    build_hoover_heat_dynamics,
    finalize_heat_dynamics_frequencies,
    heat_ramp_bath_target_K,
)
from mmml.interfaces.pycharmmInterface.mlpot.staged_workflow import _build_stage_dynamics_kw


def test_resolve_heat_ihtfrq_defaults_to_freq_cadence():
    args = argparse.Namespace(
        heat_ihtfrq=0, dyn_nprint=500, quiet=False, dyn_freq_cadence=50
    )
    assert resolve_heat_ihtfrq(args, nstep=2000) == 50


def test_resolve_heat_ihtfrq_legacy_defaults_to_dyn_nprint():
    args = argparse.Namespace(
        heat_ihtfrq=0, dyn_nprint=500, quiet=False, dyn_freq_cadence=0
    )
    assert resolve_heat_ihtfrq(args, nstep=2000) == 500


def test_resolve_heat_ihtfrq_short_stage_ensures_multiple_rescales():
    args = argparse.Namespace(
        heat_ihtfrq=0, dyn_nprint=500, quiet=False, dyn_freq_cadence=0
    )
    assert resolve_heat_ihtfrq(args, nstep=200) == 25
    assert 200 // resolve_heat_ihtfrq(args, nstep=200) >= 8


def test_resolve_heat_ihtfrq_explicit_override():
    args = argparse.Namespace(
        heat_ihtfrq=40, dyn_nprint=500, quiet=False, dyn_freq_cadence=50
    )
    assert resolve_heat_ihtfrq(args, nstep=2000) == 40


def test_resolve_heat_ihtfrq_quiet_uses_full_stage():
    args = argparse.Namespace(
        heat_ihtfrq=0, dyn_nprint=500, quiet=True, dyn_freq_cadence=50
    )
    assert resolve_heat_ihtfrq(args, nstep=123) == 123


def test_resolve_dynamics_print_kwargs_uses_cadence_not_nsavc():
    args = argparse.Namespace(
        dyn_nprint=500, dyn_iprfrq=2000, quiet=False, dyn_freq_cadence=50
    )
    dyn_print = resolve_dynamics_print_kwargs(args, nstep=500, nsavc=499)
    assert dyn_print == {"nprint": 50, "iprfrq": 50, "isvfrq": 50, "nsavv": 50}


def test_resolve_dynamics_freq_cadence_zero_disables():
    args = argparse.Namespace(dyn_freq_cadence=0)
    assert resolve_dynamics_freq_cadence(args) is None


def test_build_stage_heat_echeck_scales_for_large_cluster():
    args = argparse.Namespace(
        heat_ihtfrq=0,
        dyn_nprint=500,
        quiet=False,
        heat_thermostat="scale",
        no_echeck=False,
        no_echeck_heat=False,
        dyn_freq_cadence=0,
    )
    dyn_print = resolve_dynamics_print_kwargs(args, nstep=200)
    kw = _build_stage_dynamics_kw(
        "heat",
        args=args,
        timestep_ps=0.0005,
        nstep=200,
        save_interval_ps=0.004,
        temp=300.0,
        echeck=5000.0,
        dyn_print=dyn_print,
        restart=False,
        use_pbc=True,
        n_atoms=500,
        n_monomers=50,
    )
    assert kw["echeck"] == 10000.0
    assert kw["ihtfrq"] == 25


def test_build_stage_heat_ihtfrq_matches_dyn_nprint():
    args = argparse.Namespace(
        heat_ihtfrq=0, dyn_nprint=250, quiet=False, dyn_freq_cadence=0
    )
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


def test_normalize_dynamics_heat_ramp_kw_sets_tstruct_from_firstt():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        _normalize_dynamics_heat_ramp_kw,
    )

    kw = {"firstt": 48.0, "finalt": 240.0}
    _normalize_dynamics_heat_ramp_kw(kw)
    assert kw["tstruct"] == pytest.approx(48.0)


def test_build_heat_dynamics_sets_tstruct_to_firstt():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import build_heat_dynamics

    kw = build_heat_dynamics(temp=240.0, firstt=48.0, finalt=240.0, use_pbc=True)
    assert kw["firstt"] == pytest.approx(48.0)
    assert kw["tstruct"] == pytest.approx(48.0)


def test_apply_heat_ramp_frequencies_caps_ihtfrq_below_nstep():
    """Overlap chunks (nstep=500, stage ihtfrq=500) need interior IHTFRQ events."""
    kw = {"firstt": 1.0, "finalt": 5.0}
    apply_heat_ramp_frequencies(kw, nstep=500, ihtfrq=500)
    assert kw["ihtfrq"] < 500
    assert 500 % kw["ihtfrq"] == 0
    assert kw["TEMINC"] > 0.0


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
    chunk_kw: dict = {"TEMINC": 0.12, "ihtfrq": 100}
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
    assert chunk_kw["iasvel"] == 1
    assert chunk_kw["start"] is False


def test_hoover_cpt_heat_ramp_infers_cold_start_without_start_flag():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        _apply_overlap_chunk_dynamics_kw,
        apply_hoover_cpt_heat_ramp_overlap_chunk,
    )

    # Mimics kw after overlap helpers strip ``start`` but before dyna (cluster logs).
    chunk_kw = {
        "cpt": True,
        "hoover reft": 6.0,
        "firstt": 6.0,
        "finalt": 30.0,
        "tbath": 30.0,
    }
    _apply_overlap_chunk_dynamics_kw(chunk_kw, chunk_index=0, has_restart_read=False)
    apply_hoover_cpt_heat_ramp_overlap_chunk(
        chunk_kw,
        chunk_index=0,
        steps_done=0,
        ramp_spec={"firstt": 6.0, "finalt": 30.0},
        total_nstep=2500,
        n_chunks=10,
    )
    assert chunk_kw["iasvel"] == 1
    assert chunk_kw["start"] is True


def test_hoover_cpt_heat_ramp_preserves_cold_start_on_overlap_chunk_zero():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        _apply_overlap_chunk_dynamics_kw,
        apply_hoover_cpt_heat_ramp_overlap_chunk,
    )

    chunk_kw = {
        "start": True,
        "iasvel": 1,
        "cpt": True,
        "hoover reft": 6.0,
        "firstt": 6.0,
        "finalt": 30.0,
        "tbath": 30.0,
    }
    _apply_overlap_chunk_dynamics_kw(chunk_kw, chunk_index=0, has_restart_read=False)
    apply_hoover_cpt_heat_ramp_overlap_chunk(
        chunk_kw,
        chunk_index=0,
        steps_done=0,
        ramp_spec={"firstt": 6.0, "finalt": 30.0},
        total_nstep=250,
        n_chunks=1,
    )
    assert chunk_kw["iasvel"] == 1
    assert chunk_kw["start"] is True
    assert chunk_kw["restart"] is False


def test_hoover_cpt_heat_ramp_overlap_chunk_keeps_iasvel_zero_after_boltzmann():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        _apply_overlap_chunk_dynamics_kw,
        apply_hoover_cpt_heat_ramp_overlap_chunk,
    )

    chunk_kw = {
        "start": False,
        "iasvel": 0,
        "cpt": True,
        "hoover reft": 2.4,
        "firstt": 2.4,
        "finalt": 12.0,
        "tbath": 12.0,
    }
    _apply_overlap_chunk_dynamics_kw(chunk_kw, chunk_index=0, has_restart_read=False)
    apply_hoover_cpt_heat_ramp_overlap_chunk(
        chunk_kw,
        chunk_index=0,
        steps_done=0,
        ramp_spec={"firstt": 2.4, "finalt": 12.0},
        total_nstep=4000,
        n_chunks=8,
    )
    assert chunk_kw["iasvel"] == 0
    assert chunk_kw["start"] is False
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        apply_hoover_cpt_heat_ramp_overlap_chunk,
        hoover_cpt_heat_ramp_spec_from_kw,
        hoover_cpt_heat_ramp_target_K,
    )

    assert hoover_cpt_heat_ramp_target_K(
        firstt=0.0,
        finalt=30.0,
        step=500,
        total_nstep=25000,
        n_chunks=50,
    ) == pytest.approx(0.6)
    assert hoover_cpt_heat_ramp_target_K(
        firstt=0.0,
        finalt=30.0,
        step=0,
        total_nstep=2500,
        n_chunks=1,
    ) == pytest.approx(15.0)
    kw = build_hoover_heat_dynamics(
        temp=30.0,
        firstt=0.0,
        finalt=30.0,
        use_pbc=True,
        tmass=2000,
    )
    spec = hoover_cpt_heat_ramp_spec_from_kw(kw)
    assert spec == {"firstt": 0.0, "finalt": 30.0}
    chunk_kw: dict = {}
    apply_hoover_cpt_heat_ramp_overlap_chunk(
        chunk_kw,
        chunk_index=0,
        steps_done=0,
        ramp_spec=spec,
        total_nstep=2500,
        n_chunks=1,
    )
    assert chunk_kw["hoover reft"] == pytest.approx(15.0)
    assert chunk_kw["iasvel"] == 0
    chunk_kw = {}
    apply_hoover_cpt_heat_ramp_overlap_chunk(
        chunk_kw,
        chunk_index=1,
        steps_done=500,
        ramp_spec=spec,
        total_nstep=25000,
        n_chunks=50,
    )
    assert chunk_kw["firstt"] == pytest.approx(0.6)
    assert chunk_kw["finalt"] == 30.0
    assert chunk_kw["tbath"] == 30.0
    assert chunk_kw["hoover reft"] == pytest.approx(0.6)
    assert chunk_kw["iasvel"] == 0
    assert chunk_kw["start"] is False

    chunk_kw = {"restart": True}
    apply_hoover_cpt_heat_ramp_overlap_chunk(
        chunk_kw,
        chunk_index=1,
        steps_done=500,
        ramp_spec=spec,
        total_nstep=25000,
        n_chunks=50,
    )
    assert chunk_kw["iasvel"] == 0


def test_apply_heat_segment_ramp_kwargs_splits_ramp():
    kw = {"firstt": 0.0, "finalt": 240.0, "ihtfrq": 100, "tbath": 240.0}
    apply_heat_segment_ramp_kwargs(
        kw,
        seg_index=1,
        n_segments=4,
        heat_firstt=0.0,
        heat_finalt=240.0,
        nstep=20000,
        ihtfrq=100,
    )
    assert kw["firstt"] == 60.0
    assert kw["finalt"] == 120.0
    assert kw["TEMINC"] == 60.0 / (20000 // 100)


def test_apply_heat_segment_ramp_hoover_reft_starts_at_segment_low():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import apply_heat_segment_ramp_kwargs

    kw = {
        "firstt": 0.0,
        "finalt": 240.0,
        "hoover reft": 240.0,
        "ihtfrq": 0,
        "cpt": True,
        "tbath": 240.0,
    }
    apply_heat_segment_ramp_kwargs(
        kw,
        seg_index=1,
        n_segments=8,
        heat_firstt=0.0,
        heat_finalt=240.0,
        nstep=2500,
        ihtfrq=0,
    )
    assert kw["firstt"] == pytest.approx(30.0)
    assert kw["finalt"] == pytest.approx(60.0)
    assert kw["hoover reft"] == pytest.approx(30.0)


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
    args = argparse.Namespace(
        heat_ihtfrq=0, dyn_nprint=500, quiet=False, dyn_freq_cadence=0
    )
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
