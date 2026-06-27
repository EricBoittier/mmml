"""HEAT dynamics diagnostic summaries."""

from __future__ import annotations

import argparse

from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
    build_hoover_heat_dynamics,
    describe_heat_dynamics_setup,
    format_heat_dynamics_diagnostics,
)


def test_describe_hoover_cpt_heat_mode():
    kw = build_hoover_heat_dynamics(
        timestep_ps=0.00025,
        duration_ps=0.0375,
        save_interval_ps=0.004,
        temp=300.0,
        firstt=12.0,
        finalt=300.0,
        use_pbc=True,
        tmass=930,
    )
    kw["restart"] = False
    kw["start"] = True
    kw["iasvel"] = 1
    kw["ihtfrq"] = 0

    info = describe_heat_dynamics_setup(
        kw,
        heat_thermostat="hoover",
        use_pbc=True,
        use_memory=True,
    )
    assert "Hoover CPT NVT" in info["mode"]
    assert "single dyna" in info["velocity_init"]
    assert info["cpt"]["pmass"] == 0
    assert info["cpt"]["tmass"] == 930


def test_format_heat_diagnostics_includes_sections(capsys):
    kw = build_hoover_heat_dynamics(
        timestep_ps=0.00025,
        duration_ps=0.0375,
        save_interval_ps=0.004,
        temp=300.0,
        firstt=12.0,
        finalt=300.0,
        use_pbc=True,
        tmass=930,
    )
    kw.update(start=True, restart=False, iasvel=1, ihtfrq=0)
    args = argparse.Namespace(
        heat_thermostat="hoover",
        charmm_mm_pretreat=False,
        setup="pbc_npt",
        heat_firstt=12.0,
        heat_finalt=300.0,
        heat_ihtfrq=0,
        no_echeck_heat=False,
        no_echeck=False,
        heat_comp_damp=False,
        n_heat_segments=1,
        quiet=False,
    )
    info = describe_heat_dynamics_setup(
        kw,
        heat_thermostat="hoover",
        use_pbc=True,
        args=args,
        segment_index=0,
        n_segments=4,
    )
    text = format_heat_dynamics_diagnostics(info)
    assert "=== HEAT dynamics diagnostics (segment 1/4) ===" in text
    assert "Thermostat policy:" in text
    assert "Hoover CPT NVT" in text
    assert "Velocity / CPT initialization:" in text
    assert "CPT barostat:" in text
    assert "pmass=0" in text


def test_infer_heat_velocity_init_post_assign_scale_path():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import _infer_heat_velocity_init_label

    kw = {
        "start": True,
        "restart": False,
        "iasvel": 1,
        "ihtfrq": 500,
    }
    label = _infer_heat_velocity_init_label(kw, heat_thermostat="scale")
    assert "single dyna" in label
    assert "ihtfrq velocity scaling" in label


def test_build_heat_dynamics_dashboard_sections():
    kw = build_hoover_heat_dynamics(
        timestep_ps=0.00025,
        duration_ps=0.0375,
        save_interval_ps=0.004,
        temp=300.0,
        firstt=12.0,
        finalt=300.0,
        use_pbc=True,
        tmass=930,
    )
    kw.update(start=True, restart=False, iasvel=1, ihtfrq=0)
    info = describe_heat_dynamics_setup(
        kw,
        heat_thermostat="hoover",
        use_pbc=True,
        segment_index=0,
        n_segments=4,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        build_heat_dynamics_dashboard_sections,
    )

    sections = build_heat_dynamics_dashboard_sections(info)
    titles = [t for t, _ in sections]
    assert "Thermostat policy" in titles
    assert "Integration" in titles
    assert any("Hoover CPT NVT" in str(m) for _, m in sections)
