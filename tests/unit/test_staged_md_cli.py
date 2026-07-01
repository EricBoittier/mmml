"""Tests for staged MLpot CLI stage/PBC resolution."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
    recommend_echeck_kcal,
    resolve_charmm_use_pbc,
    resolve_echeck_for_cluster,
    resolve_flat_bottom_selection,
    resolve_loose_pbc,
    resolve_md_stages,
    resolve_mlpot_use_pbc,
    resolve_use_pbc,
)
from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
    CharmmTrajectoryFiles,
    build_hoover_heat_dynamics,
)
from mmml.interfaces.pycharmmInterface.mlpot.staged_workflow import (
    _artifact_paths,
    _build_stage_dynamics_kw,
    _configure_heat_dynamics_start,
    _configure_nve_dynamics_start,
    _overlap_for_stage,
    _publish_staged_handoff,
    _prior_restart_for_stage,
    _should_seed_heat_prior_restart,
)
from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import DynamicsOverlapConfig


def test_resolve_md_stages_pycharmm_full():
    args = argparse.Namespace(setup="pycharmm_full", md_stages=None, md_stage=None, phase="staged")
    assert resolve_md_stages(args) == ["mini", "heat", "nve", "equi", "prod"]


def test_resolve_md_stages_free_nve():
    args = argparse.Namespace(setup="free_nve", md_stages=None, md_stage=None, phase="staged")
    assert resolve_md_stages(args) == ["mini", "nve"]


def test_resolve_md_stages_override():
    args = argparse.Namespace(
        setup="pycharmm_full",
        md_stages="mini,heat",
        md_stage=None,
        phase="staged",
    )
    assert resolve_md_stages(args) == ["mini", "heat"]


def test_publish_staged_handoff_records_mini_only_state(tmp_path: Path):
    restart = tmp_path / "mini.res"
    restart.write_text("REST 1 7\n", encoding="ascii")
    args = argparse.Namespace(box_size=32.0, temperature=260.0, pressure=None)
    z = np.array([6, 1, 1], dtype=np.int32)
    handoff = object()

    with patch(
        "mmml.cli.run.md_handoff.handoff_from_charmm",
        return_value=handoff,
    ) as handoff_from_charmm, patch(
        "mmml.cli.run.md_handoff.set_handoff_out",
    ) as set_handoff_out, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation.read_restart_last_step",
        return_value=7,
    ):
        _publish_staged_handoff(
            atomic_numbers=z,
            restart_path=restart,
            args=args,
            tag="DCM20",
            stages=["mini"],
        )

    handoff_from_charmm.assert_called_once()
    _, kwargs = handoff_from_charmm.call_args
    np.testing.assert_array_equal(handoff_from_charmm.call_args.args[0], z)
    assert kwargs["restart_path"] == restart
    assert kwargs["fallback_box_side_A"] == 32.0
    assert kwargs["temperature_K"] == 260.0
    assert kwargs["step"] == 7
    assert kwargs["metadata"] == {
        "backend": "pycharmm",
        "tag": "DCM20",
        "stages": ["mini"],
    }
    set_handoff_out.assert_called_once_with(handoff)


def test_resolve_use_pbc_from_setup():
    args = argparse.Namespace(setup="pbc_nve", free_space=False, box_size=None)
    assert resolve_use_pbc(args) is True


def test_resolve_use_pbc_free_nvt_setup():
    args = argparse.Namespace(setup="free_nvt", free_space=False, box_size=None)
    assert resolve_use_pbc(args) is False


def test_resolve_use_pbc_free_space():
    args = argparse.Namespace(setup="pbc_nve", free_space=True, box_size=None)
    assert resolve_use_pbc(args) is False


def test_resolve_use_pbc_box_size():
    args = argparse.Namespace(setup="free_nve", free_space=False, box_size=40.0)
    assert resolve_use_pbc(args) is True


def test_loose_pbc_charmm_on_mlpot_off_for_free_setup_with_box():
    args = argparse.Namespace(
        setup="free_nvt",
        free_space=False,
        box_size=55.0,
        mlpot_pbc=False,
    )
    assert resolve_charmm_use_pbc(args) is True
    assert resolve_mlpot_use_pbc(args) is False
    assert resolve_loose_pbc(True, False) is True
    assert resolve_loose_pbc(True, True) is False
    assert resolve_loose_pbc(False, False) is False


def test_resolve_loose_pbc():
    assert resolve_loose_pbc(True, False) is True
    assert resolve_loose_pbc(True, True) is False
    assert resolve_loose_pbc(False, False) is False


def test_mlpot_pbc_flag_enables_mic_on_free_setup_with_box():
    args = argparse.Namespace(
        setup="free_nvt",
        free_space=False,
        box_size=55.0,
        mlpot_pbc=True,
    )
    assert resolve_charmm_use_pbc(args) is True
    assert resolve_mlpot_use_pbc(args) is True


def test_pbc_setup_enables_both():
    args = argparse.Namespace(setup="pbc_nve", free_space=False, box_size=None)
    assert resolve_charmm_use_pbc(args) is True
    assert resolve_mlpot_use_pbc(args) is True


def test_resolve_flat_bottom_selection_dcm_keeps_all_default():
    args = argparse.Namespace(composition="DCM:90", residue="ACO", fb_selection="all")
    assert resolve_flat_bottom_selection(args) == "all"


def test_resolve_flat_bottom_selection_respects_explicit_value():
    args = argparse.Namespace(
        composition="DCM:90",
        residue="ACO",
        fb_selection="TYPE CLGA1",
    )
    assert resolve_flat_bottom_selection(args) == "TYPE CLGA1"


def test_cubic_box_length_from_geometry():
    import numpy as np

    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
        cubic_box_length_from_geometry,
    )

    pos = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    side = cubic_box_length_from_geometry(pos, ml_cutoff=12.0, pad=10.0)
    assert side >= 2.0 * 12.0 + 10.0
    assert side >= 10.0 + 20.0


def test_prior_restart_for_equi_falls_back_to_heat_without_nve(tmp_path: Path):
    paths = _artifact_paths(tmp_path, "dcm_60")
    paths["heat_res"].write_text("heat\n", encoding="utf-8")

    got = _prior_restart_for_stage("equi", paths, restart_from=None)
    assert got == paths["heat_res"]


def test_prior_restart_for_equi_prefers_nve_when_present(tmp_path: Path):
    paths = _artifact_paths(tmp_path, "dcm_60")
    paths["heat_res"].write_text("heat\n", encoding="utf-8")
    paths["nve_res"].write_text("nve\n", encoding="utf-8")

    got = _prior_restart_for_stage("equi", paths, restart_from=None)
    assert got == paths["nve_res"]


def test_prior_restart_for_heat_uses_geometry_baseline(tmp_path: Path):
    paths = _artifact_paths(tmp_path, "dcm_155")
    paths["geometry_baseline_res"].write_text("baseline\n", encoding="utf-8")
    pretreat = paths["charmm_mm_prod_res"].parent
    pretreat.mkdir(parents=True, exist_ok=True)
    paths["charmm_mm_prod_res"].write_text("prod\n", encoding="utf-8")

    got = _prior_restart_for_stage("heat", paths, restart_from=None)
    assert got == paths["geometry_baseline_res"]


def test_prior_restart_for_heat_ignores_pretreat_restart_from(tmp_path: Path):
    paths = _artifact_paths(tmp_path, "dcm_155")
    paths["geometry_baseline_res"].write_text("baseline\n", encoding="utf-8")
    pretreat = paths["charmm_mm_prod_res"].parent
    pretreat.mkdir(parents=True, exist_ok=True)
    paths["charmm_mm_prod_res"].write_text("prod\n", encoding="utf-8")

    got = _prior_restart_for_stage(
        "heat", paths, restart_from=paths["charmm_mm_prod_res"]
    )
    assert got == paths["geometry_baseline_res"]


def test_prior_restart_for_heat_rejects_pretreat_without_baseline(tmp_path: Path):
    paths = _artifact_paths(tmp_path, "dcm_155")
    pretreat = paths["charmm_mm_prod_res"].parent
    pretreat.mkdir(parents=True, exist_ok=True)
    paths["charmm_mm_prod_res"].write_text("prod\n", encoding="utf-8")

    got = _prior_restart_for_stage(
        "heat", paths, restart_from=paths["charmm_mm_prod_res"]
    )
    assert got is None


def test_prior_restart_for_equi_rejects_handoff_seed_restart_from(tmp_path: Path):
    handoff = tmp_path / "handoff" / "continue_seed.res"
    handoff.parent.mkdir(parents=True)
    handoff.write_text("seed\n", encoding="utf-8")
    got = _prior_restart_for_stage(
        "equi", {}, restart_from=handoff, tag="dcm_52", n_heat_segments=1
    )
    assert got is None


def test_should_seed_heat_prior_restart_after_mini_in_memory():
    assert _should_seed_heat_prior_restart(
        seg_i=0,
        prev_restart_is_current_state=True,
        use_memory=False,
        memory_handoff_next=False,
    )
    assert not _should_seed_heat_prior_restart(
        seg_i=1,
        prev_restart_is_current_state=True,
        use_memory=False,
        memory_handoff_next=False,
    )


def test_recommend_echeck_kcal_single_monomer():
    assert recommend_echeck_kcal(1, 20) == 100.0


def test_recommend_echeck_kcal_dcm9():
    assert recommend_echeck_kcal(9, 45) == 500.0


def test_resolve_heat_firstt_finalt_defaults():
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        resolve_heat_firstt_finalt,
    )

    args = argparse.Namespace(heat_firstt=None, heat_finalt=None, heat_mode="ramp")
    assert resolve_heat_firstt_finalt(args, default_temp=300.0) == (60.0, 300.0)
    assert resolve_heat_firstt_finalt(args, default_temp=40.0) == (10.0, 40.0)


def test_resolve_heat_firstt_finalt_dcm9_soft():
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        resolve_heat_firstt_finalt,
    )

    args = argparse.Namespace(heat_firstt=0.0, heat_finalt=240.0, heat_mode="ramp")
    assert resolve_heat_firstt_finalt(args, default_temp=300.0) == (48.0, 240.0)


def test_resolve_heat_firstt_finalt_hold_mode():
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        resolve_heat_firstt_finalt,
    )

    args = argparse.Namespace(heat_firstt=60.0, heat_finalt=None, heat_mode="hold")
    assert resolve_heat_firstt_finalt(args, default_temp=300.0) == (300.0, 300.0)


def test_default_stages_for_thermalize_setups():
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        _default_stages_for_setup,
    )

    assert _default_stages_for_setup("pbc_thermalize") == ["mini", "heat", "equi"]
    assert _default_stages_for_setup("free_thermalize") == ["mini", "heat"]


def test_resolve_stage_ps_handles_explicit_none():
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import resolve_stage_ps

    args = argparse.Namespace(
        ps=1.0,
        ps_prod=None,
        ps_equi=None,
        ps_heat=None,
        ps_nve=None,
    )
    assert resolve_stage_ps(args, "prod") == 1.0
    assert resolve_stage_ps(args, "equi") == 50.0
    assert resolve_stage_ps(args, "heat") == 10.0
    assert resolve_stage_ps(args, "nve") == 1.0


def test_recommend_echeck_kcal_medium_cluster():
    assert recommend_echeck_kcal(20, 100) == 1000.0


def test_recommend_echeck_kcal_dcm90():
    assert recommend_echeck_kcal(90, 450) == 4500.0


def test_resolve_echeck_for_cluster_scales_dcm90():
    args = argparse.Namespace(echeck=500.0, no_echeck=False, no_scale_echeck=False)
    assert resolve_echeck_for_cluster(args, n_atoms=450, n_monomers=90) == 4500.0


def test_resolve_echeck_for_cluster_respects_no_scale():
    args = argparse.Namespace(echeck=500.0, no_echeck=False, no_scale_echeck=True)
    assert resolve_echeck_for_cluster(args, n_atoms=450, n_monomers=90) == 500.0


def test_resolve_echeck_for_cluster_user_floor_above_recommended():
    args = argparse.Namespace(echeck=8000.0, no_echeck=False, no_scale_echeck=False)
    assert resolve_echeck_for_cluster(args, n_atoms=450, n_monomers=90) == 8000.0


def test_build_stage_dynamics_kw_restart_omits_invalid_res_flag():
    args = argparse.Namespace()
    dyn_print = {"nprint": 100, "iprfrq": 500, "isvfrq": 500}
    kw = _build_stage_dynamics_kw(
        "nve",
        args=args,
        timestep_ps=0.0005,
        nstep=2000,
        save_interval_ps=0.05,
        temp=240.0,
        echeck=10000.0,
        dyn_print=dyn_print,
        restart=True,
    )
    assert kw["restart"] is True
    assert "res" not in kw


def test_build_stage_dynamics_kw_restart_false_sets_cold_start_flags():
    args = argparse.Namespace()
    dyn_print = {"nprint": 100, "iprfrq": 500, "isvfrq": 500}
    cold = _build_stage_dynamics_kw(
        "nve",
        args=args,
        timestep_ps=0.0002,
        nstep=1000,
        save_interval_ps=0.02,
        temp=80.0,
        echeck=500.0,
        dyn_print=dyn_print,
        restart=False,
    )
    assert cold["new"] is True
    assert cold["start"] is True
    resumed = _build_stage_dynamics_kw(
        "nve",
        args=args,
        timestep_ps=0.0002,
        nstep=1000,
        save_interval_ps=0.02,
        temp=80.0,
        echeck=500.0,
        dyn_print=dyn_print,
        restart=True,
    )
    assert resumed["new"] is False
    assert resumed["start"] is False
    assert resumed["restart"] is True


def test_build_stage_dynamics_kw_prod_restart_avoids_cold_start():
    args = argparse.Namespace(
        npt_thermostat="hoover",
        npt_pressure=1.0,
        npt_pgamma=5.0,
    )
    dyn_print = {"nprint": 100, "iprfrq": 500, "isvfrq": 500}
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics.compute_cpt_piston_masses",
        return_value=(80, 800),
    ):
        cold = _build_stage_dynamics_kw(
            "prod",
            args=args,
            timestep_ps=0.0002,
            nstep=1000,
            save_interval_ps=0.02,
            temp=80.0,
            echeck=500.0,
            dyn_print=dyn_print,
            restart=False,
            use_pbc=True,
        )
        assert cold["new"] is True
        assert cold["start"] is True
        resumed = _build_stage_dynamics_kw(
            "prod",
            args=args,
            timestep_ps=0.0002,
            nstep=1000,
            save_interval_ps=0.02,
            temp=80.0,
            echeck=500.0,
            dyn_print=dyn_print,
            restart=True,
            use_pbc=True,
        )
    assert resumed["new"] is False
    assert resumed["start"] is False
    assert resumed["restart"] is True


def test_build_stage_dynamics_kw_heat_hoover_vacuum_uses_ihtfrq_fallback():
    args = argparse.Namespace(heat_thermostat="hoover", heat_firstt=0.0, heat_finalt=240.0)
    dyn_print = {"nprint": 100, "iprfrq": 500, "isvfrq": 500}
    kw = _build_stage_dynamics_kw(
        "heat",
        args=args,
        timestep_ps=0.00025,
        nstep=80000,
        save_interval_ps=0.125,
        temp=300.0,
        echeck=100.0,
        dyn_print=dyn_print,
        restart=False,
        use_pbc=False,
    )
    assert "cpt" not in kw
    assert "hoover reft" not in kw
    assert kw["iasors"] == 0
    assert kw["ihtfrq"] > 0
    assert kw["TEMINC"] > 0


def test_build_stage_dynamics_kw_heat_hoover_pbc_disables_ihtfrq_ramp():
    args = argparse.Namespace(
        heat_thermostat="hoover",
        heat_firstt=0.0,
        heat_finalt=240.0,
        heat_mode="ramp",
    )
    dyn_print = {"nprint": 100, "iprfrq": 500, "isvfrq": 500}
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics.compute_cpt_piston_masses",
        return_value=(80, 800),
    ):
        kw = _build_stage_dynamics_kw(
            "heat",
            args=args,
            timestep_ps=0.00025,
            nstep=80000,
            save_interval_ps=0.125,
            temp=300.0,
            echeck=100.0,
            dyn_print=dyn_print,
            restart=False,
            use_pbc=True,
        )
    assert kw["hoover reft"] == 48.0
    assert kw["tmass"] == 800
    assert kw["echeck"] == 5000.0
    assert kw["pgamma"] == 0.0
    assert kw["cpt"] is True
    assert kw["ihtfrq"] == 0
    assert "TEMINC" not in kw


def test_build_stage_dynamics_kw_heat_auto_no_echeck_heat_disables_echeck():
    args = argparse.Namespace(heat_thermostat="scale", _auto_no_echeck_heat=True)
    dyn_print = {"nprint": 100, "iprfrq": 500, "isvfrq": 500}
    kw = _build_stage_dynamics_kw(
        "heat",
        args=args,
        timestep_ps=0.00025,
        nstep=80000,
        save_interval_ps=0.125,
        temp=300.0,
        echeck=8000.0,
        dyn_print=dyn_print,
        restart=False,
        use_pbc=True,
    )
    assert kw["echeck"] == -1.0


def test_build_stage_dynamics_kw_heat_no_echeck_heat_disables_echeck():
    args = argparse.Namespace(heat_thermostat="scale", no_echeck_heat=True)
    dyn_print = {"nprint": 100, "iprfrq": 500, "isvfrq": 500}
    kw = _build_stage_dynamics_kw(
        "heat",
        args=args,
        timestep_ps=0.00025,
        nstep=80000,
        save_interval_ps=0.125,
        temp=300.0,
        echeck=8000.0,
        dyn_print=dyn_print,
        restart=False,
        use_pbc=True,
    )
    assert kw["echeck"] == -1.0
    assert "cpt" not in kw
    assert kw["ihtfrq"] > 0


def test_resolve_heat_thermostat_coerces_scale_after_pretreat():
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        resolve_charmm_mm_pretreat_for_staged,
        resolve_heat_thermostat,
    )

    args = argparse.Namespace(
        heat_thermostat="scale",
        charmm_mm_pretreat=True,
        setup="pbc_npt",
        quiet=True,
    )
    assert resolve_heat_thermostat(args) == "hoover"

    args_no_pretreat = argparse.Namespace(
        heat_thermostat="scale",
        charmm_mm_pretreat=False,
        setup="pbc_npt",
        quiet=True,
    )
    assert resolve_heat_thermostat(args_no_pretreat) == "scale"


def test_resolve_charmm_mm_pretreat_for_staged_skips_handoff():
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        resolve_charmm_mm_pretreat_for_staged,
    )

    args = argparse.Namespace(charmm_mm_pretreat=True)
    assert resolve_charmm_mm_pretreat_for_staged(args, handoff_coords_in_memory=False)
    assert not resolve_charmm_mm_pretreat_for_staged(args, handoff_coords_in_memory=True)

    args_force = argparse.Namespace(
        charmm_mm_pretreat=True,
        charmm_mm_pretreat_on_handoff=True,
    )
    assert resolve_charmm_mm_pretreat_for_staged(
        args_force, handoff_coords_in_memory=True
    )


def test_build_stage_dynamics_kw_heat_scale_pbc_avoids_cpt():
    args = argparse.Namespace(heat_thermostat="scale", heat_firstt=40.0, heat_finalt=200.0)
    dyn_print = {"nprint": 100, "iprfrq": 500, "isvfrq": 500}
    kw = _build_stage_dynamics_kw(
        "heat",
        args=args,
        timestep_ps=0.00025,
        nstep=40000,
        save_interval_ps=0.125,
        temp=200.0,
        echeck=5000.0,
        dyn_print=dyn_print,
        restart=False,
        use_pbc=True,
    )
    assert "cpt" not in kw
    assert kw["ihtfrq"] > 0
    assert kw["TEMINC"] > 0


def test_build_hoover_heat_tmass_floor_for_small_psf_mass():
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics.compute_cpt_piston_masses",
        return_value=(8, 80),
    ):
        kw = build_hoover_heat_dynamics(
            firstt=10.0,
            finalt=240.0,
            use_pbc=True,
        )
    assert kw["tmass"] == 400
    assert kw["pgamma"] == 0.0
    assert kw["hoover reft"] == 10.0


def test_resolve_heat_hoover_tmass_explicit_and_clamped_default():
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import resolve_heat_hoover_tmass

    args = argparse.Namespace(heat_hoover_tmass=600)
    assert resolve_heat_hoover_tmass(args, psf_tmass=80) == 600
    args = argparse.Namespace(heat_hoover_tmass=None)
    assert resolve_heat_hoover_tmass(args, psf_tmass=80) == 400
    assert resolve_heat_hoover_tmass(args, psf_tmass=5000) == 1200


def test_build_stage_dynamics_kw_free_space_equi_uses_charmm_heat_controls():
    args = argparse.Namespace()
    dyn_print = {"nprint": 100, "iprfrq": 500, "isvfrq": 500}
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
    assert kw["restart"] is True
    assert kw["ihtfrq"] == 0
    assert kw["TEMINC"] == 0.0
    assert kw["iasvel"] == 0
    assert "firstt" not in kw
    assert "hoover reft" not in kw
    assert "tmass" not in kw
    assert "cpt" not in kw
    assert "pint pconst pref" not in kw
    assert kw["imgfrq"] == 0


def test_overlap_for_stage_enables_heat_chunking():
    cfg = DynamicsOverlapConfig(
        action="rescue",
        min_distance_A=0.4,
        check_interval=100,
        n_monomers=10,
        use_pbc=False,
    )

    assert _overlap_for_stage("heat", cfg) is cfg
    assert _overlap_for_stage("equi", cfg) is cfg
    assert _overlap_for_stage("prod", cfg) is cfg


def test_configure_heat_dynamics_start_hoover_memory_handoff_no_comp_velocities():
    """Hoover CPT after mini: single dyna start=True (no nstep=0 assign)."""
    io = CharmmTrajectoryFiles()
    kw = {
        "firstt": 0.0,
        "finalt": 240.0,
        "tbath": 240.0,
        "cpt": True,
        "hoover reft": 240.0,
        "tmass": 2000,
        "start": True,
    }

    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics.assign_velocities_at_temperature"
    ) as assign:
        _configure_heat_dynamics_start(
            kw,
            io,
            coords_in_memory=True,
            restart_from_file=False,
            timestep_ps=0.00025,
            use_pbc=True,
            quiet=True,
            heat_thermostat="hoover",
        )

    assign.assert_not_called()
    assert kw["restart"] is False
    assert kw["start"] is True
    assert kw["iasvel"] == 1


def test_configure_heat_dynamics_start_scale_memory_handoff_single_dyna():
    """Scale heat after mini: single dyna start=True (no nstep=0 assign)."""
    io = CharmmTrajectoryFiles()
    kw = {
        "firstt": 26.0,
        "finalt": 130.0,
        "tbath": 130.0,
        "ihtfrq": 500,
        "start": False,
    }

    _configure_heat_dynamics_start(
        kw,
        io,
        coords_in_memory=True,
        restart_from_file=False,
        timestep_ps=0.0002,
        use_pbc=True,
        quiet=True,
        heat_thermostat="scale",
    )

    assert kw["restart"] is False
    assert kw["new"] is False
    assert kw["start"] is True
    assert kw["iasvel"] == 1
    assert kw["iasors"] == 0


def test_configure_nve_dynamics_start_memory_handoff_single_dyna(tmp_path):
    """After mini, NVE uses start=True on the main dyna (no nstep=0 assign)."""
    res = tmp_path / "nve_dcm_5.res"
    io = CharmmTrajectoryFiles(restart_write=res)
    kw = {"restart": True, "start": True, "iasvel": 1, "firstt": 300.0}

    _configure_nve_dynamics_start(
        kw,
        io,
        coords_in_memory=True,
        restart_from_file=False,
        timestep_ps=0.00025,
        use_pbc=False,
        quiet=True,
        temp=60.0,
    )

    assert kw["restart"] is False
    assert kw["start"] is True
    assert kw["iasvel"] == 1
    assert float(kw["firstt"]) == 60.0
    assert io.restart_read is None


def _write_restartable_res(path: Path, *, jhstrt: int = 250) -> None:
    path.write_text(
        "REST     1       500\n"
        " !NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NSAVL\n"
        f"   10     0     500       1      10   {jhstrt}     297       0       0\n",
        encoding="ascii",
    )


def test_configure_heat_dynamics_start_hoover_restart_file_single_dyna(tmp_path):
    """Hoover CPT from restart file: READYN+start in one dyna (no nstep=0 assign)."""
    res = tmp_path / "mini.res"
    _write_restartable_res(res, jhstrt=0)
    io = CharmmTrajectoryFiles(restart_read=res)
    kw = {
        "firstt": 2.0,
        "finalt": 10.0,
        "cpt": True,
        "hoover reft": 2.0,
        "tmass": 840,
    }

    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics.assign_velocities_at_temperature"
    ) as assign:
        _configure_heat_dynamics_start(
            kw,
            io,
            coords_in_memory=False,
            restart_from_file=True,
            timestep_ps=0.00025,
            use_pbc=True,
            quiet=True,
            heat_thermostat="hoover",
        )

    assign.assert_not_called()
    assert kw["restart"] is True
    assert kw["start"] is True
    assert kw["iasvel"] == 1
    assert io.restart_read == res


def test_configure_heat_dynamics_start_in_place_resume_uses_dyna_restart(tmp_path):
    res = tmp_path / "heat_dcm_20.res"
    _write_restartable_res(res, jhstrt=250)
    io = CharmmTrajectoryFiles(restart_read=res, restart_write=res)
    kw = {
        "firstt": 42.4,
        "finalt": 60.0,
        "cpt": True,
        "hoover reft": 60.0,
        "tmass": 2000,
        "start": True,
    }

    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics.assign_velocities_at_temperature"
    ) as assign:
        _configure_heat_dynamics_start(
            kw,
            io,
            coords_in_memory=False,
            restart_from_file=True,
            timestep_ps=0.0001,
            use_pbc=True,
            quiet=True,
            heat_thermostat="hoover",
        )

    assign.assert_not_called()
    assert kw["restart"] is True
    assert kw["start"] is False
    assert kw["new"] is False
    assert kw["iasvel"] == 1
    assert io.restart_read == res


def test_reset_stage_restart_preserves_in_place_read(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.staged_workflow import (
        _reset_stage_restart,
    )

    res = tmp_path / "heat.res"
    _write_restartable_res(res)
    _reset_stage_restart(res, restart_read=res)
    assert res.is_file()

    _reset_stage_restart(res)
    assert not res.is_file()


def test_reset_stage_restart_preserves_memory_handoff_seed(tmp_path):
    """Seeded restart_write must survive _reset_stage_restart (EQUI/HEAT handoff)."""
    from unittest.mock import patch

    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import CharmmTrajectoryFiles
    from mmml.interfaces.pycharmmInterface.mlpot.staged_workflow import (
        _reset_stage_restart,
        _seed_restart_for_memory_handoff,
    )

    res = tmp_path / "equi_dcm_30.res"
    io = CharmmTrajectoryFiles(restart_write=res)
    kw: dict = {}
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.staged_workflow.rewrite_dynamics_restart_from_current_state"
    ):
        _seed_restart_for_memory_handoff(io, kw, stage="equi")
    res.write_text("seeded restart\n", encoding="ascii")
    assert kw.get("restart") is not True

    _reset_stage_restart(res, restart_read=res)
    assert res.is_file()
    assert res.read_text(encoding="ascii") == "seeded restart\n"


def test_charmm_trajectory_files_open_for_run_mkdirs_restart_parent(tmp_path):
    """Restart writes must create nested pretreat/ and expose Fortran paths."""
    nested = tmp_path / "pretreat" / "mini_box_equil.res"
    dcd = tmp_path / "pretreat" / "mini_box_equil.dcd"

    io = CharmmTrajectoryFiles(restart_write=nested, trajectory=dcd)
    _open_files, iokw, _aliases = io.open_for_run()

    assert nested.parent.is_dir()
    assert isinstance(iokw["iunwri"], str)
    assert isinstance(iokw["iuncrd"], str)
    assert "mini_box_equil.res" in iokw["iunwri"]
    assert "mini_box_equil.dcd" in iokw["iuncrd"]


def test_resolve_max_grms_before_dyn_scales_large_pbc_cluster():
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        resolve_max_grms_before_dyn,
    )

    args = argparse.Namespace(max_grms_before_dyn=50.0)
    grms = resolve_max_grms_before_dyn(args, 206, 1030, pbc=True)
    assert grms >= 175.0


def test_resolve_max_grms_before_dyn_no_scale_uses_base():
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        resolve_max_grms_before_dyn,
    )

    args = argparse.Namespace(max_grms_before_dyn=80.0, no_scale_max_grms=True)
    grms = resolve_max_grms_before_dyn(args, 20, 100, pbc=True)
    assert grms == 80.0


def test_heat_multiseg_memory_handoff_reset_then_seed(tmp_path):
    """Multi-segment HEAT resets scratch before seeding fly-off checkpoint."""
    from unittest.mock import patch

    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import CharmmTrajectoryFiles
    from mmml.interfaces.pycharmmInterface.mlpot.staged_workflow import (
        _reset_stage_restart,
        _seed_restart_for_memory_handoff,
    )

    res = tmp_path / "heat_dcm_206.0.res"
    io = CharmmTrajectoryFiles(restart_write=res)
    kw: dict = {}
    res.write_text("stale\n", encoding="ascii")
    _reset_stage_restart(res, restart_read=None)
    assert not res.is_file()
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.staged_workflow.rewrite_dynamics_restart_from_current_state"
    ):
        _seed_restart_for_memory_handoff(io, kw, stage="heat")
    res.write_text("seeded restart\n", encoding="ascii")
    assert res.is_file()


def test_configure_equi_dynamics_start_from_heat_restart_file(tmp_path):
    """EQUI CPT from heat restart: coords only, then dyna start (no READYN)."""
    heat = tmp_path / "heat_dcm_50.res"
    _write_restartable_res(heat, jhstrt=500)
    equi = tmp_path / "equi_dcm_50.res"
    io = CharmmTrajectoryFiles(restart_read=heat, restart_write=equi)
    kw = {
        "cpt": True,
        "hoover reft": 30.0,
        "pmass": 84,
        "tmass": 840,
        "restart": True,
        "start": False,
        "iasvel": 0,
    }

    with (
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.staged_workflow.ensure_charmm_crystal_for_cpt"
        ) as crystal,
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation.read_restart_coordinates",
            return_value=__import__("numpy").zeros((10, 3)),
        ),
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.setup.sync_charmm_positions",
        ),
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.clear_comparison_coordinates",
        ),
    ):
        from mmml.interfaces.pycharmmInterface.mlpot.staged_workflow import (
            _configure_equi_dynamics_start,
        )

        _configure_equi_dynamics_start(
            kw,
            io,
            restart_from_file=True,
            coords_in_memory=False,
            use_pbc=True,
            quiet=True,
            temp=30.0,
            box_side=40.0,
        )

    crystal.assert_called_once()
    assert kw["restart"] is False
    assert kw["start"] is True
    assert kw["iasvel"] == 1
    assert kw["firstt"] == 30.0
    assert io.restart_read is None


def test_configure_equi_dynamics_start_survives_overlap_chunk_prep(tmp_path):
    """Overlap chunk 0 must not clear restart+start after EQUI CPT barostat config."""
    heat = tmp_path / "heat_dcm_50.res"
    _write_restartable_res(heat, jhstrt=144)
    equi = tmp_path / "equi_dcm_50.res"
    io = CharmmTrajectoryFiles(restart_read=heat, restart_write=equi)
    kw = {
        "cpt": True,
        "hoover reft": 20.0,
        "pmass": 93,
        "tmass": 930,
        "restart": True,
        "start": False,
        "iasvel": 0,
    }
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        _apply_overlap_chunk_dynamics_kw,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.staged_workflow import (
        _configure_equi_dynamics_start,
    )

    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.staged_workflow.ensure_charmm_crystal_for_cpt"
    ):
        _configure_equi_dynamics_start(
            kw,
            io,
            restart_from_file=True,
            coords_in_memory=True,
            use_pbc=True,
            quiet=True,
            temp=20.0,
            box_side=40.0,
        )
    _apply_overlap_chunk_dynamics_kw(kw, chunk_index=0, has_restart_read=False)
    assert kw["restart"] is False
    assert kw["start"] is True
    assert kw["iasvel"] == 1
    assert kw["firstt"] == 20.0
    assert io.restart_read is None


def test_configure_equi_dynamics_start_skips_in_place_resume(tmp_path):
    res = tmp_path / "equi_dcm_50.res"
    _write_restartable_res(res, jhstrt=200)
    io = CharmmTrajectoryFiles(restart_read=res, restart_write=res)
    kw = {
        "cpt": True,
        "hoover reft": 300.0,
        "pmass": 84,
        "restart": True,
        "start": False,
    }

    from mmml.interfaces.pycharmmInterface.mlpot.staged_workflow import (
        _configure_equi_dynamics_start,
    )

    _configure_equi_dynamics_start(
        kw,
        io,
        restart_from_file=True,
        use_pbc=True,
        quiet=True,
        temp=300.0,
    )

    assert kw["start"] is False


def test_configure_equi_dynamics_start_skips_when_memory_handoff_already_configured():
    io = CharmmTrajectoryFiles()
    kw = {
        "cpt": True,
        "pmass": 84,
        "restart": False,
        "start": True,
        "iasvel": 1,
    }

    from mmml.interfaces.pycharmmInterface.mlpot.staged_workflow import (
        _configure_equi_dynamics_start,
    )

    _configure_equi_dynamics_start(
        kw,
        io,
        restart_from_file=False,
        use_pbc=True,
        quiet=True,
        temp=300.0,
    )

    assert kw["restart"] is False
    assert kw["start"] is True


def test_configure_npt_dynamics_start_memory_handoff_no_readyn():
    """EQUI/PROD after bonded-MM: single CPT dyna start (no nstep=0 assign, no READYN)."""
    from unittest.mock import patch

    io = CharmmTrajectoryFiles(restart_write=Path("/tmp/equi.res"))
    kw = {
        "restart": True,
        "start": True,
        "firstt": 280.0,
        "cpt": True,
        "hoover reft": 280.0,
        "tmass": 500,
        "iunrea": 3,
    }

    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics.assign_velocities_at_temperature"
    ) as assign, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.staged_workflow.ensure_charmm_crystal_for_cpt"
    ) as crystal:
        from mmml.interfaces.pycharmmInterface.mlpot.staged_workflow import (
            _configure_npt_dynamics_start,
        )

        _configure_npt_dynamics_start(
            kw,
            io,
            coords_in_memory=True,
            restart_from_file=False,
            timestep_ps=0.0002,
            use_pbc=True,
            quiet=True,
            temp=280.0,
            box_side=32.0,
        )

    assign.assert_not_called()
    crystal.assert_called_once()
    assert kw["restart"] is False
    assert kw["start"] is True
    assert kw["iasvel"] == 1
    assert kw["iunrea"] == -1
    assert io.restart_read is None


def test_equi_after_heat_overlap_memory_handoff_configures_npt_start(
    tmp_path, monkeypatch
):
    """CPT EQUI after HEAT must re-init barostat when overlap chunk 0 skips READYN."""
    from unittest.mock import MagicMock, patch

    from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import (
        DynamicsOverlapConfig,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.staged_workflow import (
        _maybe_configure_cpt_in_memory_overlap_start,
    )

    heat = tmp_path / "heat.res"
    heat.write_text("REST 1 4000\n", encoding="ascii")
    equi = tmp_path / "equi.res"
    io = CharmmTrajectoryFiles(restart_read=heat, restart_write=equi)
    kw = {
        "cpt": True,
        "hoover reft": 280.0,
        "restart": True,
        "start": False,
        "nsavc": 50,
        "iunrea": 3,
    }
    cfg = DynamicsOverlapConfig(
        action="error",
        min_distance_A=0.5,
        check_interval=500,
        n_monomers=2,
        use_pbc=True,
    )
    args = argparse.Namespace(dynamics_overlap_memory_handoff=False, quiet=True)
    monkeypatch.delenv("MMML_NO_OVERLAP_MEMORY_HANDOFF", raising=False)

    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics.assign_velocities_at_temperature"
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.staged_workflow.ensure_charmm_crystal_for_cpt"
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.staged_workflow.rewrite_dynamics_restart_from_current_state"
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.charmm_lib_links_mpi",
        return_value=True,
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._under_mpirun",
        return_value=True,
    ):
        seed = _maybe_configure_cpt_in_memory_overlap_start(
            stage="equi",
            kw=kw,
            io=io,
            use_memory=False,
            prev_restart_is_current_state=True,
            stage_overlap=cfg,
            mlpot_ctx=MagicMock(),
            nstep=4000,
            args=args,
            timestep_ps=0.00025,
            use_pbc=True,
            temp=280.0,
            box_side=32.0,
        )

    assert seed == equi
    assert kw["restart"] is False
    assert kw["iunrea"] == -1
    assert io.restart_read is None


def test_maybe_configure_cpt_skips_single_chunk_overlap(tmp_path, monkeypatch):
    from unittest.mock import MagicMock, patch

    from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import (
        DynamicsOverlapConfig,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.staged_workflow import (
        _maybe_configure_cpt_in_memory_overlap_start,
    )

    heat = tmp_path / "heat.res"
    heat.write_text("REST 1 4000\n", encoding="ascii")
    io = CharmmTrajectoryFiles(restart_read=heat, restart_write=tmp_path / "equi.res")
    kw = {"cpt": True, "hoover reft": 280.0, "restart": True, "nsavc": 50}
    cfg = DynamicsOverlapConfig(
        action="error",
        min_distance_A=0.5,
        check_interval=500,
        n_monomers=2,
        use_pbc=True,
    )
    args = argparse.Namespace(dynamics_overlap_memory_handoff=True, quiet=True)

    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.staged_workflow._configure_npt_dynamics_start"
    ) as configure_npt:
        result = _maybe_configure_cpt_in_memory_overlap_start(
            stage="equi",
            kw=kw,
            io=io,
            use_memory=False,
            prev_restart_is_current_state=True,
            stage_overlap=cfg,
            mlpot_ctx=MagicMock(),
            nstep=500,
            args=args,
            timestep_ps=0.00025,
            use_pbc=True,
            temp=280.0,
            box_side=32.0,
        )

    configure_npt.assert_not_called()
    assert result is None


def test_mlpot_profile_propagation(monkeypatch):
    from mmml.cli.run.md_system import parse_md_system_args, build_pycharmm_command
    from mmml.interfaces.pycharmmInterface.mlpot.staged_workflow import run_staged_workflow
    import os

    # 1. Test parser registers the flag
    args = parse_md_system_args(["--mlpot-profile", "--setup", "free_nve"])
    assert args.mlpot_profile is True

    # 2. Test that build_pycharmm_command forwards it
    cmd = build_pycharmm_command(args)
    assert "--mlpot-profile" in cmd

    # 3. Test that run_staged_workflow sets the env vars
    monkeypatch.delenv("MMML_MLPOT_PROFILE", raising=False)
    monkeypatch.delenv("MMML_JAX_COMPILE_TIMERS", raising=False)

    with patch("mmml.interfaces.pycharmmInterface.mlpot.staged_workflow._load_or_build_cluster", side_effect=ValueError("abort")):
        try:
            run_staged_workflow(args)
        except ValueError:
            pass

    assert os.environ.get("MMML_MLPOT_PROFILE") == "1"
    assert os.environ.get("MMML_JAX_COMPILE_TIMERS") == "1"

