"""Unit tests for CHARMM MM pretreat → MLpot handoff dashboard."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_build_pretreat_handoff_warns_restart_box_source(tmp_path: Path) -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.run_workflow import (
        build_charmm_mm_pretreat_handoff_sections,
    )

    restart = tmp_path / "equi.res"
    restart.write_text(
        "REST     0     1 CUBI\n"
        "       1 !NTITLE\n"
        "*\n"
        "\n"
        " !NATOM\n"
        "         5\n"
        "\n"
        " !CRYSTAL PARAMETERS\n"
        " 0.280000000000000D+02 0.000000000000000D+00 0.000000000000000D+00\n"
        " 0.000000000000000D+00 0.280000000000000D+02 0.000000000000000D+00\n"
        " 0.000000000000000D+00 0.000000000000000D+00 0.280000000000000D+02\n"
        "\n"
        " !X, Y, Z\n"
        " 0.100000000000000D+00 0.200000000000000D+00 0.300000000000000D+00\n"
        " 0.400000000000000D+00 0.500000000000000D+00 0.600000000000000D+00\n"
        " 0.700000000000000D+00 0.800000000000000D+00 0.900000000000000D+00\n"
        " 0.100000000000000D+01 0.110000000000000D+01 0.120000000000000D+01\n"
        " 0.130000000000000D+01 0.140000000000000D+01 0.150000000000000D+01\n"
    )
    pos = np.linspace(0.0, 10.0, 15).reshape(5, 3)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "mmml.interfaces.pycharmmInterface.mlpot.run_workflow.charmm_grms",
            lambda: 0.42,
        )
        mp.setattr(
            "mmml.interfaces.pycharmmInterface.mlpot.pbc_env._read_charmm_box_sides_A",
            lambda: (0.0, 0.0, 0.0),
        )
        mp.setattr(
            "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.charmm_crystal_is_active",
            lambda **_: False,
        )
        mp.setattr(
            "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.probe_charmm_cubic_box_side_A",
            lambda **_: (None, None),
        )
        mp.setattr(
            "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.resolve_mlpot_mic_box_side_A",
            lambda **kw: (28.0, "restart"),
        )
        def _fake_resolve(**kw):
            if kw.get("restart_path") is not None:
                return 28.0, "restart"
            if kw.get("fallback_side_A") is None:
                return 0.0, "fallback"
            return float(kw["fallback_side_A"]), "fallback"

        mp.setattr(
            "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.resolve_charmm_cubic_box_side_A",
            _fake_resolve,
        )
        mp.setattr(
            "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
            lambda: pos.copy(),
        )
        sections = build_charmm_mm_pretreat_handoff_sections(
            pos,
            n_monomers=1,
            tag="dcm_1",
            pretreat_restart=restart,
            workflow_box_side_A=28.0,
            use_pbc=True,
        )

    pbc = dict(next(s for title, s in sections if title == "PBC → MLpot handoff"))
    assert "restart" in pbc["workflow_box_L_Å"]
    warnings = dict(next(s for title, s in sections if title == "Warnings"))
    assert any("restart" in str(v).lower() for v in warnings.values())


def test_print_pretreat_handoff_panel_plain(capsys) -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.run_workflow import (
        print_charmm_mm_pretreat_handoff_panel,
    )

    pos = np.zeros((10, 3), dtype=float)
    pos[0:5] = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.5], [0.5, 0.5, 0.5]]
    pos[5:10] = pos[0:5] + np.array([5.0, 0.0, 0.0])

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "mmml.interfaces.pycharmmInterface.mlpot.run_workflow.charmm_grms",
            lambda: 1.0,
        )
        mp.setattr(
            "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
            lambda: pos.copy(),
        )
        mp.setattr(
            "mmml.utils.rich_report.rich_enabled",
            lambda **_: False,
        )
        print_charmm_mm_pretreat_handoff_panel(
            pos,
            n_monomers=2,
            tag="dcm_2",
            use_pbc=False,
            quiet=False,
        )

    out = capsys.readouterr().out
    assert "CHARMM MM pretreat → MLpot handoff" in out
    assert "Geometry" in out
    assert "dcm_2" in out
