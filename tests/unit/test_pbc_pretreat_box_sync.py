"""PBC box sync after CHARMM MM pretreat (NPT vs density estimate)."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np

from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
    find_latest_pretreat_mm_restart,
    sync_workflow_pbc_box_side_after_mm_pretreat,
)


def test_sync_workflow_pbc_box_side_after_mm_pretreat_updates_live_box() -> None:
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.resolve_charmm_cubic_box_side_A",
        return_value=(39.99, "pbound"),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.charmm_crystal_is_active",
        return_value=True,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.parse_cubic_box_side_from_charmm_restart",
        return_value=39.99,
    ):
        synced = sync_workflow_pbc_box_side_after_mm_pretreat(
            28.0,
            pretreat_restart=Path("/tmp/charmm_mm_equi_dcm_25.res"),
            quiet=True,
        )
    assert synced == 39.99


def test_sync_workflow_pbc_box_side_after_mm_pretreat_noop_when_unchanged() -> None:
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.resolve_charmm_cubic_box_side_A",
        return_value=(28.0, "fallback"),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.charmm_crystal_is_active",
        return_value=False,
    ):
        synced = sync_workflow_pbc_box_side_after_mm_pretreat(28.0, quiet=True)
    assert synced == 28.0


def test_sync_workflow_pbc_box_side_after_mm_pretreat_none_when_not_pbc() -> None:
    assert sync_workflow_pbc_box_side_after_mm_pretreat(None, quiet=True) is None


def test_find_latest_pretreat_mm_restart_prefers_prod(tmp_path: Path) -> None:
    paths = {
        "charmm_mm_heat_res": tmp_path / "heat.res",
        "charmm_mm_equi_res": tmp_path / "equi.res",
        "charmm_mm_prod_res": tmp_path / "prod.res",
    }
    for path in paths.values():
        path.write_text("stub")
    assert find_latest_pretreat_mm_restart(paths) == paths["charmm_mm_prod_res"]


def test_find_latest_pretreat_mm_restart_equi_when_no_prod(tmp_path: Path) -> None:
    paths = {
        "charmm_mm_heat_res": tmp_path / "heat.res",
        "charmm_mm_equi_res": tmp_path / "equi.res",
        "charmm_mm_prod_res": tmp_path / "prod.res",
    }
    paths["charmm_mm_heat_res"].write_text("stub")
    paths["charmm_mm_equi_res"].write_text("stub")
    assert find_latest_pretreat_mm_restart(paths) == paths["charmm_mm_equi_res"]


def test_probe_charmm_cubic_box_side_A_returns_none_when_unavailable() -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import probe_charmm_cubic_box_side_A

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env._read_charmm_box_sides_A",
        return_value=(0.0, 0.0, 0.0),
    ):
        side, source = probe_charmm_cubic_box_side_A()
    assert side is None
    assert source is None


def test_resolve_mlpot_mic_box_side_A_skips_restart_when_crystal_active(
    tmp_path: Path,
) -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import resolve_mlpot_mic_box_side_A

    restart = tmp_path / "prod.res"
    restart.write_text("stub")
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.charmm_crystal_is_active",
        return_value=True,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.resolve_charmm_cubic_box_side_A",
        return_value=(28.0, "pbound"),
    ) as mock_resolve:
        side, source = resolve_mlpot_mic_box_side_A(
            fallback_side_A=28.0,
            restart_path=restart,
        )
    assert side == 28.0
    assert source == "pbound"
    mock_resolve.assert_called_once_with(
        fallback_side_A=28.0,
        restart_path=None,
        rel_tol=1e-3,
    )


def test_pretreat_handoff_panel_tolerates_inactive_pbound(tmp_path: Path) -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.run_workflow import (
        build_charmm_mm_pretreat_handoff_sections,
    )

    restart = tmp_path / "prod.res"
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

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.run_workflow.charmm_grms",
        return_value=0.5,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env._read_charmm_box_sides_A",
        return_value=(0.0, 0.0, 0.0),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.charmm_crystal_is_active",
        return_value=False,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.probe_charmm_cubic_box_side_A",
        return_value=(None, None),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.resolve_charmm_cubic_box_side_A",
        return_value=(28.0, "restart"),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.resolve_mlpot_mic_box_side_A",
        return_value=(28.0, "restart"),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        return_value=pos.copy(),
    ):
        sections = build_charmm_mm_pretreat_handoff_sections(
            pos,
            n_monomers=1,
            tag="dcm_1",
            pretreat_restart=restart,
            workflow_box_side_A=28.0,
            use_pbc=True,
        )

    pbc = dict(next(s for title, s in sections if title == "PBC → MLpot handoff"))
    assert pbc["pbound_cubic_L_Å"] == "inactive"
    assert "28.000 (restart)" in pbc["workflow_box_L_Å"]
