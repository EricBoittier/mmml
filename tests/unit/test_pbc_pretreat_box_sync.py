"""PBC box sync after CHARMM MM pretreat (NPT vs density estimate)."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

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
