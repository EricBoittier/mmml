"""Numbered minimize snapshot naming."""

from pathlib import Path
from unittest import mock

import pytest

from mmml.interfaces.pycharmmInterface.mlpot.minimize_artifacts import (
    CHARMM_MM_PRE,
    MLPOT_MMML,
    MinimizeArtifactRegistry,
    legacy_mlpot_mini_paths,
    rescue_snapshot_spec,
    snapshot_file_paths,
)
from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import (
    DynamicsOverlapConfig,
    save_stabilized_overlap_rescue_snapshot,
)


def test_snapshot_stems() -> None:
    assert CHARMM_MM_PRE.stem("dcm_90") == "01_charmm_mm_dcm_90"
    assert MLPOT_MMML.stem("dcm_90") == "02_mlpot_mmml_dcm_90"


def test_legacy_mlpot_paths_unchanged() -> None:
    legacy = legacy_mlpot_mini_paths(Path("/tmp/out"), "dcm_90")
    assert legacy["mini_crd"].name == "mini_full_mlpot_dcm_90.crd"


def test_registry_manifest(tmp_path: Path) -> None:
    reg = MinimizeArtifactRegistry(tmp_path, "dcm_9")
    paths = snapshot_file_paths(tmp_path, MLPOT_MMML, "dcm_9")
    reg.record(MLPOT_MMML, {"crd": paths["crd"], "pdb": paths["pdb"]})
    data = reg.manifest_path.read_text(encoding="utf-8")
    assert "02_mlpot_mmml_dcm_9" in data
    assert '"kind": "MMML"' in data


def test_registry_tracks_last_rescue_crd(tmp_path: Path) -> None:
    reg = MinimizeArtifactRegistry(tmp_path, "dcm_30")
    assert reg.last_rescue_crd is None
    spec = rescue_snapshot_spec("heat segment 1/4", seq=10)
    crd = tmp_path / f"{spec.stem('dcm_30')}.crd"
    crd.write_text("dummy", encoding="utf-8")
    reg.record(spec, {"crd": crd, "pdb": crd.with_suffix(".pdb")})
    assert reg.last_rescue_crd == crd.resolve()


def test_save_stabilized_overlap_rescue_snapshot_uses_hybrid_grms(tmp_path: Path):
    reg = MinimizeArtifactRegistry(tmp_path, "dcm_20")
    cfg = DynamicsOverlapConfig(
        action="rescue",
        n_monomers=2,
        artifact_registry=reg,
    )
    ctx = mock.Mock()
    written = {
        "crd": tmp_path / "10_rescue_equi_at_step_2000_dcm_20.crd",
        "pdb": tmp_path / "10_rescue_equi_at_step_2000_dcm_20.pdb",
    }
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.resolve_mlpot_grms_kcalmol_A",
        return_value=12.5,
    ) as hybrid_grms, mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.minimize_artifacts.save_snapshot_from_charmm",
        return_value=written,
    ) as save_snap:
        save_stabilized_overlap_rescue_snapshot(
            ctx,
            cfg,
            label="EQUI at step 2000",
        )
    hybrid_grms.assert_called_once()
    save_snap.assert_called_once()
    assert save_snap.call_args.kwargs["grms_kcalmol_A"] == pytest.approx(12.5)
    assert save_snap.call_args.kwargs["title"] == "Geometry rescue: EQUI at step 2000"
