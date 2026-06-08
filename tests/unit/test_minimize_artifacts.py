"""Numbered minimize snapshot naming."""

from pathlib import Path

from mmml.interfaces.pycharmmInterface.mlpot.minimize_artifacts import (
    CHARMM_MM_PRE,
    MLPOT_MMML,
    MinimizeArtifactRegistry,
    legacy_mlpot_mini_paths,
    rescue_snapshot_spec,
    snapshot_file_paths,
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
