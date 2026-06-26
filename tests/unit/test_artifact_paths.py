"""Short artifact filename conventions."""

from pathlib import Path

from mmml.interfaces.pycharmmInterface.mlpot.artifact_paths import (
    alternate_overlap_scratch,
    is_overlap_scratch_restart_path,
    overlap_chunk_dcd_paths,
    overlap_chunk_trajectory_path,
    overlap_restart_slot_paths,
    staged_artifact_paths,
)
from mmml.interfaces.pycharmmInterface.mlpot.minimize_artifacts import (
    MLPOT_MMML,
    rescue_snapshot_spec,
    snapshot_file_paths,
)


def test_staged_paths_use_short_names(tmp_path: Path) -> None:
    paths = staged_artifact_paths(tmp_path, "dcm_8")
    assert paths["vmd_psf"].name == "model.psf"
    assert paths["heat_res"].name == "heat.res"
    assert paths["heat_dcd"].name == "heat.dcd"
    assert paths["mini_crd"].name == "mini.crd"
    assert paths["geometry_baseline_res"].name == "baseline.res"


def test_snapshot_stems_omit_tag(tmp_path: Path) -> None:
    assert MLPOT_MMML.stem("dcm_8") == "02_mini"
    paths = snapshot_file_paths(tmp_path, MLPOT_MMML, "dcm_8")
    assert paths["crd"].name == "02_mini.crd"


def test_rescue_snapshot_short_slug() -> None:
    spec = rescue_snapshot_spec("heat after early abort recovery step 640", seq=10)
    assert spec.stem("dcm_8") == "10_rescue_10"


def test_overlap_scratch_slots() -> None:
    final = Path("/tmp/heat.res")
    a, b = overlap_restart_slot_paths(final)
    assert a.name == "heat.a.res"
    assert b.name == "heat.b.res"
    assert is_overlap_scratch_restart_path(a)
    assert alternate_overlap_scratch(a) == b


def test_overlap_chunk_dcd_naming(tmp_path: Path) -> None:
    dcd = tmp_path / "heat.dcd"
    dcd.write_bytes(b"")
    chunk0 = overlap_chunk_trajectory_path(dcd, 0)
    chunk0.write_bytes(b"")
    chunk1 = overlap_chunk_trajectory_path(dcd, 1)
    chunk1.write_bytes(b"")
    assert chunk0.name == "heat.0000.dcd"
    assert overlap_chunk_dcd_paths(dcd) == [chunk0, chunk1]


def test_vmd_script_uses_basename_paths(tmp_path: Path) -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.artifact_paths import VMD_TCL
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import write_vmd_load_script

    topo = tmp_path / "model.psf"
    traj = tmp_path / "heat.0000.dcd"
    topo.write_text("psf", encoding="utf-8")
    traj.write_bytes(b"x")
    tcl = write_vmd_load_script(
        out_dir=tmp_path,
        tag="ignored",
        topology_psf=topo,
        trajectory=traj,
        n_atoms=10,
    )
    assert tcl.name == VMD_TCL
    text = tcl.read_text(encoding="utf-8")
    assert "mol new {model.psf}" in text
    assert "mol addfile {heat.0000.dcd}" in text
    assert str(tmp_path) not in text
