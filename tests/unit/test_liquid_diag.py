"""Unit tests for liquid-density diagnostic helpers."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / "workflows" / "pbc_liquid_density_dyn"
sys.path.insert(0, str(WORKFLOW / "scripts"))

import collect_diagnostics as cd  # noqa: E402
import monitor_lib as ml  # noqa: E402
import trajectory_diag as td  # noqa: E402
from mmml.utils.psf_reader import read_psf_atom_types  # noqa: E402


def _load_dcd_writer():
    path = REPO / "mmml" / "utils" / "dcd_writer.py"
    spec = importlib.util.spec_from_file_location("dcd_writer", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class _NatomsStub:
    def __init__(self, n: int) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n


def _write_dcd(path: Path, coords: np.ndarray, *, box: float = 20.0) -> None:
    """coords: (n_frames, n_atoms, 3). Pure numpy DCD writer (no ASE)."""
    dcd_writer = _load_dcd_writer()
    n_atoms = coords.shape[1]
    box_vec = np.array([box, box, box])
    boxes = [box_vec] * coords.shape[0]
    dcd_writer.save_trajectory_dcd(
        path,
        coords,
        _NatomsStub(n_atoms),
        boxes=boxes,
        dt_ps=0.001,
    )


@pytest.fixture
def smoke_psf() -> Path:
    psf = REPO / "artifacts/pbc_liquid_density_dyn_smoke/dcm_32/pycharmm_init/model.psf"
    if not psf.is_file():
        pytest.skip("smoke PSF not available")
    return psf


def test_read_psf_atom_types_smoke(smoke_psf: Path):
    types = read_psf_atom_types(smoke_psf)
    assert types.shape == (160,)
    assert set(types.tolist()) == {"CG321", "HGA2", "CLGA1"}


def test_parse_dyna_lines():
    text = """
DYNA>      100      0.0250   -1234.5    300.0    -1534.5    298.2
DYNA>      200      0.0500   -1230.1    305.0    -1535.1    299.0
"""
    rows = ml.parse_dyna_lines(text)
    assert len(rows) == 2
    assert rows[0]["step"] == 100.0
    assert rows[1]["temperature_K"] == pytest.approx(299.0)


def test_summarize_dyna_drift():
    rows = [
        {
            "step": 1,
            "time_ps": 0.0,
            "total_energy_kcal": 0.0,
            "kinetic_energy_kcal": 0.0,
            "potential_energy_kcal": -10.0,
            "temperature_K": 300.0,
        },
        {
            "step": 2,
            "time_ps": 0.1,
            "total_energy_kcal": 1.5,
            "kinetic_energy_kcal": 0.0,
            "potential_energy_kcal": -9.0,
            "temperature_K": 301.0,
        },
    ]
    summary = ml.summarize_dyna(rows)
    assert summary["n_frames"] == 2
    assert summary["total_energy_drift_kcal"] == pytest.approx(1.5)


def test_pair_label_sorted():
    assert td.pair_label("CLGA1", "CG321") == "CG321__CLGA1"


def test_compute_pair_rdf_two_types():
    types = np.array(["A", "A", "B", "B"])
    # frame 0: A-A at 3 Å, B-B at 5 Å (within 20 Å box)
    pos = np.array(
        [
            [
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [15.0, 0.0, 0.0],
            ]
        ],
        dtype=float,
    )
    box = np.array([20.0, 20.0, 20.0])
    out = td.compute_pair_rdf(pos, types, box_lengths=box, r_max=8.0, n_bins=40)
    assert out["n_frames"] == 1
    assert "A__A" in out["pairs"]
    assert "A__B" in out["pairs"]
    assert "B__B" in out["pairs"]
    assert out["pairs"]["A__A"]["peak_r_A"] is not None


def test_load_dcd_and_pair_rdf(tmp_path: Path, smoke_psf: Path):
    types = read_psf_atom_types(smoke_psf)
    n_atoms = len(types)
    # Two identical frames from origin grid (minimal synthetic coords)
    rng = np.random.default_rng(0)
    frame = rng.normal(size=(n_atoms, 3)) * 2.0 + 10.0
    coords = np.stack([frame, frame + 0.01], axis=0)
    dcd = tmp_path / "traj.dcd"
    _write_dcd(dcd, coords)

    pos, meta = td.load_dcd_trajectory(dcd, stride=1, max_frames=2)
    assert meta["n_atoms"] == n_atoms
    assert pos.shape[0] == 2

    out = td.compute_pair_rdf(
        pos,
        types,
        box_lengths=np.array([30.0, 30.0, 30.0]),
        r_max=10.0,
        n_bins=50,
    )
    assert out["n_type_pairs"] == 6  # 3 MM types -> 6 unordered pairs
    assert all(k in out["pairs"] for k in ("CG321__CG321", "CG321__CLGA1", "HGA2__HGA2"))


def test_analyze_cell_trajectories(tmp_path: Path, smoke_psf: Path):
    cell = tmp_path / "dcm_test"
    leg = cell / "pycharmm_init"
    leg.mkdir(parents=True)
    (leg / "model.psf").write_text(smoke_psf.read_text(encoding="utf-8"), encoding="utf-8")

    types = read_psf_atom_types(smoke_psf)
    coords = np.stack(
        [
            np.random.default_rng(1).normal(size=(len(types), 3)) + 10.0,
            np.random.default_rng(2).normal(size=(len(types), 3)) + 10.0,
        ]
    )
    _write_dcd(leg / "prod.dcd", coords)

    payload = td.analyze_cell_trajectories(cell, stride=1, max_frames=2)
    pair_rdf = payload["trajectory"]["pair_rdf"]
    assert pair_rdf["n_frames"] == 2
    assert pair_rdf["n_type_pairs"] >= 3
    assert payload["psf_files"] == ["pycharmm_init/model.psf"]


def test_inspect_run_from_stdout(tmp_path: Path):
    cfg_path = WORKFLOW / "config.smoke.gpu.yaml"
    if not cfg_path.is_file():
        pytest.skip("smoke gpu config missing")
    from campaign_lib import cell_from_tag, load_config, paths_for_run  # noqa: E402

    cfg = load_config(cfg_path)
    cell = cell_from_tag(cfg, "dcm_32")
    paths = paths_for_run(cfg, cell)
    out = paths["out_dir"]
    if not out.is_dir():
        out = tmp_path / "dcm_32"
        out.mkdir(parents=True)
    log = out / "stdout.log"
    log.write_text(
        "warmup-mlpot-jax\n"
        "DYNA>      100      0.0250   -1234.5    300.0    -1534.5    298.2\n"
        "DYNA>      200      0.0500   -1230.1    305.0    -1535.1    299.0\n",
        encoding="utf-8",
    )
    # Point config output to tmp if using synthetic dir
    if str(out).startswith(str(tmp_path)):
        cfg = dict(cfg)
        try:
            cfg["output_root"] = str(tmp_path.relative_to(REPO))
        except ValueError:
            cfg["output_root"] = str(tmp_path)
        import yaml

        alt = tmp_path / "cfg.yaml"
        alt.write_text(yaml.dump(cfg), encoding="utf-8")
        cfg = load_config(alt)
        cell = cell_from_tag(cfg, "dcm_32")

    monitor = ml.inspect_run(cfg, cell)
    assert monitor.dyna.get("n_frames") == 2
    assert monitor.health in {"OK", "WARN", "BAD", "—"}


def test_collect_manifest(tmp_path: Path):
    cfg_path = WORKFLOW / "config.smoke.gpu.yaml"
    if not cfg_path.is_file():
        pytest.skip("smoke gpu config missing")

    test_root = REPO / "artifacts" / "_pytest_liquid_diag_collect"
    out_art = test_root / "dcm_32"
    out_art.mkdir(parents=True, exist_ok=True)
    (out_art / "stdout.log").write_text(
        "DYNA>      100      0.0250   -100.0    50.0    -150.0    300.0\n",
        encoding="utf-8",
    )

    import yaml
    from campaign_lib import load_config  # noqa: E402

    cfg = load_config(cfg_path)
    cfg["output_root"] = str(test_root.relative_to(REPO))
    alt_cfg = tmp_path / "cfg.yaml"
    alt_cfg.write_text(yaml.dump(cfg), encoding="utf-8")

    args = cd.build_parser().parse_args(
        ["--config", str(alt_cfg), "collect", "--output-dir", str(tmp_path / "diag")]
    )
    assert cd.cmd_collect(args) == 0
    manifest = json.loads((tmp_path / "diag" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["cells_collected"] == 1
    assert manifest["cells"][0]["run_tag"] == "dcm_32"
    assert manifest["dyna_plots"] == 0  # only 1 DYNA line


def test_type_pair_indices_all_unordered_pairs():
    types = np.array(["CG321", "CG321", "HGA2", "CLGA1"])
    pairs = td._type_pair_indices(types)
    labels = {td.pair_label(a, b) for (a, b) in pairs}
    assert labels == {
        "CG321__CG321",
        "CG321__HGA2",
        "CG321__CLGA1",
        "HGA2__HGA2",
        "CLGA1__HGA2",
        "CLGA1__CLGA1",
    }


def test_mic_distances_wraps_across_box():
    box = np.array([10.0, 10.0, 10.0])
    pos_a = np.array([[0.0, 0.0, 0.0]])
    pos_b = np.array([[9.5, 0.0, 0.0]])
    d = td._mic_distances(pos_a, pos_b, box)
    assert d[0, 0] == pytest.approx(0.5)


def test_dcd_reader_frame_stride(tmp_path: Path):
    from mmml.utils.dcd_reader import read_dcd_trajectory, scan_dcd_frame_count

    coords = np.zeros((4, 5, 3), dtype=np.float32)
    for i in range(4):
        coords[i, :, 0] = float(i)
    dcd = tmp_path / "stride.dcd"
    _write_dcd(dcd, coords)

    readable, header, truncated = scan_dcd_frame_count(dcd)
    assert readable == 4
    assert header == 4
    assert truncated is False

    pos, meta = read_dcd_trajectory(dcd, max_frames=10, frame_stride=2, require_complete=False)
    assert pos.shape == (2, 5, 3)
    assert meta["frame_stride"] == 2
    assert pos[0, 0, 0] == pytest.approx(0.0)
    assert pos[1, 0, 0] == pytest.approx(2.0)


def test_box_lengths_from_handoff(tmp_path: Path):
    handoff = tmp_path / "handoff"
    handoff.mkdir()
    cell = np.diag([28.0, 28.0, 32.0])
    np.savez(handoff / "state.npz", cell=cell, positions=np.zeros((10, 3)))

    box = td.box_lengths_from_handoff(tmp_path)
    assert box is not None
    assert box.tolist() == pytest.approx([28.0, 28.0, 32.0])


def test_plot_pair_rdf_png(tmp_path: Path):
    pair_rdf = {
        "pairs": {
            "A__A": {
                "bins_A": [0.5, 1.5, 2.5],
                "g_r": [0.0, 2.0, 0.5],
                "peak_g": 2.0,
            },
            "A__B": {
                "bins_A": [0.5, 1.5, 2.5],
                "g_r": [0.1, 1.0, 0.2],
                "peak_g": 1.0,
            },
        }
    }
    png = tmp_path / "pair_rdf.png"
    assert td.plot_pair_rdf_png(pair_rdf, png, title="test") is True
    assert png.stat().st_size > 100


def test_compute_pair_rdf_atom_count_mismatch():
    pos = np.zeros((1, 3, 3))
    types = np.array(["A", "B"])
    with pytest.raises(ValueError, match="atom count"):
        td.compute_pair_rdf(pos, types, box_lengths=np.array([10.0, 10.0, 10.0]))


def test_smoke_psf_type_pair_counts(smoke_psf: Path):
    types = read_psf_atom_types(smoke_psf)
    pairs = td._type_pair_indices(types)
    assert len(types) == 160
    assert len(pairs) == 6
    assert set(types.tolist()) == {"CG321", "HGA2", "CLGA1"}
