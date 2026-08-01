"""Unit tests for neat-liquid MD analysis helpers."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from mmml.analysis.liquid_md import (
    analyze_campaign_dir,
    analyze_h5,
    density_report,
    element_pair_rdfs_from_arrays,
    infer_solvent_from_composition,
    monomer_com_msd,
    packing_density_g_cm3,
    write_analysis_outputs,
)


def test_packing_density_dcm_bulk_box():
    # N at 1.0× bulk for L=28 Å is 206 (see bulk_density helpers).
    rho = packing_density_g_cm3(n_molecules=206, mw_g_mol=84.93, box_side_A=28.0)
    assert rho == pytest.approx(1.326, rel=0.02)


def test_density_report_relative_error():
    report = density_report(n_molecules=120, box_side_A=30.0, solvent="DCM")
    assert report.solvent == "DCM"
    assert report.reference_g_cm3 == pytest.approx(1.326)
    assert report.relative_error is not None
    # 120 @ 30 Å is under-dense vs bulk (~1.326).
    assert report.density_g_cm3 < 1.326


def test_infer_solvent_from_composition():
    assert infer_solvent_from_composition("DCM:120") == "DCM"
    assert infer_solvent_from_composition({"ACO": 64}) == "ACO"
    assert infer_solvent_from_composition("DCM:10,ACO:10") is None


def test_element_pair_rdfs_ideal_gas_like():
    rng = np.random.default_rng(0)
    n_frames, n_atoms, box = 8, 40, 20.0
    positions = rng.uniform(0.0, box, size=(n_frames, n_atoms, 3))
    z = np.array([6, 1, 1, 17] * 10, dtype=int)
    rdf = element_pair_rdfs_from_arrays(
        positions, z, box_side_A=box, r_max=8.0, n_bins=40
    )
    assert rdf["n_frames"] == n_frames
    assert "Cl-Cl" in rdf["pairs"] or "C-Cl" in rdf["pairs"]
    for rec in rdf["pairs"].values():
        assert rec["peak_r_A"] is not None
        assert np.isfinite(rec["peak_g"])


def test_element_pair_rdfs_excludes_intramolecular_bonds():
    # Two DCM-like monomers with a short intramolecular C–H and a longer
    # intermolecular C–Cl contact; intermolecular mode should not peak at ~1 Å.
    pos0 = np.array(
        [
            [0.0, 0.0, 0.0],  # C
            [1.09, 0.0, 0.0],  # H
            [-0.5, 0.9, 0.0],  # H
            [0.0, 0.0, 1.77],  # Cl
            [0.0, 0.0, -1.77],  # Cl
            [5.0, 0.0, 0.0],  # C'
            [6.09, 0.0, 0.0],
            [4.5, 0.9, 0.0],
            [5.0, 0.0, 1.77],
            [5.0, 0.0, -1.77],
        ],
        dtype=np.float64,
    )
    positions = np.stack([pos0, pos0 + 0.01], axis=0)
    z = np.array([6, 1, 1, 17, 17, 6, 1, 1, 17, 17], dtype=int)
    rdf = element_pair_rdfs_from_arrays(
        positions,
        z,
        box_side_A=20.0,
        r_max=10.0,
        n_bins=50,
        atoms_per_monomer=5,
        exclude_intramolecular=True,
    )
    assert rdf["exclude_intramolecular"] is True
    ch = rdf["pairs"]["C-H"]
    assert ch["peak_r_A"] is not None
    assert ch["peak_r_A"] > 2.0


def test_monomer_com_msd_grows_with_diffusion():
    rng = np.random.default_rng(1)
    n_frames, n_mol, apm, box = 40, 12, 5, 25.0
    # Independent Brownian monomer COMs + fixed intramolecular offsets.
    com = np.cumsum(rng.normal(0.0, 0.05, size=(n_frames, n_mol, 3)), axis=0)
    offsets = rng.normal(0.0, 0.2, size=(n_mol, apm, 3))
    pos = com[:, :, None, :] + offsets[None, :, :, :]
    pos = pos.reshape(n_frames, n_mol * apm, 3)
    msd = monomer_com_msd(
        pos,
        atoms_per_monomer=apm,
        box_side_A=box,
        timestep_ps=0.05,
        fit_start_fraction=0.25,
    )
    assert msd.n_monomers == n_mol
    assert msd.msd_A2[-1] > msd.msd_A2[5]
    assert msd.diffusion_A2_per_ps >= 0.0


def _write_synthetic_h5(path: Path, *, box: float = 20.0, n_mol: int = 8) -> None:
    apm = 5
    n_atoms = n_mol * apm
    n_frames = 12
    rng = np.random.default_rng(2)
    # Mildly structured liquid-like coordinates.
    pos = rng.uniform(0.0, box, size=(n_frames, n_atoms, 3)).astype(np.float32)
    z = np.tile(np.array([6, 1, 1, 17, 17], dtype=np.int32), n_mol)
    t = np.arange(n_frames, dtype=np.float64) * 0.05
    with h5py.File(path, "w") as handle:
        handle.create_dataset("positions", data=pos)
        handle.create_dataset("time_ps", data=t)
        handle.create_dataset("temperature", data=np.full(n_frames, 300.0))
        handle.create_dataset("potential_energy", data=-100.0 + 0.01 * np.arange(n_frames))
        handle.create_dataset("total_energy", data=-90.0 + 0.001 * np.arange(n_frames))
        handle.create_dataset("kinetic_energy", data=np.full(n_frames, 10.0))
        handle.attrs["atomic_numbers"] = z
        handle.attrs["n_atoms"] = n_atoms
        handle.attrs["dt_ps"] = 0.0005
        handle.attrs["steps_per_recording"] = 100
        handle.attrs["ensemble"] = "nvt"


def test_analyze_h5_and_write_outputs(tmp_path: Path):
    h5 = tmp_path / "pbc_nvt_jaxmd_nvt.h5"
    _write_synthetic_h5(h5)
    (tmp_path / "suite_summary_jaxmd.json").write_text(
        json.dumps({"box_A": 20.0, "composition": {"DCM": 8}}),
        encoding="utf-8",
    )
    report = analyze_h5(h5, solvent="DCM", max_frames=20, r_max=8.0)
    assert report["density"]["n_molecules"] == 8
    assert report["rdf"]["top_peaks"]
    assert report["timeseries"]["temperature_mean_K"] == pytest.approx(300.0)
    out = tmp_path / "analysis"
    artifacts = write_analysis_outputs(report, out)
    assert Path(artifacts["metrics"]).is_file()
    metrics = json.loads(Path(artifacts["metrics"]).read_text(encoding="utf-8"))
    assert "peak_r_A" in next(iter(metrics["rdf"]["pairs"].values()))
    # PNGs require matplotlib; skip soft-fail if Agg backend missing in CI.
    assert "rdf_full" in artifacts


def test_analyze_campaign_dir_prefers_jaxmd_nvt(tmp_path: Path):
    campaign = tmp_path / "liquid_dcm"
    settle = campaign / "jaxmd_settle"
    prod = campaign / "jaxmd_nvt"
    settle.mkdir(parents=True)
    prod.mkdir(parents=True)
    _write_synthetic_h5(settle / "pbc_nvt_jaxmd_nvt.h5")
    _write_synthetic_h5(prod / "pbc_nvt_jaxmd_nvt.h5")
    (prod / "suite_summary_jaxmd.json").write_text(
        json.dumps({"box_A": 20.0}), encoding="utf-8"
    )
    (campaign / "campaign_plan.json").write_text(
        json.dumps({"defaults": {"composition": "DCM:8", "box_size": 20.0}}),
        encoding="utf-8",
    )
    report = analyze_campaign_dir(campaign, solvent="DCM", max_frames=10)
    assert "error" not in report
    assert "jaxmd_nvt" in report["h5"]
