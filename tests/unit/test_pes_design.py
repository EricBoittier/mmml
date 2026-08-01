from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mmml.cli.misc.pes_design import main


def test_pes_design_filters_selects_and_plots(tmp_path: Path):
    rng = np.random.default_rng(3)
    n = 90
    R = np.zeros((n, 4, 3))
    Z = np.tile(np.array([6, 1, 8, 1]), (n, 1))
    type_idx = np.tile(np.array([2, 4, 7, 4]), (n, 1))
    distances = np.concatenate([
        rng.normal(2.4, .08, 30), rng.normal(3.5, .12, 30), rng.normal(5.2, .2, 30)
    ])
    for i, d in enumerate(distances):
        R[i, 0] = [0, 0, 0]; R[i, 1] = [0.95, 0, 0]
        R[i, 2] = [d, 0.1 * np.sin(i), 0]; R[i, 3] = [d + 0.96, 0, 0]
    # One impossible collision must be rejected.
    R[0, 2] = [0.2, 0, 0]
    E = ((distances - 3.5) ** 2).reshape(-1, 1)
    inp, out, report_dir = tmp_path / "pool.npz", tmp_path / "selected.npz", tmp_path / "report"
    np.savez(inp, R=R, Z=Z, N=np.full(n, 4), E=E,
             cgenff_type_idx=type_idx, res_name=np.repeat("A,B", n))

    assert main([
        "--input", str(inp), "--output", str(out), "--report-dir", str(report_dir),
        "--n-select", "18", "--rdf-bins", "12", "--type-hash-bins", "8",
        "--pca-components", "6", "--min-distance", "0.5", "--seed", "4",
    ]) == 0
    selected = np.load(out)
    assert len(selected["R"]) == 18
    assert 0 not in set(selected["pes_design_source_index"].tolist())
    report = json.loads((report_dir / "report.json").read_text())
    assert report["n_input"] == 90
    assert report["n_physical"] == 89
    assert np.isfinite(report["d_opt_logdet_selected"])
    for name in ("descriptor_coverage.png", "coverage_cdf.png", "rdf_spectrum.png"):
        assert (report_dir / name).stat().st_size > 1000
