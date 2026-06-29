"""Dry-run for spatial MPI CPU profile functionality script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_spatial_mpi_cpu_profile_dry_run() -> None:
    root = Path(__file__).resolve().parents[2]
    script = root / "tests/functionality/mlpot/10_spatial_mpi_cpu_profile.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--dry-run"],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "DCM:100" in proc.stdout
    assert "mpi_spatial_cpu_np_sweep.sh" in proc.stdout


def test_spatial_mpi_cpu_template_exists() -> None:
    root = Path(__file__).resolve().parents[2]
    tpl = (
        root
        / "workflows/pbc_liquid_density_dyn/benchmarks/dcm100_spatial_mpi_cpu.yaml.tpl"
    )
    assert tpl.is_file()
    text = tpl.read_text(encoding="utf-8")
    assert "ml_spatial_mpi: true" in text
    assert "DCM:100" in text
