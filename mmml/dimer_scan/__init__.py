"""Supported public API for reproducible one-dimensional dimer scans."""

from __future__ import annotations

from mmml.analysis.dimer_molecules import make_oriented_scan_geometries

from .calculators import calculator_factory
from .config import (
    ORIENTATION_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    DimerScanConfig,
)
from .evaluate import evaluate_geometries
from .result import Provenance, ScanRecord, ScanResult


def run_dimer_scan(config: DimerScanConfig) -> ScanResult:
    """Run the configured scan through the canonical geometry/calculator path."""

    if config.orientation != "campaign-default":
        raise ValueError("only orientation='campaign-default' is currently supported")
    geometries = make_oriented_scan_geometries(
        config.residues[0],
        config.residues[1],
        config.distances_angstrom,
        offsets_angstrom=[0.0],
    )
    return evaluate_geometries(config, geometries, calculator_factory(config))


__all__ = [
    "DimerScanConfig",
    "ORIENTATION_SCHEMA_VERSION",
    "Provenance",
    "RESULT_SCHEMA_VERSION",
    "ScanRecord",
    "ScanResult",
    "run_dimer_scan",
]
