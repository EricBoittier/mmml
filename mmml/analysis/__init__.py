"""Analysis helpers for comparing reference and model NPZ trajectories."""

from mmml.analysis.npz_comparison import (
    align_npz_arrays,
    compare_npz_arrays,
    compute_element_force_metrics,
    compute_force_metrics,
    compute_per_atom_force_metrics,
    compute_scalar_metrics,
    write_comparison_report,
)
from mmml.analysis.dimer_scans import (
    DimerGeometry,
    assign_mol_id,
    build_rigid_dimer,
    centered_atoms,
    distance_scan_geometries,
    evaluate_scan,
    fragment_index_arrays,
    geometric_centroid,
    make_xtb_calculator,
    molecule_pair_labels,
    normalized_vector,
)

__all__ = [
    "DimerGeometry",
    "align_npz_arrays",
    "assign_mol_id",
    "build_rigid_dimer",
    "centered_atoms",
    "compare_npz_arrays",
    "compute_element_force_metrics",
    "compute_force_metrics",
    "compute_per_atom_force_metrics",
    "compute_scalar_metrics",
    "distance_scan_geometries",
    "evaluate_scan",
    "fragment_index_arrays",
    "geometric_centroid",
    "make_xtb_calculator",
    "molecule_pair_labels",
    "normalized_vector",
    "write_comparison_report",
]
