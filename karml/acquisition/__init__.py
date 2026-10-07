"""Structure acquisition for expensive reference labels.

This package implements the methods and bookkeeping for comparing how to
select unlabeled atomistic structures before running costly reference
calculations.  The foundation / teacher potential is treated as a cheap
surrogate, never as ground truth: expensive energies and forces stay
unavailable to acquisition until selection manifests are immutable.

Public entry points are the pipeline stages in :mod:`mmml.acquisition.pipeline`
and the CLI ``mmml label-acquire``.
"""

from mmml.acquisition.ids import (
    composition_key,
    geometry_fingerprint,
    structure_id,
)
from mmml.acquisition.pca import PCAFit, fit_pca, transform_pca
from mmml.acquisition.selection import (
    farthest_point_sampling,
    greedy_doptimal,
    largest_norm_indices,
    stratified_random,
)

__all__ = [
    "composition_key",
    "geometry_fingerprint",
    "structure_id",
    "PCAFit",
    "fit_pca",
    "transform_pca",
    "farthest_point_sampling",
    "greedy_doptimal",
    "largest_norm_indices",
    "stratified_random",
]
