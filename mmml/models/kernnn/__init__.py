"""KerNN: kernel-descriptor Softplus MLP (JAX/Flax).

Pairwise ABCC distances → 1D kernels (k33) → standardized features → energy;
forces via autodiff. See ``README.md`` in this package.
"""

from mmml.models.kernnn.calculator import KerNNCalculator
from mmml.models.kernnn.checkpoint import (
    H2CO_CALCULATOR_STATS,
    import_torch_state_dict,
    init_params,
    load_checkpoint,
    load_kernnn_model,
    save_checkpoint,
)
from mmml.models.kernnn.distances import get_bond_length_abcc
from mmml.models.kernnn.kernels import get_1d_kernels_k33
from mmml.models.kernnn.model import (
    FFNet,
    KerNNConfig,
    KerNNStats,
    descriptor_from_positions,
    energy_and_forces,
    energy_from_params,
)

__all__ = [
    "FFNet",
    "H2CO_CALCULATOR_STATS",
    "KerNNCalculator",
    "KerNNConfig",
    "KerNNStats",
    "descriptor_from_positions",
    "energy_and_forces",
    "energy_from_params",
    "get_1d_kernels_k33",
    "get_bond_length_abcc",
    "import_torch_state_dict",
    "init_params",
    "load_checkpoint",
    "load_kernnn_model",
    "save_checkpoint",
]
