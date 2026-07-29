"""KerNN: kernel-descriptor Softplus MLP (JAX/Flax).

Pairwise ABCC (or ABCC_sym) distances → 1D kernels (k33) → Softplus MLP → energy;
optional DualFFNet dihedral branch. Forces via autodiff.
"""

from mmml.models.kernnn.adapter import KerNNApplyAdapter
from mmml.models.kernnn.batch_apply import build_kernnn_batch_apply, is_kernnn_checkpoint
from mmml.models.kernnn.calculator import KerNNCalculator
from mmml.models.kernnn.checkpoint import (
    H2CO_CALCULATOR_STATS,
    import_torch_state_dict,
    init_params,
    load_checkpoint,
    load_kernnn_model,
    save_checkpoint,
)
from mmml.models.kernnn.distances import (
    get_bond_length_abcc,
    get_bond_length_abcc_sym,
    get_bond_length_acem,
    get_bond_length_form,
)
from mmml.models.kernnn.kernels import get_1d_kernels_k33
from mmml.models.kernnn.model import (
    DualFFNet,
    FFNet,
    KerNNConfig,
    KerNNStats,
    descriptor_from_positions,
    energy_and_forces,
    energy_from_params,
)

__all__ = [
    "DualFFNet",
    "FFNet",
    "H2CO_CALCULATOR_STATS",
    "KerNNApplyAdapter",
    "KerNNCalculator",
    "KerNNConfig",
    "KerNNStats",
    "build_kernnn_batch_apply",
    "descriptor_from_positions",
    "energy_and_forces",
    "energy_from_params",
    "get_1d_kernels_k33",
    "get_bond_length_abcc",
    "get_bond_length_abcc_sym",
    "get_bond_length_acem",
    "get_bond_length_form",
    "import_torch_state_dict",
    "init_params",
    "is_kernnn_checkpoint",
    "load_checkpoint",
    "load_kernnn_model",
    "save_checkpoint",
]
