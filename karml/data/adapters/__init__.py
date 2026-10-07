"""
Model-specific data adapters for MMML.

Adapters convert the standardized NPZ format into model-specific
batch formats for DCMNet, PhysNetJAX, and other models.
"""

from .dcmnet import prepare_dcmnet_batches
from .physnetjax import prepare_physnet_batches

__all__ = [
    'prepare_dcmnet_batches',
    'prepare_physnet_batches',
]

