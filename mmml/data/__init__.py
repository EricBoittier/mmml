"""
Unified data module for MMML.

Provides standardized data loading, conversion, and preparation for all models
(DCMNet, PhysNetJAX, etc.) in the MMML ecosystem.

Main API:
    - load_molpro_xml: Load Molpro XML files
    - molpro_to_npz: Convert MolproData to NPZ format
    - load_npz: Load NPZ files with validation
    - prepare_batches: Prepare batches for training
"""

from .npz_schema import (
    NPZSchema,
    REQUIRED_KEYS,
    OPTIONAL_KEYS,
    METADATA_KEYS,
    validate_npz,
)
from .xml_to_npz import MolproConverter, convert_xml_to_npz, batch_convert_xml
from .loaders import (
    load_npz,
    load_multiple_npz,
    DataConfig,
    train_valid_split,
    get_data_statistics
)
from .preprocessing import (
    center_coordinates,
    normalize_energies,
    denormalize_energies,
    create_esp_mask
)

__version__ = "0.1.0"

__all__ = [
    # Schema
    "NPZSchema",
    "REQUIRED_KEYS",
    "OPTIONAL_KEYS",
    "METADATA_KEYS",
    "validate_npz",
    # Converters
    "MolproConverter",
    "convert_xml_to_npz",
    "batch_convert_xml",
    # Loaders
    "load_npz",
    "load_multiple_npz",
    "DataConfig",
    "train_valid_split",
    "get_data_statistics",
    # Preprocessing
    "center_coordinates",
    "normalize_energies",
    "denormalize_energies",
    "create_esp_mask",
]

