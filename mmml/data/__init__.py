"""
Unified data module for MMML.

Provides standardized data loading, conversion, and preparation for all models
(DCMNet, PhysNetJAX, etc.) in the MMML ecosystem.

Main API:
    - load_molpro_xml: Load Molpro XML files
    - molpro_to_npz: Convert MolproData to NPZ format
    - load_npz: Load NPZ files with validation
    - prepare_batches: Prepare batches for training
    - convert_spice_alpha_hdf5: SPICE-α HDF5 → PhysNet train NPZ
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
from .rmd17 import (
    load_rmd17_npz,
    load_rmd17_official_splits,
    resolve_rmd17_splits_dir,
)
from .spice_alpha import (
    SpiceAlphaFrame,
    assert_train_npz_contract,
    check_efield_train_npz,
    classify_units_map,
    extract_des370k_hdf5,
    convert_spice_alpha_hdf5,
    iter_spice_alpha_frames,
    pad_frames,
    split_npz,
    write_physnet_npz,
)
from .efield_lc import (
    DEFAULT_LC_FRACS,
    frac_tag,
    n_train_for_frac,
    prepare_lc_grid,
    prepare_lc_split,
    subsample_frame_npz,
)
from .units import (
    CALCULATOR_UNITS,
    CANONICAL_ENERGY_UNIT,
    CANONICAL_FORCE_UNIT,
    CANONICAL_LENGTH_UNIT,
    HARTREE_BOHR_TO_EV_ANGSTROM,
    HARTREE_TO_EV,
    TRAINING_UNITS,
    UnitsManifestV2,
    convert_energy,
    convert_forces,
    energy_to_ev,
    forces_to_ev_angstrom,
    load_units_manifest,
    normalize_to_canonical,
    subtract_atom_refs,
    units_from_npz,
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
    # MD17 / rMD17
    "load_rmd17_npz",
    "load_rmd17_official_splits",
    "resolve_rmd17_splits_dir",
    # SPICE-α (Zenodo 19205036) → PhysNet train NPZ
    "SpiceAlphaFrame",
    "assert_train_npz_contract",
    "check_efield_train_npz",
    "classify_units_map",
    "extract_des370k_hdf5",
    "convert_spice_alpha_hdf5",
    "iter_spice_alpha_frames",
    "pad_frames",
    "split_npz",
    "write_physnet_npz",
    "DEFAULT_LC_FRACS",
    "frac_tag",
    "n_train_for_frac",
    "prepare_lc_grid",
    "prepare_lc_split",
    "subsample_frame_npz",
    # Units
    "CALCULATOR_UNITS",
    "CANONICAL_ENERGY_UNIT",
    "CANONICAL_FORCE_UNIT",
    "CANONICAL_LENGTH_UNIT",
    "HARTREE_BOHR_TO_EV_ANGSTROM",
    "HARTREE_TO_EV",
    "TRAINING_UNITS",
    "UnitsManifestV2",
    "convert_energy",
    "convert_forces",
    "energy_to_ev",
    "forces_to_ev_angstrom",
    "load_units_manifest",
    "normalize_to_canonical",
    "subtract_atom_refs",
    "units_from_npz",
]

