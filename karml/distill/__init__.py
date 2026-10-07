"""Teacher–student distillation helpers (PET-MAD → PhysNet NPZ).

Public API lives in ``acetone_pool``, ``teacher_label``, and ``npz_export``.
Torch is not imported here.
"""

from karml.distill.acetone_pool import (
    ATOMS_PER_ACETONE,
    AcetonePoolConfig,
    Geometry,
    build_acetone_pool,
    load_acetone_monomer,
    load_dataset_dimers,
    pool_config_for_preset,
)
from karml.distill.npz_export import write_distill_npz
from karml.distill.teacher_label import (
    ENERGY_MODE_INTERACTION,
    ENERGY_MODE_TOTAL,
    LabeledSample,
    label_geometries,
)

__all__ = [
    "ATOMS_PER_ACETONE",
    "AcetonePoolConfig",
    "ENERGY_MODE_INTERACTION",
    "ENERGY_MODE_TOTAL",
    "Geometry",
    "LabeledSample",
    "build_acetone_pool",
    "label_geometries",
    "load_acetone_monomer",
    "load_dataset_dimers",
    "pool_config_for_preset",
    "write_distill_npz",
]
