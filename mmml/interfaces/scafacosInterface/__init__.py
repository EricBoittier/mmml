"""ScaFaCoS (Scalable Fast Coulomb Solvers) optional integration for MMML.

See ``mmml/interfaces/scafacosInterface/README.md`` for installation and the
interface contract. Runtime selection is via ``long_range_backend.resolve_lr_solver``.
"""

from mmml.interfaces.scafacosInterface.scafacos_session import (
    CoulombFieldResult,
    ScaFaCoSSession,
    ScaFaCoSUnavailable,
    compute_scafacos_coulomb,
    have_scafacos,
    load_scafacos_library,
    resolve_scafacos_library_path,
)

__all__ = [
    "CoulombFieldResult",
    "ScaFaCoSSession",
    "ScaFaCoSUnavailable",
    "compute_scafacos_coulomb",
    "have_scafacos",
    "load_scafacos_library",
    "resolve_scafacos_library_path",
]
