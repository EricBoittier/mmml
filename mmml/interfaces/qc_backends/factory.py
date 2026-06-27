"""Factory for supplementary QC backends."""

from __future__ import annotations

from typing import Any

from mmml.interfaces.qc_backends.ml_backend import MLBackend, build_ml_backend
from mmml.interfaces.qc_backends.molpro import MolproBackend, build_molpro_backend
from mmml.interfaces.qc_backends.orca_qm import OrcaQMBackend, build_orca_backend
from mmml.interfaces.qc_backends.protocol import BackendSpec, QCEvaluator
from mmml.interfaces.qc_backends.pyscf_backend import PySCFBackend, build_pyscf_backend
from mmml.interfaces.qc_backends.xtb import XTBBackend, build_xtb_backend

_BUILDERS = {
    "pyscf": build_pyscf_backend,
    "ml": build_ml_backend,
    "orca": build_orca_backend,
    "orca_qm": build_orca_backend,
    "xtb": build_xtb_backend,
    "molpro": build_molpro_backend,
}


def build_backend(spec: BackendSpec) -> QCEvaluator:
    """Instantiate a backend from a BackendSpec."""
    name = spec.name.lower()
    builder = _BUILDERS.get(name)
    if builder is None:
        raise ValueError(
            f"Unknown backend {spec.name!r}. Supported: {', '.join(sorted(_BUILDERS))}"
        )
    return builder(dict(spec.options))


def backend_from_dict(entry: dict[str, Any]) -> BackendSpec:
    """Parse a YAML/JSON backend entry."""
    name = entry.get("name")
    if not name:
        raise ValueError("Each backend entry must include 'name'.")
    options = {k: v for k, v in entry.items() if k != "name"}
    return BackendSpec(name=str(name), options=options)
