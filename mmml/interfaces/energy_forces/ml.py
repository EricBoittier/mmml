"""ML checkpoint detection and provider construction."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Mapping

from mmml.interfaces.calculators.checkpoint_loading import (
    detect_checkpoint_format,
    load_checkpoint_bundle,
)
from mmml.interfaces.energy_forces.adapters import AseCalculatorProvider
from mmml.interfaces.energy_forces.protocol import (
    EnergyForcesProvider,
    ProviderCapabilities,
    ProviderKind,
)
from mmml.interfaces.qc_backends.ml_backend import MLBackend, build_ml_backend


def detect_model_kind(checkpoint: Path | str, *, config: dict[str, Any] | None = None) -> ProviderKind:
    """Classify an ML checkpoint as PhysNet, joint, E-field, etc."""
    path = Path(checkpoint).expanduser().resolve()
    cfg = dict(config) if config is not None else _peek_checkpoint_config(path)

    if cfg.get("model_type") == "efield" or cfg.get("dipole_field_coupling") is not None:
        if "Ef" in cfg or "field_scale" in cfg:
            return ProviderKind.EFIELD_PHYSNET

    if "physnet_config" in cfg and (
        "dcmnet_config" in cfg or "noneq_config" in cfg
    ):
        return ProviderKind.JOINT_PHYSNET_DCMNET

    model_type = str(cfg.get("model_type", "")).lower()
    if model_type == "spooky" or "spooky" in str(path).lower():
        return ProviderKind.SPOOKY_PHYSNET

    if model_type == "efield" or "efield" in str(path).lower():
        return ProviderKind.EFIELD_PHYSNET

    if any(k in cfg for k in ("features", "max_degree", "num_iterations", "cutoff")):
        return ProviderKind.PHYSNET

    try:
        fmt = detect_checkpoint_format(path)
    except FileNotFoundError:
        return ProviderKind.UNKNOWN

    if fmt == "pickle_joint":
        return ProviderKind.JOINT_PHYSNET_DCMNET
    if fmt in ("json", "orbax"):
        return ProviderKind.PHYSNET
    return ProviderKind.UNKNOWN


def capabilities_for_kind(kind: ProviderKind) -> ProviderCapabilities:
    """Return default capabilities for a model/backend kind."""
    mapping: dict[ProviderKind, ProviderCapabilities] = {
        ProviderKind.PHYSNET: ProviderCapabilities(
            kind=kind,
            supports_dipole=True,
            supports_charges=True,
            supports_decomposed_ml=True,
            notes="Standard PhysNet; usable in hybrid CHARMM monomer/dimer MLpot.",
        ),
        ProviderKind.SPOOKY_PHYSNET: ProviderCapabilities(
            kind=kind,
            supports_dipole=True,
            supports_charges=True,
            supports_decomposed_ml=True,
            notes="Charge/spin-conditioned PhysNet; hybrid MLpot compatible.",
        ),
        ProviderKind.JOINT_PHYSNET_DCMNET: ProviderCapabilities(
            kind=kind,
            supports_dipole=True,
            supports_charges=True,
            supports_esp=True,
            supports_decomposed_ml=True,
            notes=(
                "Hybrid MLpot uses PhysNet submodule only (E/F); "
                "DCMNet distributed charges are not evaluated in monomer/dimer batches."
            ),
        ),
        ProviderKind.EFIELD_PHYSNET: ProviderCapabilities(
            kind=kind,
            supports_dipole=True,
            supports_external_field=True,
            supports_decomposed_ml=False,
            notes="Requires external electric field vector Ef; use efield-md or zero-field inference.",
        ),
        ProviderKind.DCMNET: ProviderCapabilities(
            kind=kind,
            supports_energy=False,
            supports_forces=False,
            supports_esp=True,
            supports_decomposed_ml=False,
            notes="Distributed charge sites for ESP; not a standalone PES.",
        ),
        ProviderKind.PYSCF: ProviderCapabilities(
            kind=kind,
            supports_dipole=True,
            notes="Supplementary QC backend (batch cross-check / validation).",
        ),
        ProviderKind.ORCA: ProviderCapabilities(
            kind=kind,
            supports_dipole=True,
            notes="Supplementary QC backend via ORCA EnGrad.",
        ),
        ProviderKind.XTB: ProviderCapabilities(
            kind=kind,
            supports_dipole=True,
            notes="Supplementary semiempirical backend (tblite/xTB).",
        ),
        ProviderKind.MOLPRO: ProviderCapabilities(
            kind=kind,
            supports_dipole=True,
            notes="Supplementary QC backend via Molpro XML.",
        ),
    }
    return mapping.get(
        kind,
        ProviderCapabilities(kind=ProviderKind.UNKNOWN, notes="Unknown provider kind."),
    )


def assert_hybrid_ml_compatible(checkpoint: Path | str, *, config: dict[str, Any] | None = None) -> ProviderKind:
    """Raise ``ValueError`` when a checkpoint cannot drive CHARMM decomposed MLpot."""
    kind = detect_model_kind(checkpoint, config=config)
    if kind == ProviderKind.EFIELD_PHYSNET:
        raise ValueError(
            f"Checkpoint {checkpoint} is an E-field PhysNet model and requires an external "
            "field vector; it cannot drive standard hybrid CHARMM MLpot. "
            "Use efield-md or build_provider() instead."
        )
    if kind == ProviderKind.UNKNOWN:
        raise ValueError(
            f"Could not classify checkpoint {checkpoint} for hybrid MLpot."
        )
    caps = capabilities_for_kind(kind)
    if not caps.supports_decomposed_ml:
        raise ValueError(
            f"Checkpoint {checkpoint} ({kind.value}) cannot be used in hybrid CHARMM "
            f"monomer/dimer MLpot. {caps.notes}"
        )
    return kind


class _MlBackendWrapper:
    """Attach capabilities to :class:`MLBackend` for the unified provider protocol."""

    def __init__(self, backend: MLBackend, *, capabilities: ProviderCapabilities) -> None:
        self._backend = backend
        self.capabilities = capabilities
        self.name = backend.name
        self.method_label = backend.method_label
        self.energy_unit = backend.energy_unit
        self.force_unit = backend.force_unit

    def evaluate_batch(
        self,
        frames: list,
        *,
        properties: frozenset[str],
        context: Mapping[str, Any] | None = None,
    ) -> dict:
        _ = context
        return self._backend.evaluate_batch(frames, properties=properties)


def build_ml_provider(options: dict[str, Any]) -> EnergyForcesProvider:
    """Build an ML provider from checkpoint options (cross-check compatible)."""
    checkpoint = options.get("checkpoint")
    if checkpoint is None:
        raise ValueError("ML provider requires 'checkpoint' in options.")
    kind = detect_model_kind(checkpoint)
    backend = build_ml_backend(options)
    return _MlBackendWrapper(backend, capabilities=capabilities_for_kind(kind))


def build_ml_ase_provider(
    checkpoint: Path | str,
    *,
    cutoff: float | None = None,
    use_dcmnet_dipole: bool = False,
) -> AseCalculatorProvider:
    """Load any supported ML checkpoint as an ASE-based provider."""
    from mmml.interfaces.calculators.checkpoint_loading import create_calculator_from_checkpoint

    calc = create_calculator_from_checkpoint(
        checkpoint,
        cutoff=cutoff,
        use_dcmnet_dipole=use_dcmnet_dipole,
    )
    kind = detect_model_kind(checkpoint)
    return AseCalculatorProvider(calc, capabilities=capabilities_for_kind(kind))


def _peek_checkpoint_config(path: Path) -> dict[str, Any]:
    if path.is_file() and path.suffix == ".json":
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, dict) and "config" in payload:
            cfg = payload["config"]
            return cfg if isinstance(cfg, dict) else {}
        return payload if isinstance(payload, dict) else {}

    config_path = path / "model_config.json"
    if config_path.is_file():
        with open(config_path, encoding="utf-8") as fh:
            cfg = json.load(fh)
        return cfg if isinstance(cfg, dict) else {}

    if path.is_file() and path.suffix == ".pkl":
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        if isinstance(payload, dict):
            for key in ("config", "model_config", "physnet_config"):
                if key in payload and isinstance(payload[key], dict):
                    return payload[key]
            return {k: v for k, v in payload.items() if k.endswith("_config") and isinstance(v, dict)} or payload
    try:
        bundle = load_checkpoint_bundle(path)
        return dict(bundle.config)
    except Exception:
        return {}
