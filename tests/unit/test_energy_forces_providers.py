"""Unit tests for unified energy/forces providers."""

from __future__ import annotations

import pytest

from mmml.interfaces.energy_forces import (
    ProviderKind,
    ProviderSpec,
    assert_hybrid_ml_compatible,
    build_provider,
    capabilities_for_kind,
    detect_model_kind,
)


def test_capabilities_physnet_supports_decomposed_ml() -> None:
    caps = capabilities_for_kind(ProviderKind.PHYSNET)
    assert caps.supports_decomposed_ml is True
    assert caps.supports_energy is True
    assert caps.supports_forces is True


def test_capabilities_joint_does_not_support_decomposed_ml() -> None:
    caps = capabilities_for_kind(ProviderKind.JOINT_PHYSNET_DCMNET)
    assert caps.supports_decomposed_ml is False
    assert caps.supports_esp is True


def test_capabilities_qc_backends() -> None:
    for kind in (ProviderKind.PYSCF, ProviderKind.ORCA, ProviderKind.XTB, ProviderKind.MOLPRO):
        caps = capabilities_for_kind(kind)
        assert caps.supports_energy is True
        assert caps.supports_decomposed_ml is False


def test_detect_model_kind_physnet_config() -> None:
    kind = detect_model_kind(
        "unused-path",
        config={"features": 32, "num_iterations": 2, "cutoff": 6.0},
    )
    assert kind == ProviderKind.PHYSNET


def test_detect_model_kind_joint_config() -> None:
    kind = detect_model_kind(
        "unused-path",
        config={"physnet_config": {"natoms": 10}, "dcmnet_config": {"n_dcm": 3}},
    )
    assert kind == ProviderKind.JOINT_PHYSNET_DCMNET


def test_detect_model_kind_efield_config() -> None:
    kind = detect_model_kind(
        "unused-path",
        config={"field_scale": 0.001, "dipole_field_coupling": False, "features": 32},
    )
    assert kind == ProviderKind.EFIELD_PHYSNET


def test_assert_hybrid_ml_compatible_rejects_joint() -> None:
    with pytest.raises(ValueError, match="joint_physnet_dcmnet"):
        assert_hybrid_ml_compatible(
            "unused-path",
            config={"physnet_config": {}, "dcmnet_config": {}},
        )


def test_assert_hybrid_ml_compatible_accepts_physnet() -> None:
    kind = assert_hybrid_ml_compatible(
        "unused-path",
        config={"features": 32, "num_iterations": 2},
    )
    assert kind == ProviderKind.PHYSNET


def test_build_provider_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown provider"):
        build_provider(ProviderSpec(name="not_a_backend"))
