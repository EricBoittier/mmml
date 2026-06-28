"""Unit tests for jax-pme long-range Coulomb backend."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.long_range_backend import (
    JaxPmeLongRangeSolver,
    compute_jax_pme_coulomb,
    create_lr_solver,
    jax_pme_host_device_name,
    jax_pme_host_eval_context,
    pick_lr_solver,
    resolve_jax_pme_method,
)
from tests.functionality.long_range._common import (
    cscl_crystal,
    have_jax_pme_package,
    ion_dimer_system,
    jax_pme_coulomb_energy_forces,
)

pytestmark = pytest.mark.skipif(
    not have_jax_pme_package(),
    reason="jax-pme not installed",
)


def test_resolve_jax_pme_method_defaults_to_ewald():
    assert resolve_jax_pme_method(None) == "ewald"
    assert resolve_jax_pme_method("p3m") == "p3m"


def test_create_lr_solver_jax_pme():
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.long_range_backend.pick_lr_solver",
        return_value="jax_pme",
    ):
        solver = create_lr_solver("jax_pme")
    assert isinstance(solver, JaxPmeLongRangeSolver)
    assert solver.method == "ewald"


@pytest.mark.parametrize("method", ["ewald", "pme", "p3m"])
def test_compute_jax_pme_coulomb_matches_common_helper(method: str):
    system = ion_dimer_system(separation_A=5.0, box_length_A=30.0)
    ref = jax_pme_coulomb_energy_forces(system, method=method)  # type: ignore[arg-type]
    out = compute_jax_pme_coulomb(
        system.positions_A,
        system.charges_e,
        box_length_A=system.box_length_A,
        method=method,
    )
    np.testing.assert_allclose(out.energy_kcalmol, ref.energy_kcalmol, rtol=1e-10)
    np.testing.assert_allclose(out.forces_kcalmol_A, ref.forces_kcalmol_A, rtol=1e-8)


@pytest.mark.parametrize("method", ["ewald", "pme", "p3m"])
def test_jax_pme_methods_agree_on_cscl(method: str):
    system = cscl_crystal(box_length_A=10.0)
    ref = compute_jax_pme_coulomb(
        system.positions_A,
        system.charges_e,
        box_length_A=system.box_length_A,
        method="ewald",
    )
    test = compute_jax_pme_coulomb(
        system.positions_A,
        system.charges_e,
        box_length_A=system.box_length_A,
        method=method,
    )
    np.testing.assert_allclose(test.energy_kcalmol, ref.energy_kcalmol, rtol=2e-3)


def test_jax_pme_host_device_defaults_to_cpu(monkeypatch):
    monkeypatch.delenv("MMML_JAX_PME_DEVICE", raising=False)
    assert jax_pme_host_device_name() == "cpu"


def test_jax_pme_host_eval_context_uses_cpu_default_device(monkeypatch):
    monkeypatch.delenv("MMML_JAX_PME_DEVICE", raising=False)
    import jax

    class _FakeDevice:
        def __str__(self) -> str:
            return "cpu:0"

    fake_cpu = [_FakeDevice()]

    def _devices(kind: str):
        assert kind == "cpu"
        return fake_cpu

    with mock.patch("jax.devices", side_effect=_devices), mock.patch(
        "jax.default_device"
    ) as dd:
        dd.return_value.__enter__ = lambda self: None
        dd.return_value.__exit__ = lambda self, *a: None
        with jax_pme_host_eval_context():
            pass
        dd.assert_called_once_with(fake_cpu[0])


def test_jax_pme_host_eval_context_noop_when_gpu_requested(monkeypatch):
    monkeypatch.setenv("MMML_JAX_PME_DEVICE", "gpu")
    with jax_pme_host_eval_context():
        pass


def test_pick_lr_solver_jax_pme_when_scafacos_absent(monkeypatch):
    monkeypatch.delenv("MMML_LR_SOLVER", raising=False)
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.long_range_backend.have_scafacos",
        return_value=False,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.long_range_backend.have_jax_pme",
        return_value=True,
    ):
        assert pick_lr_solver("auto") == "jax_pme"
