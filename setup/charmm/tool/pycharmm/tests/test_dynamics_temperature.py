import math

import numpy as np
import pytest

import pycharmm.block as block
import pycharmm.coor as coor
import pycharmm.dynamics as dynamics
import pycharmm.psf as psf


class _Lib:
    def __init__(self, status):
        self.status = status
        self.calls = []

    def dynamics_exchange_temperature(self, expected, new):
        self.calls.append((expected._obj.value, new._obj.value))
        return self.status


def test_apply_rex_temperature_direct_calls_core(monkeypatch):
    fake = _Lib(1)
    monkeypatch.setattr(dynamics, "lib", fake)

    assert dynamics.apply_rex_temperature_direct(300.0, 330.0)
    assert fake.calls == [(300.0, 330.0)]


def test_validate_rex_temperature_direct_is_a_read_only_exchange(monkeypatch):
    fake = _Lib(1)
    monkeypatch.setattr(dynamics, "lib", fake)

    assert dynamics.validate_rex_temperature_direct(300.0)
    assert fake.calls == [(300.0, 300.0)]


@pytest.mark.parametrize(
    "status, message",
    [
        (-1, "requires ordinary NVT with one atomic bath"),
        (-2, "does not match"),
        (-3, "atomic velocity state is unavailable"),
        (-4, "live BLaDE temperature does not match"),
        (-5, "live BLaDE temperature state is unavailable"),
    ],
)
def test_apply_rex_temperature_direct_surfaces_core_errors(monkeypatch, status, message):
    monkeypatch.setattr(dynamics, "lib", _Lib(status))

    with pytest.raises(RuntimeError, match=message):
        dynamics.apply_rex_temperature_direct(300.0, 330.0)


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan")])
def test_apply_rex_temperature_direct_rejects_invalid_input(monkeypatch, temperature):
    monkeypatch.setattr(dynamics, "lib", _Lib(1))

    with pytest.raises(ValueError):
        dynamics.apply_rex_temperature_direct(temperature, 330.0)


def test_temperature_exchange_scales_velocities_and_continues(alanine_dipeptide_with_nbonds):
    dynamics.set_fbetas(np.full(psf.get_natom(), 5.0))

    def run_segment(temperature, iasvel):
        run = dynamics.DynamicsScript(
            velos=True,
            leap=True,
            nstep=2,
            lang=True,
            start=True,
            timest=0.001,
            firstt=temperature,
            finalt=temperature,
            tbath=temperature,
            iasvel=iasvel,
            inbfrq=-1,
            ihbfrq=0,
            ilbfrq=0,
            nprint=1,
            iprfrq=1,
            isvfrq=0,
            echeck=-1.0,
        )
        run.run()
        return run.velos

    run_segment(300.0, 1)
    before = coor.get_comparison()[["x", "y", "z"]].to_numpy()

    block.initialize(2)
    block.enable_lambda_dynamics()
    block.set_langevin(temp=0.0)
    block.end()
    assert block.get_temperature_direct() == pytest.approx(0.0)

    dynamics.apply_rex_temperature_direct(300.0, 330.0)

    after = coor.get_comparison()[["x", "y", "z"]].to_numpy()
    assert np.allclose(after, before * math.sqrt(330.0 / 300.0))
    assert dynamics.validate_rex_temperature_direct(330.0)
    assert block.get_temperature_direct() == pytest.approx(300.0)
    block.clear()

    velocities = run_segment(330.0, 0)
    assert np.isfinite(velocities.to_numpy()).all()
