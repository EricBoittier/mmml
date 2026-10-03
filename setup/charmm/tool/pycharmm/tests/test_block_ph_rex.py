import ctypes

import numpy as np

import pycharmm.block as block


class _GetLib:
    def blockdata_get_nblock(self):
        return 3

    def blockdata_ph_rex_get(
        self,
        ph,
        temperature,
        lambdas,
        biases,
        masses,
        frictions,
        fixed,
        sites,
        count,
        status,
    ):
        assert count.value == 3
        ph._obj.value = 5.5
        temperature._obj.value = 298.0
        for index in range(3):
            lambdas[index] = [1.0, 0.25, 0.75][index]
            biases[index] = [0.0, 2.0, -3.0][index]
            masses[index] = 12.0
            frictions[index] = 5.0
            fixed[index] = index == 2
            sites[index] = [0, 1, 1][index]
        status._obj.value = 0


class _SetLib:
    def __init__(self):
        self.calls = []

    def blockdata_get_nblock(self):
        return 3

    def blockdata_ph_rex_set(self, ph, biases, count, status):
        self.calls.append((ph.value, [biases[index] for index in range(count.value)]))
        status._obj.value = 0


class _ScalarFunction:
    def __init__(self, value):
        self.value = value
        self.restype = None

    def __call__(self):
        return self.value


def test_get_ph_rex_state_direct(monkeypatch):
    monkeypatch.setattr(block, "_check_blockdata_available", lambda: True)
    monkeypatch.setattr(block, "lib", _GetLib())

    state = block.get_ph_rex_state_direct()

    assert state["ph"] == 5.5
    assert state["temperature"] == 298.0
    assert np.array_equal(state["lambdas"], [1.0, 0.25, 0.75])
    assert np.array_equal(state["biases"], [0.0, 2.0, -3.0])
    assert np.array_equal(state["fixed"], [False, False, True])
    assert np.array_equal(state["sites"], [0, 1, 1])


def test_set_ph_rex_label_direct_validates_and_calls_core(monkeypatch):
    fake = _SetLib()
    monkeypatch.setattr(block, "_check_blockdata_available", lambda: True)
    monkeypatch.setattr(block, "lib", fake)

    assert block.set_ph_rex_label_direct(6.5, [0.0, 4.0, -2.0])
    assert fake.calls == [(6.5, [0.0, 4.0, -2.0])]
    assert not block.set_ph_rex_label_direct(6.5, [0.0, 4.0])
    assert not block.set_ph_rex_label_direct(6.5, [0.0, np.nan, -2.0])


def test_get_ph_direct_sets_scalar_abi(monkeypatch):
    ph = _ScalarFunction(7.25)

    class _Lib:
        blockdata_get_ph = ph

    monkeypatch.setattr(block, "_check_blockdata_available", lambda: True)
    monkeypatch.setattr(block, "lib", _Lib())

    assert block.get_ph_direct() == 7.25
    assert ph.restype is ctypes.c_double
