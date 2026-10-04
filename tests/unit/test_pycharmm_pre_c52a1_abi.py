"""Call shapes for libcharmm.so built before the c52a1 API bump."""

import ctypes

import pycharmm.crystal as crystal
import pycharmm.nbonds as nbonds


class _Getter:
    def __init__(self, value: float):
        self.value = value
        self.restype = None
        self.argtypes = None

    def __call__(self, slot):
        slot[0] = self.value
        return 1


class _OldLib:
    def __init__(self):
        self.nbonds_get_ctonnb = _Getter(7.5)
        self.calls = []

    def crystal_build(self, cutoff, ops, nops):
        self.calls.append(nops)
        return 1


class _NewLib(_OldLib):
    def blockdata_is_active(self):
        return 1

    def crystal_build(self, cutoff, ops, nops):
        self.calls.append(nops)
        return 1


def test_pre_c52a1_ctonnb_getter_reads_the_out_argument(monkeypatch):
    monkeypatch.setattr(nbonds, "lib", _OldLib())
    assert nbonds.get_ctonnb() == 7.5


def test_pre_c52a1_crystal_build_passes_nops_by_pointer(monkeypatch):
    lib = _OldLib()
    monkeypatch.setattr(crystal, "lib", lib)
    monkeypatch.setattr(crystal, "_init_ctypes", lambda: None)
    assert crystal.build(10.0) == 1
    assert len(lib.calls) == 1
    assert not isinstance(lib.calls[0], ctypes.c_int)


def test_c52a1_crystal_build_passes_nops_by_value(monkeypatch):
    lib = _NewLib()
    monkeypatch.setattr(crystal, "lib", lib)
    monkeypatch.setattr(crystal, "_init_ctypes", lambda: None)
    assert crystal.build(10.0) == 1
    assert isinstance(lib.calls[0], ctypes.c_int)
    assert lib.calls[0].value == 0
