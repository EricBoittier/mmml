"""``_charmm_collapsed_nonbonded_hint`` decorates a failed geometry gate.

No CHARMM: the helper is pure logic over one energy read, and the point is that
it never raises and never fires spuriously.
"""

from __future__ import annotations

import sys
import types

import pytest

from mmml.utils.monomer_internal_geometry import (
    COLLAPSED_ELEC_PER_ATOM_KCAL,
    charmm_collapsed_nonbonded_hint,
)


def _hint():
    return charmm_collapsed_nonbonded_hint, COLLAPSED_ELEC_PER_ATOM_KCAL


def _fake_energy(monkeypatch, elec):
    """Install a stub ``pycharmm.energy`` whose ``get_elec`` returns *elec*."""
    mod = types.ModuleType("pycharmm.energy")

    def get_elec():
        if isinstance(elec, Exception):
            raise elec
        return elec

    mod.get_elec = get_elec
    pkg = sys.modules.get("pycharmm") or types.ModuleType("pycharmm")
    monkeypatch.setitem(sys.modules, "pycharmm", pkg)
    monkeypatch.setitem(sys.modules, "pycharmm.energy", mod)
    # `import pycharmm.energy as energy` binds the package *attribute* when
    # pycharmm is already imported, so patching sys.modules alone leaves the real
    # module in place and makes this test pass or fail on collection order.
    monkeypatch.setattr(pkg, "energy", mod, raising=False)


def test_fires_on_a_collapsed_structure(monkeypatch):
    # Measured on a MEOH:4 cluster minimized with a wiped NONBONDED table.
    _fake_energy(monkeypatch, -9_065_379.116)
    hint, _ = _hint()
    hint = hint(24)
    assert "ELEC" in hint
    assert "test_charmm_param_read_contract" in hint
    assert "api_read.F90" in hint


def test_silent_on_a_healthy_structure(monkeypatch):
    # Same cluster, live nonbonded table.
    _fake_energy(monkeypatch, 43.105)
    fn, _ = _hint()
    assert fn(24) == ""


def test_silent_just_below_the_threshold(monkeypatch):
    n_atoms = 24
    fn, limit = _hint()
    _fake_energy(monkeypatch, -0.99 * limit * n_atoms)
    assert fn(n_atoms) == ""


def test_scales_with_system_size(monkeypatch):
    """The threshold is per atom, so a big healthy box must not trip it."""
    elec = -50.0 * 3000  # 3000 atoms at a hefty 50 kcal/mol each
    _fake_energy(monkeypatch, elec)
    fn, _ = _hint()
    assert fn(3000) == ""


@pytest.mark.parametrize("n_atoms", [0, 1])
def test_ignores_degenerate_systems(monkeypatch, n_atoms):
    _fake_energy(monkeypatch, -1e12)
    fn, _ = _hint()
    assert fn(n_atoms) == ""


def test_never_raises_when_charmm_cannot_be_read(monkeypatch):
    """Diagnosis must not mask the real error it is decorating."""
    _fake_energy(monkeypatch, RuntimeError("no CHARMM session"))
    fn, _ = _hint()
    assert fn(24) == ""
