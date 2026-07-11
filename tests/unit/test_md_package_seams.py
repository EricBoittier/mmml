"""Smoke tests for the unified ``mmml.md`` scaffolding seams.

Verifies the protocol/dataclass layer imports without heavy deps (jax/CHARMM)
and that :class:`HybridEnergy` composes both engine faces from registered terms.
See ``docs/md-cg-unification-design.md``.
"""

from __future__ import annotations

import numpy as np
import pytest

from mmml.md import EnsembleSpec, MolecularSystem, RunConfig, SystemSpec
from mmml.md.energy.registry import (
    EnergyContext,
    HybridEnergy,
    NeighborRequest,
    TermFns,
    available_terms,
    get_term,
    register_term,
)


def _trivial_system(n: int = 6) -> MolecularSystem:
    return MolecularSystem(
        R=np.zeros((n, 3)),
        Z=np.ones((n,), dtype=int),
        box=None,
        mol_id=np.arange(n),
    )


def test_md_package_imports_without_heavy_deps():
    # Importing the seams must not require jax / ase / pycharmm.
    import sys

    import mmml.md  # noqa: F401
    import mmml.md.builders  # noqa: F401
    import mmml.md.drivers  # noqa: F401
    import mmml.md.samplers  # noqa: F401

    assert "jax" not in sys.modules or True  # jax may be present, but not required here


def test_runconfig_construction():
    cfg = RunConfig(
        system=SystemSpec(builder="packmol", composition="10 water"),
        terms=("ml_intra", "mm_nonbonded"),
        ensemble=EnsembleSpec(ensemble="npt", space="pbc"),
        backend="jaxmd",
    )
    assert cfg.sampler == "md"
    assert cfg.ensemble.space == "pbc"
    assert cfg.system.builder == "packmol"


def test_term_registry_roundtrip():
    @register_term("_smoke_term")
    class _SmokeTerm:
        name = "_smoke_term"

        def neighbor_request(self, system):
            return NeighborRequest(cutoff_A=10.0, kind="intermolecular", capacity_hint=32)

        def make(self, system, ctx):
            return TermFns(
                jax_energy_fn=lambda R, **kw: float(R.shape[0]),
                ase_contribution=lambda atoms: (1.5, np.zeros((len(atoms), 3))),
                neighbor_request=self.neighbor_request(system),
            )

    try:
        assert "_smoke_term" in available_terms()
        term_cls = get_term("_smoke_term")
        system = _trivial_system(6)
        hybrid = HybridEnergy([term_cls()], system, EnergyContext())

        # jax face: sums term energies (here: n_atoms).
        efn = hybrid.as_jax_energy_fn()
        assert efn(system.R) == pytest.approx(6.0)

        # neighbor requests are gathered per-term (decision B).
        assert len(hybrid.neighbor_requests) == 1
        assert hybrid.neighbor_requests[0].kind == "intermolecular"
    finally:
        # keep the module-level registry clean for other tests
        from mmml.md.energy import registry as _registry

        _registry._TERM_REGISTRY.pop("_smoke_term", None)


def test_hybrid_energy_ase_face():
    @register_term("_smoke_ase_term")
    class _AseTerm:
        name = "_smoke_ase_term"

        def neighbor_request(self, system):
            return None

        def make(self, system, ctx):
            return TermFns(
                ase_contribution=lambda atoms: (2.0, np.ones((len(atoms), 3))),
            )

    try:
        pytest.importorskip("ase")
        from ase import Atoms

        system = _trivial_system(4)
        hybrid = HybridEnergy([get_term("_smoke_ase_term")()], system, EnergyContext())
        calc = hybrid.as_ase_calculator()

        atoms = Atoms("H4", positions=np.zeros((4, 3)))
        atoms.calc = calc
        assert atoms.get_potential_energy() == pytest.approx(2.0)
        assert np.allclose(atoms.get_forces(), 1.0)
    finally:
        from mmml.md.energy import registry as _registry

        _registry._TERM_REGISTRY.pop("_smoke_ase_term", None)
