"""Validation for the ML energy terms against the last-epoch example checkpoint.

Uses ``examples/sppoky-epoch-0010_params.json`` (the highest-epoch spooky
checkpoint shipped in ``examples/``). The terms are thin wrappers over the
``hybrid_energy`` factories, so we assert the wired term reproduces a direct
factory call (proving index/displacement/context wiring) and stays finite +
jittable. Skips cleanly when jax or the checkpoint are unavailable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CKPT = REPO / "examples" / "sppoky-epoch-0010_params.json"


@pytest.fixture(scope="module")
def ml_model():
    if not CKPT.exists():
        pytest.skip(f"checkpoint {CKPT.name} not present")
    try:
        from mmml.interfaces.calculators.simple_inference import (
            create_calculator_from_checkpoint,
        )
    except Exception as exc:  # pragma: no cover - env dependent
        pytest.skip(f"cannot import checkpoint loader: {exc}")
    calc = create_calculator_from_checkpoint(str(CKPT))
    model = getattr(calc, "model", getattr(calc, "_mmml_physnet_model", None))
    params = getattr(calc, "params", getattr(calc, "_mmml_physnet_params", None))
    if model is None or params is None:
        pytest.skip("could not extract model/params from calculator")
    return model, params


def _two_waters():
    from mmml.md.system import MolecularSystem

    r = np.array([
        [0.00, 0.00, 0.00], [0.96, 0.00, 0.00], [-0.24, 0.93, 0.00],  # water 0
        [3.10, 0.20, 0.10], [4.00, 0.30, 0.05], [2.90, 1.10, -0.10],  # water 1
    ])
    Z = np.array([8, 1, 1, 8, 1, 1], dtype=np.int32)
    box = np.diag([20.0, 20.0, 20.0])
    return MolecularSystem(
        R=r, Z=Z, box=box,
        mol_id=np.array([0, 0, 0, 1, 1, 1], dtype=np.int32),
        monomer_indices=[np.array([0, 1, 2]), np.array([3, 4, 5])],
    )


def _context(system):
    from jax_md import space

    from mmml.md.energy.registry import EnergyContext

    disp, _ = space.periodic(np.diag(np.asarray(system.box)))
    return EnergyContext(model=None, params=None, displacement_fn=disp), disp


def test_ml_terms_registered():
    import mmml.md.energy.terms  # noqa: F401  (importing registers the built-ins)
    from mmml.md.energy import available_terms

    assert "ml_intra" in available_terms()
    assert "ml_pep_water" in available_terms()


def test_ml_intra_matches_factory(ml_model):
    from mmml.interfaces.jaxmdInterface.hybrid_energy import make_monomer_energy_fn
    from mmml.md.energy.registry import EnergyContext
    from mmml.md.energy.terms import MLIntramolecularTerm

    model, params = ml_model
    system = _two_waters()
    _, disp = _context(system)
    ctx = EnergyContext(model=model, params=params, displacement_fn=disp)

    fn = MLIntramolecularTerm().make(system, ctx).jax_energy_fn
    got = float(fn(jnp.asarray(system.R)))

    ref_fn = make_monomer_energy_fn(
        model, params, jnp.asarray(system.Z, dtype=jnp.int32),
        [jnp.asarray(m, dtype=jnp.int32) for m in system.monomer_indices], disp,
    )
    ref = float(ref_fn(jnp.asarray(system.R)))

    assert np.isfinite(got)
    assert got == pytest.approx(ref, rel=1e-6)


def test_ml_intra_is_jittable(ml_model):
    from mmml.md.energy.registry import EnergyContext
    from mmml.md.energy.terms import MLIntramolecularTerm

    model, params = ml_model
    system = _two_waters()
    _, disp = _context(system)
    ctx = EnergyContext(model=model, params=params, displacement_fn=disp)
    fn = MLIntramolecularTerm().make(system, ctx).jax_energy_fn
    jfn = jax.jit(fn)
    assert float(jfn(jnp.asarray(system.R))) == pytest.approx(float(fn(jnp.asarray(system.R))), rel=1e-6)


def test_ml_core_group_matches_factory(ml_model):
    from mmml.interfaces.jaxmdInterface.hybrid_energy import (
        make_peptide_water_ml_energy_fn,
    )
    from mmml.md.energy.registry import EnergyContext
    from mmml.md.energy.terms import MLCoreGroupTerm

    model, params = ml_model
    system = _two_waters()
    _, disp = _context(system)
    ctx = EnergyContext(model=model, params=params, displacement_fn=disp)

    core = [0, 1, 2]
    groups = [[3, 4, 5]]
    fn = MLCoreGroupTerm(core_indices=core, group_indices=groups).make(system, ctx).jax_energy_fn
    got = float(fn(jnp.asarray(system.R)))

    ref_fn = make_peptide_water_ml_energy_fn(
        model, params, jnp.asarray(system.Z, dtype=jnp.int32),
        jnp.asarray(core, dtype=jnp.int32),
        [jnp.asarray(g, dtype=jnp.int32) for g in groups], disp,
    )
    ref = float(ref_fn(jnp.asarray(system.R)))

    assert np.isfinite(got)
    assert got == pytest.approx(ref, rel=1e-6)


def test_ml_intra_missing_model_raises():
    from mmml.md.energy.registry import EnergyContext
    from mmml.md.energy.terms import MLIntramolecularTerm

    system = _two_waters()
    with pytest.raises(ValueError, match="ML model"):
        MLIntramolecularTerm().make(system, EnergyContext())


def test_hybrid_composes_ml_and_mm(ml_model):
    """ml_intra composes with a bias term through HybridEnergy (jax face)."""
    from mmml.md.energy import HybridEnergy
    from mmml.md.energy.registry import EnergyContext
    from mmml.md.energy.terms import MLIntramolecularTerm, SMDBiasTerm

    model, params = ml_model
    system = _two_waters()
    _, disp = _context(system)
    ctx = EnergyContext(model=model, params=params, displacement_fn=disp)

    terms = [MLIntramolecularTerm(), SMDBiasTerm(0, 3, k_ev_per_A2=1.0, target=3.0)]
    hybrid = HybridEnergy(terms, system, ctx)
    total = float(hybrid.as_jax_energy_fn()(jnp.asarray(system.R)))
    parts = sum(float(t.make(system, ctx).jax_energy_fn(jnp.asarray(system.R))) for t in terms)
    assert total == pytest.approx(parts, rel=1e-6)
