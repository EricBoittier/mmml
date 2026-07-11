"""Parity/smoke tests for the shared jax-md driver."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax_md")

from mmml.md.config import EnsembleSpec
from mmml.md.drivers import JaxmdDriver
from mmml.md.energy import EnergyContext, HybridEnergy, TermFns
from mmml.md.system import MolecularSystem


class _HarmonicTerm:
    name = "harmonic"

    def neighbor_request(self, system):
        return None

    def make(self, system, ctx):
        import jax.numpy as jnp

        origin = jnp.asarray(ctx.options.get("origin", np.zeros_like(system.R)))

        def energy_fn(R, **kwargs):
            return 0.5 * jnp.sum((R - origin) ** 2)

        return TermFns(jax_energy_fn=energy_fn)


class _KeywordHarmonicTerm:
    name = "keyword_harmonic"

    def neighbor_request(self, system):
        return None

    def make(self, system, ctx):
        import jax.numpy as jnp

        def energy_fn(R, **kwargs):
            return 0.5 * kwargs["scale"] * jnp.sum(R**2)

        return TermFns(jax_energy_fn=energy_fn)


def _system():
    return MolecularSystem(
        R=np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
        Z=np.array([1, 1]),
        box=None,
        mol_id=np.array([0, 1]),
    )


def test_fire_minimization_reduces_energy(tmp_path: Path):
    system = _system()
    energy = HybridEnergy([_HarmonicTerm()], system, EnergyContext())
    output = tmp_path / "fire.npz"
    result = JaxmdDriver(record_every=5, block_size=5, output_path=output).run(
        system, energy, EnsembleSpec(ensemble="min", space="free", dt_fs=1.0, n_steps=20)
    )

    assert result.exit_code == 0
    assert result.n_frames == 5
    assert result.metadata["energies"][-1] < result.metadata["energies"][0]
    assert output.exists()


def test_nve_records_requested_final_step_and_rebuilds_neighbors():
    system = _system()
    energy = HybridEnergy([_HarmonicTerm()], system, EnergyContext())
    rebuilds = []

    def neighbor_fn(position, box):
        rebuilds.append(position.copy())
        return {}

    result = JaxmdDriver(record_every=4, block_size=3, neighbor_fn=neighbor_fn).run(
        system,
        energy,
        EnsembleSpec(
            ensemble="nve",
            space="free",
            dt_fs=0.1,
            n_steps=7,
            params={"masses": np.ones(2), "seed": 4},
        ),
    )

    assert result.metadata["steps"] == 7
    assert result.n_frames == 3  # initial + step 4 + non-aligned final step 7
    assert len(rebuilds) == 4  # initial allocation + one refresh per block


def test_nvt_nhc_runs_and_is_reproducible():
    system = _system()
    energy = HybridEnergy([_HarmonicTerm()], system, EnergyContext())
    ensemble = EnsembleSpec(
        ensemble="nvt",
        space="free",
        temperature_K=150.0,
        dt_fs=0.1,
        n_steps=3,
        params={"seed": 17},
    )

    first = JaxmdDriver(record_every=2).run(system, energy, ensemble)
    second = JaxmdDriver(record_every=2).run(system, energy, ensemble)

    np.testing.assert_allclose(first.metadata["positions"], second.metadata["positions"])
    assert np.isfinite(first.metadata["energies"]).all()
    assert first.n_frames == 3


def test_fixed_box_pbc_and_neighbor_kwargs_are_routed():
    base = _system()
    system = MolecularSystem(
        R=base.R,
        Z=base.Z,
        box=np.eye(3) * 8.0,
        mol_id=base.mol_id,
    )
    energy = HybridEnergy([_KeywordHarmonicTerm()], system, EnergyContext())
    boxes = []

    def neighbor_fn(position, box):
        boxes.append(box.copy())
        return {"scale": np.array(2.0)}

    result = JaxmdDriver(record_every=1, neighbor_fn=neighbor_fn).run(
        system,
        energy,
        EnsembleSpec(ensemble="nve", space="pbc", dt_fs=0.1, n_steps=1),
    )

    np.testing.assert_allclose(boxes[0], system.box)
    assert result.metadata["energies"][0] == pytest.approx(2.0)


def test_overlap_repair_reinitializes_at_repaired_positions():
    system = _system()
    energy = HybridEnergy([_HarmonicTerm()], system, EnergyContext())
    repaired = np.zeros_like(system.R)
    calls = []

    def repair(position, box):
        calls.append(position)
        return repaired if len(calls) == 1 else None

    result = JaxmdDriver(record_every=1).run(
        system,
        energy,
        EnsembleSpec(ensemble="min", space="free", dt_fs=0.1, n_steps=2),
        on_overlap=repair,
    )

    np.testing.assert_allclose(result.metadata["positions"][1], repaired)
    assert len(calls) == 2


@pytest.mark.parametrize(
    ("driver", "ensemble", "message"),
    [
        (JaxmdDriver(record_every=0), EnsembleSpec(), "record_every"),
        (JaxmdDriver(block_size=0), EnsembleSpec(n_steps=1), "block_size"),
        (JaxmdDriver(), EnsembleSpec(dt_fs=0), "dt_fs"),
        (JaxmdDriver(), EnsembleSpec(n_steps=-1), "n_steps"),
        (
            JaxmdDriver(),
            EnsembleSpec(n_steps=0, params={"masses": np.array([1.0, 0.0])}),
            "strictly positive",
        ),
    ],
)
def test_invalid_configuration_is_rejected(driver, ensemble, message):
    system = _system()
    energy = HybridEnergy([_HarmonicTerm()], system, EnergyContext())
    with pytest.raises(ValueError, match=message):
        driver.run(system, energy, ensemble)


def test_neighbor_and_repair_contracts_are_validated():
    system = _system()
    energy = HybridEnergy([_HarmonicTerm()], system, EnergyContext())
    ensemble = EnsembleSpec(ensemble="min", dt_fs=0.1, n_steps=1)

    with pytest.raises(TypeError, match="neighbor_fn must return a mapping"):
        JaxmdDriver(neighbor_fn=lambda position, box: None).run(system, energy, ensemble)

    with pytest.raises(ValueError, match="on_overlap returned positions with shape"):
        JaxmdDriver().run(
            system,
            energy,
            ensemble,
            on_overlap=lambda position, box: np.zeros((1, 3)),
        )


def test_npt_requires_a_box():
    system = _system()  # free space (box=None)
    energy = HybridEnergy([_HarmonicTerm()], system, EnergyContext())
    with pytest.raises(ValueError, match="npt requires a periodic box"):
        JaxmdDriver().run(system, energy, EnsembleSpec(ensemble="npt", n_steps=1))


def _periodic_system(n_side: int = 3, spacing: float = 2.5):
    grid = np.arange(n_side) * spacing
    pts = np.array([[x, y, z] for x in grid for y in grid for z in grid], dtype=float)
    box = float(n_side * spacing)
    n = len(pts)
    return MolecularSystem(
        R=pts,
        Z=np.ones(n, dtype=int),
        box=np.diag([box, box, box]),
        mol_id=np.arange(n),
    )


def test_npt_evolves_the_box():
    system = _periodic_system()
    energy = HybridEnergy([_HarmonicTerm()], system, EnergyContext())
    ensemble = EnsembleSpec(
        ensemble="npt", dt_fs=0.5, n_steps=20, temperature_K=100.0,
        pressure_bar=1.0, params={"float64": True, "seed": 1},
    )
    traj = JaxmdDriver(record_every=5).run(system, energy, ensemble)

    boxes = traj.metadata["boxes"]
    assert boxes.shape[0] == traj.n_frames
    # box stays finite and actually changes under the barostat
    assert np.all(np.isfinite(boxes))
    assert not np.allclose(boxes[0], boxes[-1])
    # positions/energies remain finite throughout
    assert np.all(np.isfinite(traj.metadata["positions"]))
    assert np.all(np.isfinite(traj.metadata["energies"]))
    assert traj.metadata["steps"] == 20
