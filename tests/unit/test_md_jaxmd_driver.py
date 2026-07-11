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
    assert result.n_frames == 2  # initial + non-aligned final step
    assert len(rebuilds) == 4  # initial allocation + one refresh per block


def test_rejects_npt_until_dynamic_box_path_is_extracted():
    system = _system()
    energy = HybridEnergy([_HarmonicTerm()], system, EnergyContext())
    with pytest.raises(NotImplementedError, match="min, nve, and nvt"):
        JaxmdDriver().run(system, energy, EnsembleSpec(ensemble="npt", n_steps=1))
