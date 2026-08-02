"""Seed relaxation + ML-region seed-force gate for hybrid umbrella windows.

Both exist because of the same production failure: windows are seeded by rigidly
displacing the solute inside a box packed at one ξ, and the whole-system max |F|
used to gate those seeds is pinned by raw Packmol solvent contacts, so it reads
the same in every window and never catches the strained ones.
"""

from __future__ import annotations

import numpy as np
import pytest

from mmml.umbrella.config import UmbrellaConfig
from mmml.umbrella.hybrid import relax_around_frozen_seed, seed_force_maxima


def _config(**overrides):
    data = {
        "checkpoint": "ckpt.json",
        "output_dir": "out",
        "engine": "hybrid_jaxmd",
        "from_psf": "x.psf",
        "from_pdb": "x.pdb",
        "box_size": 30.0,
        "atom_i": 0,
        "atom_j": 1,
        "xi_min": -1.0,
        "xi_max": 1.0,
        "n_windows": 3,
    }
    data.update(overrides)
    return UmbrellaConfig.from_dict(data)


def test_seed_force_maxima_separates_ml_region_from_solvent():
    """The solvent maximum must not mask a strained solute."""
    forces = np.zeros((6, 3), dtype=np.float64)
    forces[4, 0] = 37.1  # solvent contact, identical in every window
    forces[1, 2] = 5.0  # ML region
    fmax_ml, fmax_all = seed_force_maxima(forces, ml_indices=[0, 1, 2])
    assert fmax_ml == pytest.approx(5.0)
    assert fmax_all == pytest.approx(37.1)


def test_seed_force_maxima_handles_empty_ml_region():
    forces = np.ones((3, 3), dtype=np.float64)
    fmax_ml, fmax_all = seed_force_maxima(forces, ml_indices=[])
    assert np.isnan(fmax_ml)
    assert fmax_all == pytest.approx(1.0)


class _SpringCalculator:
    """Harmonic pull of every atom toward the origin, ASE calculator face."""

    implemented_properties = ("energy", "forces")

    def __init__(self, k: float = 1.0):
        self.k = float(k)
        self.atoms = None

    def get_potential_energy(self, atoms=None, force_consistent=False):
        r = np.asarray(atoms.get_positions(), dtype=np.float64)
        return 0.5 * self.k * float(np.sum(r**2))

    def get_forces(self, atoms=None):
        r = np.asarray(atoms.get_positions(), dtype=np.float64)
        return -self.k * r

    def get_stress(self, atoms=None):
        raise NotImplementedError

    def calculation_required(self, atoms, quantities):
        return True


def _atoms(positions):
    from ase import Atoms

    at = Atoms(numbers=[1] * len(positions), positions=np.asarray(positions, float))
    at.calc = _SpringCalculator()
    return at


def test_relax_around_frozen_seed_holds_the_solute_and_moves_the_rest():
    """Frozen atoms keep their seeded ξ; mobile atoms relax toward the minimum."""
    positions = np.array(
        [[3.0, 0.0, 0.0], [-3.0, 0.0, 0.0], [0.0, 4.0, 0.0]], dtype=np.float64
    )
    atoms = _atoms(positions)
    relaxed, n_steps = relax_around_frozen_seed(
        atoms, frozen_indices=[0, 1], fmax=1e-3, steps=200
    )
    assert n_steps > 0
    np.testing.assert_allclose(relaxed[:2], positions[:2], atol=1e-12)
    assert np.linalg.norm(relaxed[2]) < 1e-2


def test_relax_around_frozen_seed_clears_the_constraint():
    """A left-over FixAtoms would zero the forces the gate is about to read."""
    atoms = _atoms([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    relax_around_frozen_seed(atoms, frozen_indices=[0], fmax=1e-3, steps=50)
    assert atoms.constraints == []
    assert abs(atoms.get_forces()[0, 0]) == pytest.approx(2.0, abs=1e-9)


def test_relax_around_frozen_seed_respects_the_step_budget():
    atoms = _atoms([[8.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    _, n_steps = relax_around_frozen_seed(
        atoms, frozen_indices=[1], fmax=1e-12, steps=3
    )
    assert n_steps <= 3


def test_config_defaults_leave_seed_relaxation_off():
    cfg = _config()
    assert cfg.relax_seed_steps == 0
    assert cfg.relax_seed_fmax == pytest.approx(1.0)


def test_config_accepts_seed_relaxation_settings():
    cfg = _config(relax_seed_steps=300, relax_seed_fmax=0.5)
    assert cfg.relax_seed_steps == 300
    assert cfg.relax_seed_fmax == pytest.approx(0.5)


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"relax_seed_steps": -1}, "relax_seed_steps"),
        ({"relax_seed_fmax": 0.0}, "relax_seed_fmax"),
    ],
)
def test_config_rejects_bad_seed_relaxation_settings(overrides, match):
    with pytest.raises(ValueError, match=match):
        _config(**overrides)


def test_prod_yamls_enable_seed_relaxation():
    """The two production configs are the ones that hit the failure."""
    import yaml
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "examples" / "m" / "yaml"
    for name in ("umbrella_nc_acn_prod.yaml", "umbrella_nc_tip3_prod.yaml"):
        data = yaml.safe_load((root / name).read_text(encoding="utf-8"))
        assert int(data["relax_seed_steps"]) > 0, name
        assert float(data["relax_seed_fmax"]) > 0, name
