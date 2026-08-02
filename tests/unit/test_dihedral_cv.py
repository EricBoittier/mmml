"""Unit tests for DihedralCV + periodic umbrella bias."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from mmml.md.restraints import DihedralCV, cv_from_spec, periodic_delta_deg
from mmml.umbrella.energy import (
    numpy_bias_matrix_cv,
    packed_bias_energies_cv,
    packed_bias_forces_cv,
)


def _planar_frame() -> np.ndarray:
    """Four atoms with a known dihedral near 0° (trans-ish)."""
    return np.array(
        [
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def test_cv_from_spec_dihedral():
    cv = cv_from_spec({"kind": "dihedral", "atoms": [14, 16, 18, 24]})
    assert isinstance(cv, DihedralCV)
    assert cv.atoms == (14, 16, 18, 24)
    assert cv_from_spec((0, 1, 2, 3)).atoms == (0, 1, 2, 3)


def test_periodic_delta_wraps():
    d = float(periodic_delta_deg(jnp.asarray(170.0), jnp.asarray(-170.0)))
    assert abs(d - (-20.0)) < 1e-6 or abs(d - 20.0) < 1e-6


def test_dihedral_value_and_bias_forces():
    r = _planar_frame()
    cv = DihedralCV(atoms=(0, 1, 2, 3))
    phi = cv.value_numpy(r)
    assert np.isfinite(phi)

    # Pack two windows with the same geometry
    packed = np.vstack([r, r])
    targets = (float(phi), float(phi) + 30.0)
    ks = (0.05, 0.05)  # eV/deg^2
    e = np.asarray(packed_bias_energies_cv(jnp.asarray(packed), 4, cv, targets, ks))
    assert e.shape == (2,)
    assert e[0] < 1e-8
    assert e[1] > 0.0

    f = np.asarray(packed_bias_forces_cv(jnp.asarray(packed), 4, cv, targets, ks))
    assert f.shape == (8, 3)
    assert np.all(np.isfinite(f))
    # On-target window ≈ zero bias force
    assert float(np.max(np.abs(f[:4]))) < 1e-5

    w = numpy_bias_matrix_cv(r, cv, targets, ks)
    assert w.shape == (2,)
    assert w[0] < 1e-8


def test_umbrella_config_dihedral_roundtrip():
    from mmml.umbrella.config import UmbrellaConfig

    cfg = UmbrellaConfig.from_dict(
        {
            "checkpoint": "ckpt.json",
            "structure": "seeds.npz",
            "seed_mode": "frames",
            "cv_x": {"kind": "dihedral", "atoms": [14, 16, 18, 24]},
            "xi_min": -180,
            "xi_max": 180,
            "n_windows": 5,
            "k_ev_A2": 0.05,
            "output_dir": "out",
        }
    )
    cvs = cfg.resolve_cvs()
    assert len(cvs) == 1
    assert isinstance(cvs[0], DihedralCV)
    assert cvs[0].atoms == (14, 16, 18, 24)
    sched = cfg.resolve_schedule()
    assert sched.cv_specs() == [{"kind": "dihedral", "atoms": [14, 16, 18, 24]}]
    assert abs(sched.xi0[0] + 180.0) < 1e-9
    dumped = cfg.to_dict()
    assert dumped["cv_x"]["kind"] == "dihedral"


def test_stretch_seed_rejects_dihedral():
    from mmml.umbrella.structure import pack_window_seeds

    r0 = _planar_frame()
    with pytest.raises(ValueError, match="dihedral"):
        pack_window_seeds(
            positions=r0,
            atom_pairs=((0, 1),),
            targets_per_cv=((-180.0, 0.0, 180.0),),
            seed_mode="stretch",
            cvs=[{"kind": "dihedral", "atoms": [0, 1, 2, 3]}],
        )


def test_fill_u_kln_periodic_dihedral():
    from mmml.umbrella.mbar import fill_u_kln

    # Two windows, one atomless-looking frame with known φ≈±90-ish planar setup
    r = _planar_frame()
    cv = DihedralCV(atoms=(0, 1, 2, 3))
    phi = cv.value_numpy(r)
    k, n_frames = 2, 2
    positions = np.stack([r, r], axis=0)[None, ...].repeat(k, axis=0)
    positions = np.broadcast_to(r, (k, n_frames, 4, 3)).copy()
    xi0 = [phi, phi + 30.0]
    ks = [0.1, 0.1]
    u_kln, n_k = fill_u_kln(
        positions=positions,
        atom_pairs=[cv],
        targets_per_cv=[xi0],
        k_per_cv=[ks],
        temperature_K=300.0,
        ml_energy_fn=lambda _r: 0.0,
    )
    assert u_kln.shape == (2, 2, 2)
    assert n_k.tolist() == [2, 2]
    # Self-bias for window 0 (on target) ≈ 0
    assert abs(u_kln[0, 0, 0]) < 1e-8
    assert u_kln[0, 1, 0] > u_kln[0, 0, 0]
