"""Hybrid train: lr_solver="ewald" Coulomb (+ optional switched LJ) + FD checks.

Mirrors test_hybrid_nvalchemiops_pme_train.py's coverage, but the ewald path
needs no mocking -- it's pure JAX (no external library, no CUDA), so these
call the real hybrid_ewald_coulomb_energy directly.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mmml.models.hybrid_energy import HybridMMConfig, hybrid_forward

jax.config.update("jax_enable_x64", True)

SIG = jnp.array([3.5, 2.5], dtype=jnp.float64)
EPS = jnp.array([0.1, 0.05], dtype=jnp.float64)
KW = dict(mm_switch_on=8.0, mm_switch_width=5.0, ml_switch_width=1.5)


def _fake_model_apply(
    params,
    *,
    atomic_numbers,
    positions,
    dst_idx,
    src_idx,
    batch_segments,
    batch_size,
    batch_mask,
    atom_mask,
):
    def energy_fn(pos):
        e_atom = -1.0 * jnp.sum(pos * pos, axis=-1) * atom_mask
        e_per = jax.ops.segment_sum(e_atom, batch_segments, num_segments=batch_size)
        return jnp.sum(e_per), e_per

    (_, e_per), grad = jax.value_and_grad(energy_fn, has_aux=True)(positions)
    return {"energy": e_per.reshape(batch_size, 1), "forces": -grad}


def _dimer_batch(sep: float = 6.0):
    pos = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [sep, 0.0, 0.0],
            [sep + 1.0, 0.0, 0.0],
        ],
        dtype=jnp.float64,
    )
    mid = jnp.array([0, 0, 1, 1])
    tidx = jnp.array([0, 1, 0, 1])
    chg = jnp.array([0.5, -0.5, -0.5, 0.5], dtype=jnp.float64)
    n = 4
    atom_mask = jnp.ones(n, dtype=jnp.float64)
    idx = jnp.arange(n)
    dst, src = jnp.meshgrid(idx, idx, indexing="ij")
    dst, src = dst.reshape(-1), src.reshape(-1)
    keep = (dst != src).astype(jnp.float64)
    return {
        "R": pos,
        "Z": jnp.array([6, 1, 6, 1]),
        "mol_id": mid.reshape(1, n),
        "cgenff_type_idx": tidx.reshape(1, n),
        "cgenff_charge": chg.reshape(1, n),
        "atom_mask": atom_mask,
        "batch_mask": keep,
        "dst_idx": dst,
        "src_idx": src,
        "batch_segments": jnp.zeros(n, dtype=jnp.int32),
    }


def test_hybrid_mm_config_ewald_honors_include_lj_and_requires_box():
    with pytest.raises(ValueError, match="pme_box_length"):
        HybridMMConfig.coerce(
            {"master_sigmas": SIG, "master_epsilons": EPS, **KW, "lr_solver": "ewald"}
        )
    cfg = HybridMMConfig.coerce(
        {
            "master_sigmas": SIG, "master_epsilons": EPS, **KW,
            "lr_solver": "ewald", "pme_box_length": 30.0, "include_lj": True,
            "learn_mm_lj_scales": True,  # still forced off under ewald
        }
    )
    assert cfg.lr_solver == "ewald"
    assert cfg.include_lj is True
    assert cfg.learn_mm_lj_scales is False
    assert cfg.pme_box_length == pytest.approx(30.0)

    cfg_off = HybridMMConfig.coerce(
        {
            "master_sigmas": SIG, "master_epsilons": EPS, **KW,
            "lr_solver": "ewald", "pme_box_length": 30.0, "include_lj": False,
        }
    )
    assert cfg_off.include_lj is False


def test_build_hybrid_mm_config_cli_ewald(tmp_path):
    from mmml.cli.make.make_training import _build_hybrid_mm_config

    path = tmp_path / "d.npz"
    np.savez(
        path,
        cgenff_type_idx=np.zeros((2, 4), dtype=np.int32),
        mol_id=np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.int32),
        cgenff_charge=np.ones((2, 4)),
        cgenff_master_sigmas=np.array([3.5, 2.5]),
        cgenff_master_epsilons=np.array([0.1, 0.05]),
        N=np.array([4, 4]),
    )
    args = SimpleNamespace(
        hybrid_mm=True, mm_charge_mode="fixed", mm_charge_correction=False,
        charges=False, quiet=True, mm_switch_on=8.0, mm_switch_width=5.0,
        ml_switch_width=1.5, no_complementary_handoff=False,
        lr_solver="ewald", pme_box_length=None, pme_accuracy=1e-4, mm_include_lj=True,
        learn_mm_lj_scales=True,
        mm_lj_sigma_scale_min=0.95, mm_lj_sigma_scale_max=1.05,
        mm_lj_epsilon_scale_min=0.25, mm_lj_epsilon_scale_max=4.0,
        mm_lj_min_type_frames=0,
        hybrid_hamiltonian="handoff", shared_cutoff=None, cutoff=6.0,
    )
    with pytest.raises(ValueError, match="pme-box-length"):
        _build_hybrid_mm_config(args, [str(path)])

    args.pme_box_length = 28.0
    cfg = _build_hybrid_mm_config(args, [str(path)])
    assert cfg["lr_solver"] == "ewald"
    assert cfg["include_lj"] is True
    assert cfg["learn_mm_lj_scales"] is False
    assert cfg["pme_box_length"] == pytest.approx(28.0)
    assert cfg["pme_real_space_cutoff"] is None  # no estimation step for ewald


def test_hybrid_ewald_path_fd_force_energy():
    """Wiring + COM scale: F = -dE/dR with the real (unmocked) ewald Coulomb."""
    batch = _dimer_batch(sep=6.0)

    def _fwd(b):
        return hybrid_forward(
            _fake_model_apply, {}, b, 1, SIG, EPS, **KW,
            complementary_handoff=True, mm_charge_mode="fixed", short_range_wall=False,
            lr_solver="ewald", include_lj=False,
            pme_box_length=40.0, pme_accuracy=1e-6, pme_real_space_cutoff=10.0,
        )

    out = _fwd(batch)
    e0 = float(np.asarray(out["energy"]).reshape(-1)[0])
    f0 = np.asarray(out["forces"]).reshape(-1, 3)

    d = np.zeros_like(f0)
    d[0, 0] = 1.0
    d[2, 0] = -1.0
    d = d / np.linalg.norm(d)
    eps = 1e-4
    pos = np.asarray(batch["R"], dtype=np.float64)

    def energy_at(p):
        b = dict(batch)
        b["R"] = jnp.asarray(p)
        return float(np.asarray(_fwd(b)["energy"]).reshape(-1)[0])

    e_plus = energy_at(pos + eps * d)
    e_minus = energy_at(pos - eps * d)
    fd = (e_plus - e_minus) / (2.0 * eps)
    analytic = float(-np.sum(f0 * d))
    rel = abs(fd - analytic) / max(abs(analytic), 1e-8)
    assert rel < 5e-3, f"fd={fd}, -F.d={analytic}, rel={rel}, E={e0}"


def test_ewald_path_independent_of_lj_tables():
    batch = _dimer_batch(sep=7.0)
    eps_hi = jnp.array([10.0, 10.0], dtype=jnp.float64)

    e0 = hybrid_forward(
        _fake_model_apply, {}, batch, 1, SIG, EPS, **KW, short_range_wall=False,
        lr_solver="ewald", include_lj=False, pme_box_length=40.0, pme_real_space_cutoff=10.0,
    )["energy"]
    e1 = hybrid_forward(
        _fake_model_apply, {}, batch, 1, SIG, eps_hi, **KW, short_range_wall=False,
        lr_solver="ewald", include_lj=False, pme_box_length=40.0, pme_real_space_cutoff=10.0,
    )["energy"]
    np.testing.assert_allclose(np.asarray(e0), np.asarray(e1), rtol=0, atol=0)


def test_full_box_ewald_keeps_intra_monomer_coulomb():
    """Full-box many-to-many: single monomer still contributes E_ewald (no subtract)."""
    from mmml.models.ewald_hybrid_coulomb import hybrid_ewald_coulomb_energy

    pos = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float64)
    mid = jnp.array([0, 0])
    q = jnp.array([0.5, -0.5], dtype=jnp.float64)

    e = hybrid_ewald_coulomb_energy(
        pos, mid, q, box_length_A=40.0, real_space_cutoff_A=10.0, **KW,
    )
    assert float(e) != 0.0  # not subtracted away like the mic hybrid path would


def test_cross_monomer_ewald_removes_single_monomer_coulomb():
    """MIC-trained compatibility mode must not double-count monomer Coulomb."""
    from mmml.models.ewald_hybrid_coulomb import hybrid_ewald_coulomb_energy

    pos = jnp.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0]], dtype=jnp.float64)
    energy = hybrid_ewald_coulomb_energy(
        pos,
        jnp.array([0, 0]),
        jnp.array([-0.8, 0.8], dtype=jnp.float64),
        box_length_A=20.0,
        n_monomers=1,
        include_self_energy=False,
        include_intramolecular=False,
    )

    assert float(energy) == pytest.approx(0.0, abs=1.0e-5)


def test_ewald_omit_self_drops_geometry_independent_offset():
    from mmml.models.ewald_hybrid_coulomb import hybrid_ewald_coulomb_energy
    from mmml.interfaces.pycharmmInterface.ewald_native import (
        default_ewald_alpha,
        ewald_self_energy,
    )

    pos = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=jnp.float64)
    mid = jnp.array([0, 1])
    q = jnp.array([1.0, -1.0], dtype=jnp.float64)
    e_full = float(
        hybrid_ewald_coulomb_energy(
            pos, mid, q, box_length_A=40.0, real_space_cutoff_A=10.0, include_self_energy=True
        )
    )
    e_noself = float(
        hybrid_ewald_coulomb_energy(
            pos, mid, q, box_length_A=40.0, real_space_cutoff_A=10.0, include_self_energy=False
        )
    )
    import math

    from mmml.models.ewald_hybrid_coulomb import COULOMB_KCAL

    alpha = default_ewald_alpha(10.0, accuracy_exponent=math.sqrt(max(-math.log(1e-6), 1.0)))
    e_self = float(ewald_self_energy(q, alpha) * COULOMB_KCAL)
    assert e_full == pytest.approx(e_noself + e_self, rel=0, abs=1e-8)
    assert e_noself != pytest.approx(e_full, abs=1e-6)


def test_full_box_ewald_ignores_com_switch_kwargs():
    """COM MM taper is not applied (matches untapered MD many-to-many Ewald)."""
    from mmml.models.ewald_hybrid_coulomb import hybrid_ewald_coulomb_energy

    pos = jnp.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [20.0, 0.0, 0.0], [21.0, 0.0, 0.0]],
        dtype=jnp.float64,
    )
    mid = jnp.array([0, 0, 1, 1])
    q = jnp.array([0.5, -0.5, -0.5, 0.5], dtype=jnp.float64)

    e_near = hybrid_ewald_coulomb_energy(
        pos, mid, q, box_length_A=40.0, real_space_cutoff_A=10.0,
        mm_switch_on=8.0, mm_switch_width=5.0, ml_switch_width=1.5, complementary_handoff=True,
    )
    e_far_kwargs = hybrid_ewald_coulomb_energy(
        pos, mid, q, box_length_A=40.0, real_space_cutoff_A=10.0,
        mm_switch_on=100.0, mm_switch_width=1.0, ml_switch_width=1.0, complementary_handoff=False,
    )
    assert float(e_near) == pytest.approx(float(e_far_kwargs), rel=0, abs=0)


def test_cli_exposes_ewald_lr_solver_choice():
    import inspect

    from mmml.cli.make import make_training

    src = inspect.getsource(make_training)
    assert '"--lr-solver"' in src
    assert '"ewald"' in src


def test_ewald_energy_jittable_with_grad():
    """Train path must survive jit + value_and_grad -- the whole point."""
    from mmml.models.ewald_hybrid_coulomb import hybrid_ewald_coulomb_energy

    pos0 = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=jnp.float64)
    mid = jnp.array([0, 1])
    chg = jnp.array([1.0, -1.0], dtype=jnp.float64)

    def e_fn(p):
        return hybrid_ewald_coulomb_energy(
            p, mid, chg, box_length_A=40.0, accuracy=1e-6, real_space_cutoff_A=10.0,
        )

    e, g = jax.jit(jax.value_and_grad(e_fn))(pos0)
    assert bool(jnp.all(jnp.isfinite(g)))

    d = jnp.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]], dtype=jnp.float64)
    d = d / jnp.linalg.norm(d)
    eps = 1e-5
    fd = (e_fn(pos0 + eps * d) - e_fn(pos0 - eps * d)) / (2.0 * eps)
    analytic = jnp.sum(g * d)
    rel = abs(float(fd - analytic)) / max(abs(float(analytic)), 1e-12)
    assert rel < 1e-5, f"fd={fd}, dE={analytic}, rel={rel}, E={e}"
