"""Hybrid train: nvalchemiops_pme Coulomb (LJ off) + force–energy consistency."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

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


def test_hybrid_mm_config_nvalchemiops_forces_lj_off_and_requires_box():
    with pytest.raises(ValueError, match="pme_box_length"):
        HybridMMConfig.coerce(
            {
                "master_sigmas": SIG,
                "master_epsilons": EPS,
                **KW,
                "lr_solver": "nvalchemiops_pme",
            }
        )
    cfg = HybridMMConfig.coerce(
        {
            "master_sigmas": SIG,
            "master_epsilons": EPS,
            **KW,
            "lr_solver": "nvalchemiops_pme",
            "pme_box_length": 30.0,
            "include_lj": True,  # forced off
        }
    )
    assert cfg.lr_solver == "nvalchemiops_pme"
    assert cfg.include_lj is False
    assert cfg.pme_box_length == pytest.approx(30.0)


def test_build_hybrid_mm_config_cli_nvalchemiops(tmp_path):
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
        hybrid_mm=True,
        mm_charge_mode="fixed",
        mm_charge_correction=False,
        charges=False,
        quiet=True,
        mm_switch_on=8.0,
        mm_switch_width=5.0,
        ml_switch_width=1.5,
        no_complementary_handoff=False,
        lr_solver="nvalchemiops_pme",
        pme_box_length=None,
        pme_accuracy=1e-4,
        mm_include_lj=True,
    )
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.long_range_backend.have_nvalchemiops_pme",
        return_value=True,
    ):
        with pytest.raises(ValueError, match="pme-box-length"):
            _build_hybrid_mm_config(args, [str(path)])

    args.pme_box_length = 28.0
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.long_range_backend.have_nvalchemiops_pme",
        return_value=True,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.long_range_backend.estimate_nvalchemiops_pme_real_space_cutoff",
        return_value=9.0,
    ):
        cfg = _build_hybrid_mm_config(args, [str(path)])
    assert cfg["lr_solver"] == "nvalchemiops_pme"
    assert cfg["include_lj"] is False
    assert cfg["pme_box_length"] == pytest.approx(28.0)
    assert cfg["pme_real_space_cutoff"] == pytest.approx(9.0)


def test_build_hybrid_mm_config_requires_package(tmp_path):
    from mmml.cli.make.make_training import _build_hybrid_mm_config

    path = tmp_path / "d.npz"
    np.savez(
        path,
        cgenff_type_idx=np.zeros((1, 2), dtype=np.int32),
        mol_id=np.array([[0, 1]], dtype=np.int32),
        cgenff_charge=np.array([[0.5, -0.5]]),
        cgenff_master_sigmas=np.array([3.5]),
        cgenff_master_epsilons=np.array([0.1]),
        N=np.array([2]),
    )
    args = SimpleNamespace(
        hybrid_mm=True,
        mm_charge_mode="fixed",
        mm_charge_correction=False,
        charges=False,
        quiet=True,
        mm_switch_on=8.0,
        mm_switch_width=5.0,
        ml_switch_width=1.5,
        no_complementary_handoff=False,
        lr_solver="nvalchemiops_pme",
        pme_box_length=30.0,
        pme_accuracy=1e-6,
        mm_include_lj=True,
    )
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.long_range_backend.have_nvalchemiops_pme",
        return_value=False,
    ):
        with pytest.raises(ValueError, match="nvalchemiops"):
            _build_hybrid_mm_config(args, [str(path)])


def _analytic_pair_coulomb_kcal(pos, charges, **kwargs):
    """Smooth stand-in for PME: sum_{i<j} k q_i q_j / r (kcal/mol)."""
    from mmml.interfaces.pycharmmInterface.long_range_backend import CHARMM_COULOMB_KCAL

    q = jnp.asarray(charges).reshape(-1)
    n = pos.shape[0]
    iu, ju = jnp.triu_indices(n, k=1)
    d = pos[iu] - pos[ju]
    r = jnp.sqrt(jnp.maximum(jnp.sum(d * d, axis=-1), 1e-12))
    return CHARMM_COULOMB_KCAL * jnp.sum(q[iu] * q[ju] / r)


def test_hybrid_nvalchemiops_path_fd_force_energy_with_analytic_stub():
    """Wiring + COM scale: F = -dE/dR when PME kernel is a smooth stub."""
    batch = _dimer_batch(sep=6.0)

    def _fwd():
        return hybrid_forward(
            _fake_model_apply,
            {},
            batch,
            1,
            SIG,
            EPS,
            **KW,
            complementary_handoff=True,
            mm_charge_mode="fixed",
            short_range_wall=False,
            lr_solver="nvalchemiops_pme",
            include_lj=False,
            pme_box_length=40.0,
            pme_accuracy=1e-4,
            pme_real_space_cutoff=10.0,
        )

    with mock.patch(
        "mmml.models.nvalchemiops_hybrid_coulomb.nvalchemiops_pme_coulomb_energy_jax",
        side_effect=_analytic_pair_coulomb_kcal,
    ):
        out = _fwd()
        e0 = float(np.asarray(out["energy"]).reshape(-1)[0])
        f0 = np.asarray(out["forces"]).reshape(-1, 3)

        # Directional FD on a soft unit direction.
        d = np.zeros_like(f0)
        d[0, 0] = 1.0
        d[2, 0] = -1.0
        d = d / np.linalg.norm(d)
        eps = 1e-4
        pos = np.asarray(batch["R"], dtype=np.float64)

        def energy_at(p):
            b = dict(batch)
            b["R"] = jnp.asarray(p)
            with mock.patch(
                "mmml.models.nvalchemiops_hybrid_coulomb.nvalchemiops_pme_coulomb_energy_jax",
                side_effect=_analytic_pair_coulomb_kcal,
            ):
                return float(
                    np.asarray(
                        hybrid_forward(
                            _fake_model_apply,
                            {},
                            b,
                            1,
                            SIG,
                            EPS,
                            **KW,
                            complementary_handoff=True,
                            mm_charge_mode="fixed",
                            short_range_wall=False,
                            lr_solver="nvalchemiops_pme",
                            include_lj=False,
                            pme_box_length=40.0,
                            pme_accuracy=1e-4,
                            pme_real_space_cutoff=10.0,
                        )["energy"]
                    ).reshape(-1)[0]
                )

        e_plus = energy_at(pos + eps * d)
        e_minus = energy_at(pos - eps * d)
        fd = (e_plus - e_minus) / (2.0 * eps)
        analytic = float(-np.sum(f0 * d))
        rel = abs(fd - analytic) / max(abs(analytic), 1e-8)
        assert rel < 5e-3, f"fd={fd}, -F·d={analytic}, rel={rel}, E={e0}"


def test_nvalchemiops_path_independent_of_lj_tables():
    batch = _dimer_batch(sep=7.0)
    eps_hi = jnp.array([10.0, 10.0], dtype=jnp.float64)

    with mock.patch(
        "mmml.models.nvalchemiops_hybrid_coulomb.nvalchemiops_pme_coulomb_energy_jax",
        side_effect=_analytic_pair_coulomb_kcal,
    ):
        e0 = hybrid_forward(
            _fake_model_apply,
            {},
            batch,
            1,
            SIG,
            EPS,
            **KW,
            short_range_wall=False,
            lr_solver="nvalchemiops_pme",
            include_lj=False,
            pme_box_length=40.0,
            pme_real_space_cutoff=10.0,
        )["energy"]
        e1 = hybrid_forward(
            _fake_model_apply,
            {},
            batch,
            1,
            SIG,
            eps_hi,
            **KW,
            short_range_wall=False,
            lr_solver="nvalchemiops_pme",
            include_lj=False,
            pme_box_length=40.0,
            pme_real_space_cutoff=10.0,
        )["energy"]
    np.testing.assert_allclose(np.asarray(e0), np.asarray(e1), rtol=0, atol=0)


def _nval_pme_runtime_ok() -> bool:
    from mmml.interfaces.pycharmmInterface.long_range_backend import (
        have_nvalchemiops_pme,
        nvalchemiops_pme_coulomb_energy_jax,
    )

    if not have_nvalchemiops_pme():
        return False
    try:
        pos = jnp.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=jnp.float64)
        chg = jnp.array([1.0, -1.0], dtype=jnp.float64)
        e = nvalchemiops_pme_coulomb_energy_jax(
            pos,
            chg,
            box_length_A=40.0,
            accuracy=1e-4,
            real_space_cutoff_A=12.0,
            compute_forces=False,
        )
        float(np.asarray(e))
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _nval_pme_runtime_ok(), reason="nvalchemiops PME runtime unavailable")
def test_real_nvalchemiops_pme_energy_fd_conserves():
    from mmml.interfaces.pycharmmInterface.long_range_backend import (
        nvalchemiops_pme_coulomb_energy_jax,
    )

    pos0 = jnp.array([[0.0, 0.0, 0.0], [5.0, 0.2, -0.1]], dtype=jnp.float64)
    chg = jnp.array([1.0, -1.0], dtype=jnp.float64)
    box = 40.0
    cut = 12.0
    acc = 1e-4

    def e_fn(p):
        return nvalchemiops_pme_coulomb_energy_jax(
            p,
            chg,
            box_length_A=box,
            accuracy=acc,
            real_space_cutoff_A=cut,
            compute_forces=False,
        )

    e0, g = jax.value_and_grad(e_fn)(pos0)
    f = -g
    d = jnp.array([[0.3, -0.1, 0.2], [-0.3, 0.1, -0.2]], dtype=jnp.float64)
    d = d / jnp.linalg.norm(d)
    eps = 1e-4
    fd = (e_fn(pos0 + eps * d) - e_fn(pos0 - eps * d)) / (2.0 * eps)
    analytic = jnp.sum(-f * d)
    rel = abs(float(fd - analytic)) / max(abs(float(analytic)), 1e-8)
    assert rel < 2e-2, f"fd={fd}, -F·d={analytic}, rel={rel}, E={e0}"


def test_full_box_pme_keeps_intra_monomer_coulomb():
    """Full-box many-to-many: single monomer still contributes E_PME (no subtract)."""
    from mmml.models.nvalchemiops_hybrid_coulomb import (
        hybrid_nvalchemiops_pme_coulomb_energy,
    )

    pos = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float64)
    mid = jnp.array([0, 0])
    q = jnp.array([0.5, -0.5], dtype=jnp.float64)

    with mock.patch(
        "mmml.models.nvalchemiops_hybrid_coulomb.nvalchemiops_pme_coulomb_energy_jax",
        side_effect=_analytic_pair_coulomb_kcal,
    ):
        e = hybrid_nvalchemiops_pme_coulomb_energy(
            pos,
            mid,
            q,
            box_length_A=40.0,
            real_space_cutoff_A=10.0,
            **KW,
        )
    # Analytic pair Coulomb for this dimer: k * (0.5)*(-0.5) / 1.0
    from mmml.interfaces.pycharmmInterface.long_range_backend import CHARMM_COULOMB_KCAL

    assert float(e) == pytest.approx(CHARMM_COULOMB_KCAL * (-0.25), rel=1e-10)


def test_full_box_pme_ignores_com_switch_kwargs():
    """COM MM taper is not applied (matches untapered MD many-to-many Ewald)."""
    from mmml.models.nvalchemiops_hybrid_coulomb import (
        hybrid_nvalchemiops_pme_coulomb_energy,
    )

    pos = jnp.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [20.0, 0.0, 0.0], [21.0, 0.0, 0.0]],
        dtype=jnp.float64,
    )
    mid = jnp.array([0, 0, 1, 1])
    q = jnp.array([0.5, -0.5, -0.5, 0.5], dtype=jnp.float64)

    with mock.patch(
        "mmml.models.nvalchemiops_hybrid_coulomb.nvalchemiops_pme_coulomb_energy_jax",
        side_effect=_analytic_pair_coulomb_kcal,
    ) as pme:
        e_near = hybrid_nvalchemiops_pme_coulomb_energy(
            pos,
            mid,
            q,
            box_length_A=40.0,
            real_space_cutoff_A=10.0,
            mm_switch_on=8.0,
            mm_switch_width=5.0,
            ml_switch_width=1.5,
            complementary_handoff=True,
        )
        e_far_kwargs = hybrid_nvalchemiops_pme_coulomb_energy(
            pos,
            mid,
            q,
            box_length_A=40.0,
            real_space_cutoff_A=10.0,
            mm_switch_on=100.0,
            mm_switch_width=1.0,
            ml_switch_width=1.0,
            complementary_handoff=False,
        )
    assert float(e_near) == pytest.approx(float(e_far_kwargs), rel=0, abs=0)
    assert pme.call_count == 2


def test_cli_exposes_lr_solver_flags():
    import inspect

    from mmml.cli.make import make_training

    src = inspect.getsource(make_training)
    assert '"--lr-solver"' in src
    assert '"--pme-box-length"' in src
    assert "nvalchemiops_pme" in src
