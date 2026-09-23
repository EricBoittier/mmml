"""JIT safety for sparse ML dimer COM distances under PBC."""

from __future__ import annotations

import os
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters


def test_sparse_dimer_jit_with_traced_box() -> None:
    """Sparse dimer filtering must not Python-branch on traced cell values."""
    from mmml.interfaces.pycharmmInterface.mmml_calculator import setup_calculator

    n_mono = 5
    n_monomers = 8
    n_atoms = n_mono * n_monomers
    z = jnp.full((n_atoms,), 6, dtype=jnp.int32)
    rng = np.random.default_rng(0)
    r0 = rng.normal(size=(n_atoms, 3))
    box = 27.0
    fake_mm_fn = lambda *args, **kwargs: (
        jnp.array(0.0, dtype=jnp.float32),
        jnp.zeros((n_atoms, 3), dtype=jnp.float32),
    )
    fake_update_fn = lambda *args, **kwargs: (
        jnp.zeros((1, 2), dtype=jnp.int32),
        jnp.ones((1,), dtype=bool),
    )

    def fake_build_mm(*args, **kwargs):
        if kwargs.get("use_jax_md_neighbor_list", True):
            return fake_mm_fn, fake_update_fn
        return fake_mm_fn

    with patch(
        "mmml.interfaces.pycharmmInterface.mmml_calculator.build_mm_energy_forces_fn",
        side_effect=fake_build_mm,
    ):
        factory = setup_calculator(
            ATOMS_PER_MONOMER=n_mono,
            N_MONOMERS=n_monomers,
            model_restart_path=None,
            ml_potential_mode="jax_mm_clone",
            doML=True,
            doMM=False,
            doML_dimer=True,
            MAX_ATOMS_PER_SYSTEM=10,
            cell=box,
            defer_xla_gpu_warmup=True,
            verbose=False,
            ml_sparse_dimers=True,
            ml_max_active_dimers=28,
        )
        _, spherical_fn, _ = factory(
            atomic_numbers=z,
            atomic_positions=jnp.asarray(r0),
            n_monomers=n_monomers,
            cutoff_params=CutoffParameters(mm_switch_on=12.0),
            doML=True,
            doMM=False,
            doML_dimer=True,
            backprop=False,
            create_ase_calculator=False,
        )
        box_mat = jnp.asarray(
            [[box, 0.0, 0.0], [0.0, box, 0.0], [0.0, 0.0, box]],
            dtype=jnp.float64,
        )
        out = spherical_fn(
            atomic_numbers=z,
            positions=jnp.asarray(r0),
            n_monomers=n_monomers,
            cutoff_params=CutoffParameters(mm_switch_on=12.0),
            doML=True,
            doMM=False,
            doML_dimer=True,
            box=box_mat,
        )
    assert bool(jnp.isfinite(out.energy))
    assert out.forces.shape == (n_atoms, 3)
    assert bool(jnp.all(jnp.isfinite(out.forces)))


@pytest.mark.parametrize("close_pair", ["first", "last"])
def test_sparse_dimer_padded_slots_add_no_forces(close_pair: str) -> None:
    """Sparse and dense dimer paths give the same forces when the cap has spare slots.

    Unused sparse slots carry index ``n_dimers``; a raw gather clamps it to the
    last pair, so a close last pair used to get its switched force added once
    per spare slot.
    """
    from mmml.interfaces.pycharmmInterface.mmml_calculator import setup_calculator

    n_mono, n_monomers, box = 5, 6, 40.0
    n_atoms = n_mono * n_monomers
    z = jnp.full((n_atoms,), 6, dtype=jnp.int32)
    rng = np.random.default_rng(0)
    base = rng.normal(scale=0.7, size=(n_mono, 3))
    far = [[2, 2, 2], [20, 2, 2], [2, 20, 2], [2, 2, 20]]
    if close_pair == "last":  # pair (4, 5) about 4 Å apart, all others > 6 Å
        centers = far + [[20, 20, 20], [24, 20, 20]]
    else:  # pair (0, 1) about 3.5 Å apart
        centers = [[2, 2, 2], [5.5, 2, 2], [2, 20, 2], [2, 2, 20], [20, 20, 20], [20, 20, 2]]
    r0 = jnp.asarray(
        np.concatenate([np.asarray(c, float) + base + rng.normal(scale=0.05, size=base.shape) for c in centers])
    )
    fake_mm_fn = lambda *a, **k: (jnp.array(0.0), jnp.zeros((n_atoms, 3)))
    fake_update_fn = lambda *a, **k: (jnp.zeros((1, 2), dtype=jnp.int32), jnp.ones((1,), dtype=bool))

    def fake_build_mm(*args, **kwargs):
        if kwargs.get("use_jax_md_neighbor_list", True):
            return fake_mm_fn, fake_update_fn
        return fake_mm_fn

    cp = CutoffParameters(mm_switch_on=6.0, ml_switch_width=1.5)

    def evaluate(sparse: bool, positions=None):
        pos = r0 if positions is None else jnp.asarray(positions)
        with patch(
            "mmml.interfaces.pycharmmInterface.mmml_calculator.build_mm_energy_forces_fn",
            side_effect=fake_build_mm,
        ):
            factory = setup_calculator(
                ATOMS_PER_MONOMER=n_mono,
                N_MONOMERS=n_monomers,
                model_restart_path=None,
                ml_potential_mode="jax_mm_clone",
                doML=True,
                doMM=False,
                doML_dimer=True,
                MAX_ATOMS_PER_SYSTEM=10,
                cell=box,
                defer_xla_gpu_warmup=True,
                verbose=False,
                ml_sparse_dimers=sparse,
                ml_max_active_dimers=6 if sparse else None,
            )
            _, spherical_fn, _ = factory(
                atomic_numbers=z,
                atomic_positions=r0,
                n_monomers=n_monomers,
                cutoff_params=cp,
                doML=True,
                doMM=False,
                doML_dimer=True,
                backprop=False,
                create_ase_calculator=False,
            )
            return spherical_fn(
                atomic_numbers=z,
                positions=pos,
                n_monomers=n_monomers,
                cutoff_params=cp,
                doML=True,
                doMM=False,
                doML_dimer=True,
                box=jnp.eye(3) * box,
            )

    dense, sparse = evaluate(False), evaluate(True)
    np.testing.assert_allclose(float(sparse.energy), float(dense.energy), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(sparse.forces), np.asarray(dense.forces), atol=1e-3)

    # Energy stencil on the close pair: sparse and dense must agree. jax_mm_clone
    # is a bonded spoof (not a clean FD target for analytic F); the switch
    # product rule is FD-checked in test_ml_switch_support and
    # test_ase_fragment_hybrid. PBC sparse trajectories recorded before #252
    # (3e8c9478c), including ETOH student NVE with ml_sparse_dimers True,
    # should be revalidated. That bug inflated forces only on the two
    # highest-index molecules when that pair was inside mm_switch_on; it does
    # not explain every observed NVE instability.
    atom = (n_monomers - 1) * n_mono if close_pair == "last" else 0
    h = 1.0e-3
    pos = np.asarray(r0, dtype=np.float64)
    plus, minus = pos.copy(), pos.copy()
    plus[atom, 0] += h
    minus[atom, 0] -= h
    fd_dense = -(float(evaluate(False, plus).energy) - float(evaluate(False, minus).energy)) / (2.0 * h)
    fd_sparse = -(float(evaluate(True, plus).energy) - float(evaluate(True, minus).energy)) / (2.0 * h)
    assert fd_sparse == pytest.approx(fd_dense, rel=2e-2, abs=5e-2)


def test_sparse_dimer_cap_overflow_fails_closed() -> None:
    """Two in-range pairs and cap=1 must abort, not drop a dimer."""
    from mmml.interfaces.pycharmmInterface.mlpot.mlpot_sparse_dimer_policy import (
        SparseDimerCapOverflow,
    )
    from mmml.interfaces.pycharmmInterface.mmml_calculator import setup_calculator

    n_mono, n_monomers, box = 5, 3, 40.0
    n_atoms = n_mono * n_monomers
    z = jnp.full((n_atoms,), 6, dtype=jnp.int32)
    rng = np.random.default_rng(1)
    base = rng.normal(scale=0.2, size=(n_mono, 3))
    centers = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.5, 2.5, 0.0]]
    r0 = jnp.asarray(
        np.concatenate([np.asarray(c) + base for c in centers])
    )
    fake_mm_fn = lambda *a, **k: (jnp.array(0.0), jnp.zeros((n_atoms, 3)))
    fake_update_fn = lambda *a, **k: (
        jnp.zeros((1, 2), dtype=jnp.int32),
        jnp.ones((1,), dtype=bool),
    )

    def fake_build_mm(*args, **kwargs):
        if kwargs.get("use_jax_md_neighbor_list", True):
            return fake_mm_fn, fake_update_fn
        return fake_mm_fn

    cp = CutoffParameters(mm_switch_on=6.0, ml_switch_width=1.5)
    with patch(
        "mmml.interfaces.pycharmmInterface.mmml_calculator.build_mm_energy_forces_fn",
        side_effect=fake_build_mm,
    ):
        factory = setup_calculator(
            ATOMS_PER_MONOMER=n_mono,
            N_MONOMERS=n_monomers,
            model_restart_path=None,
            ml_potential_mode="jax_mm_clone",
            doML=True,
            doMM=False,
            doML_dimer=True,
            MAX_ATOMS_PER_SYSTEM=10,
            cell=box,
            defer_xla_gpu_warmup=True,
            verbose=False,
            ml_sparse_dimers=True,
            ml_max_active_dimers=1,
        )
        _, spherical_fn, _ = factory(
            atomic_numbers=z,
            atomic_positions=r0,
            n_monomers=n_monomers,
            cutoff_params=cp,
            doML=True,
            doMM=False,
            doML_dimer=True,
            backprop=False,
            create_ase_calculator=False,
        )
        with pytest.raises((SparseDimerCapOverflow, Exception), match="cap saturated"):
            spherical_fn(
                atomic_numbers=z,
                positions=r0,
                n_monomers=n_monomers,
                cutoff_params=cp,
                doML=True,
                doMM=False,
                doML_dimer=True,
                box=jnp.eye(3) * box,
            )

def test_dimer_active_margin_does_not_change_energy_or_forces() -> None:
    """Pairs past mm_switch_on have zero switched weight, so the old 1.5 A margin is dead work.

    Centroid distances cover the full-ML region, the switching band (4.5-6.0 A)
    and the old margin band (6.0-7.5 A); energy and forces must match to float
    precision with the margin at 0 and at ml_switch_width.
    """
    from mmml.interfaces.pycharmmInterface.mmml_calculator import setup_calculator

    n_mono, box = 5, 40.0
    rng = np.random.default_rng(1)
    base = rng.normal(scale=0.7, size=(n_mono, 3))
    # chain along x: consecutive separations 4.0, 5.2, 5.8, 6.3, 7.1, 5.5 A; far copies elsewhere
    xs = np.cumsum([2.0, 4.0, 5.2, 5.8, 6.3, 7.1, 5.5])
    centers = [[x, 2.0, 2.0] for x in xs] + [[20.0, 20.0, 20.0]]
    n_monomers = len(centers)
    n_atoms = n_mono * n_monomers
    z = jnp.full((n_atoms,), 6, dtype=jnp.int32)
    r0 = jnp.asarray(
        np.concatenate([np.asarray(c, float) + base + rng.normal(scale=0.05, size=base.shape) for c in centers])
    )
    fake_mm_fn = lambda *a, **k: (jnp.array(0.0), jnp.zeros((n_atoms, 3)))
    fake_update_fn = lambda *a, **k: (jnp.zeros((1, 2), dtype=jnp.int32), jnp.ones((1,), dtype=bool))

    def fake_build_mm(*args, **kwargs):
        if kwargs.get("use_jax_md_neighbor_list", True):
            return fake_mm_fn, fake_update_fn
        return fake_mm_fn

    cp = CutoffParameters(mm_switch_on=6.0, ml_switch_width=1.5)

    def evaluate(margin: float):
        with patch(
            "mmml.interfaces.pycharmmInterface.mmml_calculator.build_mm_energy_forces_fn",
            side_effect=fake_build_mm,
        ):
            factory = setup_calculator(
                ATOMS_PER_MONOMER=n_mono,
                N_MONOMERS=n_monomers,
                model_restart_path=None,
                ml_potential_mode="jax_mm_clone",
                doML=True,
                doMM=False,
                doML_dimer=True,
                MAX_ATOMS_PER_SYSTEM=10,
                cell=box,
                defer_xla_gpu_warmup=True,
                verbose=False,
                ml_sparse_dimers=True,
                ml_max_active_dimers=12,
                ml_dimer_active_margin=margin,
            )
            _, spherical_fn, _ = factory(
                atomic_numbers=z,
                atomic_positions=r0,
                n_monomers=n_monomers,
                cutoff_params=cp,
                doML=True,
                doMM=False,
                doML_dimer=True,
                backprop=False,
                create_ase_calculator=False,
            )
            return spherical_fn(
                atomic_numbers=z,
                positions=r0,
                n_monomers=n_monomers,
                cutoff_params=cp,
                doML=True,
                doMM=False,
                doML_dimer=True,
                box=jnp.eye(3) * box,
            )

    old, new = evaluate(1.5), evaluate(0.0)
    # Bit-identical in float64; float32 differs only by summation order over the slots.
    f_scale = float(jnp.max(jnp.abs(old.forces)))
    assert f_scale > 1e-3  # the check is not vacuous
    np.testing.assert_allclose(float(new.energy), float(old.energy), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(new.forces), np.asarray(old.forces), atol=1e-5 * f_scale)


def test_monomer_own_pad_matches_shared_padded_batch(monkeypatch) -> None:
    """Monomers evaluated at their own size give the same energy/forces as padded to dimer size.

    Needs a real PhysNet (the jax_mm_clone stand-in bypasses the PhysNet apply);
    uses the repo's DESdimers checkpoint.
    """
    from pathlib import Path
    from mmml.interfaces.pycharmmInterface.mmml_calculator import setup_calculator

    n_mono, box = 5, 40.0
    rng = np.random.default_rng(2)
    base = rng.normal(scale=0.7, size=(n_mono, 3))
    xs = np.cumsum([2.0, 4.0, 5.2, 5.8, 3.9, 7.1, 5.5, 4.6])
    centers = [[x, 2.0, 2.0] for x in xs] + [[20.0, 20.0, 20.0], [20.0, 24.5, 20.0]]
    n_monomers = len(centers)
    n_atoms = n_mono * n_monomers
    z = jnp.asarray(rng.choice([1, 6, 8], size=n_atoms), dtype=jnp.int32)
    r0 = jnp.asarray(
        np.concatenate([np.asarray(c, float) + base + rng.normal(scale=0.05, size=base.shape) for c in centers])
    )
    fake_mm_fn = lambda *a, **k: (jnp.array(0.0), jnp.zeros((n_atoms, 3)))
    fake_update_fn = lambda *a, **k: (jnp.zeros((1, 2), dtype=jnp.int32), jnp.ones((1,), dtype=bool))

    def fake_build_mm(*args, **kwargs):
        if kwargs.get("use_jax_md_neighbor_list", True):
            return fake_mm_fn, fake_update_fn
        return fake_mm_fn

    cp = CutoffParameters(mm_switch_on=6.0, ml_switch_width=1.5)
    ckpt = Path(__file__).resolve().parents[2] / "examples" / "ckpts_json" / "DESdimers_params.json"
    if not ckpt.is_file():
        pytest.skip(f"missing {ckpt}")

    def evaluate(own_pad: str):
        monkeypatch.setenv("MMML_ML_MONOMER_OWN_PAD", own_pad)
        with patch(
            "mmml.interfaces.pycharmmInterface.mmml_calculator.build_mm_energy_forces_fn",
            side_effect=fake_build_mm,
        ):
            factory = setup_calculator(
                ATOMS_PER_MONOMER=n_mono,
                N_MONOMERS=n_monomers,
                model_restart_path=str(ckpt),
                ml_potential_mode="physnet",
                doML=True,
                doMM=False,
                doML_dimer=True,
                MAX_ATOMS_PER_SYSTEM=10,
                cell=box,
                defer_xla_gpu_warmup=True,
                verbose=False,
                ml_sparse_dimers=True,
                ml_max_active_dimers=16,  # < 45 pairs: sparse path
                ml_batch_size=4,  # chunked: 10 monomers + 16 slots > 4
            )
            _, spherical_fn, _ = factory(
                atomic_numbers=z,
                atomic_positions=r0,
                n_monomers=n_monomers,
                cutoff_params=cp,
                doML=True,
                doMM=False,
                doML_dimer=True,
                backprop=False,
                create_ase_calculator=False,
            )
            layout = getattr(spherical_fn, "ml_chunk_layout", None)
            assert layout is not None  # chunked sparse path with a host chunk budget
            assert layout.n_monomers == (0 if own_pad == "1" else n_monomers)
            return spherical_fn(
                atomic_numbers=z,
                positions=r0,
                n_monomers=n_monomers,
                cutoff_params=cp,
                doML=True,
                doMM=False,
                doML_dimer=True,
                box=jnp.eye(3) * box,
            )

    shared, own = evaluate("0"), evaluate("1")
    f_scale = float(jnp.max(jnp.abs(shared.forces)))
    assert f_scale > 1e-3
    np.testing.assert_allclose(float(own.energy), float(shared.energy), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(own.forces), np.asarray(shared.forces), atol=1e-5 * f_scale)
