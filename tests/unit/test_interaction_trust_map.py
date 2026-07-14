"""Learned per-element-pair trust map for the neural interaction energy.

The trust map is a (6,6) log-lambda matrix over (H,C,N,O,S,Cl), fit by an
evidence-balanced NLL so that at the stationary point lambda_c ~ evidence / <r^2>_c:
small shrinkage where the data justifies a large neural correction, large where it does
not. These tests pin the parameter plumbing and that empirical-Bayes stationary point.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "train_so3lr_spooky_extxyz.py"


@pytest.fixture(scope="module")
def trainer():
    spec = importlib.util.spec_from_file_location("train_so3lr_spooky_extxyz", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(**overrides) -> argparse.Namespace:
    base = dict(
        features=16, max_degree=1, num_iterations=1, num_basis_functions=8, cutoff=6.0,
        max_atomic_number=18, predict_charges=False, n_res=1, no_zbl=True,
        trainable_zbl=False, zbl_cuton=0.1, zbl_cutoff=0.6, efa=False,
        use_energy_bias=False, electrostatics_damping_sigma=4.0,
        fixed_cgenff_vdw=True, interaction_trust_map=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_parameter_only_exists_when_enabled(trainer):
    def has_param(flag):
        model = trainer.create_model(_args(interaction_trust_map=flag), max_atoms=3)
        p = model.init(
            jax.random.PRNGKey(0),
            atomic_numbers=jnp.array([8, 1, 1]), charges=jnp.zeros(3), spins=jnp.zeros(3),
            positions=jnp.zeros((3, 3)), dst_idx=jnp.array([0, 1]), src_idx=jnp.array([1, 0]),
            batch_segments=jnp.zeros(3, jnp.int32), batch_size=1,
            batch_mask=jnp.ones(2), atom_mask=jnp.ones(3),
        )
        leaves = jax.tree_util.tree_flatten_with_path(p)[0]
        return any("neural_interaction_log_lambda" in str(path) for path, _ in leaves)

    assert has_param(False) is False
    assert has_param(True) is True


def test_matrix_is_6x6_over_the_declared_elements(trainer):
    model = trainer.create_model(_args(interaction_trust_map=True), max_atoms=3)
    p = model.init(
        jax.random.PRNGKey(0),
        atomic_numbers=jnp.array([8, 1, 1]), charges=jnp.zeros(3), spins=jnp.zeros(3),
        positions=jnp.zeros((3, 3)), dst_idx=jnp.array([0, 1]), src_idx=jnp.array([1, 0]),
        batch_segments=jnp.zeros(3, jnp.int32), batch_size=1,
        batch_mask=jnp.ones(2), atom_mask=jnp.ones(3),
    )
    lam = next(v for path, v in jax.tree_util.tree_flatten_with_path(p)[0]
               if "neural_interaction_log_lambda" in str(path))
    assert lam.shape == (len(trainer.TRUST_MAP_ELEMENTS), len(trainer.TRUST_MAP_ELEMENTS))


def _one_dimer_batch():
    """Two atoms per monomer, both element C (slot 1), 3 A apart -> one C-C contact."""
    Z = jnp.array([6, 6, 6, 6])
    R = jnp.array([[0, 0, 0], [1.5, 0, 0], [0, 0, 3.0], [1.5, 0, 3.0]], dtype=jnp.float32)
    mol_id = jnp.array([0, 0, 1, 1])
    n = 4
    dst, src = zip(*[(i, j) for i in range(n) for j in range(n) if i != j])
    return Z, R, mol_id, jnp.array(dst), jnp.array(src)


def test_empirical_bayes_stationary_point(trainer):
    """Fitting log_lambda to a fixed residual r must converge to lambda ~ evidence/r^2
    on the contacted bucket (C-C here). This is the property that makes the matrix a
    trust map: the learned shrinkage encodes how large a correction the data supported.
    """
    Z, R, mol_id, dst, src = _one_dimer_batch()
    n_el = len(trainer.TRUST_MAP_ELEMENTS)
    r = jnp.array([[0.5]])  # fixed neural interaction, eV
    evidence = 1.0
    log_lambda = jnp.zeros((n_el, n_el))

    def loss(ll):
        term, _ = trainer._interaction_trust_map_loss(
            r, ll, Z=Z, R=R, dst_idx=dst, src_idx=src, mol_id=mol_id,
            batch_segments=jnp.zeros(4, jnp.int32), batch_mask=jnp.ones(len(dst)),
            batch_size=1, cutoff=6.0, evidence=evidence, hyperprior=0.0,
        )
        return term

    g = jax.grad(loss)
    for _ in range(4000):
        log_lambda = log_lambda - 0.05 * g(log_lambda)

    _, lam = trainer._interaction_trust_map_loss(
        r, log_lambda, Z=Z, R=R, dst_idx=dst, src_idx=src, mol_id=mol_id,
        batch_segments=jnp.zeros(4, jnp.int32), batch_mask=jnp.ones(len(dst)),
        batch_size=1, cutoff=6.0, evidence=evidence, hyperprior=0.0,
    )
    slot_c = trainer.TRUST_MAP_ELEMENTS.index(6)
    expected = evidence / float(r[0, 0]) ** 2  # gamma / r^2 = 1/0.25 = 4.0
    assert lam[slot_c, slot_c] == pytest.approx(expected, rel=0.05)


def test_larger_residual_gives_smaller_lambda(trainer):
    """Monotonicity: a bucket whose data supports a bigger correction is shrunk less."""
    Z, R, mol_id, dst, src = _one_dimer_batch()
    n_el = len(trainer.TRUST_MAP_ELEMENTS)

    def fit(r_val):
        ll = jnp.zeros((n_el, n_el))
        r = jnp.array([[r_val]])
        def loss(x):
            t, _ = trainer._interaction_trust_map_loss(
                r, x, Z=Z, R=R, dst_idx=dst, src_idx=src, mol_id=mol_id,
                batch_segments=jnp.zeros(4, jnp.int32), batch_mask=jnp.ones(len(dst)),
                batch_size=1, cutoff=6.0, evidence=1.0, hyperprior=0.0,
            )
            return t
        g = jax.grad(loss)
        for _ in range(3000):
            ll = ll - 0.05 * g(ll)
        _, lam = trainer._interaction_trust_map_loss(
            r, ll, Z=Z, R=R, dst_idx=dst, src_idx=src, mol_id=mol_id,
            batch_segments=jnp.zeros(4, jnp.int32), batch_mask=jnp.ones(len(dst)),
            batch_size=1, cutoff=6.0, evidence=1.0, hyperprior=0.0,
        )
        return float(lam[1, 1])

    assert fit(1.0) < fit(0.2)  # bigger residual -> less shrinkage


def test_dump_script_reads_and_ranks(trainer, tmp_path):
    import json, subprocess
    n_el = len(trainer.TRUST_MAP_ELEMENTS)
    ll = np.zeros((n_el, n_el))
    ll[1, 1] = 3.0  # C-C: large log_lambda -> strongly shrunk
    ckpt = {"params": {"params": {"neural_interaction_log_lambda": ll.tolist()}}}
    p = tmp_path / "step-00000001_params.json"
    p.write_text(json.dumps(ckpt))
    out = subprocess.run(
        [sys.executable, str(_SCRIPT.parent / "dump_trust_map.py"), str(p)],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    assert "C-C" in out.stdout and "trusted" in out.stdout.lower()
