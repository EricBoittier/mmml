"""Batched Langevin gas-phase sampler used for Delta H_vap."""

import numpy as np
import pytest

from mmml.distill.gas_monomer_md import KB_KCAL, block_mean_std, delta_hvap, langevin_gas_md


class _Trap:
    """Isotropic harmonic trap per atom: E = k/2 |x - x0|^2 -> <E> = 3N/2 kT."""

    def __init__(self, x0, k=50.0):
        self.x0, self.k = np.asarray(x0), k

    def evaluate(self, structures):
        e, f = [], []
        for _, x in structures:
            d = x - self.x0
            e.append(0.5 * self.k * float(np.sum(d * d)))
            f.append(-self.k * d)
        return np.array(e), f


def test_equipartition_in_harmonic_trap():
    z = np.array([6, 8, 1])
    x0 = np.array([[0.0, 0, 0], [1.4, 0, 0], [2.0, 0.8, 0]])
    res = langevin_gas_md(_Trap(x0), z, x0, temperature_K=300.0, dt_fs=0.5, friction_per_fs=0.05,
                          n_copies=64, n_steps=3000, equil_steps=500, sample_every=5, seed=3)
    expect = 1.5 * len(z) * KB_KCAL * 300.0
    assert res.mean == pytest.approx(expect, rel=0.05)
    assert res.T_mean == pytest.approx(300.0, rel=0.05)


def test_delta_hvap_and_blocks():
    assert delta_hvap(-10.0, -20.0, 298.15) == pytest.approx(10.0 + KB_KCAL * 298.15)
    m, s = block_mean_std(np.arange(10.0), n_blocks=5)
    assert m == pytest.approx(4.5) and s > 0
