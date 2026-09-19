"""MLpot callback PBC wrap: lattice-only, never writes CHARMM coordinates."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot import DecomposedMlpotCalculator


def test_callback_wrap_is_lattice_only_and_leaves_charmm_arrays():
    L, apm = 20.0, 3
    rng = np.random.default_rng(0)
    pos = rng.uniform(-9, 9, (12, 3))
    pos[:apm] += np.array([L * 0.55, 0.0, 0.0])  # molecule 0 COM outside [-L/2, L/2]
    x, y, z = pos[:, 0].copy(), pos[:, 1].copy(), pos[:, 2].copy()
    fake = SimpleNamespace(_cell=L, _atoms_per_monomer=[apm] * 4)
    out = DecomposedMlpotCalculator._maybe_rewrap_primary_cell_in_callback(fake, pos, len(pos), x, y, z)
    assert np.array_equal(x, pos[:, 0]) and np.array_equal(y, pos[:, 1]) and np.array_equal(z, pos[:, 2])
    shift = (out - pos).reshape(4, apm, 3)
    assert np.allclose(shift, shift[:, :1, :])  # rigid per molecule
    assert np.allclose(shift / L, np.round(shift / L))  # integer lattice vectors
    com = out.reshape(4, apm, 3).mean(1)
    assert np.all(com >= -L / 2 - 1e-9) and np.all(com < L / 2 + 1e-9)
