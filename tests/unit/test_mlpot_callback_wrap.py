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


def test_callback_wrap_uses_live_box_not_stale_cell():
    """NPT: wrap with the current pbound L, not last step's ``_cell``.

    COM at x=12 is inside a 26 Å CHARMM frame ([-13, 13]) and must stay.
    The same COM is outside a stale 20 Å frame ([-10, 10]) and would shift by −20.
    """
    apm = 3
    pos = np.zeros((6, 3), dtype=np.float64)
    pos[:apm, 0] = [11.8, 12.0, 12.2]  # COM = 12.0
    pos[apm:, 0] = [0.0, 0.1, -0.1]
    x, y, z = pos[:, 0].copy(), pos[:, 1].copy(), pos[:, 2].copy()
    fake = SimpleNamespace(_cell=20.0, _atoms_per_monomer=[apm, apm])
    stale = DecomposedMlpotCalculator._maybe_rewrap_primary_cell_in_callback(
        fake, pos, len(pos), x, y, z
    )
    live = DecomposedMlpotCalculator._maybe_rewrap_primary_cell_in_callback(
        fake, pos, len(pos), x, y, z, box_side_A=26.0
    )
    assert stale[:apm, 0].mean() == np.float64(12.0) - 20.0
    assert live[:apm, 0].mean() == np.float64(12.0)
    assert np.array_equal(x, pos[:, 0])


def test_sync_callback_pbc_box_refreshes_cell_before_wrap(monkeypatch):
    """``calculate_charmm`` must read pbound, then wrap with that side."""
    from mmml.interfaces.pycharmmInterface.mlpot import pbc_env

    fake = SimpleNamespace(
        _cell=20.0,
        _current_box=None,
        _atoms_per_monomer=[3, 3],
        _npt_restart_read=None,
        _parent_model=None,
        _periodic_mm_config=None,
        _requires_callback_pbc_box=lambda: False,
        _callback_box_resolution_inputs=lambda: (20.0, None),
    )
    monkeypatch.setattr(
        pbc_env,
        "resolve_mlpot_mic_box_side_A",
        lambda **_kw: (26.0, "pbound"),
    )
    box = DecomposedMlpotCalculator._sync_callback_pbc_box(fake)
    assert fake._cell == 26.0
    assert box is not None
    assert float(np.diag(np.asarray(fake._current_box))[0]) == 26.0

    pos = np.zeros((6, 3), dtype=np.float64)
    pos[:3, 0] = [11.8, 12.0, 12.2]
    x = pos[:, 0].copy()
    live = DecomposedMlpotCalculator._maybe_rewrap_primary_cell_in_callback(
        fake, pos, 6, x, pos[:, 1], pos[:, 2], box_side_A=float(fake._cell)
    )
    assert live[:3, 0].mean() == np.float64(12.0)
