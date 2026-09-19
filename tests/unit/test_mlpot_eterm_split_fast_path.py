"""MLpot MM eterm split: exact zero fast path when CHARMM's live nonbond params are zero."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.mlpot import charmm_eterm_routing as routing
from mmml.interfaces.pycharmmInterface.mm_energy_forces import (
    decompose_mlpot_mm_nb_eterms_kcalmol,
)

_LIVE = "mmml.interfaces.pycharmmInterface.mm_system_energy._live_charmm_nonbonded_arrays"
_DECOMPOSE = "mmml.interfaces.pycharmmInterface.mm_energy_forces.decompose_mlpot_mm_nb_eterms_kcalmol"


class _NoHostCopy:
    """Pair-list stand-in that fails if the routing code copies it to the host."""

    shape = (4, 2)

    def __array__(self, *args, **kwargs):
        raise AssertionError("pair list must not be copied to the host")


def _calc(n_mono: int, atoms_per: int):
    calc = MagicMock()
    calc._do_mm = True
    calc._cached_update_fn = None  # no hybrid JAX split, even if opted in
    calc._get_update_fn = None
    calc._atoms_per_monomer = [atoms_per] * n_mono
    calc.cutoff_params = MagicMock(
        mm_switch_on=6.0,
        mm_switch_width=1.5,
        ml_switch_width=1.5,
        complementary_handoff=True,
    )
    return calc


def test_zero_live_params_skip_pair_pass(monkeypatch):
    monkeypatch.setenv("MMML_MLPOT_ROUTE_MM_ETERMS", "1")
    n = 18
    calc = _calc(2, 9)
    pushed = []
    zeros = np.zeros(n)
    with patch(_LIVE, return_value=(zeros, zeros.copy(), np.ones(n) * 1.8)), patch(
        _DECOMPOSE
    ) as mock_decompose, patch.object(
        routing,
        "push_mlpot_nb_components_to_charmm",
        side_effect=lambda **kw: pushed.append(kw),
    ):
        user = routing.decompose_and_route_mlpot_mm_from_callback(
            calc,
            np.zeros((n, 3)),
            _NoHostCopy(),
            _NoHostCopy(),
            np.diag([26.0] * 3),
            -123.5,
            use_mm_pairs=True,
        )
    mock_decompose.assert_not_called()
    assert user == -123.5
    assert calc._last_mm_nb_components_kcalmol["mm_total"] == 0.0
    # CHARMM still gets its (zero) buckets staged every step, as before.
    assert pushed == [
        {
            "vdw_primary_kcal": 0.0,
            "vdw_image_kcal": 0.0,
            "elec_primary_kcal": 0.0,
            "elec_image_kcal": 0.0,
            "route": True,
        }
    ]


@pytest.mark.parametrize("which", ["charges", "eps"])
def test_any_nonzero_live_param_keeps_full_split(which, monkeypatch):
    monkeypatch.setenv("MMML_MLPOT_ROUTE_MM_ETERMS", "1")
    n = 18
    calc = _calc(2, 9)
    q = np.zeros(n)
    eps = np.zeros(n)
    (q if which == "charges" else eps)[3] = -0.1
    comps = {
        "vdw_primary": 1.0,
        "vdw_image": 0.5,
        "elec_primary": 2.0,
        "elec_image": 0.25,
        "mm_total": 3.75,
    }
    with patch(_LIVE, return_value=(q, eps, np.ones(n) * 1.8)), patch(
        _DECOMPOSE, return_value=comps
    ) as mock_decompose, patch.object(routing, "push_mlpot_nb_components_to_charmm"):
        user = routing.decompose_and_route_mlpot_mm_from_callback(
            calc,
            np.zeros((n, 3)),
            np.array([[0, 9]], dtype=np.int32),
            np.array([True]),
            None,
            10.0,
            use_mm_pairs=True,
        )
    mock_decompose.assert_called_once()
    assert user == pytest.approx(6.25)


def test_fast_path_matches_full_split_on_zeroed_params():
    """The skipped pass would have returned exactly zero for every bucket."""
    rng = np.random.default_rng(3)
    n_mono, per = 12, 9
    n = n_mono * per
    L = 20.0
    pos = rng.random((n, 3)) * L
    ii, jj = np.triu_indices(n, k=1)
    pair_idx = np.stack([ii, jj], axis=1).astype(np.int32)
    out = decompose_mlpot_mm_nb_eterms_kcalmol(
        pos,
        pair_idx,
        np.ones(len(pair_idx), dtype=bool),
        np.diag([L, L, L]),
        charges_e=np.zeros(n),
        rmins_A=rng.random(n) + 1.0,
        epsilons_kcal=np.zeros(n),
        monomer_id=np.repeat(np.arange(n_mono), per),
        mm_switch_on=6.0,
        mm_switch_width=1.5,
    )
    assert out == routing._zero_nb_components()


@pytest.mark.parametrize("source", [None, "charmm", "hybrid"])
def test_hybrid_split_only_when_opted_in(source, monkeypatch):
    monkeypatch.setenv("MMML_MLPOT_ROUTE_MM_ETERMS", "1")
    if source is None:
        monkeypatch.delenv("MMML_MLPOT_ETERM_SPLIT_SOURCE", raising=False)
    else:
        monkeypatch.setenv("MMML_MLPOT_ETERM_SPLIT_SOURCE", source)
    calc = _calc(2, 9)
    calc._cached_update_fn = MagicMock(mm_eterm_split=lambda *a: np.array([-1.0, -0.5, 2.0, 0.25]))
    zeros = np.zeros(18)
    with patch(_LIVE, return_value=(zeros, zeros.copy(), np.ones(18) * 1.8)), patch.object(
        routing, "push_mlpot_nb_components_to_charmm"
    ) as push:
        user = routing.decompose_and_route_mlpot_mm_from_callback(
            calc, np.zeros((18, 3)), np.array([[0, 9]]), np.array([True]), None, 10.0, use_mm_pairs=True
        )
    if source != "hybrid":  # default: zeroed live params (all-ML) -> #226 fast path, all in USER
        assert user == 10.0 and calc._last_mm_nb_components_kcalmol["mm_total"] == 0.0
    else:
        assert user == pytest.approx(10.0 - 0.75)
        assert push.call_args.kwargs["vdw_primary_kcal"] == -1.0
        assert push.call_args.kwargs["elec_image_kcal"] == 0.25
