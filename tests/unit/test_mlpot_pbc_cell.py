"""Tests for MIC PBC cell threading in PyCHARMM MLpot backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters
from mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot import (
    DecomposedMlpotCalculator,
    DecomposedMlpotModel,
    build_decomposed_mlpot_model,
    warmup_decomposed_mlpot,
)


def test_build_decomposed_mlpot_passes_cell_to_setup_calculator():
    z = np.array([6, 1, 1, 1, 6, 1, 1, 1], dtype=int)
    per = [4, 4]
    factory = MagicMock(
        return_value=(
            None,
            MagicMock(),
            None,
        )
    )
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot.setup_calculator",
        return_value=factory,
    ) as mock_setup, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot.unpack_factory_result",
        return_value=(None, MagicMock(), None),
    ), patch(
        "mmml.interfaces.pycharmmInterface.jax_device_policy.mlpot_jax_device_context",
        return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()),
    ):
        model = build_decomposed_mlpot_model(
            "/tmp/fake_ckpt.json",
            z,
            per,
            2,
            cell=20.0,
        )
    mock_setup.assert_called_once()
    assert mock_setup.call_args.kwargs["cell"] == 20.0
    assert model._cell == 20.0


def test_build_decomposed_mlpot_vacuum_cell_false():
    z = np.array([6, 1, 1, 1, 6, 1, 1, 1], dtype=int)
    per = [4, 4]
    get_update_fn = MagicMock()
    factory = MagicMock(return_value=(None, MagicMock(), None))
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot.setup_calculator",
        return_value=factory,
    ) as mock_setup, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot.unpack_factory_result",
        return_value=(None, MagicMock(), get_update_fn),
    ), patch(
        "mmml.interfaces.pycharmmInterface.jax_device_policy.mlpot_jax_device_context",
        return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()),
    ):
        model = build_decomposed_mlpot_model(
            "/tmp/fake_ckpt.json",
            z,
            per,
            2,
        )
    assert mock_setup.call_args.kwargs["cell"] is False
    assert model._cell is False
    assert model._get_update_fn is get_update_fn


def test_decomposed_mlpot_defers_jax_factory_until_get_calculator():
    z = np.array([6, 1, 1, 1, 6, 1, 1, 1], dtype=int)
    per = [4, 4]
    factory = MagicMock(
        return_value=(
            None,
            MagicMock(),
            MagicMock(),
        )
    )
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot.setup_calculator",
        return_value=factory,
    ) as mock_setup, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot.unpack_factory_result",
        return_value=(None, MagicMock(), MagicMock()),
    ) as mock_unpack, patch(
        "mmml.utils.jax_gpu_warmup.ensure_xla_gpu_warmed",
    ) as mock_xla_warm, patch(
        "mmml.interfaces.pycharmmInterface.jax_device_policy.mlpot_jax_device_context",
        return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()),
    ):
        model = build_decomposed_mlpot_model(
            "/tmp/fake_ckpt.json",
            z,
            per,
            2,
            defer_jax_until_mlpot_registered=True,
        )
        assert model._spherical_fn is None
        mock_unpack.assert_not_called()
        model.get_pycharmm_calculator()
    mock_setup.assert_called_once()
    assert mock_setup.call_args.kwargs["defer_xla_gpu_warmup"] is True
    mock_xla_warm.assert_called_once()
    mock_unpack.assert_called_once()
    assert model._spherical_fn is not None


def test_setup_calculator_defer_skips_terminal_xla_gpu_warmup():
    """Factory build with defer must not touch GPU before CHARMM register_mlpot."""
    ckpt = Path(__file__).resolve().parents[2] / "examples/ckpts_json/DESdimers_params.json"
    if not ckpt.is_file():
        pytest.skip("DESdimers_params.json checkpoint missing")

    from mmml.interfaces.pycharmmInterface.mmml_calculator import setup_calculator

    with patch(
        "mmml.interfaces.pycharmmInterface.mmml_calculator.ensure_xla_gpu_warmed",
        return_value=False,
    ) as mock_warm:
        factory = setup_calculator(
            ATOMS_PER_MONOMER=5,
            N_MONOMERS=2,
            doML=True,
            doMM=False,
            model_restart_path=str(ckpt),
            MAX_ATOMS_PER_SYSTEM=10,
            cell=38.0,
            defer_xla_gpu_warmup=True,
            verbose=False,
        )
    mock_warm.assert_not_called()
    assert callable(factory)


def test_register_mlpot_context_forwards_cell():
    from mmml.interfaces.pycharmmInterface.mlpot import run_workflow

    z = np.zeros(8, dtype=int)
    r = np.zeros((8, 3), dtype=float)
    ckpt = Path("/tmp/fake_ckpt.json")
    fake_model = DecomposedMlpotModel(MagicMock(), CutoffParameters(), 2, z, cell=20.0)
    fake_ctx = MagicMock()
    fake_sel = MagicMock()
    call_order: list[str] = []

    def _register(*args, **kwargs):
        call_order.append("register")
        return fake_ctx

    def _warmup(*args, **kwargs):
        call_order.append("warmup")

    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.run_workflow.load_physnet_mlpot_bundle",
        return_value=(None, None, fake_model),
    ) as mock_load, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.run_workflow.register_mlpot",
        side_effect=_register,
    ) as mock_register, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.run_workflow.select_all_atoms",
        return_value=fake_sel,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.run_workflow.defer_jax_warmup_until_after_mlpot_sd",
        return_value=False,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot.warmup_decomposed_mlpot",
        side_effect=_warmup,
    ) as mock_warmup, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.run_workflow.refresh_nbonds_after_mlpot_pbc",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.run_workflow.sync_charmm_positions",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.run_workflow.get_charmm_positions_array",
        return_value=r,
    ):
        ctx, model = run_workflow._register_mlpot_context(
            z,
            r,
            ckpt,
            len(z),
            2,
            cubic_box_side_A=20.0,
            mlpot_use_pbc=True,
            verbose=True,
        )

    mock_load.assert_called_once()
    assert mock_load.call_args.kwargs["cell"] == 20.0
    mock_register.assert_called_once_with(
        fake_model, z, fake_sel, use_pbc=True, mm_internal_scale=0.0
    )
    mock_warmup.assert_called_once()
    assert mock_warmup.call_args.kwargs["cell"] == 20.0
    assert call_order == ["register", "warmup"]
    assert ctx is fake_ctx
    assert model is fake_model


def test_warmup_decomposed_mlpot_passes_box_when_cell_set():
    z = np.zeros(8, dtype=int)
    model = DecomposedMlpotModel(
        MagicMock(), CutoffParameters(), 2, z, cell=20.0, do_mm=False
    )
    r = np.zeros((8, 3), dtype=float)

    with patch(
        "mmml.utils.jax_gpu_warmup.warmup_hybrid_spherical_cutoff",
    ) as mock_warmup, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot._warmup_value_and_grad_for_model",
    ) as mock_vg_warmup, patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.recover_mpi_for_charmm_after_jax",
    ):
        warmup_decomposed_mlpot(model, r, verbose=False)

    mock_warmup.assert_called_once()
    box = mock_warmup.call_args.kwargs["box"]
    assert box is not None
    assert float(box[0, 0]) == pytest.approx(20.0)
    mock_vg_warmup.assert_called_once()


def test_warmup_decomposed_mlpot_do_mm_uses_get_update_fn():
    z = np.zeros(8, dtype=int)
    get_update_fn = MagicMock(return_value=MagicMock(return_value=(None, None)))
    model = DecomposedMlpotModel(
        MagicMock(),
        CutoffParameters(),
        2,
        z,
        cell=20.0,
        do_mm=True,
        get_update_fn=get_update_fn,
    )
    r = np.zeros((8, 3), dtype=float)

    with patch(
        "mmml.utils.jax_gpu_warmup.warmup_hybrid_spherical_cutoff",
    ) as mock_warmup, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot._warmup_value_and_grad_for_model",
    ) as mock_vg_warmup, patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.recover_mpi_for_charmm_after_jax",
    ):
        warmup_decomposed_mlpot(model, r, verbose=False)

    get_update_fn.assert_called_once()
    mock_warmup.assert_called_once()
    assert mock_warmup.call_args.kwargs["doMM"] is True
    mock_vg_warmup.assert_called_once()


def test_decomposed_calculator_initializes_mm_before_spherical_fn():
    z = np.zeros(8, dtype=int)
    get_update_fn = MagicMock(return_value=MagicMock(return_value=(None, None)))
    calc = DecomposedMlpotCalculator(
        MagicMock(),
        CutoffParameters(),
        2,
        z,
        do_mm=True,
        get_update_fn=get_update_fn,
    )
    mock_energy = jnp.array(0.0)
    mock_grad = jnp.zeros((8, 3))
    mock_vg = MagicMock(return_value=(mock_energy, mock_grad))
    calc._get_value_and_grad_fn = MagicMock(return_value=mock_vg)
    n = 8
    x = np.zeros(n, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    zc = np.zeros(n, dtype=np.float64)
    dx = np.zeros(n, dtype=np.float64)
    dy = np.zeros(n, dtype=np.float64)
    dz = np.zeros(n, dtype=np.float64)

    with patch(
        "mmml.interfaces.pycharmmInterface.jax_device_policy.mlpot_jax_device_context",
        return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()),
    ):
        calc.calculate_charmm(
            n, 0, 0, None, x, y, zc, dx, dy, dz, 0, 0, None, None, None, None, None, None, None
        )

    get_update_fn.assert_called_once()
    mock_vg.assert_called_once()


def test_pbc_nbond_cutoffs_respects_half_box():
    from mmml.interfaces.pycharmmInterface.nbonds_config import pbc_nbond_cutoffs

    cuts35 = pbc_nbond_cutoffs(35.0)
    assert cuts35.cutnb < 17.5
    assert cuts35.cutim <= 17.5
    assert cuts35.cutim >= cuts35.cutnb
    assert cuts35.ctonnb < cuts35.ctofnb < cuts35.cutnb
    assert cuts35.ctexnb == pytest.approx(cuts35.cutnb)

    cuts38 = pbc_nbond_cutoffs(38.0)
    assert cuts38.cutnb == pytest.approx(18.0)
    assert cuts38.cutim == pytest.approx(18.0)
    assert cuts38.cutim < 0.5 * 38.0
    assert cuts38.was_capped

    cuts55 = pbc_nbond_cutoffs(55.0)
    assert cuts55.cutnb == pytest.approx(18.0)
    assert cuts55.cutim == pytest.approx(22.0)
    assert cuts55.ctonnb == pytest.approx(13.0)
    assert cuts55.ctofnb == pytest.approx(17.0)
    assert not cuts55.was_capped


def test_pbc_nbond_kwargs_includes_ctexnb_when_capped():
    from mmml.interfaces.pycharmmInterface.nbonds_config import pbc_nbond_cutoffs

    cuts = pbc_nbond_cutoffs(35.0)
    kw = cuts.as_pbc_nbond_kwargs()
    assert kw["ctexnb"] == pytest.approx(cuts.cutnb)
    assert kw["cutim"] == pytest.approx(cuts.cutim)


def test_charmm_ctypes_scalar_accepts_int_and_ctypes_wrapper():
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import _charmm_ctypes_scalar

    assert _charmm_ctypes_scalar(1) == pytest.approx(1.0)
    wrapper = MagicMock(value=39.5)
    assert _charmm_ctypes_scalar(wrapper) == pytest.approx(39.5)


def test_cubic_box_matrix_from_side():
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import cubic_box_matrix_from_side

    m = cubic_box_matrix_from_side(40.0)
    assert m.shape == (3, 3)
    assert float(m[0, 0]) == pytest.approx(40.0)
    assert float(m[1, 1]) == pytest.approx(40.0)


def test_is_cubic_box_sides():
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import _is_cubic_box_sides

    assert _is_cubic_box_sides(40.0, 40.0, 40.0)
    assert not _is_cubic_box_sides(40.0, 41.0, 40.0)
    assert not _is_cubic_box_sides(0.0, 40.0, 40.0)


def test_resolve_charmm_cubic_box_side_A_uses_fallback():
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import resolve_charmm_cubic_box_side_A

    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env._read_charmm_box_sides_A",
        return_value=(0.0, 0.0, 0.0),
    ):
        side, source = resolve_charmm_cubic_box_side_A(fallback_side_A=40.0)
    assert side == pytest.approx(40.0)
    assert source == "fallback"


def test_resolve_charmm_cubic_box_side_A_uses_restart_file():
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import resolve_charmm_cubic_box_side_A

    heat_res = Path(
        "examples/other/notebooks/ffFIT/example-general/heat.res"
    ).resolve()
    if not heat_res.is_file():
        pytest.skip("example heat.res not available")
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env._read_charmm_box_sides_A",
        return_value=(0.0, 0.0, 0.0),
    ):
        side, source = resolve_charmm_cubic_box_side_A(restart_path=heat_res)
    assert side == pytest.approx(31.0)
    assert source == "restart"


def test_parse_cubic_box_side_from_charmm_restart_vacuum_returns_none():
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
        parse_cubic_box_side_from_charmm_restart,
    )

    nve_res = Path(
        "tests/functionality/mlpot/output/dynamics/nve_aco_4mer.res"
    ).resolve()
    if not nve_res.is_file():
        pytest.skip("nve restart fixture not available")
    assert parse_cubic_box_side_from_charmm_restart(nve_res) is None


def test_sync_mlpot_pbc_cell_from_charmm_updates_model():
    from mmml.interfaces.pycharmmInterface.mlpot import run_workflow

    z = np.zeros(8, dtype=int)
    model = DecomposedMlpotModel(MagicMock(), CutoffParameters(), 2, z, cell=40.0)
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.resolve_charmm_cubic_box_side_A",
        return_value=(39.25, "pbound"),
    ):
        side = run_workflow.sync_mlpot_pbc_cell_from_charmm(model, verbose=False)
    assert side == pytest.approx(39.25)
    assert model._cell == pytest.approx(39.25)


def test_decomposed_calculator_passes_charmm_box_to_spherical_fn():
    z = np.zeros(8, dtype=int)
    calc = DecomposedMlpotCalculator(
        MagicMock(),
        CutoffParameters(),
        2,
        z,
        cell=40.0,
    )
    captured: dict[str, Any] = {}

    def _fake_vg_fn(*, n_atoms, atomic_numbers_jax, box_jax):
        captured["box_jax"] = box_jax

        def _eval(positions_jax, mm_pair_idx, mm_pair_mask, use_mm_pairs):
            return jnp.array(0.0), jnp.zeros((8, 3))

        return _eval

    calc._get_value_and_grad_fn = MagicMock(side_effect=_fake_vg_fn)
    n = 8
    x = np.zeros(n, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    zc = np.zeros(n, dtype=np.float64)
    dx = np.zeros(n, dtype=np.float64)
    dy = np.zeros(n, dtype=np.float64)
    dz = np.zeros(n, dtype=np.float64)
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.resolve_charmm_cubic_box_side_A",
        return_value=(39.0, "pbound"),
    ), patch(
        "mmml.interfaces.pycharmmInterface.jax_device_policy.mlpot_jax_device_context",
        return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()),
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.recover_mpi_for_charmm_after_jax",
    ):
        calc.calculate_charmm(
            n, 0, 0, None, x, y, zc, dx, dy, dz, 0, 0, None, None, None, None, None, None, None
        )
    assert calc._cell == pytest.approx(39.0)
    box = captured["box_jax"]
    assert box is not None
    assert float(box[0, 0]) == pytest.approx(39.0)


def test_value_and_grad_fn_cached_across_callbacks(monkeypatch):
    monkeypatch.setenv("MMML_MLPOT_DEVICE", "cpu")
    z = np.zeros(8, dtype=int)

    def spherical_fn(**kwargs):
        pos = kwargs["positions"]
        energy = jnp.sum(pos**2)

        class Out:
            pass

        out = Out()
        out.energy = energy
        return out

    calc = DecomposedMlpotCalculator(
        spherical_fn,
        CutoffParameters(),
        2,
        z,
        do_mm=False,
    )
    n = 8
    x = np.linspace(0.1, 0.8, n, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    zc = np.zeros(n, dtype=np.float64)
    dx = np.zeros(n, dtype=np.float64)
    dy = np.zeros(n, dtype=np.float64)
    dz = np.zeros(n, dtype=np.float64)

    calc.calculate_charmm(
        n, 0, 0, None, x, y, zc, dx, dy, dz, 0, 0, None, None, None, None, None, None, None
    )
    first_key = calc._grad_cache_owner()._vg_cache_key
    calc.calculate_charmm(
        n, 0, 0, None, x + 0.01, y, zc, dx, dy, dz, 0, 0, None, None, None, None, None, None, None
    )
    second_key = calc._grad_cache_owner()._vg_cache_key

    assert first_key is not None
    assert second_key == first_key


def test_build_ml_exclusion_lists_upper_triangle():
    from mmml.interfaces.pycharmmInterface.mlpot.setup import _build_ml_exclusion_lists

    iblo, inb = _build_ml_exclusion_lists([0, 2, 4], natom=6)
    assert list(iblo) == [2, 2, 3, 3, 3, 3]
    assert inb == [3, 5, 5]


def test_register_mlpot_pbc_installs_exclusions_before_block():
    from mmml.interfaces.pycharmmInterface.mlpot import setup as mlpot_setup

    call_order: list[str] = []
    fake_pycharmm = MagicMock()
    fake_pycharmm.coor.get_natom.return_value = 4
    fake_sel = MagicMock()
    fake_sel.get_atom_indexes.return_value = [0, 1, 2, 3]

    def _install(_sel):
        call_order.append("install_exclusions")

    def _block(*args, **kwargs):
        call_order.append("block")
        return "all"

    class _FakeMLpot:
        def __init__(self, *, skip_iblo_inb_update=False, **kwargs):
            call_order.append("mlpot")
            if skip_iblo_inb_update:
                call_order.append("skip_iblo")

    with patch.object(mlpot_setup, "_import_pycharmm", return_value=fake_pycharmm), patch.object(
        mlpot_setup, "_install_ml_exclusions", side_effect=_install
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.block_terms.apply_mlpot_energy_block",
        side_effect=_block,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.mlpot_limits.validate_mlpot_system_size",
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_levels.charmm_relaxed_bomlev",
        return_value=MagicMock(__enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False)),
    ):
        fake_pycharmm.MLpot = _FakeMLpot
        mlpot_setup.register_mlpot(
            MagicMock(),
            [1, 1, 1, 1],
            fake_sel,
            use_pbc=True,
        )

    assert call_order == ["install_exclusions", "block", "mlpot", "skip_iblo"]


def test_register_mlpot_vacuum_skips_pre_block_exclusions():
    from mmml.interfaces.pycharmmInterface.mlpot import setup as mlpot_setup

    call_order: list[str] = []
    fake_pycharmm = MagicMock()
    fake_sel = MagicMock()
    fake_sel.get_atom_indexes.return_value = [0, 1]

    def _block(*args, **kwargs):
        call_order.append("block")
        return "all"

    def _mlpot(**kwargs):
        call_order.append("mlpot")
        if kwargs.get("skip_iblo_inb_update"):
            call_order.append("skip_iblo")
        return MagicMock()

    with patch.object(mlpot_setup, "_import_pycharmm", return_value=fake_pycharmm), patch.object(
        mlpot_setup, "_install_ml_exclusions"
    ) as mock_install, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.block_terms.apply_mlpot_energy_block",
        side_effect=_block,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.mlpot_limits.validate_mlpot_system_size",
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_levels.charmm_relaxed_bomlev",
        return_value=MagicMock(__enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False)),
    ), patch.object(fake_pycharmm, "MLpot", side_effect=_mlpot), patch.object(
        fake_pycharmm, "UpdateNonBondedScript"
    ) as mock_nb:
        mock_nb.return_value.run = MagicMock()
        mlpot_setup.register_mlpot(
            MagicMock(),
            [1, 1],
            fake_sel,
            use_pbc=False,
        )

    mock_install.assert_not_called()
    assert call_order == ["block", "mlpot"]


def test_register_mlpot_pbc_requires_skip_iblo_parameter():
    from mmml.interfaces.pycharmmInterface.mlpot import setup as mlpot_setup

    fake_sel = MagicMock()
    fake_sel.get_atom_indexes.return_value = [0, 1]

    class _OldMLpot:
        def __init__(self, **kwargs):
            pass

    class _FakePycharmm:
        MLpot = _OldMLpot

    with patch.object(mlpot_setup, "_import_pycharmm", return_value=_FakePycharmm()), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.block_terms.apply_mlpot_energy_block",
        return_value="all",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.mlpot_limits.validate_mlpot_system_size",
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_levels.charmm_relaxed_bomlev",
        return_value=MagicMock(__enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False)),
    ), pytest.raises(RuntimeError, match="skip_iblo_inb_update"):
        mlpot_setup.register_mlpot(
            MagicMock(),
            [1, 1],
            fake_sel,
            use_pbc=True,
        )
