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
    maybe_warmup_deferred_decomposed_mlpot,
    warmup_decomposed_mlpot,
)


def _hybrid_compat_patch():
    return patch(
        "mmml.interfaces.energy_forces.ml.assert_hybrid_ml_compatible",
        return_value=MagicMock(),
    )


def test_build_decomposed_mlpot_verbose_with_cell_does_not_crash():
    """Regression: periodic_mm_config was read before assignment (UnboundLocalError)."""
    z = np.array([6, 1, 1, 1, 6, 1, 1, 1], dtype=int)
    per = [4, 4]
    factory = MagicMock(return_value=(None, MagicMock(), None))
    args = type(
        "Args",
        (),
        {"mm_nonbond_mode": "jax_mic", "ml_spatial_mpi": None, "max_pairs": None},
    )()
    with _hybrid_compat_patch(), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot.setup_calculator",
        return_value=factory,
    ), patch(
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
            cell=30.0,
            verbose=True,
            args=args,
        )
    assert model._cell == 30.0


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
    with _hybrid_compat_patch(), patch(
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
    with _hybrid_compat_patch(), patch(
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
    with _hybrid_compat_patch(), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot.setup_calculator",
        return_value=factory,
    ) as mock_setup, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot.unpack_factory_result",
        return_value=(None, MagicMock(), MagicMock()),
    ) as mock_unpack, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot.ensure_xla_gpu_warmed",
    ) as mock_xla_warm, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot.mlpot_jax_device_context",
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


def test_decomposed_mlpot_sd_defer_uses_cpu_until_promote():
    z = np.array([6, 1, 1, 1, 6, 1, 1, 1], dtype=int)
    per = [4, 4]
    factory = MagicMock(return_value=(None, MagicMock(), MagicMock()))
    cpu_ctx = MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
    gpu_ctx = MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
    with _hybrid_compat_patch(), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot.setup_calculator",
        return_value=factory,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot.unpack_factory_result",
        return_value=(None, MagicMock(), MagicMock()),
    ) as mock_unpack, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot.ensure_xla_gpu_warmed",
    ) as mock_xla_warm, patch(
        "mmml.interfaces.pycharmmInterface.jax_device_policy.jax_cpu_until_mlpot_registered",
        return_value=cpu_ctx,
    ), patch(
        "mmml.interfaces.pycharmmInterface.jax_device_policy.mlpot_jax_device_context",
        return_value=gpu_ctx,
    ):
        model = build_decomposed_mlpot_model(
            "/tmp/fake_ckpt.json",
            z,
            per,
            2,
            defer_jax_until_mlpot_registered=True,
            defer_jax_until_after_sd=True,
        )
        model.get_pycharmm_calculator()
        mock_xla_warm.assert_not_called()
        cpu_ctx.__enter__.assert_called()
        gpu_ctx.__enter__.assert_not_called()
        assert model._jax_on_gpu is False
        assert model._pending_factory is factory

        model.promote_jax_factory_to_gpu()
        mock_xla_warm.assert_called_once()
        gpu_ctx.__enter__.assert_called()
        assert model._jax_on_gpu is True
        assert model._pending_factory is None
        assert mock_unpack.call_count == 2


def test_maybe_promote_deferred_jax_on_hybrid_eval_without_jax_pme():
    """periodic_external / ScaFaCoS (no jax-pme mesh) must promote on first hybrid ENER."""
    z = np.array([6, 1, 1, 6, 1, 1], dtype=int)
    model = DecomposedMlpotModel(
        MagicMock(),
        CutoffParameters(),
        2,
        z,
        cell=40.0,
        do_mm=False,
        defer_jax_until_after_sd=True,
    )
    model._jax_on_gpu = False
    calc = MagicMock(spec=DecomposedMlpotCalculator)
    calc._spherical_forward_fn = "cached"
    calc._forward_cache_key = ("k",)
    calc.spherical_fn = MagicMock()
    calc._get_update_fn = None
    calc._cached_update_fn = "cached_update"

    with patch.object(model, "promote_jax_factory_to_gpu") as mock_promote:
        model._maybe_promote_deferred_jax_on_hybrid_eval(calc)

    mock_promote.assert_called_once()
    assert calc._spherical_forward_fn is None
    assert calc._forward_cache_key is None


def test_maybe_warmup_deferred_decomposed_mlpot_skips_when_already_on_gpu():
    z = np.zeros(8, dtype=int)
    model = DecomposedMlpotModel(
        MagicMock(), CutoffParameters(), 2, z, defer_jax_until_after_sd=True
    )
    model._jax_on_gpu = True
    r = np.zeros((8, 3), dtype=float)

    with patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.defer_jax_warmup_until_after_mlpot_sd",
        return_value=True,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot.warmup_decomposed_mlpot",
    ) as mock_warmup:
        maybe_warmup_deferred_decomposed_mlpot(
            model, r, n_monomers=2, verbose=False
        )

    mock_warmup.assert_not_called()


def test_maybe_warmup_deferred_decomposed_mlpot_calls_warmup_when_deferred():
    z = np.zeros(8, dtype=int)
    model = DecomposedMlpotModel(
        MagicMock(), CutoffParameters(), 2, z, defer_jax_until_after_sd=True
    )
    model._jax_on_gpu = False
    r = np.zeros((8, 3), dtype=float)

    with patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.defer_jax_warmup_until_after_mlpot_sd",
        return_value=True,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot.warmup_decomposed_mlpot",
    ) as mock_warmup:
        maybe_warmup_deferred_decomposed_mlpot(
            model, r, cell=40.0, n_monomers=2, verbose=True
        )

    mock_warmup.assert_called_once_with(model, r, cell=40.0, verbose=True)


def test_setup_calculator_defer_skips_terminal_xla_gpu_warmup():
    """Factory build with defer must not touch GPU before CHARMM MLpot SD gete."""
    ckpt = Path(__file__).resolve().parents[2] / "examples/ckpts_json/DESdimers_params.json"
    if not ckpt.is_file():
        pytest.skip("DESdimers_params.json checkpoint missing")

    from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters
    from mmml.interfaces.pycharmmInterface.mmml_calculator import setup_calculator

    z = jnp.array([6, 1, 1, 1, 1, 6, 1, 1, 1, 1], dtype=jnp.int32)
    r0 = np.zeros((10, 3), dtype=np.float64)

    with patch(
        "mmml.utils.jax_gpu_warmup.ensure_xla_gpu_warmed",
        return_value=False,
    ) as mock_warm:
        factory = setup_calculator(
            ATOMS_PER_MONOMER=5,
            N_MONOMERS=2,
            doML=True,
            doMM=True,
            model_restart_path=str(ckpt),
            MAX_ATOMS_PER_SYSTEM=10,
            cell=32.0,
            defer_xla_gpu_warmup=True,
            verbose=False,
        )
        factory(
            atomic_numbers=z,
            atomic_positions=jnp.asarray(r0),
            n_monomers=2,
            cutoff_params=CutoffParameters(),
            doML=True,
            doMM=True,
            doML_dimer=True,
            backprop=False,
            create_ase_calculator=False,
        )
    mock_warm.assert_not_called()
    assert callable(factory)


def test_runtime_box_is_forwarded_to_lazy_mm_builder():
    """Regression: deferred MLpot + JAX-PME must build MM with callback PBC box."""
    from mmml.interfaces.pycharmmInterface.mmml_calculator import setup_calculator

    class DummyModel:
        def __init__(self, **kwargs):
            self.natoms = int(kwargs.get("natoms", 8))
            self.use_pbc = False

        def apply(self, params, **kwargs):  # pragma: no cover - not evaluated here
            R = kwargs["positions"]
            return {"energy": jnp.zeros((1,), dtype=R.dtype), "forces": jnp.zeros_like(R)}

    dummy_checkpoint = {
        "params": {},
        "config": {
            "features": 32,
            "max_degree": 3,
            "num_iterations": 2,
            "num_basis_functions": 16,
            "cutoff": 6.0,
            "max_atomic_number": 118,
            "charges": False,
            "include_electrostatics": False,
            "natoms": 8,
            "total_charge": 0,
            "n_res": 3,
            "zbl": True,
            "debug": False,
            "efa": False,
            "use_energy_bias": False,
            "use_pbc": True,
        },
    }
    calls: list[Any] = []
    fake_mm_fn = MagicMock(return_value=(jnp.array(0.0), jnp.zeros((8, 3))))
    fake_update_fn = MagicMock(return_value=(jnp.zeros((1, 2), dtype=jnp.int32), jnp.ones((1,), dtype=bool)))

    def fake_build_mm(*args, **kwargs):
        calls.append(kwargs)
        if kwargs.get("use_jax_md_neighbor_list", True):
            return fake_mm_fn, fake_update_fn
        return fake_mm_fn

    restart_path = MagicMock(spec=Path)
    restart_path.is_file.return_value = True
    restart_path.suffix = ".json"
    restart_path.resolve.return_value = restart_path
    z = jnp.array([6, 1, 1, 1, 6, 1, 1, 1], dtype=jnp.int32)
    r0 = np.zeros((8, 3), dtype=np.float64)
    box = jnp.asarray([[31.0, 0.0, 0.0], [0.0, 31.0, 0.0], [0.0, 0.0, 31.0]])

    with patch("mmml.utils.model_checkpoint.load_model_checkpoint", return_value=dummy_checkpoint), patch(
        "mmml.models.physnetjax.physnetjax.models.model.PhysNet", DummyModel
    ), patch(
        "mmml.interfaces.pycharmmInterface.mmml_calculator.build_mm_energy_forces_fn",
        side_effect=fake_build_mm,
    ):
        factory = setup_calculator(
            ATOMS_PER_MONOMER=[4, 4],
            N_MONOMERS=2,
            doML=True,
            doMM=True,
            model_restart_path=restart_path,
            MAX_ATOMS_PER_SYSTEM=8,
            cell=False,
            defer_xla_gpu_warmup=True,
            verbose=False,
        )
        _, _, get_update_fn = factory(
            atomic_numbers=z,
            atomic_positions=jnp.asarray(r0),
            n_monomers=2,
            cutoff_params=CutoffParameters(),
            doML=True,
            doMM=True,
            doML_dimer=True,
            backprop=False,
            create_ase_calculator=False,
        )
        assert get_update_fn is not None
        assert get_update_fn(r0, CutoffParameters(), box=box) is fake_update_fn

    assert len(calls) == 2
    for call in calls:
        np.testing.assert_allclose(np.asarray(call["pbc_cell"]), np.asarray(box))


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
        "mmml.interfaces.pycharmmInterface.charmm_mpi.defer_jax_warmup_until_after_mlpot_sd",
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
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.assert_mlpot_user_active",
        return_value=-1.0,
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
        fake_model,
        z,
        fake_sel,
        use_pbc=True,
        mm_internal_scale=0.0,
        mm_nonbond_mode="jax_mic",
        periodic_charmm_vdw=True,
        cubic_box_side_A=20.0,
        verbose=False,
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
    mock_forces = jnp.zeros((8, 3))
    mock_forward = MagicMock(return_value=(mock_energy, mock_forces))
    calc._get_spherical_forward_fn = MagicMock(return_value=mock_forward)
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
    mock_forward.assert_called_once()


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
    # Capped box: bump would hit L/2, so ctexnb stays at cutnb
    assert kw["ctexnb"] == pytest.approx(cuts.cutnb)
    assert kw["cutim"] == pytest.approx(cuts.cutim)


def test_pbc_nbond_kwargs_bumps_ctexnb_when_box_allows():
    from mmml.interfaces.pycharmmInterface.nbonds_config import pbc_nbond_cutoffs

    cuts = pbc_nbond_cutoffs(55.0)
    kw = cuts.as_pbc_nbond_kwargs()
    assert kw["ctexnb"] == pytest.approx(cuts.cutnb + 1.0)


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


def test_box_numpy_for_update_accepts_matrix_and_vector():
    import jax.numpy as jnp

    from mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot import _box_numpy_for_update
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import cubic_box_matrix_from_side

    matrix = jnp.asarray(cubic_box_matrix_from_side(28.0), dtype=jnp.float32)
    vector = jnp.array([28.0, 28.0, 28.0], dtype=jnp.float32)
    expected = np.array([28.0, 28.0, 28.0], dtype=np.float64)
    np.testing.assert_allclose(_box_numpy_for_update(matrix), expected)
    np.testing.assert_allclose(_box_numpy_for_update(vector), expected)
    assert _box_numpy_for_update(None) is None


def test_mlpot_spherical_forces_passes_cubic_box_matrix():
    from unittest import mock

    import jax.numpy as jnp

    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import mlpot_spherical_forces_ev_angstrom
    from mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot import (
        DecomposedMlpotCalculator,
        DecomposedMlpotModel,
    )

    calc = mock.Mock(spec=DecomposedMlpotCalculator)
    calc.atomic_numbers = np.array([6, 1], dtype=int)
    calc.n_monomers = 1
    calc.cutoff_params = mock.Mock()
    calc.do_mm = True
    calc.spherical_fn = mock.Mock(
        return_value=mock.Mock(forces=jnp.zeros((2, 3), dtype=jnp.float32))
    )
    calc._resolve_mm_pairs = mock.Mock(
        return_value=(jnp.zeros((1, 2), dtype=jnp.int32), jnp.zeros((1,), dtype=jnp.bool_), False)
    )
    model = mock.Mock(spec=DecomposedMlpotModel)
    model.get_pycharmm_calculator.return_value = calc
    pos = np.zeros((2, 3), dtype=np.float64)

    forces = mlpot_spherical_forces_ev_angstrom(model, positions=pos, use_pbc=True, box_A=28.0)

    assert forces is not None
    box_arg = calc._resolve_mm_pairs.call_args.args[1]
    assert np.asarray(box_arg).shape == (3, 3)


def test_mlpot_spherical_forces_loose_pbc_jax_pme_passes_box():
    from unittest import mock

    import jax.numpy as jnp

    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import mlpot_spherical_forces_ev_angstrom
    from mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot import (
        DecomposedMlpotCalculator,
        DecomposedMlpotModel,
    )

    calc = mock.Mock(spec=DecomposedMlpotCalculator)
    calc.atomic_numbers = np.array([6, 1], dtype=int)
    calc.n_monomers = 1
    calc.cutoff_params = mock.Mock()
    calc.do_mm = True
    calc._get_update_fn = None
    calc.spherical_fn = mock.Mock(
        return_value=mock.Mock(forces=jnp.zeros((2, 3), dtype=jnp.float32))
    )
    calc._resolve_mm_pairs = mock.Mock(
        return_value=(jnp.zeros((1, 2), dtype=jnp.int32), jnp.zeros((1,), dtype=jnp.bool_), True)
    )
    model = mock.Mock(spec=DecomposedMlpotModel)
    model._cell = False
    model._jax_pme_lr_active = mock.Mock(return_value=True)
    model._periodic_mm_config = None
    model.get_pycharmm_calculator.return_value = calc
    pos = np.zeros((2, 3), dtype=np.float64)

    forces = mlpot_spherical_forces_ev_angstrom(
        model, positions=pos, use_pbc=False, box_A=31.994
    )

    assert forces is not None
    calc._resolve_mm_pairs.assert_called_once()
    box_arg = calc._resolve_mm_pairs.call_args.args[1]
    assert float(np.asarray(box_arg)[0, 0]) == pytest.approx(31.994)
    assert "box" in calc.spherical_fn.call_args.kwargs


def test_decomposed_calculator_jax_pme_uses_charmm_box_fallback():
    z = np.zeros(8, dtype=int)
    get_update_fn = MagicMock(return_value=MagicMock(return_value=(None, None)))
    calc = DecomposedMlpotCalculator(
        MagicMock(),
        CutoffParameters(),
        2,
        z,
        cell=False,
        do_mm=True,
        get_update_fn=get_update_fn,
    )
    parent = MagicMock()
    parent._jax_pme_lr_active.return_value = True
    parent._cell = False
    parent._charmm_box_side_A = 31.994
    parent._npt_restart_read = None
    calc._parent_model = parent
    calc._get_spherical_forward_fn = MagicMock(
        return_value=lambda *args, **kwargs: (jnp.array(0.0), jnp.zeros((8, 3)))
    )
    n = 8
    x = np.zeros(n, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    zc = np.zeros(n, dtype=np.float64)
    dx = np.zeros(n, dtype=np.float64)
    dy = np.zeros(n, dtype=np.float64)
    dz = np.zeros(n, dtype=np.float64)
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.resolve_mlpot_mic_box_side_A",
        return_value=(31.994, "fallback"),
    ) as mock_resolve, patch(
        "mmml.interfaces.pycharmmInterface.jax_device_policy.mlpot_jax_device_context",
        return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()),
    ):
        calc.calculate_charmm(
            n, 0, 0, None, x, y, zc, dx, dy, dz, 0, 0, None, None, None, None, None, None, None
        )

    mock_resolve.assert_called_once()
    assert mock_resolve.call_args.kwargs["fallback_side_A"] == pytest.approx(31.994)
    assert calc._cell == pytest.approx(31.994)


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


def test_probe_charmm_cubic_box_side_A_returns_none_when_unavailable():
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import probe_charmm_cubic_box_side_A

    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env._read_charmm_box_sides_A",
        return_value=(0.0, 0.0, 0.0),
    ):
        side, source = probe_charmm_cubic_box_side_A()
    assert side is None
    assert source is None


def test_resolve_mlpot_mic_box_side_A_skips_restart_when_crystal_active(tmp_path: Path):
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import resolve_mlpot_mic_box_side_A

    restart = tmp_path / "prod.res"
    restart.write_text("dummy")
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.charmm_crystal_is_active",
        return_value=True,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.resolve_charmm_cubic_box_side_A",
        return_value=(28.0, "pbound"),
    ) as mock_resolve:
        side, source = resolve_mlpot_mic_box_side_A(
            fallback_side_A=28.0,
            restart_path=restart,
        )
    assert side == pytest.approx(28.0)
    assert source == "pbound"
    mock_resolve.assert_called_once_with(
        fallback_side_A=28.0,
        restart_path=None,
        rel_tol=1e-3,
    )


def test_sync_mlpot_pbc_cell_from_charmm_updates_model():
    from mmml.interfaces.pycharmmInterface.mlpot import run_workflow

    z = np.zeros(8, dtype=int)
    model = DecomposedMlpotModel(MagicMock(), CutoffParameters(), 2, z, cell=40.0)
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.resolve_mlpot_mic_box_side_A",
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

    def _fake_forward_fn(*, n_atoms, atomic_numbers_jax, box_jax):
        captured["box_jax"] = box_jax

        def _eval(
            positions_jax,
            mm_pair_idx,
            mm_pair_mask,
            use_mm_pairs,
            spatial_monomer_indices,
            spatial_dimer_indices,
            use_spatial,
        ):
            return jnp.array(0.0), jnp.zeros((8, 3))

        return _eval

    calc._get_spherical_forward_fn = MagicMock(side_effect=_fake_forward_fn)
    n = 8
    x = np.zeros(n, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    zc = np.zeros(n, dtype=np.float64)
    dx = np.zeros(n, dtype=np.float64)
    dy = np.zeros(n, dtype=np.float64)
    dz = np.zeros(n, dtype=np.float64)
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.resolve_mlpot_mic_box_side_A",
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


def test_decomposed_calculator_queries_box_when_jax_pme_active_without_cached_cell():
    z = np.zeros(8, dtype=int)
    get_update_fn = MagicMock(return_value=MagicMock(return_value=(None, None)))
    calc = DecomposedMlpotCalculator(
        MagicMock(),
        CutoffParameters(),
        2,
        z,
        cell=False,
        do_mm=True,
        get_update_fn=get_update_fn,
    )
    parent = MagicMock()
    parent._jax_pme_lr_active.return_value = True
    calc._parent_model = parent
    captured: dict[str, Any] = {}

    def _fake_forward_fn(*, n_atoms, atomic_numbers_jax, box_jax):
        captured["box_jax"] = box_jax

        def _eval(
            positions_jax,
            mm_pair_idx,
            mm_pair_mask,
            use_mm_pairs,
            spatial_monomer_indices,
            spatial_dimer_indices,
            use_spatial,
        ):
            return jnp.array(0.0), jnp.zeros((8, 3))

        return _eval

    calc._get_spherical_forward_fn = MagicMock(side_effect=_fake_forward_fn)
    n = 8
    x = np.zeros(n, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    zc = np.zeros(n, dtype=np.float64)
    dx = np.zeros(n, dtype=np.float64)
    dy = np.zeros(n, dtype=np.float64)
    dz = np.zeros(n, dtype=np.float64)
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.resolve_mlpot_mic_box_side_A",
        return_value=(37.5, "pbound"),
    ) as mock_resolve, patch(
        "mmml.interfaces.pycharmmInterface.jax_device_policy.mlpot_jax_device_context",
        return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()),
    ):
        calc.calculate_charmm(
            n, 0, 0, None, x, y, zc, dx, dy, dz, 0, 0, None, None, None, None, None, None, None
        )

    mock_resolve.assert_called_once()
    assert calc._cell == pytest.approx(37.5)
    box = captured["box_jax"]
    assert box is not None
    assert float(box[0, 0]) == pytest.approx(37.5)
    get_update_fn.assert_called_once()
    forwarded_box = get_update_fn.call_args.kwargs["box"]
    assert forwarded_box is not None
    assert float(forwarded_box[0, 0]) == pytest.approx(37.5)


def test_decomposed_calculator_propagates_box_sync_failure():
    z = np.zeros(8, dtype=int)
    calc = DecomposedMlpotCalculator(
        MagicMock(),
        CutoffParameters(),
        2,
        z,
        cell=40.0,
    )
    n = 8
    x = np.zeros(n, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    zc = np.zeros(n, dtype=np.float64)
    dx = np.zeros(n, dtype=np.float64)
    dy = np.zeros(n, dtype=np.float64)
    dz = np.zeros(n, dtype=np.float64)
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.resolve_mlpot_mic_box_side_A",
        side_effect=RuntimeError("CHARMM box is not cubic"),
    ), patch(
        "mmml.interfaces.pycharmmInterface.jax_device_policy.mlpot_jax_device_context",
        return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()),
    ):
        with pytest.raises(RuntimeError, match="not cubic"):
            calc.calculate_charmm(
                n, 0, 0, None, x, y, zc, dx, dy, dz, 0, 0, None, None, None, None, None, None, None
            )
    assert calc._cell == pytest.approx(40.0)


def test_spherical_forward_fn_cached_across_callbacks(monkeypatch):
    monkeypatch.setenv("MMML_MLPOT_DEVICE", "cpu")
    z = np.zeros(8, dtype=int)

    def spherical_fn(**kwargs):
        pos = kwargs["positions"]
        energy = jnp.sum(pos**2)

        class Out:
            pass

        out = Out()
        out.energy = energy
        out.forces = 2.0 * pos
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
    first_key = calc._grad_cache_owner()._forward_cache_key
    calc.calculate_charmm(
        n, 0, 0, None, x + 0.01, y, zc, dx, dy, dz, 0, 0, None, None, None, None, None, None, None
    )
    second_key = calc._grad_cache_owner()._forward_cache_key

    assert first_key is not None
    assert second_key == first_key


def test_build_ml_exclusion_lists_upper_triangle():
    from mmml.interfaces.pycharmmInterface.mlpot.setup import _build_ml_exclusion_lists

    iblo, inb = _build_ml_exclusion_lists([0, 2, 4], natom=6)
    assert list(iblo) == [2, 2, 3, 3, 3, 3]
    assert inb == [3, 5, 5]


def test_calculator_wrapping_translation_invariance():
    from mmml.interfaces.pycharmmInterface.mmml_calculator import setup_calculator
    from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters

    # 1. Setup a dummy model that computes a distance-based energy
    class DummyModel:
        def __init__(self, **kwargs):
            self.natoms = 8
            self.use_pbc = False
            
        def apply(self, params, **kwargs):
            R = kwargs.get("positions")
            # R shape: (batch_size * natoms, 3)
            R_3d = R.reshape(-1, self.natoms, 3)
            r = R_3d[:, 0] - R_3d[:, 4]
            dist = jnp.linalg.norm(r, axis=-1)
            
            grad = r / jnp.expand_dims(jnp.maximum(dist, 1e-10), -1)
            forces_3d = jnp.zeros_like(R_3d)
            forces_3d = forces_3d.at[:, 0].set(-grad)
            forces_3d = forces_3d.at[:, 4].set(grad)
            
            return {"energy": dist, "forces": forces_3d.reshape(-1, 3)}

    dummy_checkpoint = {
        "params": {},
        "config": {
            "features": 32,
            "max_degree": 3,
            "num_iterations": 2,
            "num_basis_functions": 16,
            "cutoff": 6.0,
            "max_atomic_number": 118,
            "charges": False,
            "include_electrostatics": False,
            "natoms": 8,
            "total_charge": 0,
            "n_res": 3,
            "zbl": True,
            "debug": False,
            "efa": False,
            "use_energy_bias": False,
            "use_pbc": True,
        }
    }

    # 2. Evaluate energy and forces for initial coordinates
    atomic_numbers = jnp.array([6, 1, 1, 1, 6, 1, 1, 1], dtype=jnp.int32)
    # Monomer A around [0, 0, 0], Monomer B around [29, 0, 0] (distance 1 across periodic boundary of size 30)
    R_initial = jnp.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.2, 0.0, 0.0],
        [0.3, 0.0, 0.0],
        [29.0, 0.0, 0.0],
        [29.1, 0.0, 0.0],
        [29.2, 0.0, 0.0],
        [29.3, 0.0, 0.0],
    ], dtype=jnp.float32)

    restart_path = MagicMock(spec=Path)
    restart_path.is_file.return_value = True
    restart_path.suffix = ".json"
    restart_path.resolve.return_value = restart_path

    with patch("mmml.utils.model_checkpoint.load_model_checkpoint", return_value=dummy_checkpoint), \
         patch("mmml.models.physnetjax.physnetjax.models.model.PhysNet", DummyModel):
        factory = setup_calculator(
            ATOMS_PER_MONOMER=[4, 4],
            N_MONOMERS=2,
            cell=30.0,
            model_restart_path=restart_path,
            doMM=False,
        )
        calculator, configured_spherical_cutoff, update_fn_factory = factory(
            atomic_numbers,
            R_initial,
            2,
            create_ase_calculator=True,
        )

    # Reconstruct box as passed in JAX-MD
    box = jnp.array([30.0, 30.0, 30.0], dtype=jnp.float32)

    cutoff_params = CutoffParameters(
        ml_switch_width=0.5,
        mm_switch_on=6.0,
        mm_switch_width=4.0,
    )

    out_initial = configured_spherical_cutoff(
        positions=R_initial,
        atomic_numbers=atomic_numbers,
        n_monomers=2,
        cutoff_params=cutoff_params,
        box=box,
    )

    # 3. Evaluate on shifted coordinates (Monomer B shifted by [30.0, 0, 0])
    R_shifted = R_initial.at[4:8].add(jnp.array([30.0, 0.0, 0.0]))

    out_shifted = configured_spherical_cutoff(
        positions=R_shifted,
        atomic_numbers=atomic_numbers,
        n_monomers=2,
        cutoff_params=cutoff_params,
        box=box,
    )

    # Assert energy and forces are exactly the same
    assert np.allclose(out_initial.energy, out_shifted.energy, atol=1e-5)
    assert np.allclose(out_initial.forces, out_shifted.forces, atol=1e-5)

