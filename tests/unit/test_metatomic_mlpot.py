"""Unit tests for metatomic CHARMM MLpot adapter (dummy ASE, no torch/CHARMM)."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
from ase.calculators.calculator import Calculator, all_changes

from mmml.data.units import EV_TO_KCAL_MOL
from mmml.interfaces.pycharmmInterface.mlpot.metatomic_mlpot import (
    MetatomicMlpotCalculator,
    MetatomicMlpotModel,
    build_metatomic_mlpot_model,
    resolve_metatomic_eval_mode,
    should_use_metatomic_mlpot,
)


class DummyAseCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, energy_ev: float = 1.0, force_ev: float = 0.25):
        super().__init__()
        self.energy_ev = float(energy_ev)
        self.force_ev = float(force_ev)

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        n = len(atoms)
        forces = np.full((n, 3), self.force_ev, dtype=np.float64)
        self.results = {"energy": self.energy_ev, "forces": forces}


def test_should_use_metatomic_mlpot_mode_and_suffix(tmp_path: Path) -> None:
    assert should_use_metatomic_mlpot(None, Namespace(ml_potential_mode="metatomic"))
    assert should_use_metatomic_mlpot(None, Namespace(ml_potential_mode="metatensor"))
    assert not should_use_metatomic_mlpot(None, Namespace(ml_potential_mode="physnet"))
    pt = tmp_path / "export.pt"
    pt.write_bytes(b"stub")
    assert should_use_metatomic_mlpot(pt, Namespace(ml_potential_mode="physnet"))
    assert not should_use_metatomic_mlpot(tmp_path / "params.json")


def test_resolve_metatomic_eval_mode_defaults_and_rejects() -> None:
    assert resolve_metatomic_eval_mode() == "fragments"
    assert resolve_metatomic_eval_mode(Namespace(metatomic_eval_mode=None)) == "fragments"
    assert resolve_metatomic_eval_mode(explicit="whole-system") == "whole_system"
    with pytest.raises(ValueError, match="metatomic_eval_mode"):
        resolve_metatomic_eval_mode(explicit="batches")


def test_calculate_charmm_kcal_and_force_sign() -> None:
    dummy = DummyAseCalculator(energy_ev=2.0, force_ev=0.5)
    n = 4
    calc = MetatomicMlpotCalculator(
        dummy,
        atomic_numbers=np.array([6, 1, 1, 1], dtype=int),
        atoms_per_monomer=[4],
        eval_mode="whole_system",
        do_ml=True,
        do_ml_dimer=False,
        do_mm=False,
    )
    x = [float(i) for i in range(n)]
    y = [0.0] * n
    z = [0.0] * n
    dx = [0.0] * n
    dy = [0.0] * n
    dz = [0.0] * n
    energy = calc.calculate_charmm(
        n, 0, 0, None, x, y, z, dx, dy, dz, 0, 0, None, None, None, None, None, None, None
    )
    assert energy == pytest.approx(2.0 * EV_TO_KCAL_MOL)
    expected_f = 0.5 * EV_TO_KCAL_MOL
    assert dx[0] == pytest.approx(-expected_f)
    assert dy[0] == pytest.approx(-expected_f)
    assert dz[0] == pytest.approx(-expected_f)


def test_calculate_charmm_respects_ml_atom_indices() -> None:
    dummy = DummyAseCalculator(energy_ev=1.0, force_ev=1.0)
    calc = MetatomicMlpotCalculator(
        dummy,
        atomic_numbers=np.array([8, 1], dtype=int),
        atoms_per_monomer=[2],
        eval_mode="whole_system",
        ml_atom_indices=[1, 3],
    )
    n = 5
    x = [0.0] * n
    y = [0.0] * n
    z = [0.0] * n
    dx = [0.0] * n
    dy = [0.0] * n
    dz = [0.0] * n
    calc.calculate_charmm(
        n, 0, 0, None, x, y, z, dx, dy, dz, 0, 0, None, None, None, None, None, None, None
    )
    expected = -1.0 * EV_TO_KCAL_MOL
    assert dx[1] == pytest.approx(expected)
    assert dx[3] == pytest.approx(expected)
    assert dx[0] == pytest.approx(0.0)
    assert dx[2] == pytest.approx(0.0)
    assert dx[4] == pytest.approx(0.0)


def test_build_metatomic_mlpot_model_injected_calculator(tmp_path: Path) -> None:
    ckpt = tmp_path / "export.pt"
    ckpt.write_bytes(b"stub")
    dummy = DummyAseCalculator()
    model = build_metatomic_mlpot_model(
        ckpt,
        np.array([1, 1], dtype=int),
        [1, 1],
        2,
        calculator=dummy,
        do_mm=False,
        eval_mode="fragments",
    )
    assert isinstance(model, MetatomicMlpotModel)
    calc = model.get_pycharmm_calculator(ml_atom_indices=[0, 1])
    assert isinstance(calc, MetatomicMlpotCalculator)
    assert calc.eval_mode == "fragments"
    model.set_cell(12.0)
    assert model._cell == pytest.approx(12.0)
    assert calc._cell == pytest.approx(12.0)


def test_build_decomposed_mlpot_metatomic_early_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mmml.interfaces.pycharmmInterface.mlpot import hybrid_mlpot

    sentinel = object()

    def _fake_build(*_a, **_k):
        return sentinel

    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.metatomic_mlpot.build_metatomic_mlpot_model",
        _fake_build,
    )
    ckpt = tmp_path / "export.pt"
    ckpt.write_bytes(b"stub")
    model = hybrid_mlpot.build_decomposed_mlpot_model(
        ckpt,
        np.array([8, 1, 8, 1], dtype=int),
        [2, 2],
        2,
        args=Namespace(ml_potential_mode="metatomic", include_mm=False),
    )
    assert model is sentinel


def test_warmup_decomposed_mlpot_skips_metatomic_model(tmp_path: Path) -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot import (
        warmup_decomposed_mlpot,
    )

    ckpt = tmp_path / "export.pt"
    ckpt.write_bytes(b"stub")
    dummy = DummyAseCalculator()
    model = build_metatomic_mlpot_model(
        ckpt,
        np.array([1, 1], dtype=int),
        [1, 1],
        2,
        calculator=dummy,
        do_mm=False,
    )
    warmup_decomposed_mlpot(model, np.zeros((2, 3)), verbose=False)


def test_md_system_parser_accepts_metatomic() -> None:
    from mmml.cli.run.md_system import build_parser, build_pycharmm_command

    from tests.unit.test_md_system_pycharmm_cmd import _pycharmm_args

    args = build_parser().parse_args(
        ["--ml-potential-mode", "metatomic", "--metatomic-eval-mode", "whole_system"]
    )
    assert args.ml_potential_mode == "metatomic"
    assert args.metatomic_eval_mode == "whole_system"
    cmd = build_pycharmm_command(
        _pycharmm_args(
            ml_potential_mode="metatomic",
            metatomic_eval_mode="fragments",
        )
    )
    assert "--ml-potential-mode" in cmd
    assert "metatomic" in cmd
    assert "--metatomic-eval-mode" in cmd
    assert "fragments" in cmd


def test_setup_calculator_metatomic_rejects_do_ml() -> None:
    try:
        from jax_md import space as _jax_md_space
    except Exception as exc:
        pytest.skip(f"jax_md import unavailable ({exc})")
    del _jax_md_space
    from mmml.interfaces.pycharmmInterface.mmml_calculator import setup_calculator

    with pytest.raises(ValueError, match="MetatomicMlpotCalculator"):
        setup_calculator(
            [3, 3],
            N_MONOMERS=2,
            doML=True,
            doMM=False,
            doML_dimer=False,
            model_restart_path="/tmp/fake-metatomic.pt",
            ml_potential_mode="metatomic",
        )


def test_ic_scan_and_dimer_scan_parsers_include_metatomic() -> None:
    from mmml.cli.misc.dimer_scan import build_parser as dimer_parser
    from mmml.cli.misc.ic_scan import SUPPORTED_CALCULATORS

    dimer_action = next(item for item in dimer_parser()._actions if item.dest == "calculator")
    assert "metatomic" in set(dimer_action.choices)
    assert "metatomic" in set(SUPPORTED_CALCULATORS)
