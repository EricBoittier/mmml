from __future__ import annotations

from pathlib import Path

from mmml.dimer_scan.calculators import calculator_factory
from mmml.dimer_scan.config import DimerScanConfig


def test_physnet_factory_forwards_charge_and_spin(monkeypatch, tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text("{}")
    captured = {}
    sentinel = object()

    def fake_loader(path, **kwargs):
        captured.update(path=path, **kwargs)
        return sentinel

    monkeypatch.setattr(
        "mmml.interfaces.calculators.simple_inference.create_calculator_from_checkpoint",
        fake_loader,
    )
    config = DimerScanConfig(
        residues=("MEOH", "MEOH"),
        calculator="physnet",
        checkpoint=checkpoint,
        distances_angstrom=(3.0,),
        charge=-1.0,
        spin=2.0,
    )

    calculator = calculator_factory(config)()

    assert calculator is sentinel
    assert captured == {
        "path": checkpoint.resolve(),
        "charge": -1.0,
        "spin": 2.0,
    }
