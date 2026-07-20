"""Explicit calculator selection for supported dimer scans."""

from __future__ import annotations

from collections.abc import Callable

from ase.calculators.calculator import Calculator

from mmml.analysis.dimer_scans import make_xtb_calculator

from .config import DimerScanConfig


def calculator_factory(config: DimerScanConfig) -> Callable[[], Calculator]:
    """Validate calculator requirements and return an ASE calculator factory."""

    if config.calculator == "xtb":
        if config.checkpoint is not None:
            raise ValueError("the xtb calculator does not accept a checkpoint")
        return make_xtb_calculator
    if config.calculator == "physnet":
        if config.checkpoint is None:
            raise ValueError("the physnet calculator requires --checkpoint")
        checkpoint = config.checkpoint.resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")

        def create() -> Calculator:
            from mmml.interfaces.calculators.simple_inference import (
                create_calculator_from_checkpoint,
            )

            return create_calculator_from_checkpoint(
                checkpoint,
                charge=config.charge,
                spin=config.spin,
            )

        return create
    raise ValueError(
        f"unsupported calculator {config.calculator!r}; choose 'physnet' or 'xtb'"
    )
