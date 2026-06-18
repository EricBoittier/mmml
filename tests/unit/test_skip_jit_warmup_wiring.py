"""Unit tests for --skip-jit-warmup forwarding and defer_xla_gpu_warmup wiring."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.mark.parametrize("skip", [True, False])
def test_setup_calculator_respects_defer_xla_gpu_warmup(skip: bool):
    ckpt = Path(__file__).resolve().parents[2] / "examples/ckpts_json/DESdimers_params.json"
    if not ckpt.is_file():
        pytest.skip("DESdimers_params.json checkpoint missing")

    from mmml.interfaces.pycharmmInterface.mmml_calculator import setup_calculator

    with patch(
        "mmml.interfaces.pycharmmInterface.mmml_calculator.ensure_xla_gpu_warmed",
        return_value=False,
    ) as mock_warm:
        setup_calculator(
            ATOMS_PER_MONOMER=5,
            N_MONOMERS=2,
            doML=True,
            doMM=False,
            model_restart_path=str(ckpt),
            MAX_ATOMS_PER_SYSTEM=10,
            cell=38.0,
            defer_xla_gpu_warmup=skip,
            verbose=False,
        )
    if skip:
        mock_warm.assert_not_called()
    else:
        mock_warm.assert_called()
