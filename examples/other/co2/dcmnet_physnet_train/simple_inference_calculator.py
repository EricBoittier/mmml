#!/usr/bin/env python3
"""Compatibility shim for simple inference calculator.

The implementation has moved to ``mmml.calculators.simple_inference``.
This module simply re-exports the public API to preserve backwards
compatibility with existing scripts under ``examples/co2``.
"""

from mmml.calculators.simple_inference import (  # noqa: F401
    SimpleInferenceCalculator,
    create_calculator_from_checkpoint,
)

