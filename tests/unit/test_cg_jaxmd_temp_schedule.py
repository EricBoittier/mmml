"""Unit tests for NVT temperature staging schedules in cg_jaxmd.py."""

import importlib.util
from pathlib import Path
import pytest
import sys

def _load_cg_jaxmd():
    path = Path(__file__).resolve().parents[2] / "examples"
    sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location("cg_jaxmd_for_tests", str(path / "cg_jaxmd.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

cg_jaxmd = _load_cg_jaxmd()

def test_parse_temp_schedule():
    # 1. Constant temperature
    f = cg_jaxmd.parse_temp_schedule("298.0", 1000)
    assert f(0) == 298.0
    assert f(500) == 298.0
    assert f(1000) == 298.0

    # 2. Linear ramp
    f = cg_jaxmd.parse_temp_schedule("298.0->398.0", 1000)
    assert f(0) == 298.0
    assert f(500) == 348.0
    assert f(1000) == 398.0

    # 3. Complex staged schedule
    # 25% ramp from 200 to 300, 50% hold at 300, 25% cool from 300 to 200
    f = cg_jaxmd.parse_temp_schedule("200.0->300.0:0.25, 300.0:0.5, 300.0->200.0:0.25", 1000)
    assert f(0) == 200.0
    assert f(250) == 300.0
    assert f(500) == 300.0
    assert f(750) == 300.0
    assert f(1000) == 200.0
    
    # check mid-ramp values
    assert abs(f(125) - 250.0) < 1e-3
    assert abs(f(875) - 250.0) < 1e-3
