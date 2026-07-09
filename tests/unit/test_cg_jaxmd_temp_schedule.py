"""Unit tests for NVT temperature staging schedules."""

import sys
from pathlib import Path
import pytest

# Add examples folder to path to import cg_common
path = Path(__file__).resolve().parents[2] / "examples"
sys.path.insert(0, str(path))

from cg_common import parse_temp_schedule

def test_parse_temp_schedule():
    # 1. Constant temperature
    f = parse_temp_schedule("298.0", 1000)
    assert f(0) == 298.0
    assert f(500) == 298.0
    assert f(1000) == 298.0

    # 2. Linear ramp
    f = parse_temp_schedule("298.0->398.0", 1000)
    assert f(0) == 298.0
    assert f(500) == 348.0
    assert f(1000) == 398.0

    # 3. Complex staged schedule
    # 25% ramp from 200 to 300, 50% hold at 300, 25% cool from 300 to 200
    f = parse_temp_schedule("200.0->300.0:0.25, 300.0:0.5, 300.0->200.0:0.25", 1000)
    assert f(0) == 200.0
    assert f(250) == 300.0
    assert f(500) == 300.0
    assert f(750) == 300.0
    assert f(1000) == 200.0
    
    # check mid-ramp values
    assert abs(f(125) - 250.0) < 1e-3
    assert abs(f(875) - 250.0) < 1e-3
