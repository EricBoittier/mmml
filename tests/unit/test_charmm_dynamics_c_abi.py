"""Regression checks for the PyCHARMM dynamics velocity-buffer C ABI."""

from pathlib import Path
import re


REPO = Path(__file__).resolve().parents[2]
API = REPO / "setup/charmm/source/api/api_dynamics.F90"
DYNOPT = REPO / "setup/charmm/source/dynamc/dcntrl.F90"


def _function_block(name: str) -> str:
    text = API.read_text(encoding="utf-8")
    match = re.search(
        rf"integer\(c_int\) function {name}\b.*?end function {name}",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    assert match is not None
    return match.group(0)


def test_ctypes_velocity_buffers_use_raw_pointer_abi():
    """ctypes arrays are ``double *`` and must not be read as CFI descriptors."""
    for name in ("dynamics_run", "dynamics_run_kw"):
        block = _function_block(name)
        assert "dimension(*)" in block
        assert "dimension(:)" not in block

    dynopt_head = DYNOPT.read_text(encoding="utf-8").split("!     begin", 1)[0]
    dummy_decl = re.search(
        r"real\(c_double\),\s*dimension\((.*?)\),\s*optional\s*::\s*&\s*"
        r"in_vx,\s*in_vy,\s*in_vz,\s*&\s*out_vx,\s*out_vy,\s*out_vz",
        dynopt_head,
        flags=re.IGNORECASE | re.DOTALL,
    )
    assert dummy_decl is not None
    assert dummy_decl.group(1).strip() == "*"


def test_dynamics_run_returns_xyz_to_distinct_buffers():
    block = _function_block("dynamics_run")
    assert "out_vx, out_vy, out_vz)" in block
    assert "out_vz, out_vy, out_vz)" not in block
