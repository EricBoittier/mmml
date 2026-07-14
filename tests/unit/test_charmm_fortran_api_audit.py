from pathlib import Path

from scripts.audit_charmm_fortran_api import audit_argument, scan


REPO = Path(__file__).resolve().parents[2]


def test_assumed_shape_ctypes_hazard_is_an_error():
    arg = audit_argument(
        "values",
        "real(c_double), dimension(:), intent(in) :: values",
    )
    assert any(x["code"] == "assumed_shape_array" and x["severity"] == "error" for x in arg.issues)


def test_assumed_size_raw_pointer_is_not_descriptor_error():
    arg = audit_argument(
        "values",
        "real(c_double), dimension(*), intent(in) :: values",
    )
    assert not any(x["code"] == "assumed_shape_array" for x in arg.issues)


def test_repository_api_surface_includes_fixed_dynamics_entrypoints():
    report = scan(
        REPO / "setup/charmm/source/api",
        REPO / "setup/charmm/tool/pycharmm/pycharmm",
    )
    rows = {row["symbol"]: row for row in report["routines"]}
    for symbol in ("dynamics_run", "dynamics_run_kw"):
        assert symbol in rows
        assert not any(x["code"] == "assumed_shape_array" for x in rows[symbol]["issues"])
        velocity_args = [x for x in rows[symbol]["arguments"] if x["name"].startswith(("in_v", "out_v"))]
        assert len(velocity_args) == 6
        assert all(x["dimension"] == "*" for x in velocity_args)

