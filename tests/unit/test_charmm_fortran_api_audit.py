from pathlib import Path

import scripts.audit_charmm_fortran_api as api_audit
from scripts.audit_charmm_fortran_api import audit_argument, probe_shared_library, scan


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


def test_explicit_bounds_struct_array_is_not_descriptor_error():
    arg = audit_argument(
        "label",
        "character(kind=c_char, len=1), dimension(1:9) :: label",
    )
    assert not any(x["code"] == "assumed_shape_array" for x in arg.issues)


def test_entity_declarator_array_shape_is_recorded():
    arg = audit_argument(
        "label",
        "character(kind=c_char), intent(in) :: label(*)",
    )
    assert arg.dimension == "*"
    assert not any(x["code"] == "character_length" for x in arg.issues)


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


def test_repository_api_surface_includes_interoperable_types_and_enums():
    report = scan(
        REPO / "setup/charmm/source/api",
        REPO / "setup/charmm/tool/pycharmm/pycharmm",
    )
    rows = {row["name"]: row for row in report["data_types"]}
    assert "dynamics_settings" in rows
    fields = {field["name"]: field for field in rows["dynamics_settings"]["components"]}
    assert {"ieqfrq", "ntrfrq", "ichecw", "tbath", "iasvel"} <= fields.keys()
    assert fields["tbath"]["type_spec"] == "real(c_double)"
    assert not rows["dynamics_settings"]["issues"]
    assert report["summary"]["bind_c_types"] == 7
    assert report["summary"]["bind_c_enums"] == 2
    assert report["summary"]["total_bind_c_surface_entries"] == 345


def test_optional_shared_library_probe_reports_missing_exports(monkeypatch, tmp_path):
    class FakeLibrary:
        known_symbol = object()

    monkeypatch.setattr(api_audit.ctypes, "CDLL", lambda _: FakeLibrary())
    result = probe_shared_library(
        tmp_path / "libcharmm.so",
        {"routines": [{"symbol": "known_symbol"}, {"symbol": "missing_symbol"}]},
    )
    assert result["found_symbols"] == 1
    assert result["missing_symbols"] == ["missing_symbol"]
    assert result["load_error"] is None
