import json
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import pycharmm.dynamics as dynamics


def _record(payload):
    marker = struct.pack("<i", len(payload))
    return marker + payload + marker


def _write_lmd(
    path,
    form=0,
    nblocks=3,
    nsites=2,
    magic=b"MSLD",
    lambda_temperature=298.15,
    bielam=None,
    write_frames=True,
):
    icntrl = [0] * 20
    icntrl[1] = 5
    icntrl[2] = 5
    icntrl[6] = nblocks
    icntrl[10] = nsites
    icntrl[11] = form
    theta_count = nsites - 1 if form == 2 else nblocks - 1
    raw = bytearray()
    raw += _record(struct.pack("<4s20i", magic, *icntrl))
    raw += _record(struct.pack("<f", 1.0))
    raw += _record(b"title")
    raw += _record(struct.pack("<i", 0))
    raw += _record(b"")
    raw += _record(struct.pack(f"<{nblocks}i", *range(nblocks)))
    raw += _record(struct.pack("<f", lambda_temperature))
    if bielam is None:
        bielam = [0.0] * nblocks
    raw += _record(struct.pack(f"<{nblocks}f", *bielam))
    if write_frames:
        first = [1.0] + [0.25] * (nblocks - 1)
        second = [1.0] + [0.50] * (nblocks - 1)
        raw += _record(struct.pack(f"<{nblocks}f", *first))
        raw += _record(struct.pack(f"<{theta_count}f", *([0.0] * theta_count)))
        raw += _record(struct.pack(f"<{nblocks}f", *second))
        raw += _record(struct.pack(f"<{theta_count}f", *([0.0] * theta_count)))
    path.write_bytes(raw)


def test_lambda_parquet_frame_reads_native_lmd(tmp_path):
    path = tmp_path / "lambda.lmd"
    _write_lmd(
        path,
        lambda_temperature=310.0,
        bielam=[0.0, 2.0, -3.0],
    )

    frame = dynamics._lambda_parquet_frame(path)

    delta_t = 0.0488882129
    assert list(frame.columns) == ["time", "LAM01", "LAM02"]
    assert frame["time"].tolist() == pytest.approx([5 * delta_t, 10 * delta_t])
    assert frame["LAM01"].tolist() == pytest.approx([0.25, 0.50])
    assert frame["LAM02"].tolist() == pytest.approx([0.25, 0.50])
    assert frame.attrs["lambda_temperature"] == pytest.approx(310.0)
    assert frame.attrs["bielam"] == pytest.approx([0.0, 2.0, -3.0])


def test_lambda_parquet_preserves_replica_state_metadata(tmp_path):
    pyarrow_parquet = pytest.importorskip("pyarrow.parquet")
    lmd_path = tmp_path / "lambda.lmd"
    parquet_path = tmp_path / "lambda.parquet"
    _write_lmd(
        lmd_path,
        lambda_temperature=310.0,
        bielam=[0.0, 2.0, -3.0],
    )

    dynamics._write_lambda_parquet(
        lmd_path,
        parquet_path,
        ph=5.5,
        mpi_rank=2,
    )

    metadata = pyarrow_parquet.read_schema(parquet_path).metadata
    assert float(metadata[b"charmm.lambda_temperature"]) == pytest.approx(310.0)
    assert json.loads(metadata[b"charmm.bielam"]) == pytest.approx([0.0, 2.0, -3.0])
    assert float(metadata[b"charmm.ph"]) == pytest.approx(5.5)
    assert int(metadata[b"charmm.mpi_rank"]) == 2
    frame = pd.read_parquet(parquet_path)
    assert frame.attrs["lambda_temperature"] == pytest.approx(310.0)
    assert frame.attrs["bielam"] == pytest.approx([0.0, 2.0, -3.0])
    assert frame.attrs["ph"] == pytest.approx(5.5)
    assert frame.attrs["mpi_rank"] == 2


def test_lambda_parquet_empty_file_keeps_numeric_schema(tmp_path):
    pyarrow = pytest.importorskip("pyarrow")
    lmd_path = tmp_path / "empty.lmd"
    parquet_path = tmp_path / "empty.parquet"
    _write_lmd(lmd_path, write_frames=False)

    dynamics._write_lambda_parquet(lmd_path, parquet_path)

    schema = pyarrow.parquet.read_schema(parquet_path)
    assert schema.field("time").type == pyarrow.float64()
    assert schema.field("LAM01").type == pyarrow.float32()


@pytest.mark.parametrize(
    "lambda_temperature,bielam",
    [
        (float("nan"), [0.0, 2.0, -3.0]),
        (310.0, [0.0, float("nan"), -3.0]),
    ],
)
def test_lambda_parquet_rejects_nonfinite_metadata(
    tmp_path,
    lambda_temperature,
    bielam,
):
    pytest.importorskip("pyarrow")
    lmd_path = tmp_path / "lambda.lmd"
    _write_lmd(
        lmd_path,
        lambda_temperature=lambda_temperature,
        bielam=bielam,
    )

    with pytest.raises(ValueError, match="Out of range"):
        dynamics._write_lambda_parquet(
            lmd_path,
            tmp_path / "lambda.parquet",
        )


def test_lambda_parquet_frame_reads_2sin_msld_records(tmp_path):
    path = tmp_path / "lambda_2sin.lmd"
    _write_lmd(path, form=2, nblocks=4, nsites=2)

    frame = dynamics._lambda_parquet_frame(path)

    assert frame.shape == (2, 4)
    assert list(frame.columns) == ["time", "LAM01", "LAM02", "LAM03"]


def test_lambda_parquet_frame_rejects_non_msld_magic(tmp_path):
    path = tmp_path / "lambda.lmd"
    _write_lmd(path, magic=b"LAMB")

    with pytest.raises(ValueError, match="MSLD lambda files only"):
        dynamics._lambda_parquet_frame(path)


def test_lambda_parquet_frame_rejects_truncated_lmd(tmp_path):
    path = tmp_path / "lambda.lmd"
    path.write_bytes(struct.pack("<i", 20) + b"short")

    with pytest.raises(ValueError, match="truncated"):
        dynamics._lambda_parquet_frame(path)


def test_lambda_parquet_frame_rejects_corrupt_frame_marker(tmp_path):
    path = tmp_path / "lambda.lmd"
    _write_lmd(path)
    raw = bytearray(path.read_bytes())
    struct.pack_into("<i", raw, len(raw) - 4, 123)
    path.write_bytes(raw)

    with pytest.raises(ValueError, match="frame record marker mismatch"):
        dynamics._lambda_parquet_frame(path)


def test_lambda_parquet_path_requires_one_file_per_mpi_rank(
    tmp_path,
    monkeypatch,
):
    import pycharmm.replica_exchange as replica_exchange

    comm = SimpleNamespace(Get_rank=lambda: 2, Get_size=lambda: 4)
    monkeypatch.setattr(replica_exchange, "_default_comm", lambda: comm)

    assert (
        dynamics._lambda_parquet_path_for_rank(tmp_path / "lambda_{rank}.parquet")
        == tmp_path / "lambda_2.parquet"
    )
    with pytest.raises(ValueError, match=r"\{rank\}"):
        dynamics._lambda_parquet_path_for_rank(tmp_path / "lambda.parquet")


def test_lambda_parquet_preflight_rejects_unknown_codec(tmp_path):
    pytest.importorskip("pyarrow")

    with pytest.raises(ValueError, match="unsupported.*codec"):
        dynamics._preflight_lambda_parquet(
            tmp_path / "lambda.parquet",
            "not-a-codec",
        )


def test_lambda_parquet_preflight_rejects_directory(tmp_path):
    pytest.importorskip("pyarrow")

    with pytest.raises(IsADirectoryError, match="destination is a directory"):
        dynamics._preflight_lambda_parquet(tmp_path, "snappy")


@pytest.mark.parametrize("replica_suffix", [False, True])
def test_dynamics_lambda_parquet_manages_native_output(
    tmp_path,
    monkeypatch,
    replica_suffix,
):
    import pycharmm.block as block
    import pycharmm.replica_exchange as replica_exchange

    seen = {}
    comm = SimpleNamespace(Get_rank=lambda: 2, Get_size=lambda: 4)
    monkeypatch.setattr(replica_exchange, "_default_comm", lambda: comm)
    monkeypatch.setattr(
        block,
        "get_ph_direct",
        lambda: float(np.float32(6.3)),
    )
    monkeypatch.setattr(
        dynamics,
        "_preflight_lambda_parquet",
        lambda *args, **kwargs: None,
    )

    def fake_run(self, append=""):
        seen["iunldm"] = self.opts["iunldm"]
        return self

    monkeypatch.setattr(dynamics.script.CommandScript, "run", fake_run)
    lambda_path = tmp_path / "lambda_tmp.lmd"

    def fake_tempfile(**kwargs):
        lambda_path.touch()
        return SimpleNamespace(name=str(lambda_path), close=lambda: None)

    monkeypatch.setattr(
        dynamics.tempfile,
        "NamedTemporaryFile",
        fake_tempfile,
    )

    class FakeCharmmFile:
        def __init__(self, file_name, file_unit, read_only, formatted):
            seen["opened"] = (file_name, file_unit, read_only, formatted)
            if replica_suffix:
                Path(file_name + "_2").touch()
            self.file_unit = 77
            self.is_open = True

        def close(self):
            seen["closed"] = True
            return True

    monkeypatch.setattr(dynamics, "CharmmFile", FakeCharmmFile)

    def fake_write(
        lmd_path,
        filepath,
        compression="snappy",
        ph=None,
        mpi_rank=None,
    ):
        seen["write"] = (lmd_path, filepath, compression, ph, mpi_rank)
        return filepath

    monkeypatch.setattr(dynamics, "_write_lambda_parquet", fake_write)

    output = tmp_path / "lambda_{rank}.parquet"
    dyn = dynamics.DynamicsScript(
        lambda_parquet=output,
        lambda_parquet_ph=6.3,
        iunldm=-1,
        nstep=1,
        nsavl=1,
    )

    assert dyn.run()
    assert dyn.fill_lambdata is False
    assert seen["iunldm"] == "iunldm 77 -\n"
    assert seen["opened"] == (
        str(tmp_path / "lambda_tmp.lmd"),
        -1,
        False,
        False,
    )
    assert seen["closed"]
    expected_lambda_path = Path(str(lambda_path) + ("_2" if replica_suffix else ""))
    assert seen["write"] == (
        expected_lambda_path,
        tmp_path / "lambda_2.parquet",
        "snappy",
        6.3,
        2,
    )
    assert not lambda_path.exists()
    assert not expected_lambda_path.exists()
    assert dyn.opts["iunldm"] == "iunldm -1 -\n"


@pytest.mark.parametrize("active_ph", [6.5, float("nan")])
def test_lambda_parquet_rejects_stale_ph_label(
    tmp_path,
    monkeypatch,
    active_ph,
):
    import pycharmm.block as block

    monkeypatch.setattr(
        block,
        "get_ph_direct",
        lambda: active_ph,
    )
    dyn = dynamics.DynamicsScript(
        lambda_parquet=tmp_path / "lambda.parquet",
        lambda_parquet_ph=5.5,
        nstep=1,
        nsavl=1,
    )

    with pytest.raises(ValueError, match="does not match"):
        dyn.run()


def test_lambda_parquet_keeps_existing_positional_arguments():
    dyn = dynamics.DynamicsScript(
        False,
        False,
        False,
        True,
    )

    assert dyn.fill_msldata
    assert dyn.lambda_parquet_ph is None


@pytest.mark.parametrize(
    "is_open, close_ok, message, dynamics_runs",
    [
        (False, True, "failed to open", False),
        (True, False, "failed to close", True),
    ],
)
def test_lambda_parquet_rejects_native_file_io_failure(
    tmp_path,
    monkeypatch,
    is_open,
    close_ok,
    message,
    dynamics_runs,
):
    temporary = tmp_path / "lambda_tmp.lmd"
    temporary.touch()
    ran = []
    monkeypatch.setattr(
        dynamics,
        "_preflight_lambda_parquet",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        dynamics.tempfile,
        "NamedTemporaryFile",
        lambda **kwargs: SimpleNamespace(
            name=str(temporary),
            close=lambda: None,
        ),
    )
    native_file = SimpleNamespace(
        file_unit=77,
        is_open=is_open,
        close=lambda: close_ok,
    )
    monkeypatch.setattr(dynamics, "CharmmFile", lambda **kwargs: native_file)
    monkeypatch.setattr(
        dynamics.script.CommandScript,
        "run",
        lambda self, append="": ran.append(True),
    )

    dyn = dynamics.DynamicsScript(
        lambda_parquet=tmp_path / "lambda.parquet",
        nstep=1,
        nsavl=1,
    )
    with pytest.raises(OSError, match=message):
        dyn.run()

    assert bool(ran) is dynamics_runs
    assert "iunldm" not in dyn.opts


def test_lambda_parquet_preserves_requested_lambdata(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(
        dynamics,
        "_preflight_lambda_parquet",
        lambda *args, **kwargs: None,
    )
    fake_lib = SimpleNamespace(
        lambdata_on=lambda: events.append("on"),
        lambdata_off=lambda: events.append("off"),
        lambdata_del=lambda: events.append("del"),
    )
    monkeypatch.setattr(dynamics, "lib", fake_lib)
    monkeypatch.setattr(
        dynamics.script.CommandScript,
        "run",
        lambda self, append="": self,
    )
    monkeypatch.setattr(
        dynamics,
        "get_lambdata_bias",
        lambda: pd.DataFrame({"TIME": [0.0], "BIAS": [1.0]}),
    )
    monkeypatch.setattr(
        dynamics,
        "get_lambdata_bixlamsq",
        lambda: pd.DataFrame({"TIME": [0.0], "LAM01": [0.2]}),
    )
    monkeypatch.setattr(
        dynamics.tempfile,
        "NamedTemporaryFile",
        lambda **kwargs: SimpleNamespace(
            name=str(tmp_path / "lambda_tmp.lmd"),
            close=lambda: None,
        ),
    )

    class FakeCharmmFile:
        file_unit = 91
        is_open = True

        def __init__(self, **kwargs):
            pass

        def close(self):
            return True

    monkeypatch.setattr(dynamics, "CharmmFile", FakeCharmmFile)
    monkeypatch.setattr(
        dynamics,
        "_write_lambda_parquet",
        lambda *args, **kwargs: args[1],
    )

    dyn = dynamics.DynamicsScript(
        lambda_parquet=tmp_path / "lambda.parquet",
        lambdata=True,
        nstep=1,
        nsavl=1,
    )

    assert dyn.run()
    assert events == ["on", "off", "del"]
    assert dyn.lambdata_bixlamsq is not None


@pytest.mark.parametrize("iunldm", [77, "iunldm 77 -\n"])
def test_lambda_parquet_rejects_explicit_iunldm(
    tmp_path,
    monkeypatch,
    iunldm,
):
    monkeypatch.setattr(
        dynamics.script.CommandScript,
        "run",
        lambda *args, **kwargs: pytest.fail("dynamics should not run"),
    )
    kwargs = {"iunldm": iunldm} if isinstance(iunldm, int) else {}
    dyn = dynamics.DynamicsScript(
        lambda_parquet=tmp_path / "lambda.parquet",
        **kwargs,
    )
    if not isinstance(iunldm, int):
        dyn.opts["iunldm"] = iunldm

    with pytest.raises(ValueError, match="explicit iunldm"):
        dyn.run()


def test_lambda_parquet_retains_data_after_dynamics_error(
    tmp_path,
    monkeypatch,
):
    temporary = tmp_path / "lambda_tmp.lmd"
    temporary.write_bytes(b"temporary")
    closed = []
    monkeypatch.setattr(
        dynamics,
        "_preflight_lambda_parquet",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        dynamics.tempfile,
        "NamedTemporaryFile",
        lambda **kwargs: SimpleNamespace(
            name=str(temporary),
            close=lambda: None,
        ),
    )

    class FakeCharmmFile:
        file_unit = 88
        is_open = True

        def __init__(self, **kwargs):
            pass

        def close(self):
            closed.append(True)
            return True

    monkeypatch.setattr(dynamics, "CharmmFile", FakeCharmmFile)

    def fail_run(*args, **kwargs):
        raise RuntimeError("failed")

    monkeypatch.setattr(
        dynamics.script.CommandScript,
        "run",
        fail_run,
    )

    dyn = dynamics.DynamicsScript(
        lambda_parquet=tmp_path / "lambda.parquet",
        nstep=1,
        nsavl=1,
    )
    with pytest.warns(RuntimeWarning, match="retained at"):
        with pytest.raises(RuntimeError, match="failed"):
            dyn.run()

    assert closed
    assert temporary.read_bytes() == b"temporary"
    assert "iunldm" not in dyn.opts


@pytest.mark.parametrize(
    "interrupt_at, write_fails",
    [
        ("never", True),
        ("before", True),
        ("during", True),
        ("during", False),
    ],
)
def test_lambda_parquet_conversion_interrupt_handling(
    tmp_path,
    monkeypatch,
    interrupt_at,
    write_fails,
):
    import pycharmm.blade as blade

    interrupt_state = {"set": interrupt_at == "before"}
    temporary = tmp_path / "lambda_tmp.lmd"
    temporary.write_bytes(b"native lambda data")
    monkeypatch.setattr(
        dynamics,
        "_preflight_lambda_parquet",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        dynamics.tempfile,
        "NamedTemporaryFile",
        lambda **kwargs: SimpleNamespace(
            name=str(temporary),
            close=lambda: None,
        ),
    )

    class FakeCharmmFile:
        file_unit = 88
        is_open = True

        def __init__(self, **kwargs):
            pass

        def close(self):
            return True

    monkeypatch.setattr(dynamics, "CharmmFile", FakeCharmmFile)
    monkeypatch.setattr(
        dynamics.script.CommandScript,
        "run",
        lambda self, append="": self,
    )

    def fail_write(*args, **kwargs):
        if interrupt_at == "during":
            interrupt_state["set"] = True
        if write_fails:
            raise ValueError("bad parquet")

    monkeypatch.setattr(dynamics, "_write_lambda_parquet", fail_write)
    monkeypatch.setattr(blade, "install_signal_handler", lambda: None)
    monkeypatch.setattr(blade, "restore_signal_handler", lambda: None)
    monkeypatch.setattr(blade, "set_interrupt", lambda value: None)
    monkeypatch.setattr(
        blade,
        "check_interrupt",
        lambda: interrupt_state["set"],
    )

    dyn = dynamics.DynamicsScript(
        lambda_parquet=tmp_path / "lambda.parquet",
        nstep=1,
        nsavl=1,
        blade=True,
        warn_restraints=False,
    )
    if interrupt_at != "never" and write_fails:
        with pytest.warns(RuntimeWarning, match="not written.*retained"):
            assert not dyn.run()
    elif interrupt_at != "never":
        assert not dyn.run()
    else:
        with pytest.warns(RuntimeWarning, match="not written.*retained"):
            with pytest.raises(RuntimeError, match="retained at"):
                dyn.run()

    if write_fails:
        assert temporary.read_bytes() == b"native lambda data"
    else:
        assert not temporary.exists()
    assert "iunldm" not in dyn.opts


@pytest.mark.parametrize(
    "kwargs",
    [
        {"nstep": 1},
        {"nstep": 0, "nsavl": 1},
        {"nstep": 1, "nsavl": 0},
        {"nstep": 1, "nsavl": 2},
    ],
)
def test_lambda_parquet_requires_writable_frame_interval(
    tmp_path,
    monkeypatch,
    kwargs,
):
    monkeypatch.setattr(
        dynamics.script.CommandScript,
        "run",
        lambda *args, **kwargs: pytest.fail("dynamics should not run"),
    )
    dyn = dynamics.DynamicsScript(
        lambda_parquet=tmp_path / "lambda.parquet",
        **kwargs,
    )

    with pytest.raises(ValueError, match="nstep|nsavl"):
        dyn.run()
