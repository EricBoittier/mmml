import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from pycharmm.loader import lib

_RUN_RESTART = textwrap.dedent(
    """
    import sys
    import numpy as np
    from pycharmm import CharmmFile, DynamicsScript, NonBondedScript
    from pycharmm import block, crystal, dyn, gen, ic, psf, read

    data_dir, restart_path, mode = sys.argv[1:]
    read.rtf(data_dir + "/top_all36_prot.rtf")
    read.prm(data_dir + "/par_all36_prot.prm", flex=True)
    read.sequence_string("ALA")
    gen.new_segment("ADP", "ACE", "CT3", setup_ic=True)
    ic.prm_fill(False)
    ic.seed(1, "CAY", 1, "CY", 1, "N")
    ic.build()
    crystal.define_cubic(60.0)
    crystal.build(30.0)
    NonBondedScript(
        cutnb=18.0,
        ctonnb=13.0,
        ctofnb=15.0,
        eps=1.0,
        cdie=True,
        atom=True,
        vatom=True,
        fswitch=True,
        vfswitch=True,
    ).run()
    dyn.set_fbetas(np.ones(psf.get_natom()))
    block.initialize(3)
    block.call(2, "bynum 1")
    block.call(3, f"bynum {psf.get_natom()}")
    block.enable_lambda_dynamics(theta=True)
    block.ldin(1, lambda_sq=1.0, velocity=0.0, mass=12.0, bias=0.0)
    block.ldin(2, lambda_sq=0.5, velocity=0.0, mass=12.0, bias=0.0)
    block.ldin(3, lambda_sq=0.5, velocity=0.0, mass=12.0, bias=0.0)
    block.msld(site_assignments=[0, 1, 1], fnex=5.5)
    block.set_langevin(temp=298.15)
    block.rmla("bond", "theta", "dihed", "impr")
    block.coef(1, 2, 1.0)
    block.coef(1, 3, 1.0)
    block.coef(2, 3, 0.0)
    block.end()

    lambda_output = CharmmFile(
        restart_path + ".lmd",
        read_only=False,
        formatted=False,
    )
    options = dict(
        blade=True,
        lang=True,
        nstep=1,
        timestep=0.001,
        firstt=298.15,
        finalt=298.15,
        tbath=298.15,
        tstruc=298.15,
        iasors=0,
        iasvel=1,
        ichecw=0,
        iscale=0,
        iscvel=0,
        inbfrq=-1,
        imgfrq=-1,
        ilbfrq=0,
        nsavc=0,
        nsavv=0,
        nsavl=1,
        ntrfrq=0,
        echeck=-1.0,
        iunldm=lambda_output.file_unit,
    )
    if mode == "start":
        restart = CharmmFile(
            restart_path,
            read_only=False,
            formatted=True,
        )
        options.update(
            start=True,
            iunwri=restart.file_unit,
            isvfrq=1,
        )
    else:
        restart = CharmmFile(
            restart_path,
            read_only=True,
            formatted=True,
        )
        options.update(
            restart=True,
            iunrea=restart.file_unit,
            iunwri=-1,
            isvfrq=0,
        )

    try:
        if not DynamicsScript(**options).run():
            raise RuntimeError("BLaDE dynamics was interrupted")
    finally:
        lambda_output.close()
        restart.close()
    """
)


def _run_restart(data_dir, restart_path, mode):
    return subprocess.run(
        [
            sys.executable,
            "-c",
            _RUN_RESTART,
            str(data_dir),
            str(restart_path),
            mode,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.slow
@pytest.mark.skipif(
    not hasattr(lib, "blade_init_system"),
    reason="requires a CHARMM build with BLaDE",
)
def test_blade_rejects_nonfinite_restart_state(scratch_dir):
    data_dir = Path(__file__).resolve().parent / "data"
    restart_path = scratch_dir / "blade.res"

    written = _run_restart(data_dir, restart_path, "start")
    assert written.returncode == 0, written.stdout + written.stderr

    valid = _run_restart(data_dir, restart_path, "restart")
    assert valid.returncode == 0, valid.stdout + valid.stderr

    # READYN's legacy caller/formal argument order swaps the file's XOLD and X
    # records. Test both the current coordinates and the state passed to BLaDE
    # as its restart velocity, plus the MSLD theta state.
    records = (
        ("!XOLD, YOLD, ZOLD", 1, "coordinates", "non-finite restart coordinates or velocities"),
        ("!X, Y, Z", 1, "blade_velocity", "non-finite restart coordinates or velocities"),
        ("!THETA_V", 1, "theta", "non-finite MSLD theta or theta velocity"),
        ("!THETA_V", 3, "theta_velocity", "non-finite MSLD theta or theta velocity"),
    )
    for header, offset, state, expected in records:
        lines = restart_path.read_text().splitlines(keepends=True)
        record_header = next(index for index, line in enumerate(lines) if line.strip() == header)
        value_line = record_header + offset
        lines[value_line] = "Infinity".rjust(22) + lines[value_line][22:]
        invalid_path = scratch_dir / f"blade_nonfinite_{state}.res"
        invalid_path.write_text("".join(lines))

        invalid = _run_restart(data_dir, invalid_path, "restart")
        output = invalid.stdout + invalid.stderr
        assert invalid.returncode != 0, f"{state}: {output}"
        assert expected in output
        assert "ABNORMAL TERMINATION" in output
