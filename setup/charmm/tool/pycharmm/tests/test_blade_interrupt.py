"""BLaDE dynamics interrupt regression."""

import numpy as np
import pytest

from pycharmm.loader import lib


@pytest.mark.slow
@pytest.mark.skipif(
    not hasattr(lib, "blade_check_interrupt"),
    reason="requires CHARMM compiled with BLaDE",
)
def test_charmm_blade_loop_consumes_interrupt(alanine_dipeptide_with_nbonds):
    from pycharmm import blade, coor, crystal, dyn, image, psf

    crystal.define_cubic(100.0)
    crystal.build(18.0)
    image.setup_segment(0.0, 0.0, 0.0, "ADP")
    dyn.set_fbetas(np.full(psf.get_natom(), 5.0))
    initial = coor.get_positions().to_numpy(copy=True)

    blade.set_interrupt(1)
    try:
        completed = blade.dynamics(
            nstep=1000,
            timestep=0.001,
            finalt=300.0,
            handle_interrupt=False,
            start=True,
            lang=True,
            firstt=300.0,
            tbath=300.0,
            inbfrq=-1,
            imgfrq=-1,
            echeck=-1,
        )
    finally:
        blade.set_interrupt(0)

    assert completed is False
    np.testing.assert_allclose(
        coor.get_positions().to_numpy(),
        initial,
        atol=1e-7,
        rtol=0,
    )
