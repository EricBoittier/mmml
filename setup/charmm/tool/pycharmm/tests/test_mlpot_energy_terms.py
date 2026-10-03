"""MLpot reports into its own energy terms (MLPO/MLEL), not USER/ELEC.

MLpot's energy reaches CHARMM through two registered callbacks, so the
term plumbing can be exercised without a trained model, torch, or
asparagus: register stubs that return known constants and check where
those constants land.

This covers the contract that adding energy terms must not disturb the
default table:

  * with no model registered, MLPO/MLEL stay zero and are not named;
  * with a model registered, MLPO/MLEL carry exactly the callback values
    and USER/ELEC do not;
  * EPOT accounts for the MLpot contribution.
"""

import ctypes

import pytest

import pycharmm.energy as energy
from pycharmm import reset
from pycharmm.loader import lib

# Values chosen to be unmistakable and not plausibly a real energy.
ML_POT_ENERGY = 12.5
ML_ELEC_ENERGY = -3.25

# The potential energy is energy.EPROP_EPOT (index 3, printed as ENER).
# Deliberately not the deprecated PROP_ENER, which is 1: that is TOTE, and
# it reads 0.0 after a plain energy evaluation.
EPROP_EPOT = energy.EPROP_EPOT

# Signature of the MLpot energy callback, mirroring
# pycharmm.energy_mlpot: c_double return, then 19 arguments.
_FUNC_TYPE = ctypes.CFUNCTYPE(
    ctypes.c_double,
    ctypes.c_int,  # Natom
    ctypes.c_int,  # Ntrans
    ctypes.c_int,  # Natim
    ctypes.POINTER(ctypes.c_int),  # image -> central atom index
    ctypes.POINTER(ctypes.c_double),  # x
    ctypes.POINTER(ctypes.c_double),  # y
    ctypes.POINTER(ctypes.c_double),  # z
    ctypes.POINTER(ctypes.c_double),  # dE/dx
    ctypes.POINTER(ctypes.c_double),  # dE/dy
    ctypes.POINTER(ctypes.c_double),  # dE/dz
    ctypes.c_int,  # Nmlp
    ctypes.c_int,  # Nmlmmp
    ctypes.POINTER(ctypes.c_int),  # idxi
    ctypes.POINTER(ctypes.c_int),  # idxj
    ctypes.POINTER(ctypes.c_int),  # idxjp
    ctypes.POINTER(ctypes.c_int),  # idxu
    ctypes.POINTER(ctypes.c_int),  # idxv
    ctypes.POINTER(ctypes.c_int),  # idxup
    ctypes.POINTER(ctypes.c_int),  # idxvp
)

_ELEC_TYPE = ctypes.CFUNCTYPE(ctypes.c_double)


class _MLpotStub:
    """Registers stub MLpot callbacks and releases them exactly once.

    The ctypes callback objects must outlive the CHARMM call, so they are
    held on the instance rather than as locals. ``release`` is only a
    no-op when nothing was registered -- calling ``mlpot_unset`` without a
    prior ``mlpot_set_func`` is avoided deliberately.
    """

    def __init__(self):
        self._registered = False
        self._energy_cb = None
        self._elec_cb = None

    def register(self, e_pot=None, e_elec=None):
        e_pot = ML_POT_ENERGY if e_pot is None else e_pot
        e_elec = ML_ELEC_ENERGY if e_elec is None else e_elec

        def ml_energy(
            natom,
            ntrans,
            natim,
            idx,
            x,
            y,
            z,
            dx,
            dy,
            dz,
            nmlp,
            nmlmmp,
            idxi,
            idxj,
            idxjp,
            idxu,
            idxv,
            idxup,
            idxvp,
        ):
            return e_pot

        def ml_elec():
            return e_elec

        self._energy_cb = _FUNC_TYPE(ml_energy)
        self._elec_cb = _ELEC_TYPE(ml_elec)
        lib.mlpot_set_func(self._energy_cb, self._elec_cb)

        # One ML atom; indices are 1-based on the Fortran side.
        n_ml = (ctypes.c_int * 1)(1)
        ml_index = (ctypes.c_int * 1)(1)
        ml_z = (ctypes.c_int * 1)(6)
        lib.mlpot_set_properties(n_ml, ml_index, ml_z)
        self._registered = True

    def release(self):
        if self._registered:
            reset.user_energy()
            self._registered = False


def test_mlpot_energy_term_lifecycle(alanine_dipeptide):
    """MLPO/MLEL are inert unless a model is registered, then carry its values.

    Written as one test with ordered phases rather than three separate
    tests: CHARMM state is global and the PSF is only wiped between
    modules, so requesting the build fixture from several tests in one
    file appends a second segment and SEED fails with "COORDINATES
    ALREADY KNOWN".
    """
    stub = _MLpotStub()
    try:
        # MLpot is gated on qeterm(MLPO) .or. qeterm(MLEL), so this test
        # needs the QETERM mask in its default all-enabled state. Do not
        # assume it: CHARMM's energy-term mask is global, and an earlier
        # test in the same session can leave it narrowed. grid.py does
        # exactly that, issuing "skipe all excl vdw elec" for probe
        # energies without restoring it, which leaves only VDW, ELEC,
        # USER, GRVD and GREL enabled. Running the full suite in one
        # process then reaches this test with MLPO/MLEL disabled and the
        # callbacks never invoked. SKIP NONE re-enables every slot
        # including the unnamed ones, since that branch of SKIPE assigns
        # QETERM(1:LENENT) wholesale rather than by name.
        reset.energy_terms()

        # Phase 1 -- nothing registered: the terms must be inert, and the
        # energy table must look exactly as it did before MLPO/MLEL existed.
        energy.show()
        assert energy.get_eterm(energy.TERM_MLPO) == pytest.approx(0.0)
        assert energy.get_eterm(energy.TERM_MLEL) == pytest.approx(0.0)
        elec_clean = energy.get_eterm(energy.TERM_ELEC)
        epot_clean = energy.get_eprop(EPROP_EPOT)

        # Phase 2 -- registered: each term carries exactly its callback's
        # value, and the MM electrostatic term is untouched.
        stub.register()
        energy.show()
        assert energy.get_eterm(energy.TERM_MLPO) == pytest.approx(ML_POT_ENERGY)
        assert energy.get_eterm(energy.TERM_MLEL) == pytest.approx(ML_ELEC_ENERGY)
        assert energy.get_eterm(energy.TERM_ELEC) == pytest.approx(elec_clean)

        # The MLpot contribution reaches the total potential energy.
        assert energy.get_eprop(EPROP_EPOT) == pytest.approx(
            epot_clean + ML_POT_ENERGY + ML_ELEC_ENERGY
        )

        # Phase 3 -- unregistered via the reset helper: contribution stops.
        stub.release()
        energy.show()
        assert energy.get_eterm(energy.TERM_MLPO) == pytest.approx(0.0)
        assert energy.get_eterm(energy.TERM_MLEL) == pytest.approx(0.0)
        assert energy.get_eprop(EPROP_EPOT) == pytest.approx(epot_clean)
    finally:
        stub.release()
