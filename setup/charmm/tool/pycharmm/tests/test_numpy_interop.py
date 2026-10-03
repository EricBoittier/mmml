"""Smoke test that NonBondedScript accepts both Python and numpy floats.

A regression check that pyCHARMM's typed-API arguments coerce numpy
scalar types correctly. Previous bugs surfaced when numpy.float64 values
weren't recognized as floats by the C bridge.

Original by C. L. Brooks III, April 2019.
"""

import numpy

from pycharmm import NonBondedScript, energy, minimize, settings


def test_nonbonded_script_accepts_numpy_floats(alanine_dipeptide_with_nbonds):
    """NonBondedScript with cutnb as numpy.float64 runs without error."""
    old_verb = settings.set_verbosity(0)
    settings.set_verbosity(old_verb)
    minimize.run_abnr(nstep=700, tolgrd=1e-3)

    # The actual regression checks: cutnb as a Python float, then numpy.float64.
    NonBondedScript(cutnb=20.0).run()
    NonBondedScript(cutnb=numpy.float64(20.0), cdie=True).run()

    # If we got here without an exception, the conversion worked.
    assert energy.get_total() is not None
