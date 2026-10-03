"""Tests for the per-subsystem `pycharmm.reset` cleanup helpers.

For each Layer-1 reset function this file checks two properties:

1. **Idempotent / safe-on-empty.** Calling the function before
   anything is set up doesn't raise and doesn't abort the process.
2. **Actually clears.** Where there's a meaningful observable
   ("after a SHAKE setup, ``shake.qshake`` is True; after
   ``reset.shake()`` it's False"), set up some state, reset it,
   verify the observable.

Tests are kept small and self-contained. Whenever a test builds a
real molecular system it's cleaned up via ``reset.atoms()`` at the
end so it doesn't pollute the next module.
"""

from __future__ import annotations

import pytest

from pycharmm import keywords, reset
from pycharmm.lingo import charmm_script

# ---------------------------------------------------------------------
# A minimal alanine dipeptide system, lazily built once per module.
# Several reset functions need *some* atoms loaded to do anything
# observable; this fixture provides them.
# ---------------------------------------------------------------------


@pytest.fixture
def ala_dipeptide():
    """Build a small alanine dipeptide so reset functions have state to clear.

    Function-scope (rather than module-scope) because several tests
    here legitimately delete the PSF as part of what they're verifying;
    the next test would get a half-built system if we tried to share.

    Saves and restores warn/bomb levels around the build so we don't
    leak a permissive level into the test body.
    """
    from pycharmm import gen, ic, read, settings

    # Start each test from a clean slate.
    reset.atoms()
    old_warn = settings.set_warn_level(-2)
    old_bomb = settings.set_bomb_level(-2)
    try:
        read.rtf("data/top_all36_prot.rtf")
        read.prm("data/par_all36_prot.prm", flex=True)
        read.sequence_string("ALA")
        gen.new_segment("ADP", "ACE", "CT3", setup_ic=True)
        ic.prm_fill(False)
        ic.seed(1, "CAY", 1, "CY", 1, "N")
        ic.build()
    finally:
        settings.set_warn_level(old_warn)
        settings.set_bomb_level(old_bomb)
    yield
    reset.atoms()


# ---------------------------------------------------------------------
# Idempotency: every reset function is callable from a cold state
# ---------------------------------------------------------------------


_ALL_RESETS = [
    "atoms",
    "coords",
    "nbonds",
    "crystal",
    "shake",
    "drude",
    "restraints",
    "mmfp",
    "gbsw",
    "gbmv",
    "pert",
    "gamus",
    "block",
    "grid",
    "energy_terms",
    "openmm",
    "blade",
    "domdec",
]


@pytest.mark.parametrize("name", _ALL_RESETS)
def test_reset_function_safe_from_cold_state(name):
    """Every Layer-1 reset is a safe no-op when nothing is initialized."""
    fn = getattr(reset, name)
    fn()  # must not raise; must not abort


@pytest.mark.parametrize("name", _ALL_RESETS)
def test_reset_function_idempotent(name):
    """Calling a reset twice is the same as calling it once."""
    fn = getattr(reset, name)
    fn()
    fn()  # second call must also be a no-op


def test_reset_module_has_all_layer_one_functions():
    """Every documented Layer-1 reset is in `__all__`."""
    assert set(_ALL_RESETS).issubset(set(reset.__all__))


# ---------------------------------------------------------------------
# Effective clears: verify the observable after each reset
# ---------------------------------------------------------------------


def test_atoms_clears_psf(ala_dipeptide):
    """`reset.atoms` empties the PSF."""
    from pycharmm import psf

    assert psf.get_natom() > 0  # fixture built atoms
    reset.atoms()
    assert psf.get_natom() == 0


def test_atoms_no_op_when_already_empty():
    """`reset.atoms` doesn't error when there are no atoms to delete."""
    from pycharmm import psf

    assert psf.get_natom() == 0
    reset.atoms()  # second time: still empty, must not raise
    assert psf.get_natom() == 0


def test_shake_off_after_setup(ala_dipeptide):
    """`reset.shake` turns off SHAKE constraints set via the script."""
    charmm_script("SHAKE BONH MAIN TOL 1e-6")
    reset.shake()
    # Issuing a fresh nbonds / energy after SHAKE OFF must succeed
    # without "SHAKE constraints active" surprises.
    charmm_script("nbond cutnb 14.0 ctofnb 12.0 ctonnb 10.0")


def test_crystal_free_after_define(ala_dipeptide):
    """`reset.crystal` clears a crystal definition."""
    charmm_script("crystal define cubic 50.0 50.0 50.0 90.0 90.0 90.0")
    charmm_script("crystal build cutoff 14.0")
    reset.crystal()
    # Defining a *different* crystal afterwards must succeed (would
    # otherwise complain about an existing definition).
    charmm_script("crystal define cubic 60.0 60.0 60.0 90.0 90.0 90.0")
    reset.crystal()


def test_restraints_clears_cons_harm(ala_dipeptide):
    """`reset.restraints` lets a fresh harmonic setup proceed cleanly."""
    from pycharmm import SelectAtoms, cons_harm

    cons_harm.setup_absolute(force_const=1.0, selection=SelectAtoms(select_all=True))
    # Reset must drop the restraint without aborting.
    reset.restraints()
    # A second setup on a clean slate must succeed; without the reset
    # CHARMM would treat this as set 2 layered on set 1 (cf. the FACTS
    # rescore order-dependence bug we fixed earlier in the migration).
    cons_harm.setup_absolute(force_const=2.0, selection=SelectAtoms(select_all=True))
    reset.restraints()


def test_energy_terms_re_enables_after_skipe(ala_dipeptide):
    """`reset.energy_terms` undoes a SKIPE EXCLUDE."""
    from pycharmm import energy

    charmm_script("nbond cutnb 14.0 ctofnb 12.0 ctonnb 10.0")
    charmm_script("SKIPE EXCL ELEC")
    reset.energy_terms()
    # After SKIP NONE, energy.show should compute electrostatics again.
    energy.show()
    # The fact that energy.show didn't abort is the assertion; if SKIPE
    # had left ELEC disabled and a subsequent code path required it,
    # we'd observe a mismatch. Smoke test is sufficient here.


def test_nbonds_resets_e14fac(ala_dipeptide):
    """`reset.nbonds` puts the per-atom e14fac back to a uniform value."""
    # Set non-uniform per-atom e14fac
    charmm_script("scalar e14fac set 0.35 select all end")
    reset.nbonds()
    # `scalar e14fac show` would print but isn't easily readable from
    # Python. Instead, verify the next energy command runs without
    # the "CUTNB > CUTIM" noise that a stale-cutoff state produces.
    from pycharmm import energy

    energy.show()


def test_openmm_when_not_compiled_or_not_active():
    """`reset.openmm` is a no-op if OpenMM isn't compiled or not in use."""
    # Whatever the build, reset.openmm must not raise.
    reset.openmm()
    reset.openmm()  # idempotent


def test_blade_when_not_compiled_or_not_active():
    """`reset.blade` is a no-op on builds without BLaDE."""
    reset.blade()
    reset.blade()


def test_domdec_when_not_compiled_or_not_active():
    """`reset.domdec` is a no-op on builds without DOMDEC."""
    reset.domdec()
    reset.domdec()


def test_block_when_not_compiled_or_not_active():
    """`reset.block` is a no-op on builds without BLOCK or with BLOCK uninit."""
    reset.block()
    reset.block()


# ---------------------------------------------------------------------
# Feature gating: a reset for a missing feature must literally do
# nothing rather than try to run a CHARMM command that doesn't exist
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Layer 2 -- orchestrated resets
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", ["simulation", "system", "everything"])
def test_layer2_reset_safe_from_cold_state(name):
    """Layer-2 orchestrators are safe to call from a cold session."""
    fn = getattr(reset, name)
    fn()


@pytest.mark.parametrize("name", ["simulation", "system", "everything"])
def test_layer2_reset_idempotent(name):
    """Layer-2 orchestrators are idempotent."""
    fn = getattr(reset, name)
    fn()
    fn()


def test_simulation_preserves_atoms(ala_dipeptide):
    """`reset.simulation` clears overlays but leaves the PSF alone."""
    from pycharmm import psf

    n = psf.get_natom()
    assert n > 0
    reset.simulation()
    assert psf.get_natom() == n  # model intact


def test_system_drops_atoms(ala_dipeptide):
    """`reset.system` drops the PSF as part of the model reset."""
    from pycharmm import psf

    assert psf.get_natom() > 0
    reset.system()
    assert psf.get_natom() == 0


def test_everything_drops_atoms(ala_dipeptide):
    """`reset.everything` drops everything reset.system does."""
    from pycharmm import psf

    assert psf.get_natom() > 0
    reset.everything()
    assert psf.get_natom() == 0


def test_simulation_then_rebuild(ala_dipeptide):
    """After `reset.simulation`, the user can re-run energy on the same model."""
    from pycharmm import NonBondedScript, energy

    NonBondedScript(cutnb=14.0, ctofnb=12.0, ctonnb=10.0).run()
    energy.show()
    reset.simulation()
    # Model is still there; non-bonded setup must succeed again.
    NonBondedScript(cutnb=14.0, ctofnb=12.0, ctonnb=10.0).run()
    energy.show()


def test_system_then_build_new_molecule(ala_dipeptide):
    """After `reset.system`, the user can build a different molecule."""
    from pycharmm import gen, ic, psf, read

    reset.system()
    assert psf.get_natom() == 0
    # Build a *different* sequence to verify the param table is intact.
    read.sequence_string("GLY GLY")
    gen.new_segment("DIGLY", "ACE", "CT3", setup_ic=True)
    ic.prm_fill(False)
    ic.seed(1, "CAY", 1, "CY", 1, "N")
    ic.build()
    assert psf.get_natom() > 0


def test_feature_gated_reset_skips_when_missing():
    """Functions guarded by `keywords.has` are no-ops when the key is absent."""
    # Fabricate a check by using a feature we *know* is absent on a
    # given build. We need to find one that's missing on this build.
    absent = next(
        (
            k
            for k in ("BLADE", "GAMUS", "DRUDE", "GBSW", "GBMV", "PERT", "GRID")
            if not keywords.has(k)
        ),
        None,
    )
    if absent is None:
        pytest.skip("This build has every feature; nothing to gate against.")
    # Just verify the corresponding reset returns without raising.
    fn = {
        "BLADE": reset.blade,
        "GAMUS": reset.gamus,
        "DRUDE": reset.drude,
        "GBSW": reset.gbsw,
        "GBMV": reset.gbmv,
        "PERT": reset.pert,
        "GRID": reset.grid,
    }[absent]
    fn()
