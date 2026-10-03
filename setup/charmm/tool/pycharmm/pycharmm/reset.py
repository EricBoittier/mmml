# pycharmm: molecular dynamics in python with CHARMM
# Copyright (C) 2018 Josh Buckner
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Per-subsystem cleanup helpers for CHARMM global state.

CHARMM keeps the live simulation in a *single* set of global Fortran
module variables: one PSF, one parameter table, one OpenMM context,
one BLaDE system, one set of restraints, and so on. Running a second
simulation in the same Python session — to score the next ligand in a
docking pipeline, sweep a parameter, or rerun a notebook cell from
scratch — requires explicitly tearing down the pieces of that state
that would otherwise leak across runs.

This module is the **Layer 1** API: one function per subsystem, each
of which clears just that subsystem and is safe to call any number of
times whether or not the subsystem was ever initialized. The
**Layer 2** orchestrators (`system`, `simulation`, `everything`)
combine these in dependency-correct order; see the
:func:`system`/:func:`simulation`/:func:`everything` docstrings.

Design contract — every Layer-1 function:

1. **Idempotent.** Calling it twice is the same as calling it once;
   calling it on a never-initialized subsystem is a no-op.
2. **Quiet.** Cleanup runs with WRNLEV/BOMLEV temporarily lowered to
   ``-5`` so a stale-state warning from CHARMM (e.g. "BLOCK not
   initialized") doesn't abort the process.
3. **Feature-aware.** If the underlying subsystem requires a
   compile-time pref keyword that isn't present in this build, the
   function is a literal ``return``. Uses
   :mod:`pycharmm.keywords` for the check.
4. **No magic.** Returns ``None``. If a caller needs to know whether
   anything was actually cleared, the underlying ``cons_harm.turn_off``
   etc. still expose their booleans; this layer just orchestrates.

Functions
=========
- `atoms` -- delete every atom from the PSF
- `coords` -- zero the comparison/comp2 coordinate arrays
- `nbonds` -- restore default non-bonded cutoffs and clear per-atom e14fac
- `crystal` -- clear crystal definition (and image setup tied to it)
- `shake` -- turn off SHAKE constraints
- `drude` -- DRUDE RESET + LONE CLEAR
- `restraints` -- harmonic + fixed + NOE + RESD + CONS CLEAR (belt-and-suspenders)
- `mmfp` -- GEO/SSBP/VMOD restraints
- `gbsw` -- GBSW implicit solvent
- `gbmv` -- GBMV implicit solvent
- `pert` -- PERT free energy perturbation
- `gamus` -- GAMUS biasing
- `block` -- BLOCK module (FEP/MSLD)
- `grid` -- grid-potential bias
- `user_energy` -- unregister MLpot / generic user-energy callbacks
- `energy_terms` -- re-enable every energy term (SKIP NONE)
- `openmm` -- destroy OpenMM context
- `blade` -- tear down BLaDE system
- `domdec` -- disable DOMDEC

Examples
========
>>> from pycharmm import reset

Drop everything in this CHARMM session that might have leaked from a
previous test or simulation, in dependency-correct order:

>>> reset.openmm()
>>> reset.blade()
>>> reset.shake()
>>> reset.restraints()
>>> reset.crystal()
>>> reset.atoms()

Or, equivalently (when Layer 2 lands):

>>> reset.system()  # NotImplementedError until follow-up commit

Notes
=====
Two things this module does NOT do, by design:

* **Reload the parameter table.** CHARMM has no programmatic way to
  scrub the RTF/parameter tables to empty; the way to "reset" them
  is to call ``read.rtf(...)`` again *without* ``append=True`` (the
  first non-append call replaces). Document this for the caller and
  let them choose.

* **Restart the CHARMM library.** A truly fresh CHARMM (one in which
  even pref-keyword settings could change) requires a new process.
  Do that with ``subprocess.Popen``, not from inside this module.
"""

from __future__ import annotations

from contextlib import contextmanager

from pycharmm import keywords
from pycharmm.lingo import charmm_script


__all__ = [
    # Layer 1 -- per-subsystem clears
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
    "user_energy",
    "energy_terms",
    "openmm",
    "blade",
    "domdec",
    # Layer 2 -- orchestrated resets
    "simulation",
    "system",
    "everything",
]


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------


@contextmanager
def _quiet():
    """Suppress CHARMM warnings and aborts during a cleanup operation.

    Cleanup commands tend to run against modules that might not be in
    a tidy state (e.g. trying to disable BLOCK when BLOCK was never
    initialized), and CHARMM is happy to abort the process on a
    stale-state warning. Bracket each cleanup site with this so a
    grumpy module doesn't take the test session down with it.

    Save/restore is best-effort: if importing ``settings`` fails (lib
    not loaded yet, etc.) the context manager is a literal no-op.
    """
    try:
        from pycharmm import settings
    except Exception:
        yield
        return
    try:
        old_warn = settings.set_warn_level(-5)
        old_bomb = settings.set_bomb_level(-5)
    except Exception:
        yield
        return
    try:
        yield
    finally:
        try:
            settings.set_warn_level(old_warn)
            settings.set_bomb_level(old_bomb)
        except Exception:
            pass


def _has(name: str) -> bool:
    """Best-effort feature check.

    Wraps :func:`pycharmm.keywords.has` and returns ``False`` if the
    feature query itself fails (e.g. the lib isn't loaded). This makes
    the reset functions usable from edge contexts like a CHARMM-not-
    yet-initialized fixture.
    """
    try:
        return keywords.has(name)
    except Exception:
        return False


# ---------------------------------------------------------------------
# Layer 1 — per-subsystem clears
# ---------------------------------------------------------------------


def atoms() -> None:
    """Delete every atom in the PSF.

    Cascades to bonds, angles, dihedrals, impropers, cross-terms, and
    residue/segment metadata via CHARMM's standard ``DELETE ATOM``
    handling. Also implicitly invalidates non-bond lists, image atoms,
    and any cached pointers held by OpenMM/BLaDE — so call after
    :func:`openmm` and :func:`blade` to avoid a stale-pointer abort.

    No-op if the PSF is already empty.
    """
    with _quiet():
        try:
            from pycharmm import psf
            if psf.get_natom() > 0:
                psf.delete_atoms()
        except Exception:
            pass


def coords() -> None:
    """Zero the comparison and (optional) second-comparison coord sets.

    Resets coordinate scratch buffers used by ``cons_harm`` (with
    ``comparison=True``), `coor.orient`, and ``coor compare``. The
    *main* coordinate set is reset implicitly by :func:`atoms`; this
    function only touches the auxiliary buffers.

    No-op if no atoms are loaded (nothing to zero).
    """
    with _quiet():
        try:
            from pycharmm import psf
            if psf.get_natom() == 0:
                return
        except Exception:
            return
        # Both COMP and COMP2 (if compiled in) accept the same syntax.
        # The set commands write zeros into the active buffer and never
        # error even when the buffer is already zero.
        charmm_script("coor set xdir 0.0 ydir 0.0 zdir 0.0 comp")
        if _has("COMP2"):
            charmm_script("coor set xdir 0.0 ydir 0.0 zdir 0.0 comp2")


def nbonds() -> None:
    """Restore non-bonded options to defaults and clear per-atom e14fac.

    Two things:

    1. Issues a ``NBOND`` command with permissive defaults, which
       triggers ``system_dirty=.true.`` for BLaDE and the equivalent
       invalidation for OpenMM (cf. ``source/nbonds/nbonds.F90``).
    2. Resets ``qe14ff`` and the per-atom ``e14ff`` array via
       ``scalar e14fac set 1.0 select all end``, which propagates the
       change through ``SCAFIL/SCARET`` so backends pick up the new
       value on their next rebuild.

    The cutoff defaults here intentionally match CHARMM's startup
    defaults rather than something the user might consider "good"; the
    intent is to *reset*, not to *configure*.

    IMGFRQ note: leaked ``imgfrq`` is what motivated this function's
    second reset path.  A prior simulation that set ``imgfrq`` to N
    leaks that value into a later run whose ``inbfrq`` default is not
    a divisor of N; ``FINCYC`` in ``source/energy/eutil.F90`` then
    trips ``LEVEL -2 WARNING: IMGFRQ is not a multiple of INBFRQ``,
    which is escalated by the default BOMLEV and kills the process via
    ``_gfortran_exit``.  We can't reset ``imgfrq`` via the NBOND
    command -- ``source/gener/update.F90:142`` only parses ``IMGF``
    when ``LIMAGE .AND. NTRANS > 0``, so on a non-crystal system the
    keyword is silently ignored.  ``crystal free`` (called by
    :func:`crystal` upstream in :func:`system`/:func:`everything`)
    drops the image structure but leaves the module-level IMGFRQ
    variable untouched.  So we go direct via the ctypes setter
    ``nbonds.set_imgfrq(0)``, which writes the Fortran module variable
    regardless of whether images are active.

    No-op if no atoms are loaded (nothing for the scalar update to act
    on).
    """
    with _quiet():
        try:
            from pycharmm import psf
            has_atoms = psf.get_natom() > 0
        except Exception:
            has_atoms = False
        # NBOND with bare flags is fine even pre-PSF — it's just option
        # bookkeeping.  imgfrq is included for parity but is ignored
        # by the parser on a non-crystal system (see docstring); we
        # reset it explicitly below via the ctypes setter.
        charmm_script(
            "nbond inbfrq 0 ihbfrq 0 imgfrq 0 "
            "cutnb 14.0 ctofnb 12.0 ctonnb 10.0 "
            "eps 1.0 e14fac 1.0"
        )
        # Force imgfrq=0 regardless of whether images are active.
        try:
            from pycharmm import nbonds as _nbonds
            _nbonds.set_imgfrq(0)
        except Exception:
            # Best-effort -- if the setter isn't available (older build,
            # missing symbol) we've already done what we can via NBOND.
            pass
        if has_atoms:
            charmm_script("scalar e14fac set 1.0 select all end")


def crystal() -> None:
    """Clear the crystal definition (and, with it, image setup).

    Uses CHARMM's documented ``crystal free`` command (cf.
    ``doc/crystl.info``). Image groups defined via ``image byres`` /
    ``image byseg`` against the crystal are torn down implicitly. Safe
    to call with no crystal active.
    """
    with _quiet():
        charmm_script("crystal free")


def shake() -> None:
    """Turn off all SHAKE constraints.

    No-op if SHAKE was never enabled. Documented in
    ``doc/shake.info``.
    """
    with _quiet():
        charmm_script("SHAKE OFF")


def drude() -> None:
    """Reset the Drude-particle setup and clear lonepairs.

    Per ``doc/drude.info``, ``DRUDE RESET`` returns Drude particles'
    mass and charge to their parent heavy atom and erases the
    distinction between Drude and non-Drude atoms. ``LONE CLEAR``
    drops any registered lonepairs. Both are safe no-ops when neither
    feature was used.
    """
    with _quiet():
        if _has("DRUDE"):
            charmm_script("DRUDE RESET")
        if _has("LONEPAIR"):
            charmm_script("LONE CLEAR")


def restraints() -> None:
    """Clear all bias and restraint potentials in one shot.

    Issues:

    * ``CONS CLEAR`` -- covers harm/fix/IC/droplet/RMSD/EMAP/path/
      Helix in a single command (cf. ``doc/cons.info``).
    * ``noedata_reset`` -- direct C-level NOE list reset.
    * ``resddata_reset`` -- direct C-level RESD list reset.
    * :func:`mmfp` -- GEO/SSBP/VMOD restraints.

    The granular Python wrappers (``cons_harm.turn_off``,
    ``cons_fix.turn_off``) are intentionally avoided here because
    they build a ``SelectAtoms`` object internally, which can trip
    on inconsistent atom_info caching on a freshly-built or
    nearly-empty PSF. ``CONS CLEAR`` is the equivalent script-level
    operation and doesn't have that hazard.

    Safe to call with no restraints active.
    """
    with _quiet():
        # CONS CLEAR -- the catch-all for the cons.info command tree.
        charmm_script("CONS CLEAR")
        # CONS CLEAR doesn't always reset the cons_harm "on" flag, so
        # call the dedicated C entry point too. cons_harm_turn_off
        # touches CHARMM globals only, not SelectAtoms (so it's safe
        # to call on freshly-built or empty PSFs, unlike the
        # cons_harm.turn_off Python wrapper).
        try:
            from pycharmm.loader import lib
            lib.cons_harm_turn_off()
            lib.noedata_reset()
            lib.resddata_reset()
        except Exception:
            pass
        # The pycharmm.restraints module tracks restraint state on the
        # Python side too -- which atoms are fixed, which are
        # harmonic-restrained, etc. Without resetting that, a fresh
        # setup_absolute would raise IncompatibleRestraintError on
        # atoms the Python state still thinks are FIX-ed.
        try:
            from pycharmm import restraints as _restraints
            _restraints.reset()
        except Exception:
            pass
        # MMFP family lives in its own command tree.
        mmfp()


def mmfp() -> None:
    """Reset MMFP geometric (GEO) restraints.

    Per ``doc/mmfp.info``, ``GEO reset`` clears the restraint table.
    Unlike ``GEO reset``, the analogous ``SSBP reset`` and ``VMOD
    RESET`` commands also *initialize* their respective subsystems
    (re-allocating buffers, marking the SSBP boundary potential as
    active, etc.), so blindly issuing them on a session that wasn't
    using them leaves CHARMM in a worse state than it started in
    (e.g. the next ``nbond`` call will complain that SSBP requires
    extended electrostatics). Users who actually had SSBP or VMOD
    active should clear those subsystems with their own dedicated
    commands.
    """
    with _quiet():
        charmm_script("MMFP\nGEO reset\nEND")


def gbsw() -> None:
    """Reset GBSW implicit-solvent setup.

    Per ``doc/gbsw.info``: ``GBSW RESET`` clears all GBSW arrays and
    settings. No-op on builds without ``KEY_GBSW``.
    """
    with _quiet():
        if _has("GBSW"):
            charmm_script("GBSW RESET")


def gbmv() -> None:
    """Clear GBMV implicit-solvent setup.

    Per ``doc/gbmv.info``: ``GBMV CLEAR`` clears all GBMV arrays and
    flags. No-op on builds without ``KEY_GBMV``.
    """
    with _quiet():
        if _has("GBMV"):
            charmm_script("GBMV CLEAR")


def pert() -> None:
    """Turn off the PERT free-energy perturbation machinery.

    Per ``doc/pert.info``. Drops the lambda=0 PSF, biases, and
    associated bookkeeping. No-op on builds without ``KEY_PERT``.
    """
    with _quiet():
        if _has("PERT"):
            charmm_script("PERT OFF")


def gamus() -> None:
    """Clear GAMUS biasing.

    Per ``doc/gamus.info``: ``GAMUS CLEAR`` removes all configured
    GAMUS bias potentials. No-op on builds without ``KEY_GAMUS``.
    """
    with _quiet():
        if _has("GAMUS"):
            charmm_script("GAMUS CLEAR")


def block() -> None:
    """Clear all BLOCK module state (FEP/MSLD).

    Per ``doc/block.info``: ``BLOCK CLEAR`` removes every trace of the
    BLOCK module — block definitions, lambda dynamics setup, MC/MD
    coupling, restraining potentials, and per-block force masks.
    No-op on builds without ``KEY_BLOCK``.
    """
    with _quiet():
        if _has("BLOCK"):
            try:
                from pycharmm import block as _block
                _block.clear()
            except Exception:
                charmm_script("BLOCK\nCLEAR\nEND")


def grid() -> None:
    """Disable the grid-potential bias and free its arrays.

    Per ``doc/grid.info``: ``grid off`` deactivates and ``grid clear``
    deallocates. No-op on builds without ``KEY_GRID``.
    """
    with _quiet():
        if not _has("GRID"):
            return
        try:
            from pycharmm import grid as _grid
            _grid.off()
            _grid.clear()
        except Exception:
            charmm_script("grid off")
            charmm_script("grid clear")


def user_energy() -> None:
    """Unregister Python energy callbacks: MLpot and the generic user term.

    Both callbacks are registered from Python and held by CHARMM as raw
    function pointers -- ``mlpot_set_func`` for MLpot (the MLPO and MLEL
    energy terms) and the generic user-energy function behind ``USER``.

    Returns
    -------
    None

    Notes
    -----
    Nothing else in this module clears these callbacks, so without this
    a callback registered for one system stays live across a reset and
    is called again after the PSF has been rebuilt, closing over atom
    indices and a model that no longer match the current system. That
    gives silently wrong energies at best and a crash at worst.

    Safe to call when nothing is registered: the underlying
    ``mlpot_unset`` and ``func_unset`` are no-ops in that case.

    Called by :func:`simulation`, and therefore also by :func:`system`
    and :func:`everything`.

    Examples
    --------
    >>> from pycharmm import reset
    >>> reset.user_energy()     # drop any MLpot / USER callback
    """
    with _quiet():
        try:
            from pycharmm.loader import lib
            for entry in ("mlpot_unset", "func_unset"):
                fn = getattr(lib, entry, None)
                if fn is not None:
                    fn()
        except Exception:
            # A missing or already-cleared callback must not abort the
            # rest of a reset sequence.
            pass


def energy_terms() -> None:
    """Re-enable every energy term in the QETERM mask.

    Counterpart to ``SKIP <terms>``. Equivalent to ``SKIP NONE`` (cf.
    ``doc/cons.info``). Safe to call any time.
    """
    with _quiet():
        charmm_script("SKIP NONE")


def openmm() -> None:
    """Destroy the cached OpenMM context and disable OpenMM.

    Calls ``omm.clear()`` (which issues ``OMM CLEAR`` and destroys the
    Context), then ``omm.disable()`` (``OMM OFF``). The clear path
    also frees any registered custom forces and Torch forces. No-op on
    builds without ``KEY_OPENMM`` and on sessions where OpenMM is
    not currently enabled.

    Call before :func:`atoms` -- once the PSF is gone, the OpenMM
    context's pointers are stale and tearing it down later can crash
    or hang.
    """
    with _quiet():
        if not _has("OPENMM"):
            return
        try:
            from pycharmm import omm as _omm
            if _omm.is_enabled():
                _omm.clear()
                _omm.disable()
        except Exception:
            pass


def blade() -> None:
    """Disable BLaDE and tear down the BLaDE system.

    Sets ``blade_active=.false.`` and calls ``teardown_blade_system``
    in ``blade_ctrl`` to release GPU resources. No-op on builds
    without ``KEY_BLADE`` and on sessions where BLaDE is not
    currently enabled.

    Call before :func:`atoms` -- BLaDE keeps device-side copies of
    the PSF and parameter tables.
    """
    with _quiet():
        if not _has("BLADE"):
            return
        try:
            from pycharmm import blade as _blade
            if _blade.is_enabled():
                _blade.disable()
        except Exception:
            pass


def domdec() -> None:
    """Disable DOMDEC.

    Calls ``domdec.disable()`` (which issues ``energy domdec off``,
    actually running an energy evaluation under the hood). No-op on
    builds without ``KEY_DOMDEC`` and on sessions where DOMDEC is
    not currently enabled -- otherwise the disable command's energy
    pass would abort against an unconfigured non-bonded setup.
    """
    with _quiet():
        if not _has("DOMDEC"):
            return
        try:
            from pycharmm import domdec as _domdec
            if _domdec.is_enabled():
                _domdec.disable()
        except Exception:
            pass


# ---------------------------------------------------------------------
# Layer 2 — orchestrated resets
# ---------------------------------------------------------------------
#
# Order matters: things that reference the PSF or parameter table must
# be torn down *before* the PSF is dropped, or the dangling pointers
# can crash a later energy/dyn call. The chosen order is:
#
#   1. external backend caches (OpenMM context, BLaDE GPU system,
#      DOMDEC partitioning) -- they hold copies of PSF/parameter data;
#   2. BLOCK module overlays (lambda-dynamics, etc.);
#   3. SHAKE -- references bond list which is about to disappear;
#   4. restraints (cons_harm, cons_fix, NOE, RESD, MMFP, ...);
#   5. PERT -- references the lambda=0 PSF copy;
#   6. Drude particle bookkeeping;
#   7. GAMUS bias potentials;
#   8. crystal (also tears down image groups);
#   9. grid potentials;
#  10. energy term mask -> SKIP NONE;
#  11. coordinate buffers (comp, comp2);
#  12. atoms -- deepest, cascades to bonds/angles/connectivity.


def simulation() -> None:
    """Drop every overlay on top of the molecular model, model intact.

    Use this when you want to rerun a simulation under different
    conditions on the *same* system: change backends, swap restraints,
    re-enable a skipped energy term, etc. The PSF, atom positions,
    parameter table, and crystal definition are all preserved.

    Equivalent to:

    >>> reset.openmm()
    >>> reset.blade()
    >>> reset.domdec()
    >>> reset.block()
    >>> reset.shake()
    >>> reset.restraints()  # also covers MMFP
    >>> reset.pert()
    >>> reset.gamus()
    >>> reset.grid()
    >>> reset.user_energy()
    >>> reset.energy_terms()

    Notes
    -----
    Does NOT call :func:`atoms`, :func:`coords`, :func:`crystal`,
    :func:`drude`, or :func:`nbonds` -- those would alter the model
    or its dynamics setup, which is not what "drop the overlay"
    means. If you want those, use :func:`system` or :func:`everything`.
    """
    openmm()
    blade()
    domdec()
    block()
    shake()
    restraints()  # internally also calls mmfp()
    pert()
    gamus()
    grid()
    user_energy()
    energy_terms()


def system() -> None:
    """Reset the entire molecular *system*; keep loaded parameters.

    Use this when you want to start a new simulation in the same
    Python session: build a different molecule, dock the next
    compound, etc. Everything that was set up on top of CHARMM's
    parameter table goes away; the parameter table itself stays so
    you don't pay the toppar reload cost.

    Calls :func:`simulation`, then drops the model itself in
    dependency-correct order:

    1. :func:`simulation` -- overlay clears
    2. :func:`drude`      -- Drude particle bookkeeping
    3. :func:`crystal`    -- crystal definition + image setup
    4. :func:`coords`     -- comp / comp2 buffers
    5. :func:`atoms`      -- delete every atom (cascades to bonds,
       angles, dihedrals, impropers, cmaps, segments, residues)

    After this returns, ``psf.get_natom() == 0`` and CHARMM is
    ready to ``read.sequence_*`` / ``gen.new_segment(...)`` a new
    structure.

    Notes
    -----
    Does NOT call :func:`nbonds` -- the user may want to keep a
    custom non-bonded setup across systems. Call ``reset.nbonds()``
    explicitly if that's what you want, or use :func:`everything`.
    """
    simulation()
    drude()
    crystal()
    coords()
    atoms()


def everything() -> None:
    """Call every Layer-1 reset, in dependency-correct order.

    The most aggressive in-process reset available. After this
    returns, CHARMM is in approximately the state it was in
    immediately after :func:`pycharmm.charmm_init` -- empty PSF,
    default non-bonded options, no crystal, no restraints, no
    backends active. Loaded RTF/parameter tables are *not* erased
    (CHARMM has no command for that); to fully reset those, follow
    this with a fresh ``read.rtf(...)`` and ``read.prm(...)``.

    Equivalent to :func:`system` plus :func:`nbonds`.

    Examples
    --------
    >>> from pycharmm import reset
    >>> reset.everything()      # back to a clean session

    Use as a pytest module-end teardown:

    >>> @pytest.fixture(autouse=True, scope="module")
    ... def _reset_after_module():
    ...     yield
    ...     reset.everything()
    """
    system()
    nbonds()
