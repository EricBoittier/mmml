"""Shared pytest setup for pycharmm tests.

Tests run against the pip-installed pycharmm package (installed during
the CMake build).  The installed loader.py has the CHARMM library path
baked in by configure_file(), so CHARMM_LIB_DIR does not need to be
set manually.

The working directory is changed to tests/ so that relative paths such
as `data/...` resolve correctly.
"""

import os
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent

os.chdir(TESTS_DIR)


# All test files are now proper pytest tests; nothing to ignore.
collect_ignore = []


def pytest_report_header(config):
    """Say so when `torch' is a leftover directory rather than a module.

    A directory under site-packages with no ``__init__.py`` is a namespace
    package, so ``import torch`` succeeds, ``torch.__file__`` is None and
    ``torch.nn`` does not exist.  Every guard of the form
    ``importorskip("torch")`` passes and the failure surfaces later as
    ``module 'torch' has no attribute 'nn'``, which reads like a torch
    problem rather than an installation one.  The torch tests ask for
    ``torch.nn`` and skip; this line explains why.
    """
    import importlib.util

    try:
        spec = importlib.util.find_spec("torch")
    except (ImportError, ValueError):
        return None
    if spec is None or spec.origin is not None:
        return None
    where = ""
    if spec.submodule_search_locations:
        where = f" at {list(spec.submodule_search_locations)[0]}"
    return (f"torch: leftover directory{where} with no __init__.py -- "
            "importable but empty, so the torch tests will skip")


_DATA_DIR = TESTS_DIR / "data"


@pytest.fixture(autouse=True)
def _no_writes_to_shared_data():
    """Catch tests that accidentally write to ``tests/data/``.

    The shared test-data directory holds RTF / PRM / PDB / PSF files
    that other tests *read*; a stray write to it (a typo'd path, a
    `write.coor_pdb('data/foo.pdb')` that should have been
    `scratch_dir / 'foo.pdb'`) corrupts the inputs for every later
    test in the session.

    This autouse fixture takes a one-pass snapshot of the data
    directory's file mtimes before each test and re-checks after.
    Any new or modified file under ``tests/data/`` raises an
    AssertionError naming the offender, so the bug is caught at the
    test that wrote the file rather than three CI runs later.

    Implementation notes:
      * mtime-based comparison is O(N) for N files in tests/data;
        on the dev box that's ~30 files, well under a millisecond.
      * Symlinks are followed (`stat()` resolves them) -- the fixture
        intentionally treats the real targets as the protected set.
      * Disable for a specific test by parametrizing or by mocking
        the data dir; we have not yet needed an opt-out marker.
    """
    if not _DATA_DIR.is_dir():
        yield
        return
    before = {p: p.stat().st_mtime_ns for p in _DATA_DIR.rglob("*") if p.is_file()}
    yield
    new = []
    modified = []
    for p in _DATA_DIR.rglob("*"):
        if not p.is_file():
            continue
        if p not in before:
            new.append(p)
        elif p.stat().st_mtime_ns != before[p]:
            modified.append(p)
    if new or modified:

        def rel(paths):
            return sorted(str(p.relative_to(_DATA_DIR)) for p in paths)

        raise AssertionError(
            "Test wrote to the shared tests/data/ directory. Use the "
            "`scratch_dir` (or `scratch_chdir`) fixture instead.\n"
            f"  new files: {rel(new)}\n"
            f"  modified:  {rel(modified)}"
        )


@pytest.fixture
def scratch_dir(tmp_path):
    """Per-test scratch directory; auto-cleaned by pytest.

    Use this for *any* file the test needs to write -- ``write.coor_pdb``,
    ``CharmmFile`` output, ``write.psf_card``, restart files, DCDs, etc.
    Returns a ``pathlib.Path`` you can pass directly or convert with
    ``str(scratch_dir / "foo.pdb")``.

    Pytest deletes scratch directories after a few sessions (3 by
    default; configurable via ``--basetemp`` / pytest.ini), so a
    failed run's artifacts stick around long enough for debugging
    but don't accumulate forever.

    For tests that use *relative*-path writes (legacy CHARMM-script
    style, e.g. ``write.coor_pdb('pdb/foo.pdb')``), use
    :func:`scratch_chdir` instead -- that fixture additionally chdir's
    into the scratch directory so relative paths resolve correctly.
    """
    return tmp_path


@pytest.fixture
def scratch_chdir(tmp_path, monkeypatch):
    """Per-test scratch directory + chdir into it.

    Use when the test or the underlying CHARMM script writes to
    relative paths (``write.coor_pdb('pdb/foo.pdb')``,
    ``CharmmFile(file_name='res/restart.res', ...)``). Each
    subdirectory the test wants under the scratch root must be
    created explicitly with e.g. ``(scratch_chdir / "pdb").mkdir()``.

    The original cwd is restored automatically at test teardown via
    ``monkeypatch``.
    """
    monkeypatch.chdir(tmp_path)
    return tmp_path


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests that need a pref keyword absent from this build.

    Implements the ``@pytest.mark.requires_feature('NAME')`` marker
    declared in ``pytest.ini``. Each marked test (or class, or
    module) is skipped when ``pycharmm.keywords.has(NAME)`` returns
    False; matching is case-insensitive, with leading/trailing
    whitespace stripped (cf. :func:`pycharmm.keywords.has`).
    """
    from pycharmm import keywords as _keywords

    for item in items:
        for mark in item.iter_markers(name="requires_feature"):
            for name in mark.args:
                if not _keywords.has(name):
                    item.add_marker(
                        pytest.mark.skip(
                            reason=(
                                f"requires CHARMM pref keyword "
                                f"{name.strip().upper()!r}, not "
                                f"compiled into this build"
                            )
                        )
                    )
                    break


@pytest.fixture
def alanine_dipeptide():
    """Build a single ACE-ALA-CT3 alanine dipeptide in CHARMM's PSF.

    Reads ``data/top_all36_prot.rtf`` and ``data/par_all36_prot.prm``,
    generates one ALA residue capped with ACE/CT3, fills internal
    coordinates from the parameter file (``ic.prm_fill(False)``), seeds
    the build at CAY-CY-N of residue 1, and runs ``ic.build()``.

    This is the *minimal* build sequence shared by ~17 test files. It
    does not load water/ion topology, does not read the NBFIX patch,
    does not orient the structure, and does not run a NonBondedScript;
    tests that need any of those should either add them inline after
    requesting this fixture or use :func:`alanine_dipeptide_with_nbonds`
    which bundles the full standard setup.

    Function-scope: each test gets a fresh build. The shared
    ``_clear_charmm_state_between_modules`` fixture wipes PSF/atom
    state at the end of the module so successive modules don't leak
    state into each other.
    """
    from pycharmm import gen, ic, read

    read.rtf("data/top_all36_prot.rtf")
    read.prm("data/par_all36_prot.prm", flex=True)
    read.sequence_string("ALA")
    gen.new_segment("ADP", "ACE", "CT3", setup_ic=True)
    ic.prm_fill(False)
    ic.seed(1, "CAY", 1, "CY", 1, "N")
    ic.build()


@pytest.fixture
def alanine_dipeptide_with_nbonds():
    """Full standard alanine dipeptide setup with solvent params and nonbonded list.

    Extends :func:`alanine_dipeptide` with the steps duplicated across
    most existing tests:

      * appends ``data/water_ions.rtf`` / ``data/water_ions.prm``
      * appends ``data/sodium_oxygen_nbfixes.prm`` with warn/bomb levels
        temporarily lowered to -1 (the NBFIX file emits warnings about
        atoms not in the topology that are harmless here)
      * orients the structure with ``coor.orient(by_rms=False,
        by_mass=False, by_noro=False)``
      * runs ``NonBondedScript(cutnb=18.0, ctonnb=15.0, ctofnb=13.0,
        eps=1.0, cdie=True, atom=True, vatom=True, fswitch=True,
        vfswitch=True).run()``

    Use this when a test needs the standard production-ish nonbonded
    setup. Tests that need different cutoffs or that skip the NBFIX
    read should request :func:`alanine_dipeptide` and add their own
    extras inline.
    """
    from pycharmm import NonBondedScript, coor, gen, ic, read, settings

    read.rtf("data/top_all36_prot.rtf")
    read.prm("data/par_all36_prot.prm", flex=True)
    read.rtf("data/water_ions.rtf", append=True)
    read.prm("data/water_ions.prm", append=True, flex=True)
    old_warn = settings.set_warn_level(-1)
    old_bomb = settings.set_bomb_level(-1)
    read.prm("data/sodium_oxygen_nbfixes.prm", append=True, flex=True)
    settings.set_warn_level(old_warn)
    settings.set_bomb_level(old_bomb)
    read.sequence_string("ALA")
    gen.new_segment("ADP", "ACE", "CT3", setup_ic=True)
    ic.prm_fill(False)
    ic.seed(1, "CAY", 1, "CY", 1, "N")
    ic.build()
    coor.orient(by_rms=False, by_mass=False, by_noro=False)
    NonBondedScript(
        cutnb=18.0,
        ctonnb=15.0,
        ctofnb=13.0,
        eps=1.0,
        cdie=True,
        atom=True,
        vatom=True,
        fswitch=True,
        vfswitch=True,
    ).run()


@pytest.fixture
def _torch_seed():
    """Override-me hook for the per-file np.random seed.

    Each torch test file overrides this fixture with its own
    deterministic seed (e.g. ``np.random.seed(8412)``); the
    :func:`dummy_peptide_15` fixture depends on this hook so the
    seed is in place before the fixture's ``np.random.rand`` calls
    run. Seeds are intentionally distinct across files: a regression
    that breaks one file's random sample shouldn't be masked by
    another file passing on the same set of atoms.
    """
    return None


@pytest.fixture
def dummy_peptide_15(_torch_seed):
    """Build a 15-residue dummy ('DUMB') peptide for the torch test files.

    The torch tests share an identical ~70-line preamble: a CHARMM script
    string defining a one-atom 'DUMB' residue (mass 1.0, single particle
    type 'j'), then 15 of those residues in segment 'test', then a random
    reference coordinate set, a random perturbed set, and a random
    selection of 5 atoms to restrain. This fixture rolls all of that up.

    Depends on :func:`_torch_seed` (a stub no-op in conftest, overridden
    per-file with ``np.random.seed(...)``); pytest's dependency ordering
    guarantees the per-file seed is set before this fixture's random
    calls run.

    Returns a ``types.SimpleNamespace`` with attributes:

      coors_ref          : ``np.ndarray (15, 3)`` of reference positions in A
      coors_perturbed    : ``np.ndarray (15, 3)`` perturbed by 0..0.5 A
      atoms_restrained   : ``SelectAtoms`` with 5 randomly chosen atoms
      atom_mask          : ``np.ndarray (5,)`` boolean/index mask matching
                           ``atoms_restrained`` for use as a positions
                           index in torch / numpy

    Important: this fixture deliberately does **not** clear OpenMM
    context state on entry. Several torch tests are stateful (run
    via ``-m stateful``) precisely because they rely on the OMM
    context state set up earlier; clearing it here breaks the
    intended energy/force comparison flow.
    """
    import types

    import numpy as np
    import pandas as pd

    from pycharmm import (
        SelectAtoms,
        coor,
        generate,
        lingo,
        psf,
        read,
    )

    lingo.charmm_script(
        """
    read rtf card
    * rtf for dummy particles
    *
    36  1

    mass  -1  j          1.0

    auto angles dihe patch

    resi dumb 0.0
    atom du j 0.0

    end

    read param card flex
    * parameters for dummy particles
    *

    atom
    mass  -1  j          1.0

    nonbonded nbxmod 5 cdiel  fshift vatom vdistance vfswitch -
    cutnb 14.0 ctofnb 12.0 ctonnb 10.0 eps 1.0 e14fac 1.0 wmin 1.5

    j      0.0  -0.0    1.0

    end
        """
    )
    read.sequence_string(
        "DUMB DUMB DUMB DUMB DUMB DUMB DUMB DUMB DUMB DUMB DUMB DUMB DUMB DUMB DUMB"
    )
    generate.new_segment("test")

    natom = psf.get_natom()
    coors_ref = np.random.rand(natom, 3) * 10
    coors_perturbed = coors_ref + np.random.rand(natom, 3) * 0.5

    coor.set_positions(pd.DataFrame(coors_ref, columns=["x", "y", "z"]))
    coor.set_comparison(coor.get_main())
    coor.set_positions(pd.DataFrame(coors_perturbed, columns=["x", "y", "z"]))

    atoms_restrained_list = np.random.choice(natom, size=5, replace=False)
    atoms_restrained = SelectAtoms().by_atom_nums(atoms_restrained_list)
    atom_mask = np.array(list(atoms_restrained))

    return types.SimpleNamespace(
        coors_ref=coors_ref,
        coors_perturbed=coors_perturbed,
        atoms_restrained=atoms_restrained,
        atom_mask=atom_mask,
    )


@pytest.fixture(autouse=True)
def _restore_energy_term_mask():
    """Restore CHARMM's energy-term mask after each test.

    QETERM is global process state and a `SKIPE` narrowing outlives the
    test that issued it, so a later test can silently evaluate a subset
    of the energy. Several tests narrow it and do not restore it
    (`test_user_e`, `test_user_harm`, `test_custom_dynam`,
    `test_custom_forces_collective`), as did `grid.generate()` until it
    was fixed to restore the mask itself. The symptom is order
    dependence: a test passes alone and fails in a full-suite run, with
    a term reading 0.0 for no visible reason.

    `SKIPE INIT` assigns QETERM(1:LENENT) wholesale and returns without
    printing the "SKIPE> The following energy terms will be computed"
    listing, so this neither adds output nor depends on a term having
    been named -- terms named lazily on first use, such as the MLpot
    MLPO/MLEL pair, are unreachable by a name-based restore.
    """
    yield
    try:
        from pycharmm import lingo, settings

        old_prn = settings.set_verbosity(0)
        lingo.charmm_script("skipe init")
        settings.set_verbosity(old_prn)
    except Exception:
        pass


@pytest.fixture(autouse=True, scope="module")
def _clear_charmm_state_between_modules():
    """Wipe CHARMM PSF/atom state at the end of each test module.

    CHARMM holds a single global PSF; tests that build a system leave
    atoms behind, which causes the next module's `read.rtf` calls to
    hang or abort. Module-scope (rather than function-scope) keeps any
    intra-module module-scoped fixture intact, while still preventing
    cross-module pollution.

    Note: deliberately conservative -- only the PSF is cleared. The
    full `pycharmm.reset.everything()` would also tear down OpenMM
    contexts, BLaDE systems, etc., but several existing tests rely on
    the *previous* module's setup of those subsystems still being
    live. Tests that need a fully clean session should call
    `pycharmm.reset.everything()` explicitly in their own fixture.
    """
    yield
    try:
        from pycharmm import psf, settings

        old_warn = settings.set_warn_level(-5)
        old_bomb = settings.set_bomb_level(-5)
        if psf.get_natom() > 0:
            psf.delete_atoms()
        settings.set_warn_level(old_warn)
        settings.set_bomb_level(old_bomb)
    except Exception:
        pass
