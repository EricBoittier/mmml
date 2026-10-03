# pycharmm test suite

A guide to running, navigating, and adding to the pycharmm pytest
suite. Aimed at someone who has just cloned the repo, has a CHARMM
build with pycharmm installed, and wants to be productive in 10
minutes.

## Quick reference

| Command | What it does |
|---|---|
| `pytest` | The default sweep. ~670 tests in ~70 s. Skips slow + stateful tests. |
| `pytest -m slow` | Long-running tests (e.g. e14fac OpenMM/BLaDE comparison). Skipped by default. |
| `pytest -m stateful` | Tests that leak CHARMM global state and have to be run alone. Each one passes individually. |
| `pytest -m "slow or stateful or not slow"` | Everything. |
| `pytest tests/test_x.py` | One file. |
| `pytest tests/test_x.py::test_y` | One test. |
| `pytest tests/test_x.py -v --tb=long` | Verbose with full traceback (default is `--tb=short`). |
| `pytest --collect-only -q` | Show every test pytest sees, without running. Useful for catching collection errors. |
| `python -m pyflakes tests/test_*.py` | Lint the suite. Should print nothing. |

The default sweep is what runs in CI. **Don't** push a change that
regresses the default-sweep count or introduces a new pyflakes
warning.

## Markers

The suite uses three markers, declared in `pytest.ini` and enforced
by `--strict-markers` (a typo like `@pytest.mark.statefull` is now a
collection error, not a silent skip).

### `@pytest.mark.slow`

Long-running tests, &gt; 60 s wall time. Skipped by default. Apply when
the test is expected to take meaningful real time even on a fast
machine — e.g. a 100-step MD comparison, a 5×5 grid sweep over force
constants. Keep the body itself fast in CI; if a test is slow because
it's *broken*, fix it instead of marking it.

### `@pytest.mark.stateful`

Tests that leave CHARMM in a state the next test can't reliably
recover from. Common causes:

* persists a `crystal.define_*` definition that the next test's `nbond`
  call trips against (CUTNB &gt; CUTIM check)
* leaves an OpenMM context whose pointers reference a PSF that's
  about to be deleted
* configures BLaDE per-atom params that no in-process reset can fully
  unwind

These tests pass when run alone but fail when run after siblings.
Marking them lets the cumulative sweep stay green; explicit `-m
stateful` runs them one-at-a-time.

Before reaching for `@stateful`, try the `pycharmm.reset` module —
`reset.everything()` clears most things. The marker is for state we
can't currently clean up programmatically.

### `@pytest.mark.requires_feature("KEYWORD")`

Skips a test if a particular CHARMM `pref.dat` keyword wasn't
compiled into the running binary. Implemented via a collection hook
in `conftest.py` that calls `pycharmm.keywords.has(KEYWORD)`. Useful
because pycharmm runs against many CHARMM build configurations:

```python
@pytest.mark.requires_feature("BLADE")
def test_blade_thing(): ...


@pytest.mark.requires_feature("OPENMM", "OMMTORCH")
def test_torch_thing(): ...
```

Multiple positional args are AND-ed (every named feature must be
present).

## Where to put files

CHARMM test scripts have a long history of writing output files into
the working directory. Don't.

### `data/` is read-only

The `tests/data/` subdirectory holds RTF, PRM, PDB, PSF, etc. that
many tests *read*. An autouse fixture in `conftest.py`
(`_no_writes_to_shared_data`) snapshots that directory's mtimes
before each test and raises if any file there changes — so you find
out at the test that wrote the file, not three CI runs later.

If you see

```
AssertionError: Test wrote to the shared tests/data/ directory.
Use the `scratch_dir` (or `scratch_chdir`) fixture instead.
```

you put a write target in `data/`. Move it to a scratch path.

### `scratch_dir` fixture

```python
def test_round_trip(scratch_dir):
    out = scratch_dir / "foo.crd"
    write.coor_card(str(out), title="...")
    assert out.is_file()
```

`scratch_dir` is a per-test `pathlib.Path` under `/tmp/pytest-of-<user>/`
that pytest cleans up after a few sessions. Pass it to anything that
takes a filename: `write.coor_pdb`, `CharmmFile`, restart files,
DCDs.

### `scratch_chdir` fixture

For tests that use *relative*-path writes (legacy CHARMM script style
like `write.coor_pdb('pdb/foo.pdb')`), use `scratch_chdir` — same
return value but with `monkeypatch.chdir` already applied. Each
subdirectory you want under it must be created explicitly:

```python
def test_legacy_relative_writes(scratch_chdir):
    (scratch_chdir / "pdb").mkdir()
    write.coor_pdb("pdb/foo.pdb")
```

## Shared fixtures

These live in `conftest.py` and are available to any test file. They
exist because the same setup was duplicated across &gt; 15 test files;
if you find yourself copy-pasting a CHARMM build, reach for one of
these first.

### `alanine_dipeptide`

The minimal alanine dipeptide build:

```python
def test_x(alanine_dipeptide):
    # PSF is built. No nonbonded list yet, no waters, no orientation.
    assert psf.get_natom() > 0
```

Reads `data/top_all36_prot.rtf` + `data/par_all36_prot.prm`,
generates one ALA capped with ACE/CT3, fills internal coords from the
parameter file, and builds. Function-scope: each test gets a fresh
build. The cross-module state cleanup in `conftest.py` wipes the PSF
between modules, so this fixture is safe to use anywhere.

### `alanine_dipeptide_with_nbonds`

Same as above plus:

* appends water-ions toppar
* appends `data/sodium_oxygen_nbfixes.prm` with warn levels lowered
* runs `coor.orient(by_rms=False, by_mass=False, by_noro=False)`
* runs the standard `NonBondedScript(cutnb=18, ctonnb=15, ctofnb=13,
  ...).run()`

Use this when the test wants the standard production-ish nonbonded
setup. If you need different cutoffs, request the bare
`alanine_dipeptide` and configure inline.

### `dummy_peptide_15`

A 15-residue chain of artificial particles (CHARMM-script-defined
dummy atom type with mass 1.0), random coordinates, random restraint
selection. Returns a small dataclass with `coors_ref`,
`coors_perturbed`, `atoms_restrained`, `atom_mask`. Used by all six
torch tests; reduces ~70 lines of identical setup per file to one
fixture invocation.

### `_clear_charmm_state_between_modules` (autouse, module-scope)

Calls `psf.delete_atoms()` at module teardown. Module-scope rather
than function-scope so an intra-module module-scoped fixture can
build state that's shared across tests in the same file, while still
preventing cross-module pollution.

Deliberately conservative — only the PSF is cleared. The full
`pycharmm.reset.everything()` would also tear down OpenMM contexts
and BLaDE systems, but several existing tests rely on the *previous*
module's setup of those subsystems still being live. If a test needs
a fully clean session, call `pycharmm.reset.everything()` explicitly
in its own fixture.

### `_no_writes_to_shared_data` (autouse, function-scope)

The `data/` write guard described above.

## The reset module

`pycharmm.reset` exposes a layered API for clearing CHARMM global
state without restarting the process. Useful in fixtures that need
guaranteed-clean state, and in interactive notebook workflows.

```python
from pycharmm import reset

# Layer 1 — per-subsystem
reset.atoms()  # delete every atom
reset.openmm()  # destroy OpenMM context
reset.blade()  # tear down BLaDE
reset.restraints()  # cons_harm + cons_fix + NOE + RESD + MMFP
reset.crystal()  # crystal definition
reset.shake()  # SHAKE constraints
# ... 14 in total; see the module docstring

# Layer 2 — orchestrated
reset.simulation()  # backends + restraints + biases. Model intact.
reset.system()  # simulation() + atoms + crystal. Toppar intact.
reset.everything()  # the deepest in-process reset.
```

Every function is idempotent, quiet (warn levels suppressed during
cleanup), and feature-gated (calling `reset.blade()` on a non-BLaDE
build is a no-op).

## How to add a new test

1. **Pick the right file.** See the index below. Keep one logical
   subsystem per file; if you need to add a new subsystem, make a new
   `test_*.py`.

2. **Use the smallest fixture that does the job.** If the
   `alanine_dipeptide_with_nbonds` fixture covers your setup, use it.
   If you need a different molecule, copy the inline build into your
   test rather than expanding the shared fixture (subtle differences
   in setup details have caused real bugs).

3. **Write to `scratch_dir`, not `data/` or cwd.** The guard will
   catch you if you forget.

4. **Seed your randomness.** If your test uses `np.random` or
   `random`, seed it inside the test body (file-name-derived seeds
   work fine — `np.random.seed(abs(hash(__file__)) % 10000)`).

5. **Don't `from pycharmm import *`.** Be explicit about what you
   import — pyflakes will flag the star imports, and they hide which
   names actually matter.

6. **Pick a marker carefully.**
   * `@pytest.mark.slow` for &gt;60 s tests
   * `@pytest.mark.stateful` *only* if you've tried `reset.everything()`
     and it doesn't unstick the next test
   * `@pytest.mark.requires_feature("X")` for build-conditional tests
   * No marker for the common case — it'll run in the default sweep.

7. **Run before you push.**

   ```sh
   cd tool/pycharmm
   pytest                              # default sweep, must be 0 failed
   python -m pyflakes tests/test_*.py  # must print nothing
   ```

## Common pitfalls

* **"My test passes alone but fails in the sweep."** Cumulative
  CHARMM state from a previous test. Try `reset.everything()` at
  test start; if that doesn't work, mark `@stateful`.

* **"My test passes locally but fails in CI."** Different CHARMM
  build configuration. Check whether you depend on a feature that
  isn't always compiled in (`OPENMM`, `BLADE`, `DOMDEC`, `FFTDOCK`,
  `OMMTORCH`); add `@requires_feature` if so.

* **"My test wrote to `data/` and broke a later test's read."** The
  guard now catches this immediately. Switch to `scratch_dir`.

* **"Pytest collected 0 tests for my file."** Probably an import
  error. Run `pytest --collect-only tests/test_x.py` to see the
  traceback.

* **"My new fixture isn't being used."** Pytest fixtures are
  resolved by name. Make sure the test function takes the fixture
  name as a parameter, and that the fixture is in `conftest.py`
  (not just in the test file).

## Index

What each test file covers, one line each. Keep this current as you
add new files.

| File | Subsystem |
|---|---|
| test_ala_dipeptide_build.py | End-to-end alanine dipeptide build + minimize + DCD write integration test |
| test_block.py | BLOCK module — basic init / coef / lambda dynamics setup |
| test_block_integration.py | BLOCK module — context manager, MSLD, lambda dynamics integration |
| test_card_io.py | CARD-format coordinate I/O round-trips |
| test_cdocker.py | Rigid + Flexible CDOCKER FFTDOCK pipeline against benzene/T4 |
| test_charmm_file.py | The CharmmFile context manager |
| test_charmm_script.py | Smoke test: build alanine dipeptide and write a DCD via lingo |
| test_coordinates.py | coor module — set/get/show, comparison set, orient |
| test_correl.py | The correl module — Series, ACF/CCF/MSD, transforms, spectra |
| test_crystal.py | Crystal lattice definitions + backend compatibility matrix |
| test_crystal_image.py | Cubic crystal + image setup + PME/Ewald nonbonded comparison |
| test_custom_dynam.py | CustomDynam Python-callback integrator |
| test_custom_forces_basic.py | OpenMM custom forces — bond/angle/torsion/external/NB/GB/Hbond/many-particle |
| test_custom_forces_collective.py | OpenMM custom forces — compound-bond/centroid-bond/CV/volume/RMSD/Rg |
| test_custom_forces_misc.py | OpenMM custom forces — tabulated functions, introspection, force groups |
| test_custom_forces_integration.py | Custom forces integrated with dynamics |
| test_custom_forces_torch.py | TorchForce integration smoke test |
| test_custom_forces_buckets.py | Per-bucket ETERM dispatch (CFIN/CFNB/CFEX/CFMB/CFCV) |
| test_dimens.py | Lazy CHARMM library init + dimens module |
| test_dynamics.py | Langevin dynamics on alanine + 350 TIP3 waters |
| test_e14fac_openmm_blade.py | Three-way energy comparison CHARMM/OpenMM/BLaDE for varying e14fac |
| test_energy.py | Energy module — get_total, get_grms, term breakdown |
| test_errors.py | Safeguard error-classification table + interpret_error |
| test_facts_dock_rescore.py | FACTS_rescore against benzene/T4 — receptor + binding energy |
| test_facts_score.py | FACTS state isolation across successive scoring calls |
| test_grid.py | CDOCKER grid generation + on/off/clear state machine |
| test_grid_ommd.py | OMM-docking grid + soft/hard grid layering |
| test_hydrogen_mass_repartition.py | psf.hmr() preserves total mass |
| test_join.py | Joining peptide + waterbox segments |
| test_lingo.py | charmm_script + get_energy_value + variable get/set |
| test_minimize.py | ABNR vs OpenMM minimization energy agreement |
| test_ml_potential.py | Asparagus ML potential workflow (skipped without asparagus) |
| test_numpy_interop.py | numpy array coercion through pycharmm typed APIs |
| test_openmm.py | OpenMM minimization, stricter tolerance variant |
| test_pnoe_blade.py | Point NOE restraints in BLaDE |
| test_reset.py | The reset module — per-subsystem clears + Layer-2 orchestrators |
| test_restraints_atoms.py | Restraints — harmonic + fixed-atom constraints |
| test_restraints_meta.py | Restraints — SCAT/backend-compat/state/namespace API/logging |
| test_restraints_noe.py | Restraints — NOE distance restraints (basic + PNOE + moving PNOE) |
| test_restraints_resd.py | Restraints — RESD distance/dihedral restraints |
| test_safeguards.py | Optional safeguards system |
| test_script_factory.py | The script_factory module |
| test_select.py | select module — selection helpers (hydrogen, lone, initial, ...) |
| test_select_atoms.py | SelectAtoms class — set arithmetic, accessors |
| test_select_script.py | Persistent named selections + SelectAtoms metadata |
| test_torch_constantforce.py | Torch constant-force restraint via TorchForce |
| test_torch_force_multiple.py | Torch harmonic restraint with multiple force registrations |
| test_torch_harmonic.py | Torch SimpleHarmonic vs CHARMM cons_harm |
| test_torch_harmonic_forces.py | Torch SimpleHarmonic with explicit force return |
| test_torch_harmonic_scaled.py | Torch ScaledHarmonic — energy only |
| test_torch_harmonic_scaled_forces.py | Torch ScaledHarmonic with global parameter scaling |
| test_user_e.py | User-defined energy term |
| test_user_harm.py | User-defined harmonic restraint |
