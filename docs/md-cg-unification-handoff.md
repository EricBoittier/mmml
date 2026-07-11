# Handoff: finishing the `md-system` / `cg_jaxmd` unification

**Purpose:** everything a fresh session needs to complete the remaining work on
the `mmml/md/` unified MD stack. Read this together with
[the design doc](md-cg-unification-design.md) (§0 is the live status; §11 is the
granular checklist).

Baseline commit at handoff: `593c92832` (working tree clean).

---

## 1. What is DONE (built, committed, unit-tested)

The whole shared stack exists in `mmml/md/` and runs end-to-end from a
`RunConfig`:

```
config ─lowering→ RunConfig ─assemble→ builder → HybridEnergy ─auto→ neighbor_fn → { JaxmdDriver | RigidBodySampler }
```

| Piece | Module | Tests |
|---|---|---|
| Topology / FF state (`MolecularSystem`, `FFParams`) | `mmml/md/system.py` | `test_md_package_seams.py`, `test_md_builders.py` |
| Config (`RunConfig`, `EnsembleSpec`) | `mmml/md/config.py` | seams |
| Energy terms (all 6) | `mmml/md/energy/terms/` | `test_md_energy_terms.py`, `test_md_mm_nonbonded.py`, `test_md_ml_terms.py` |
| Capacity / dtype policy | `mmml/md/energy/capacity.py` | `test_md_capacity.py` |
| Builders + FF bridge | `mmml/md/builders/` | `test_md_builders.py` |
| Driver incl. **NPT** | `mmml/md/drivers/jaxmd.py` | `test_md_jaxmd_driver.py` |
| Assembly glue | `mmml/md/assemble.py` | `test_md_assemble.py` |
| Lowering adapters | `mmml/md/lowering.py` | `test_md_lowering.py` |
| Neighbor-list factory | `mmml/md/neighbors.py` | `test_md_neighbors.py` |
| Rigid-body `Sampler` (MC) | `mmml/md/samplers/rigid.py` | `test_md_samplers.py` |
| Cross-platform `libcharmm` loader | `pycharmm/lib.py` | `test_md_pycharmm_lib_loader.py` |

The six energy terms: `ml_intra`, `ml_pep_water`, `mm_nonbonded`, `vdw_core`,
`smd`, `dihedral` — all registered, box-aware where relevant, parity-tested
against the `cg_jaxmd` originals / the reference nonbonded / the example ML
checkpoint (`examples/sppoky-epoch-0010_params.json`).

Full `mmml/md` package unit suite (the tests above + `test_cg_jaxmd_unified.py`,
§2a): **102 tests, all green** (running the whole `tests/unit/test_md_*.py` glob,
which also includes pre-existing legacy `md-system` tests unrelated to this
package, gives 324 passed / 1 skipped — also green, no regressions).

---

## 2. What REMAINS

### 2a. `examples/cg_jaxmd.py` swap — DONE (as a new parallel script)

**`examples/cg_jaxmd_unified.py`** is a thin, validated front-end over
`assemble_and_run` for cg_jaxmd-style peptide-water runs: JSON config →
`runconfig_from_cg_config(cfg, phase)` → `assemble_and_run(...)`, chaining
`fire → nvt → nve` phases with positions carried forward
(`dataclasses.replace(system, R=...)` between phases; velocities are **not**
carried — a documented simplification). It is a **new script alongside** the
original `examples/cg_jaxmd.py` (untouched), not an in-place rewrite — safer to
validate, and the original stays available for feature parity (φ/ψ restraints,
H-X bond repair, DCD export, diagnostics — all explicitly out of scope for the
new front-end; it raises `NotImplementedError` for `constrain_phi_psi` rather
than silently diverging).

Validated end-to-end against real CHARMM builds
(`PeptideWaterSystemBuilder(n_molecules=4, box_size=15.0)`) + the example
checkpoint (`examples/sppoky-epoch-0010_params.json`): single-phase runs,
multi-phase chaining (`nvt`'s start energy matches `fire`'s end energy exactly),
and the `peptide_water_ml` + `peptide_water_ml_core_vdw` combination. Tests in
`tests/unit/test_cg_jaxmd_unified.py` (11 tests, ~1.5 min including 3 real
CHARMM+checkpoint integration tests).

**Two real bugs were found and fixed via this end-to-end validation** (details
in design doc §0):
1. `JaxmdDriver`'s fixed-box NVE/NVT/FIRE path silently wrapped real-space (Å)
   positions as fractional coordinates (`space.periodic_general` defaults to
   `fractional_coordinates=True`), causing instant divergence on the first
   integration step. Fixed in `mmml/md/drivers/jaxmd.py`.
2. `assemble_and_run`'s auto-wired neighbor list didn't exclude peptide-water
   pairs from `mm_nonbonded` when `ml_pep_water` was active, double-counting
   that interaction. Fixed in `mmml/md/assemble.py` (checks
   `"ml_pep_water" in config.terms`).

**Gotcha for the next session:** CHARMM has persistent global state across
builds within one Python process — the "same" builder call with the "same"
seed can produce a different geometry depending on what was built earlier in
the same process/pytest session. Don't assert absolute energy magnitudes in
CHARMM-integration tests; assert relative/structural properties instead (see
`test_end_to_end_peptide_water_ml_no_double_counting` for the pattern: compare
with-exclusion vs. without-exclusion on the *same* built geometry).

Also remember: only pass `driver=` to `assemble_and_run` if you've wired the
neighbor_fn yourself — otherwise the auto-wiring (only triggered
`if driver is None`) is skipped and `mm_nonbonded`'s host pair-build path will
raise `TracerArrayConversionError` under jit.

### 2b. `mmml/cli/run/md_system.py --backend jaxmd` swap (still open, highest value remaining)

`run_backend()` currently dispatches to `md_pbc_suite/jaxmd.py`. Route it
through `runconfig_from_md_system_args(args)` → `assemble_and_run(...)` instead,
then retire the duplicate inline loop once trajectories match.

**Validation gate:** run a short trajectory through BOTH the old and new paths
on the *same* small solvated system and compare energies/frames. Because a
peptide-only PSF is a single molecule (one `mol_id` → zero intermolecular
pairs), use a **solvated multi-molecule build** (packmol or
`PeptideWaterSystemBuilder`), not `pept.psf` alone — same lesson as 2a. The
`cg_jaxmd_unified.py` front-end (§2a) is a good template for how to wire a
front-end onto `assemble_and_run`.

### 2c. RDF validation of rigid sampling
Run `RigidBodySampler` vs. flexible MD on a small liquid box and compare the
radial distribution function. Needs a real force field run. (§11 "Rigid
sampling", last unchecked item.)

### 2d. apocharmm GPU driver
Decided design is in the design doc (§4, §10, §11): device-side ML forces via a
custom `ForceManager`, DLPack transport, apocharmm owns the loop. Needs the
pybind11 GPU build first — **not doable on this Mac** (no CUDA). Blocked on
hardware.

---

## 3. Environment / how to run

- **Python:** use `.venv/bin/python` (jax 0.10.2, jax-md installed).
- **CHARMM:** `libcharmm.dylib` lives at `setup/charmm/`. `pycharmm/lib.py`
  auto-discovers it, but exporting `CHARMM_LIB_DIR=/Users/ericboittier/mmml/setup/charmm`
  is the safe belt-and-braces for test runs.
- **Run the md suite:**
  ```bash
  export CHARMM_LIB_DIR=/Users/ericboittier/mmml/setup/charmm
  .venv/bin/python -m pytest tests/unit/test_md_*.py -p no:cacheprovider -q
  ```
- **Lint:** `.venv/bin/python -m ruff check mmml/md/`
- **Docs preview:** `.venv/bin/mkdocs serve -a 127.0.0.1:8000` (the design +
  decomposition pages are in the nav).

---

## 4. Conventions / gotchas (don't relearn these the hard way)

1. **Import-lightness is a hard invariant.** `import mmml.md` must NOT pull in
   jax/ASE/CHARMM. Heavy imports live *inside* `make()` / `run()` / factory
   functions, never at module top. There is a test asserting this
   (`test_md_package_seams.py`); keep it green.
2. **Term registration is lazy.** Terms register when `mmml.md.energy.terms` is
   imported. `build_hybrid_energy` imports it; standalone `available_terms()`
   tests must import it first.
3. **jit rules for terms** (see `docs/hybrid-mlmm-decomposition.md` §6): static
   shapes, pad don't resize, `mask`-multiply not boolean-index, clamp masked
   distances before `sqrt`/`1/r`, close over constants in `make()`.
4. **Dtype policy:** all float math is float64; only indices (`int32`) and masks
   (`int8`) are low precision. Constants in `mmml/md/energy/capacity.py`.
5. **Box-aware terms:** `mm_nonbonded`, `vdw_core`, `smd` accept an optional
   `box` kwarg (threaded by the driver for NPT). Non-PBC terms ignore it via
   `**kwargs`. Don't break this — NPT depends on it.
6. **`mm_nonbonded` needs a neighbor_fn under jit** (its host pair-build path is
   numpy and won't trace). `assemble_and_run` auto-wires one; a hand-rolled
   driver call must pass `neighbor_fn=make_intermolecular_neighbor_fn(...)`.
7. **CHARMM `rmin` is Rmin/2** (a half-value); the LJ sigma pair rule is a
   *sum*, not the usual σ combining. `FFParams.rmin_half` preserves this.
8. **Validation style:** every extracted term was checked against its original
   (the cg_jaxmd formula or the reference `nonbonded_energy_and_forces`), not
   just self-consistency. Keep that bar for the entrypoint swaps (compare to the
   existing trajectory).

---

## 5. Suggested order for the next session

1. ~~`cg_jaxmd` entrypoint swap~~ — done (§2a), as `examples/cg_jaxmd_unified.py`.
2. `md_system --backend jaxmd` swap + comparison, then delete the inline loop (§2b).
3. RDF validation of rigid sampling (§2c).
4. apocharmm — only once a CUDA box + build are available (§2d).

Update `docs/md-cg-unification-design.md` §0 and §11 as each lands (that has been
the running convention).
