# Handoff: finishing the `md-system` / `cg_jaxmd` unification

**Purpose:** everything a fresh session needs to complete the remaining work on
the `mmml/md/` unified MD stack. Read this together with
[the design doc](md-cg-unification-design.md) (§0 is the live status; §11 is the
granular checklist).

Baseline commit at original handoff: `593c92832`. Both entrypoint swaps (§2a,
§2b) have since landed on top of that baseline — see design doc §0 for the
running status.

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

### 2b. `mmml/cli/run/md_system.py --backend jaxmd` swap — DONE (opt-in flag)

**`--jaxmd-unified`** on `mmml md-system --backend jaxmd` routes through
`mmml/cli/run/md_system_unified.py`: `runconfig_from_md_system_args(args)` →
`assemble_and_run(...)`, instead of the legacy `md_pbc_suite/jaxmd.py` inline
loop. It is **opt-in** (`main()` in `md_system.py` checks
`getattr(args, "jaxmd_unified", False)` right before `build_command`/
`run_backend`) — the default (unflagged) path is completely untouched; 66
pre-existing `md_system` CLI tests still pass unchanged.

Only the **packmol composition builder** is wired (`--builder pyxtal`,
`--template-pdb`, `--continue-from` all raise `NotImplementedError` rather
than silently diverging — see `check_md_system_args_supported`). Validated
end-to-end via the real CLI for both `pbc_nve` and `pbc_nvt`:

```bash
mmml md-system --backend jaxmd --jaxmd-unified \
  --setup pbc_nve --composition "TIP3:4" --box-size 15.0 \
  --checkpoint examples/sppoky-epoch-0010_params.json \
  --dt-fs 1.0 --ps 0.01 --seed 6
```

Tests in `tests/unit/test_md_system_unified.py` (9 tests: 6 fast config
validation + 3 real end-to-end CHARMM+checkpoint runs).

**Two gaps found and fixed while validating this** (the packmol composition
path exercises code the peptide-water path in §2a never touched):

1. `build_packmol_composition_cluster` (used by the packmol `SystemBuilder`)
   **never writes a PSF file** — unlike `build_trialanine_water_box_in_charmm`
   (§2a's builder), it only leaves the system live in CHARMM. Without a PSF
   file, `FFParams` can't be resolved and `mm_nonbonded` can't run.
   `build_packmol_system_with_ffparams()` in `md_system_unified.py` fixes this
   by writing the live PSF to a scratch file (`pycharmm.write.psf_card(...)`)
   right after the packmol build, then lowering it exactly like
   `_lower_optional_psf` does.
2. The packmol composition builder **does not self-bootstrap CHARMM** either
   (no internal `ensure_pycharmm_loaded()` call) — it assumes the caller
   already did. `run_unified_jaxmd()` now calls `ensure_pycharmm_loaded()`
   explicitly (it's idempotent) before building. Symptom if you skip this:
   `AttributeError: 'NoneType' object has no attribute 'lingo'` deep inside
   CHARMM's `generate`/`ic` calls — not obviously connected to a missing
   bootstrap call unless you already know the fix.

**Still open, lower priority:** wiring `--builder pyxtal` / `--template-pdb`
/ `--continue-from` (handoff) into the unified path, and eventually retiring
the legacy inline loop once the unified path's coverage is a strict superset.

### 2b′. Cross-backend sweep — DONE — `workflows/unified_backend_sweep/`

A Snakemake workflow smoke-testing every driver/sampler + ensemble reachable
through `assemble_and_run` (`jaxmd_min`, `jaxmd_nve`, `jaxmd_nvt`, `jaxmd_npt`,
`rigid_mc`) on the same small real system (4-water TIP3 box, `ml_intra` +
`mm_nonbonded`, the example checkpoint), across several seeds. See its own
README for usage.

**This is what caught the `RigidBodySampler` neighbor_fn gap** (§11 "Rigid
sampling"): the sampler had no way to supply `mm_nonbonded`'s pair list, so any
intermolecular-pair term raised `TracerArrayConversionError` under its
`jax.jit`. Fixed by giving `RigidBodySampler` a `neighbor_fn` /
`neighbor_refresh_every` mechanism mirroring `JaxmdDriver` (rebuilt once per N
sweeps — rebuilding on the host per proposed move would dominate runtime), and
extending `assemble_and_run`'s auto-wiring (refactored into a shared
`_auto_neighbor_fn` helper) to cover the rigid path too.

**Also found:** `jaxmd_npt` takes ~4 minutes to JIT-compile with a real ML
model on a 12-atom system, vs. ~15–30 seconds for FIRE/NVE/NVT/rigid-MC. Not a
bug — the fractional↔real coordinate bridge plus the ML model's autodiff graph
is just markedly more expensive to trace/compile for NPT. The workflow's
`config.yaml` gives `jaxmd_npt` a longer `runtime_min` for this reason. Budget
for it if you add more backends/larger systems to this sweep.

### 2c. RDF validation of rigid sampling
Run `RigidBodySampler` vs. flexible MD on a small liquid box and compare the
radial distribution function. Needs a real force field run. (§11 "Rigid
sampling", last unchecked item.) The `unified_backend_sweep` workflow's
`rigid_mc` + `jaxmd_nvt` settings are a starting point — extend it to compute
and compare an actual RDF rather than just an energy/finite-ness check.

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
9. **A legacy builder function working standalone does NOT mean it self-
   bootstraps CHARMM or persists a PSF.** Different legacy build paths make
   different implicit assumptions about caller state
   (`build_trialanine_water_box_in_charmm` self-bootstraps and writes a PSF;
   `build_packmol_composition_cluster` does neither). Check both explicitly
   for any *new* builder you wire up, rather than assuming the pattern from
   one builder generalizes to the next.
10. **CHARMM's persistent global state biases test order.** A builder call with
    the "same" seed can produce different geometry depending on what was built
    earlier in the same process. Don't assert absolute energies in
    CHARMM-integration tests — assert relative/structural properties on the
    *same* built geometry instead (§2a's `test_end_to_end_peptide_water_ml_no_double_counting`
    is the pattern to copy).

---

## 5. Suggested order for the next session

1. ~~`cg_jaxmd` entrypoint swap~~ — done (§2a), as `examples/cg_jaxmd_unified.py`.
2. ~~`md_system --backend jaxmd` swap~~ — done (§2b), as the opt-in
   `--jaxmd-unified` flag / `mmml/cli/run/md_system_unified.py`. Only the
   packmol composition builder is wired; extending to pyxtal/template-PDB/
   handoff and eventually retiring the legacy inline loop remains open.
3. RDF validation of rigid sampling (§2c).
4. apocharmm — only once a CUDA box + build are available (§2d).

Update `docs/md-cg-unification-design.md` §0 and §11 as each lands (that has been
the running convention).
