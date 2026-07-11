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

Full `mmml/md` unit suite: ~90 tests, all green.

---

## 2. What REMAINS

All three items are integration- or hardware-gated — they need real runs, not
more unit-testable library code.

### 2a. Swap the two legacy entrypoints onto `assemble_and_run` (highest value)

The lowering + assembly layers exist; the actual entrypoints still run their own
inline loops.

- **`examples/cg_jaxmd.py`** (do this first — smaller, checkpoint/build path
  already exercised). Make it a thin front-end: read JSON →
  `runconfig_from_cg_config(cfg, phase)` → `assemble_and_run(...)`. The energy
  model comes from `EnergyContext(model=..., params=...)` via
  `create_calculator_from_checkpoint` (see how `test_md_ml_terms.py` loads it).
  The builder is `peptide_water` (`PeptideWaterSystemBuilder`).
- **`mmml/cli/run/md_system.py`** `--backend jaxmd`: `run_backend()` currently
  dispatches to `md_pbc_suite/jaxmd.py`. Route it through
  `runconfig_from_md_system_args(args)` → `assemble_and_run(...)` instead, then
  retire the duplicate inline loop once trajectories match.

**Validation gate (important):** run a short trajectory through BOTH the old and
new paths on the *same* small solvated system and compare energies/frames.
Because a peptide-only PSF is a single molecule (one `mol_id` → zero
intermolecular pairs), you need a **solvated multi-molecule build** for a
meaningful comparison — use `PeptideWaterSystemBuilder` or a packmol build, not
`pept.psf` alone.

### 2b. RDF validation of rigid sampling
Run `RigidBodySampler` vs. flexible MD on a small liquid box and compare the
radial distribution function. Needs a real force field run. (§11 "Rigid
sampling", last unchecked item.)

### 2c. apocharmm GPU driver
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

1. `cg_jaxmd` entrypoint swap + short solvated-run comparison (§2a).
2. `md_system --backend jaxmd` swap + comparison, then delete the inline loop.
3. RDF validation of rigid sampling (§2b).
4. apocharmm — only once a CUDA box + build are available (§2c).

Update `docs/md-cg-unification-design.md` §0 and §11 as each lands (that has been
the running convention).
