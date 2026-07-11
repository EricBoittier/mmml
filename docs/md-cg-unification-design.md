# Unifying `md-system` and `cg_jaxmd`: calculator & builder schema

**Status:** In progress — the shared stack is built and tested; both
front-end entrypoint swaps are landed (opt-in for `md-system`); apocharmm and
RDF validation remain (see §0 and §11).
**Scope:** How to split the calculators and system builders shared by
[`mmml/cli/run/md_system.py`](https://github.com/EricBoittier/mmml/blob/main/mmml/cli/run/md_system.py) and
[`examples/cg_jaxmd.py`](https://github.com/EricBoittier/mmml/blob/main/examples/cg_jaxmd.py) so the two can run on one
architecture.

---

## 0. Implementation status

The design below is largely realized in the `mmml/md/` package. What exists,
end-to-end from a config:

```
config ──lowering──▶ RunConfig ──assemble──▶ builder → HybridEnergy → JaxmdDriver
```

| Layer | Module | State |
|---|---|---|
| Topology / FF state | `mmml/md/system.py` (`MolecularSystem`, `FFParams`) | ✅ tested |
| Config | `mmml/md/config.py` (`RunConfig`, `EnsembleSpec`) | ✅ |
| Energy terms (all 6) | `mmml/md/energy/terms/` | ✅ parity-tested |
| Capacity / dtype policy | `mmml/md/energy/capacity.py` | ✅ tested |
| Builders + FF bridge | `mmml/md/builders/` | ✅ CHARMM-integration tested |
| Driver incl. **NPT** | `mmml/md/drivers/jaxmd.py` | ✅ tested |
| Assembly glue | `mmml/md/assemble.py` | ✅ tested |
| Lowering adapters | `mmml/md/lowering.py` | ✅ tested |
| Neighbor-list factory | `mmml/md/neighbors.py` | ✅ tested |
| Rigid-body `Sampler` (MC) | `mmml/md/samplers/rigid.py` | ✅ tested |

**Energy terms:** `ml_intra`, `ml_pep_water`, `mm_nonbonded`, `vdw_core`, `smd`,
`dihedral` — each registered, box-aware where relevant, and validated against
the `cg_jaxmd` originals / the reference nonbonded / the example ML checkpoint.

**End-to-end:** `assemble_and_run` runs the full pipeline as real dynamics —
`mm_nonbonded` declares an intermolecular `NeighborRequest`, `assemble_and_run`
auto-builds the padded neighbor list (`make_intermolecular_neighbor_fn`), and
the `JaxmdDriver` propagates it (NVE smoke test on a multi-molecule box).

**Cross-cutting:** the cross-platform `libcharmm` loader (`pycharmm/lib.py`)
now self-discovers `setup/charmm` on both `.dylib`/`.so`; `import mmml.md`
stays free of jax/CHARMM (heavy deps are lazy inside `make()` / `run()`); the
whole `tests/unit/test_md_*.py` glob (incl. pre-existing legacy `md-system`
tests) is green (333 passed, 1 skipped).

**Bug fix found via end-to-end validation:** `JaxmdDriver`'s fixed-box
NVE/NVT/FIRE path called `space.periodic_general(box)` without
`fractional_coordinates=False`. jax-md defaults that to `True`, so real-space
(Å) positions were silently treated as fractional coordinates and wrapped
modulo 1 by the integrator's `shift_fn` on the very first step — a discontinuous,
unphysical jump invisible to energy-only checks at `R` but catastrophic once any
step was taken. Found by running the full `assemble_and_run` pipeline
(`PeptideWaterSystemBuilder` + the example ML checkpoint + `ml_intra` +
`mm_nonbonded`) end-to-end: FIRE minimization exploded to ~1e15 eV within a few
steps. Fixed by passing `fractional_coordinates=False` for the fixed-box branch
(the NPT branch was already correct, since it explicitly bridges fractional↔real
itself). Regression-tested in
`tests/unit/test_md_jaxmd_driver.py::test_fixed_box_pbc_keeps_real_space_positions_unwrapped`
(verified to fail without the fix, pass with it). With the fix, a real
CHARMM-built peptide-water system + ML checkpoint runs a stable 50-step FIRE
minimization end-to-end (energy monotonically decreasing, no divergence).

**Second bug fix found the same way:** `assemble_and_run`'s auto-wired
neighbor list did not pass `peptide_water_ml=True` to
`make_intermolecular_neighbor_fn` when the `ml_pep_water` term was selected, so
`mm_nonbonded` still included core–solvent pairs even though `ml_pep_water`
already scores them — a silent double-count of the peptide-water interaction
(classical LJ/Coulomb *and* the ML dimer term). Found while validating the
`peptide_water_ml` toggle path in `examples/cg_jaxmd_unified.py`: energy jumped
to ~2e4 eV instead of the expected MD/MM scale. Fixed in `assemble.py` by
checking `"ml_pep_water" in config.terms`. Regression-tested by comparing the
excluded vs. unexcluded pair list and energy on the same geometry (see
`tests/unit/test_cg_jaxmd_unified.py::test_end_to_end_peptide_water_ml_no_double_counting`) —
note that this test asserts the *relative* with/without-exclusion difference
rather than an absolute energy threshold, since CHARMM's persistent global
state across builds in one process makes absolute geometry (and thus energy)
depend on prior test execution order within the same session.

**Third fix, same technique:** validating `--jaxmd-unified` end-to-end found
that `build_packmol_composition_cluster` (the packmol composition builder used
by `md-system`) neither writes a PSF file nor bootstraps CHARMM itself — both
assumptions the legacy `md_pbc_suite/jaxmd.py` path could rely on implicitly
because its caller already did so, but which a fresh, explicit front-end
cannot. `mmml/cli/run/md_system_unified.py` makes both explicit: it writes the
live CHARMM PSF to a scratch file to resolve `FFParams`, and calls
`ensure_pycharmm_loaded()` before building.

**Remaining (§11):** the apocharmm driver (needs a GPU build) and the RDF
validation of rigid sampling (needs a real force-field production run) are the
two items still gated on hardware/scope rather than more unit-testable code.
The `--jaxmd-unified` flag also doesn't yet cover the pyxtal/template-PDB
builders or handoff/continuation — see §11 for the exact gaps.

---

## 1. Motivation

Today there are two parallel MD stacks that solve ~80% of the same problem but
share almost nothing above the leaf-helper level.

### `md-system` — the orchestrator
- File: `mmml/cli/run/md_system.py` (~3.3k lines).
- Pure dispatcher: `parse_args → build_command() → run_backend()` hands off
  in-process to one of three backend modules:
  - `mmml/cli/run/md_pbc_suite/ase.py` (ASE dynamics)
  - `mmml/cli/run/md_pbc_suite/jaxmd.py` (jax-md)
  - `mmml/cli/run/md_pbc_suite/pycharmm_mlpot.py` (CHARMM dynamics)
- Speaks in **ASE `Calculator` objects**
  (`MonomerSumCalculator` + `JAXIntermolecularCalculator`, see
  `mmml/interfaces/calculators/hybrid.py`).
- Generic liquid/crystal builder (packmol, pyxtal, composition, box sizing)
  plus a campaign / handoff / manifest layer.
- Setups: `{free,pbc}_{nve,nvt,thermalize}`, `pbc_npt`, `pycharmm_*`, `lambda_ti`.

### `cg_jaxmd` — the research script
- File: `examples/cg_jaxmd.py` (~2.6k lines), driven by
  `workflows/cg_jaxmd_ala_water_sweep` (Snakemake).
- Monolithic, single-system (trialanine in a water box).
- Speaks in a **jax-md `energy_fn(R)`** hand-composed from many terms:
  ML intramolecular, ML peptide–water dimers, MM nonbonded, SMD bias,
  φ/ψ restraints, vdW core; plus CHARMM rescue/repair, per-molecule ML charges,
  and inline NHC / NVE / FIRE loops.
- Config = JSON emitted by `scripts/run_setting.py`, swept via Snakemake.
- **Not integrated** with `md-system`; reuses only leaf helpers
  (`jaxmdInterface/hybrid_energy.py`, `calculators/simple_inference.py`,
  the pycharmm builders).

The goal: a shared architecture where `cg_jaxmd` is just a specific
builder + term selection + driver, and `md-system --backend jaxmd` uses the
**same** driver and term registry.

---

## 2. The core tension

There are **two irreconcilable energy contracts**. This is the one place a
single abstraction should *not* be forced — instead we bridge them.

| | ASE side | jax-md side |
|---|---|---|
| Contract | `Calculator.calculate(atoms)` → `results["energy"/"forces"]` | `energy_fn(R, neighbor, box, **kw)` pure & jittable |
| State | stateful, numpy / float64 | pure, static shapes, padded pairs |
| Gradients | finite-diff or per-calc | `jax.grad`, autodiff |
| Neighbors | ASE / rebuilt each call | jax-md `neighbor_list`, capacity-padded |

Everything **below** the energy layer (topology, builders) and **around** it
(config, drivers, term selection) can and should be shared. The schema splits
exactly along those seams.

---

## 3. Design constraints

1. **Two energy faces** must both stay first-class (ASE `Calculator` ⇄ jax
   `energy_fn`). Bridge, don't merge.
2. **CHARMM is stateful, global, CPU, non-jittable** — usable only for
   *building* (PSF / topology / FF params) and *rescue* (minimize / repair),
   never inside a jitted loop.
3. **jax-md needs static shapes** → dynamic intermolecular / peptide–water
   pairs require padding + masks (`cg_jaxmd` already does this with
   `_pad_pairs`, `_pad_peptide_water_slots`).
4. **Energy composition is per-experiment** — SMD, restraints, ML-vs-MM
   intramolecular are toggles. Needs a pluggable **term registry**, not a
   hardcoded sum.
5. **ML models vary** (physnet vs spooky: `charges` / `spins` kwargs) — already
   probed via signature inspection in `hybrid_energy.py`; keep that behind the
   term factory.
6. **Ensembles / space are orthogonal to energy**:
   {free, PBC} × {NVE, NVT (NHC | Langevin), NPT} × {minimize, FIRE}.
7. **Config comes from two mouths** (argparse CLI + Snakemake JSON) — both must
   lower to one internal `RunConfig`.

---

## 4. Available backends

- **Integrator / driver backends:** ASE, jax-md, PyCHARMM (Fortran/pyCHARMM),
  and — proposed — **apocharmm** (GPU-only CHARMM, C++/CUDA + pybind11).
- **Sampler backends** (orthogonal to MD integrators): standard MD, and —
  proposed — **rigid-body sampling** (rigid monomers: translation + rotation
  DOF, SHAKE/SETTLE constraints, MC or rigid MD moves).
- **Builder backends:** packmol (`packmol_placement`, `tip3_liquid_box`,
  `dcm_liquid_box`), pyxtal, `peptide_builder` / `protein_charmm_build`,
  `trialanine_water_box`, template-PDB, `setupBox` / `setupRes`.
- **Energy backends:** physnetjax ML (intramolecular monomer, peptide–water
  dimer), CHARMM MM nonbonded (`nonbonded_energy_and_forces`, jax), CHARMM
  bonded (`cgenff_bonded`), biases (SMD, flat-bottom, φ/ψ, COM restraint).
- **QC / reference backends** (eval only, out of scope for the MD loop): orca,
  pyscf, molpro.

### apocharmm (GPU CHARMM) — interface target

[apocharmm](https://github.com/EricBoittier/apocharmm) is a GPU-only CHARMM MD
package (C++ ~27%, CUDA ~20%, pybind11 Python bindings). It is a natural
fourth `Driver` backend, entered at the **C++/CUDA level** rather than through
the Fortran/pyCHARMM path. Concrete entry points (from `include/`):

- **Container:** `CharmmContext` (holds `CharmmPSF`, `CharmmParameters`,
  `CharmmCrd`, `Coordinates`) — the object a `MolecularSystem` would lower to.
- **Integrators:** `CudaLangevinPistonIntegrator` (NPT),
  `CudaLangevinThermostatIntegrator` / `CudaNoseHooverThermostatIntegrator`
  (NVT), `CudaVelocityVerletIntegrator`, `CudaLeapFrogIntegrator`,
  `CudaBAOABIntegrator`.
- **Forces / neighbor lists:** `CudaNeighborList`, `CudaPMEDirectForce`,
  `CudaPMEReciprocalForce`, `CudaBondedForce` — native PBC + PME on device.
- **Free energy:** `EDSForceManager`, `FEPEIForceManager`, `MBARForceManager`,
  `DualTopologyForceManager`, `PertForceManager` — directly relevant to the
  existing `lambda_dynamics` / `lambda_mbar` / `lambda_ti` code.
- **I/O:** `Subscriber` family (`DcdSubscriber`, `NetCDFSubscriber`,
  `RestartSubscriber`, `StateSubscriber`, `CheckpointSubscriber`).
- **Restraints/constraints:** `GeometricRestraintForce`,
  `HarmonicRestraintForce`, `Constraints` (SHAKE) — the constraint machinery a
  rigid-body sampler would build on.

**Committed direction (decided):** the ML forces must enter apocharmm's
**device force loop as a custom `ForceManager` contribution** — forces stay on
device, no per-step host round-trips. This is the deep C++/CUDA integration,
not host-side evaluation between steps. The two interface levels below are
therefore a **staging order**, not an either/or:

1. **Python (pybind11), host-side ML — validation stepping stone only.** Drive
   `CharmmContext` + `Cuda*Integrator` from an `ApoCharmmDriver`, evaluate jax
   ML forces on the host and add them per step. Used to pin down correctness
   (energy/force parity) before touching CUDA. *Not* the shipping path.
2. **C++/CUDA, device-side ML — the target.** MMML's jax ML forces enter the
   device force loop as a custom `ForceManager` so forces never leave the GPU.
   The buffer boundary crosses via **DLPack capsules** (decided, §10) — not raw
   device pointers — to get shape/dtype/device metadata and proper ownership
   semantics against jax's async allocator. The custom force is subscribed into
   apocharmm's `ForceManagerComposite` (`subscribe<ForceType>(...)`), so
   apocharmm owns the step and the boundary lives in one place. The only
   remaining question is the cross-stream **sync** mechanism (§10).

---

## 5. Proposed schema — 6 layers

```
┌─ orchestration ─────────────────────────────────────────────┐
│  RunConfig (single dataclass)  ← argparse CLI | Snakemake     │
│  campaign / sweep / handoff / manifest                       │
└─────────────────────────────────────────────────────────────┘
             │ builds                    │ selects
             ▼                           ▼
┌─ builders ─────────────┐   ┌─ energy terms (registry) ───────┐
│ SystemBuilder.build()  │   │ EnergyTerm.make(system, ctx)    │
│  packmol / pyxtal /    │   │  ml_intra, ml_pep_water,        │
│  peptide_water / tmpl  │   │  mm_nonbonded, smd, dihedral,   │
└───────────┬────────────┘   │  vdw_core, flat_bottom ...      │
            │ produces       └───────────────┬─────────────────┘
            ▼                                 │ composed into
┌─ MolecularSystem (shared, immutable) ──┐    ▼
│ R, Z, box, mol_id, monomer_indices,    │  HybridEnergy
│ water_indices, psf, ff_params, excl    │  ├─ .as_jax_energy_fn()  → jax-md
└────────────────────────────────────────┘  └─ .as_ase_calculator() → ASE
                                                 │ consumed by
                                                 ▼
┌─ drivers (integrators) ──────────────────────────────────────┐
│ Driver.run(system, energy_provider, ensemble) → Trajectory   │
│  AseDriver | JaxmdDriver (unify runner + cg loop) | CharmmDrv │
└──────────────────────────────────────────────────────────────┘
```

### Layer responsibilities

- **Orchestration** — lowers CLI args *and* Snakemake JSON into one
  `RunConfig`; owns campaign/sweep/handoff/manifest.
- **Builders** — `SystemSpec → MolecularSystem`; thin wrappers over the
  existing `pycharmmInterface` build modules.
- **MolecularSystem** — immutable, backend-agnostic topology artifact that
  every layer above the builders reads.
- **Energy terms** — one class per physics term, registry-keyed, each able to
  emit a jax contribution and/or an ASE contribution.
- **HybridEnergy** — composes the selected terms and exposes both faces
  (`as_jax_energy_fn`, `as_ase_calculator`).
- **Drivers** — one per integrator engine (ASE, jax-md, PyCHARMM, apocharmm);
  consume a `HybridEnergy` and an `EnsembleSpec`.
- **Samplers** — a sibling of drivers, selected by `RunConfig`. MD is the
  default sampler; **rigid-body sampling** is an alternative that moves whole
  monomers as rigid bodies (translation + rotation), via MC moves or
  constrained (SHAKE/SETTLE) rigid MD. Reuses the same `HybridEnergy` and
  `MolecularSystem`; only the propagator differs.

---

## 6. Protocol sketches

```python
# 0. FF parameters — resolved ONCE by the builder, carried as data (decision A, §10).
#    Mirrors NonbondedSystemData field-for-field (see hybrid-mlmm-decomposition §2).
#    No energy term re-derives exclusions / e14 / LJ tables at runtime.
@dataclass(frozen=True)
class FFParams:
    charges: np.ndarray           # (N,) partial charges       ← nbdata.charges
    epsilon: np.ndarray           # (N,) LJ epsilon (kcal/mol) ← nbdata.epsilon
    rmin_half: np.ndarray         # (N,) LJ Rmin/2 (Å)         ← nbdata.rmin
    at_codes: np.ndarray          # (N,) nonbonded type code   ← nbdata.at_codes
    exclusions: np.ndarray        # (M,2) 1-2/1-3 excluded     ← nbdata.excluded_pairs
    e14_pairs: np.ndarray         # (K,2) 1-4 pairs            ← nbdata.e14_pairs
    # FFParams.from_nonbonded_system_data(nbdata) does the conversion (duck-typed).
    # 1-4 SCALING (e14_scale / vdw14) is a pair-build detail, not per-atom FF data.

# 1. Topology — backend-agnostic, immutable; what builders emit / everyone reads
@dataclass(frozen=True)
class MolecularSystem:
    R: np.ndarray                 # (N,3) positions
    Z: np.ndarray                 # (N,) atomic numbers
    box: np.ndarray | None        # (3,3), or None for free space
    mol_id: np.ndarray            # (N,) molecule membership
    monomer_indices: list[np.ndarray]
    water_indices: list[np.ndarray]
    psf_path: Path | None
    ff_params: FFParams | None    # fully-resolved FF state (decision A)
    metadata: dict

# 2. Builders — SystemSpec -> MolecularSystem (also the one place FFParams is built)
class SystemBuilder(Protocol):
    def build(self, spec: SystemSpec) -> MolecularSystem: ...
# PackmolLiquidBuilder, PyxtalCrystalBuilder, PeptideWaterBuilder, TemplatePdbBuilder

# 3. Energy terms — composable, registry-keyed, engine-agnostic factory.
#    Each term declares the neighbor/pair capacity it needs (decision B): the term
#    owns its own padding (peptide-water slots vs. intermolecular pairs differ).
@dataclass(frozen=True)
class NeighborRequest:
    cutoff_A: float
    kind: str                     # "intermolecular" | "peptide_water" | ...
    capacity_hint: int | None     # padded slot count; term sizes its own buffers

class TermFns(NamedTuple):
    jax_energy_fn: Callable | None       # energy_fn(R, nbrs, box, **kw), jittable
    ase_contribution: Callable | None    # numpy energy/forces for ASE
    neighbor_request: NeighborRequest | None

class EnergyTerm(Protocol):
    name: str
    def neighbor_request(self, system: MolecularSystem) -> NeighborRequest | None: ...
    def make(self, system: MolecularSystem, ctx: EnergyContext) -> TermFns: ...
# registry: {"ml_intra", "ml_pep_water", "mm_nonbonded", "smd",
#            "dihedral", "vdw_core", "flat_bottom", ...}

class HybridEnergy:
    # Gathers each term's NeighborRequest, allocates the union of neighbor lists,
    # and routes the right (padded) list to each term at call time.
    def __init__(self, terms: list[EnergyTerm], system, ctx): ...
    def as_jax_energy_fn(self) -> Callable: ...      # for JaxmdDriver
    def as_ase_calculator(self) -> Calculator: ...   # sum of contributions

# 4. Drivers — one per integrator engine
class Driver(Protocol):
    def run(self, system, energy: HybridEnergy, ensemble: EnsembleSpec) -> Trajectory: ...
# AseDriver, JaxmdDriver, CharmmDriver, ApoCharmmDriver
```

---

## 7. Target directory layout

Mostly *moves*, not new code.

```
mmml/md/
  system.py          # MolecularSystem, SystemSpec, FFParams
  builders/          # SystemBuilder impls (wrap existing pycharmmInterface builders)
  energy/
    registry.py      # EnergyTerm protocol + term registry
    terms/           # ml_intra.py, ml_pep_water.py, mm_nonbonded.py,
                     #   smd.py, dihedral_restraint.py, vdw_core.py  ← from cg_jaxmd
    hybrid.py        # HybridEnergy + as_jax_energy_fn / as_ase_calculator
  drivers/
    ase.py           # ← md_pbc_suite/ase.py
    jaxmd.py         # ← unify jaxmd_runner.set_up_nhc_sim_routine + cg_jaxmd loop
    charmm.py        # ← md_pbc_suite/pycharmm_mlpot.py
  config.py          # RunConfig (target of both argparse and Snakemake JSON)

mmml/cli/run/md_system.py   # thin: argv -> RunConfig -> Driver
examples/cg_jaxmd.py        # thin: JSON -> RunConfig -> PeptideWaterBuilder + terms + JaxmdDriver
```

---

## 8. How the two converge

- **`cg_jaxmd`** becomes:
  `PeptideWaterBuilder`
  + `HybridEnergy([ml_intra | ml_pep_water, mm_nonbonded, smd, dihedral, vdw_core])`
  + `JaxmdDriver`.
- **`md-system --backend jaxmd`** becomes:
  `PackmolLiquidBuilder`
  + `HybridEnergy([ml_intra, mm_nonbonded])`
  + the **same** `JaxmdDriver`.

The sweep's two energy-mode toggles (`use_ml_intramolecular`,
`peptide_water_ml`) become term-selection flags in the registry — no code fork.

---

## 9. Highest-leverage moves (recommended order)

1. **Extract `cg_jaxmd`'s energy terms** (SMD, φ/ψ, vdW-core, peptide–water)
   into `mmml/md/energy/terms/` behind the `EnergyTerm` interface. This is what
   un-forks the science and makes the terms reusable from `md-system`.
2. **Merge the two jax-md loops** — `jaxmd_runner.set_up_nhc_sim_routine` and
   `cg_jaxmd`'s inline NHC / NVE / FIRE — into one `JaxmdDriver`. They are
   currently two independently-drifting integrators; this is the biggest drift
   risk.
3. **Introduce `MolecularSystem` + `SystemBuilder`** as thin wrappers over the
   existing pycharmm builders, so both entry points build through one seam.
4. **Lower both configs into `RunConfig`**; make `md_system.py` and
   `examples/cg_jaxmd.py` thin front-ends.

### Suggested migration sequence (non-breaking)

- Land the protocols/dataclasses (`system.py`, `energy/registry.py`,
  `config.py`) with no behavior change.
- Wrap existing builders → `builders/`; keep old call sites delegating.
- Extract terms one at a time; validate energy parity against `cg_jaxmd`
  diagnostics (`diagnose_energy`, `run_force_and_nl_diagnostics`) at each step.
- Build `JaxmdDriver` from `jaxmd_runner`, then port `cg_jaxmd` onto it behind a
  flag; compare trajectories before removing the inline loop.
- Flip `md_system --backend jaxmd` onto `JaxmdDriver`; retire the duplicate.

---

## 10. Decisions & open questions

### Decided

- **`FFParams` boundary → captured as data (option A).** CHARMM FF state
  (charges, LJ tables, exclusions, e14 / vdw14) is resolved **once by the
  builder** and carried on `MolecularSystem.ff_params` (see §6). Energy terms
  read it; none re-derive it at runtime. This removes the inline recomputation
  `cg_jaxmd` currently does and makes term evaluation pure w.r.t. FF state.
- **Neighbor-list ownership → per-term capacities (option B).** Each
  `EnergyTerm` declares a `NeighborRequest` (cutoff, kind, padded capacity) and
  owns its padding; `HybridEnergy` allocates the union and routes the right
  list to each term. Peptide–water slots and intermolecular pairs keep
  independent capacities rather than sharing one driver-owned list.
- **apocharmm ML forces → device-side custom `ForceManager` (decided).** ML
  forces enter apocharmm's device force loop; they do **not** stay host-side.
  Host-side pybind11 evaluation is only a validation stepping stone (§4).
- **apocharmm buffer transport → DLPack capsule (decided).** jax↔apocharmm
  device buffers cross as DLPack capsules (`jax.dlpack.to_dlpack` /
  `from_dlpack`), **not** raw CUDA device pointers. Rationale: DLPack carries
  shape/dtype/device/strides and a capsule deleter, so it is safe against jax's
  async allocator and buffer donation (a raw pointer pulled from jax can be
  reused/freed under us, and layout mismatches fail silently). A thin pybind11
  shim wraps apocharmm's `CudaContainer`/`DeviceVector` buffers as
  `DLManagedTensor`. The unavoidable `float4 xyzq` (f32) ↔ `N×3` / `double`
  layout+precision transform is a small custom kernel on top of the wrap.
- **apocharmm owns the step (decided).** The custom `MLForceManager` is
  subscribed into apocharmm's `ForceManagerComposite`
  (`subscribe<ForceType>(tag, stream, values, energyVirial)`) with its own
  stream; apocharmm's integrator drives the loop and the DLPack handoff lives
  in one place (`MLForceManager::calcForce`), accumulating (`+=`) into its
  force `values`. The jax-owns-loop / XLA-custom-call alternative is rejected —
  apocharmm is a stateful CUDA engine, not a leaf op.
- **Rescue path → explicit driver hook.** CHARMM repair/minimize is modelled as
  a driver hook (`on_overlap`), never hidden inside an energy term, so energy
  terms stay pure and the impurity is confined to the driver.
- **Rigid-body DOF → quaternions (decided).** Rigid-monomer orientation is
  represented as a unit quaternion (translation = COM), renormalized after each
  move. Preferred over rotation matrices: 4 numbers vs. 9, no orthogonality
  drift to project out, and clean composition for MC rotation moves.
- **Rigid sampling → `Sampler` peer, not a driver constraint mode (decided).**
  Rigid-body sampling is its own `Sampler` (peer of the MD `Driver`, selected by
  `RunConfig`), reusing the same `MolecularSystem` + `HybridEnergy`. It is not
  bolted into each integrator as a constraint mode, so it composes with any
  energy term and any backend without touching the drivers.

### Still open

- **Cross-stream sync mechanism** — how to order the jax ML-force stream against
  apocharmm's force-manager stream. Options: (1) host barrier per step
  (`block_until_ready` + `cudaStreamSynchronize`) — correct, on-device, but
  serializes; (2) event-based overlap (`cudaEventRecord`/`cudaStreamWaitEvent`)
  — blocked by jax not exposing its CUDA stream publicly; (3) DLPack
  stream-aware protocol (`__dlpack__(stream=...)`, needs recent jax + shim
  support). **Plan: ship the barrier (1) first, measure, and only pursue
  overlap (3) if throughput-bound** — the transport (DLPack) and structure
  (apocharmm-owns-loop) are settled; only the ordering optimization is open.

This is the only genuinely open design question; it is deferred to
implementation/measurement rather than blocked. All other seams are decided.

### PyCHARMM / MPI runtime contract

PyCHARMM is not an ordinary optional Python import in this stack. The vendored
package loads a native `libcharmm` that may itself be linked to OpenMPI and
Fortran. A native `STOP`, `MPI_Abort`, loader mismatch, or segmentation fault
terminates the interpreter; pytest cannot convert it into a Python traceback.
Consequently, **successful import of the MMML bootstrap module is not proof that
live CHARMM is available**. The authoritative runtime signal is
`import_pycharmm.PYCHARMM_AVAILABLE`.

There are three supported execution modes:

| Mode | Intended work | Contract |
|---|---|---|
| Offline/unit | restart parsing and text patching, builders mocked at the CHARMM boundary | pytest sets `MMML_WARMUP_MLPOT_JAX_ONLY=1`; do not enter native PyCHARMM |
| Live, one rank | PSF/parameter reads, restart writes, CHARMM parity tests | launch with `MMML_MPI_NP=1 ./scripts/mmml-charmm-mpirun.sh ...` when `libcharmm` is MPI-linked |
| Live, multiple ranks | DOMDEC / spatial ML MPI | launch with the same wrapper and an explicit `MMML_MPI_NP`; validate Tier 2 first |

The launcher is part of the runtime ABI, not merely a convenience wrapper: it
selects the OpenMPI installation compatible with `libcharmm`, establishes the
library search path, constrains OpenMP threads, and orders MPI/mpi4py/JAX
initialization. Do not run an MPI-linked `libcharmm` from a plain `pytest` or
`python` process merely because `import pycharmm` appears to work.

Preflight and test recipes:

```bash
# Loader/OpenMPI/mpi4py survey before launching a job
mmml mpi-check --strict

# Live one-rank PyCHARMM smoke in the real runtime
MMML_MPI_NP=1 ./scripts/mmml-charmm-mpirun.sh \
  python -m pytest tests/charmm_mpi/test_mpi_live_energy.py -q

# Spatial MPI validation before a multi-rank MD run
mmml mpi-check --tier2 --prelaunch --strict
MMML_MPI_NP=2 MMML_MLPOT_SPATIAL_MPI=1 \
  ./scripts/mmml-charmm-mpirun.sh mpi-check --tier2 --strict
```

For normal runs, the implemented `mpi-launch` front end keeps environment,
rank topology, and JAX policy separate. `uv run` resolves Python once and the
selected `sys.executable` is forwarded through the existing ABI-aware wrapper:

```bash
uv run mmml mpi-launch --preset single -- md-system --config run.yaml
uv run mmml mpi-launch --preset cpu --jax-cpu-threads 16 -- \
  md-system --config run.yaml
uv run mmml mpi-launch --preset spatial --mpi-ranks 4 -- \
  md-system --config run.yaml --ml-spatial-mpi
```

The presets are aliases over independent `--mpi-ranks`, `--jax-mode`,
`--jax-cpu-threads`, and `--charmm-omp-threads` settings. This also supports
multi-rank CHARMM with rank-0 JAX and generic GPU-per-rank JAX without claiming
that either is spatial decomposition. See [`mpi-launch`](cli/commands/mpi-launch.md).

`MMML_NO_CHARMM_MPI=1` is only valid for a genuinely serial `libcharmm`; it must
not be used to disguise an MPI-linked build. Likewise,
`MMML_WARMUP_MLPOT_JAX_ONLY=1` means "bootstrap/import seams only" and must
never be interpreted as permission to call CHARMM state, coordinate, energy,
or restart APIs.

The handoff boundary follows this rule explicitly: with a usable restart
template and `PYCHARMM_AVAILABLE=False`, `save_handoff_to_res` performs the
offline, validated text patch. It enters `_write_handoff_restart_via_charmm`
only when native CHARMM is genuinely loaded. This prevents the former failure
where `test_save_handoff_to_res_with_template` ended at native level with exit
code 1 and no pytest traceback.

For the full operational matrix and cluster troubleshooting, see
[PyCHARMM + MPI](pycharmm-mpi.md), [`mpi-check`](cli/commands/mpi-check.md), and
[`health-check`](cli/commands/health-check.md).

---

## 11. Roadmap checklist

Tracking checklist for the unification, plus the two newly-scoped capabilities
(rigid sampling, apocharmm interface). Migration order is deliberate:
land seams non-breaking, extract terms with parity checks, then add backends.

### Core unification (from §9)

- [x] Land protocols/dataclasses (`system.py`, `energy/registry.py`,
      `config.py`) with no behavior change. *(Done: `mmml/md/` package —
      `system.py` (`FFParams`/`MolecularSystem`/`SystemSpec`), `config.py`
      (`RunConfig`/`EnsembleSpec`), `results.py` (`Trajectory`),
      `energy/registry.py` (`EnergyTerm`/`TermFns`/`NeighborRequest`/
      `HybridEnergy` + term registry), and `builders`/`drivers`/`samplers`
      Protocols. Imports pull in no jax/ASE/CHARMM; smoke-tested in
      `tests/unit/test_md_package_seams.py`.)*
- [~] Wrap existing builders into `builders/` (`SystemBuilder`); old call sites
      delegate. *(In progress.)*
  - [x] `PsfSystemBuilder` — builds a `MolecularSystem` from a PSF + params +
        coords; populates `FFParams` field-for-field from `NonbondedSystemData`
        (`FFParams.from_nonbonded_system_data`) and partitions molecules from the
        PSF bond graph (`molecule_ids_from_bonds`). CHARMM-integration tested on
        `pept.psf`/`pept.pdb`: `tests/unit/test_md_builders.py`.
  - [x] Packmol / PyXtal / peptide-water placement wrappers — legacy builders
        now lower PSF-ordered coordinates, molecule partitions, box, residue
        identity, and water groups into `MolecularSystem`. Packmol/PyXtal can
        optionally lower a supplied emitted PSF to `FFParams`; the
        trialanine-water wrapper always delegates its emitted PSF/parameter
        files through `PsfSystemBuilder`. Backend calls are lazy and mock-backed
        seam tests require no CHARMM/Packmol/PyXtal:
        `tests/unit/test_md_builders.py`. Old front-end call sites still need to
        delegate before the parent item is complete.
- [x] Extract `cg_jaxmd` energy terms into `energy/terms/` one at a time,
      validating at each step. *(Complete — all six terms extracted, registered,
      and parity-tested.)*
  - [x] `smd` — `SMDBiasTerm` (moving harmonic end-to-end restraint, PBC/free,
        `lambda_t` steering). Parity-tested vs. the cg_jaxmd formula + ASE
        forces vs. finite difference: `tests/unit/test_md_energy_terms.py`.
  - [x] `dihedral` — `DihedralRestraintTerm` (φ/ψ backbone restraints).
        Parity-tested vs. the cg_jaxmd formula.
  - [x] `vdw_core` — `RepulsiveCoreVdwTerm` (repulsive LJ wall between a core
        atom group and molecule groups, smooth cutoff, optional padded slots).
        Parity-tested vs. the cg_jaxmd formula; takes explicit ε/Rmin arrays now,
        to be sourced from `FFParams` by the builder.
  - [x] `mm_nonbonded` — `MMNonbondedTerm`. Faithful (B)-block extraction reusing
        the shared switching kernels (`_pair_vdw_energy` / `_pair_elec_energy`):
        jittable padded-pair jax face (`pair_i/pair_j/pair_mask/e14/vdw14`) plus
        an ASE face wrapping `nonbonded_energy_and_forces`. Consumes `FFParams`.
        Parity-tested to ~1e-9 vs. the reference (energy + forces, host/padded/
        jit/ASE): `tests/unit/test_md_mm_nonbonded.py`.
  - [x] `ml_intra` — `MLIntramolecularTerm` (per-monomer ML energy) and
        `ml_pep_water` — `MLCoreGroupTerm` (core–group ML dimers over padded
        `active_group_slots`). Thin wrappers over
        `jaxmdInterface.hybrid_energy`; model+params come from the run
        `EnergyContext` (model-agnostic). Validated against the factory calls +
        jit + composition using the last-epoch example checkpoint
        (`examples/sppoky-epoch-0010_params.json`):
        `tests/unit/test_md_ml_terms.py`.
- [x] Build `JaxmdDriver` from `jaxmd_runner.set_up_nhc_sim_routine`. Shared
      driver in `mmml/md/drivers/jaxmd.py`: lazy optional imports, free/PBC NVE,
      NVT-NHC, FIRE, **NPT (Nosé–Hoover barostat)**, block-boundary per-term
      neighbor refresh, overlap-repair hook, partial-final-block recording, and
      optional NPZ output. NPT keeps the real-space terms via a fractional↔real
      bridge and box-aware terms (`mm_nonbonded` / `vdw_core` / `smd` accept a
      `box` kwarg; box threaded each step). Tests (FIRE/NVE/NVT, fixed-box PBC,
      **box-evolving NPT**): `tests/unit/test_md_jaxmd_driver.py`.
- [~] Wire the front-ends onto the shared stack. *(Assembly glue landed:
      `mmml/md/assemble.py` — builder registry (`get_builder` /
      `available_builders` / `build_system`), `build_hybrid_energy` (term
      registry + per-term kwargs), and `assemble_and_run` (RunConfig → builder →
      HybridEnergy → JaxmdDriver). End-to-end tested from a `RunConfig`:
      `tests/unit/test_md_assemble.py`.)*
  - [x] Lowering adapters (`mmml/md/lowering.py`): `runconfig_from_md_system_args`
        (argparse `Namespace` → `RunConfig`) and `runconfig_from_cg_config`
        (cg_jaxmd JSON + phase → `RunConfig`), with `terms_from_cg_config`
        implementing the sweep-toggle → term-selection mapping (doc §8). Pure /
        numpy-only; tested in `tests/unit/test_md_lowering.py`.
  - [x] `mmml/cli/run/md_system_unified.py` — an **opt-in** `--jaxmd-unified`
        flag on `mmml md-system --backend jaxmd` that routes through
        `runconfig_from_md_system_args` → `assemble_and_run` instead of the
        legacy `md_pbc_suite/jaxmd.py` inline loop. Opt-in (not the default)
        so the existing path is untouched until this covers more of
        md-system's feature surface — only the packmol composition builder is
        wired; `--builder pyxtal`, `--template-pdb`, and `--continue-from`
        raise `NotImplementedError` rather than silently diverging.
        `build_packmol_system_with_ffparams` closes a real gap: the packmol
        composition builder (`build_packmol_composition_cluster`) never writes
        a PSF file, so it writes the live CHARMM PSF to a scratch file to
        resolve `FFParams` (mirrors `_lower_optional_psf`). Also explicitly
        calls `ensure_pycharmm_loaded()` before building — unlike
        `PeptideWaterSystemBuilder`'s underlying builder, the packmol
        composition builder does not self-bootstrap CHARMM. Validated
        end-to-end via the real CLI (`mmml md-system --backend jaxmd
        --jaxmd-unified ...`) for both `pbc_nve` and `pbc_nvt` with a real
        checkpoint; existing (non-flagged) `--backend jaxmd` behavior is
        unaffected (66 pre-existing `md_system` CLI tests still pass). Tests
        in `tests/unit/test_md_system_unified.py` (9 tests: 6 fast config
        validation + 3 real end-to-end CHARMM+checkpoint runs, including
        `FFParams` resolution and both `pbc_nve`/`pbc_nvt`).
        The full legacy dispatch remains the default; retiring the duplicate
        inline loop is deferred until the unified path covers pyxtal/template
        builders and handoff.
  - [x] `examples/cg_jaxmd_unified.py` — a thin, validated front-end over
        `assemble_and_run` for cg_jaxmd-style peptide-water runs: reads a
        cg_jaxmd JSON config, chains `fire → nvt → nve` phases (carrying
        positions forward between phases via `dataclasses.replace`), and wires
        `ml_intra` / `mm_nonbonded` / `ml_pep_water` / `vdw_core` / `smd` term
        kwargs from the config toggles. Deliberately raises
        `NotImplementedError` for `constrain_phi_psi` (not yet wired) rather
        than silently diverging from the legacy script. This is a **new
        parallel script**, not an in-place rewrite of the 2624-line
        `examples/cg_jaxmd.py` — safer to validate, and `cg_jaxmd.py` is
        untouched. Validated end-to-end against real CHARMM builds
        (`PeptideWaterSystemBuilder`) + the example checkpoint: single-phase
        runs, multi-phase position chaining (`nvt`'s start energy matches
        `fire`'s end energy exactly), and the `peptide_water_ml` +
        `peptide_water_ml_core_vdw` combination. Tests in
        `tests/unit/test_cg_jaxmd_unified.py` (11 tests: config validation,
        pure `term_kwargs` derivation, and 3 real end-to-end CHARMM+checkpoint
        integration tests).

### Rigid sampling

- [x] Define a `Sampler` protocol (peer of `Driver`) selected by `RunConfig`;
      MD is the default sampler. `assemble_and_run` dispatches
      `config.sampler == "rigid"` to the sampler.
- [x] Rigid-body state: per-monomer COM + unit-quaternion orientation, groups
      from `MolecularSystem.monomer_indices` (`quat_from_axis_angle` /
      `quat_to_matrix` in `mmml/md/samplers/rigid.py`).
- [x] Rigid-move propagator: Metropolis MC translation + quaternion rotation,
      reusing the jitted `HybridEnergy`. Rigid moves preserve every
      intramolecular distance (tested to 1e-8): `tests/unit/test_md_samplers.py`.
- [x] Acceptance uses the composed energy, so existing bias terms (SMD, etc.)
      participate automatically (exercised via `assemble_and_run` in
      `tests/unit/test_md_assemble.py`).
- [ ] Validate rigid sampling reproduces liquid structure (RDF) vs. flexible MD
      on a small box. *(Needs a real forcefield run — deferred.)*
- [ ] Optional: constrained (SHAKE/SETTLE) rigid MD variant. *(MC path landed
      first; not required for the sampler to be usable.)*

### apocharmm (GPU CHARMM) interface

- [ ] Build apocharmm and confirm the pybind11 module imports in the MMML env
      (CUDA 11.1.1+, GCC 10.1+, NetCDF4).
- [ ] `MolecularSystem` → `CharmmContext` lowering (`CharmmPSF`,
      `CharmmParameters`, `CharmmCrd`).
- [ ] `ApoCharmmDriver` (Python/pybind11): map `EnsembleSpec` to
      `CudaLangevinPistonIntegrator` (NPT) /
      `CudaLangevinThermostatIntegrator` / `CudaNoseHooverThermostatIntegrator`
      / velocity-Verlet.
- [ ] Wire `Subscriber` I/O (`DcdSubscriber` / `NetCDFSubscriber` /
      `RestartSubscriber`) into the shared `Trajectory` output.
- [ ] **Host-side ML** (validation stepping stone, not shipping): evaluate MMML
      jax forces between steps and add them via a pybind11 force hook; use only
      to pin energy/force parity before touching CUDA.
- [ ] **Device-side ML — the target (committed, §10):** expose MMML ML forces
      as a custom apocharmm `ForceManager` so forces never leave the GPU.
  - [ ] pybind11 shim wrapping apocharmm `CudaContainer`/`DeviceVector` buffers
        as DLPack `DLManagedTensor` (both directions); confirm zero-copy
        `jax.dlpack.from_dlpack` / `to_dlpack` round-trip on a toy buffer.
  - [ ] Layout+precision kernel: `float4 xyzq` (f32) → `N×3` for the model, and
        model forces → accumulate (`+=`) into the `double` force buffer.
  - [ ] Implement `MLForceManager` and subscribe it into the
        `ForceManagerComposite` (`subscribe<ForceType>(tag, stream, values,
        energyVirial)`) with its own stream; DLPack handoff + jax call live in
        `calcForce`.
  - [ ] Sync: **host barrier first** (`block_until_ready` +
        `cudaStreamSynchronize`) — correct and on-device; validate forces match
        the host-side reference bitwise-close.
  - [ ] Benchmark step throughput vs. host-side path; **only if throughput-bound**,
        move to DLPack stream-aware sync (`__dlpack__(stream=...)`) for overlap.
- [ ] Bridge free-energy managers (`FEPEIForceManager`, `MBARForceManager`,
      `EDSForceManager`, `DualTopologyForceManager`) to the existing
      `lambda_dynamics` / `lambda_mbar` / `lambda_ti` paths.
- [ ] Parity check: apocharmm vs. PyCHARMM vs. jax-md energies/forces on a
      shared PBC box.
