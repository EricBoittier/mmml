# Unifying `md-system` and `cg_jaxmd`: calculator & builder schema

**Status:** Proposed (design only — no code changes yet)
**Scope:** How to split the calculators and system builders shared by
[`mmml/cli/run/md_system.py`](https://github.com/EricBoittier/mmml/blob/main/mmml/cli/run/md_system.py) and
[`examples/cg_jaxmd.py`](https://github.com/EricBoittier/mmml/blob/main/examples/cg_jaxmd.py) so the two can run on one
architecture.

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
   Requires a CUDA interop boundary (device-pointer handoff or DLPack between
   jax and apocharmm buffers). The remaining question is *how* to build that
   boundary, not *whether* — see §10.

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
#    No energy term re-derives exclusions / e14 / vdw14 / LJ tables at runtime.
@dataclass(frozen=True)
class FFParams:
    charges: np.ndarray           # (N,) partial charges
    lj_eps: np.ndarray            # (N,) or type-indexed LJ epsilon
    lj_sigma: np.ndarray          # (N,) or type-indexed LJ sigma
    lj_type_index: np.ndarray     # (N,) index into LJ tables
    exclusions: np.ndarray        # pair list of excluded (i,j)
    e14_pairs: np.ndarray         # 1-4 pair list
    e14_scale: np.ndarray         # per-pair electrostatic 1-4 scaling
    vdw14: np.ndarray             # 1-4 LJ params / scaling
    # sourced from the PSF + CHARMM params at BUILD time; immutable thereafter.

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
  Host-side pybind11 evaluation is only a validation stepping stone (§4). The
  open part is the interop mechanism, not the direction (below).
- **Rescue path → explicit driver hook.** CHARMM repair/minimize is modelled as
  a driver hook (`on_overlap`), never hidden inside an energy term, so energy
  terms stay pure and the impurity is confined to the driver.

### Still open

- **apocharmm device interop mechanism** — *how* to hand jax device buffers to
  a custom apocharmm `ForceManager`: DLPack capsule exchange vs. raw
  CUDA device-pointer handoff, who owns the force buffer, and stream/sync
  semantics between the jax XLA stream and apocharmm's CUDA stream. (Direction
  is settled; this is the implementation crux — see §11 apocharmm checklist.)
- **Rigid-body DOF representation** — quaternions vs. rotation matrices for the
  rigid moves, and whether rigid sampling is a `Sampler` peer of the MD
  `Driver` or a constraint mode inside each driver.

---

## 11. Roadmap checklist

Tracking checklist for the unification, plus the two newly-scoped capabilities
(rigid sampling, apocharmm interface). Migration order is deliberate:
land seams non-breaking, extract terms with parity checks, then add backends.

### Core unification (from §9)

- [ ] Land protocols/dataclasses (`system.py`, `energy/registry.py`,
      `config.py`) with no behavior change.
- [ ] Wrap existing builders into `builders/` (`SystemBuilder`); old call sites
      delegate.
- [ ] Extract `cg_jaxmd` energy terms into `energy/terms/` one at a time
      (`ml_intra`, `ml_pep_water`, `mm_nonbonded`, `smd`, `dihedral`,
      `vdw_core`), validating against `diagnose_energy` /
      `run_force_and_nl_diagnostics` at each step.
- [ ] Build `JaxmdDriver` from `jaxmd_runner.set_up_nhc_sim_routine`; port
      `cg_jaxmd` onto it behind a flag and compare trajectories.
- [ ] Flip `md_system --backend jaxmd` onto `JaxmdDriver`; retire the duplicate
      inline loop.
- [ ] Lower argparse CLI **and** Snakemake JSON into one `RunConfig`; make
      `md_system.py` and `examples/cg_jaxmd.py` thin front-ends.

### Rigid sampling

- [ ] Define a `Sampler` protocol (peer of `Driver`) selected by `RunConfig`;
      MD is the default sampler.
- [ ] Rigid-body state: per-monomer center-of-mass + orientation (quaternion),
      derived from `MolecularSystem.monomer_indices`.
- [ ] Rigid-move propagators: MC translation/rotation moves and/or constrained
      (SHAKE/SETTLE) rigid MD, reusing the existing `HybridEnergy`.
- [ ] Acceptance / bias hooks compatible with the existing bias terms
      (flat-bottom, COM restraint, SMD).
- [ ] Validate rigid sampling reproduces liquid structure (RDF) vs. flexible MD
      on a small box.

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
  - [ ] Prototype jax↔apocharmm buffer exchange (DLPack capsule vs. raw CUDA
        device pointer) on a toy force; settle ownership + stream/sync semantics.
  - [ ] Implement the custom `ForceManager` (subclass / composite) that reads
        device coordinates and writes the ML force contribution in place.
  - [ ] Validate device-side ML forces match the host-side reference bitwise-
        close, then benchmark step throughput vs. the host-side path.
- [ ] Bridge free-energy managers (`FEPEIForceManager`, `MBARForceManager`,
      `EDSForceManager`, `DualTopologyForceManager`) to the existing
      `lambda_dynamics` / `lambda_mbar` / `lambda_ti` paths.
- [ ] Parity check: apocharmm vs. PyCHARMM vs. jax-md energies/forces on a
      shared PBC box.
