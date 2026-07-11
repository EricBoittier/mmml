# Unifying `md-system` and `cg_jaxmd`: calculator & builder schema

**Status:** Proposed (design only — no code changes yet)
**Scope:** How to split the calculators and system builders shared by
[`mmml/cli/run/md_system.py`](../mmml/cli/run/md_system.py) and
[`examples/cg_jaxmd.py`](../examples/cg_jaxmd.py) so the two can run on one
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

- **Integrator / driver backends:** ASE, jax-md, PyCHARMM.
- **Builder backends:** packmol (`packmol_placement`, `tip3_liquid_box`,
  `dcm_liquid_box`), pyxtal, `peptide_builder` / `protein_charmm_build`,
  `trialanine_water_box`, template-PDB, `setupBox` / `setupRes`.
- **Energy backends:** physnetjax ML (intramolecular monomer, peptide–water
  dimer), CHARMM MM nonbonded (`nonbonded_energy_and_forces`, jax), CHARMM
  bonded (`cgenff_bonded`), biases (SMD, flat-bottom, φ/ψ, COM restraint).
- **QC / reference backends** (eval only, out of scope for the MD loop): orca,
  pyscf, molpro.

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
- **Drivers** — one per integrator engine; consume a `HybridEnergy` and an
  `EnsembleSpec`.

---

## 6. Protocol sketches

```python
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
    ff_params: FFParams | None    # charges, LJ, exclusions, e14/vdw14
    metadata: dict

# 2. Builders — SystemSpec -> MolecularSystem
class SystemBuilder(Protocol):
    def build(self, spec: SystemSpec) -> MolecularSystem: ...
# PackmolLiquidBuilder, PyxtalCrystalBuilder, PeptideWaterBuilder, TemplatePdbBuilder

# 3. Energy terms — composable, registry-keyed, engine-agnostic factory
class EnergyTerm(Protocol):
    name: str
    def make(self, system: MolecularSystem, ctx: EnergyContext) -> TermFns: ...
# TermFns bundles a jax energy_fn AND/OR an ASE contribution.
# registry: {"ml_intra", "ml_pep_water", "mm_nonbonded", "smd",
#            "dihedral", "vdw_core", "flat_bottom", ...}

class HybridEnergy:
    def __init__(self, terms: list[EnergyTerm], system, ctx): ...
    def as_jax_energy_fn(self) -> Callable: ...      # for JaxmdDriver
    def as_ase_calculator(self) -> Calculator: ...   # sum of contributions

# 4. Drivers — one per integrator engine
class Driver(Protocol):
    def run(self, system, energy: HybridEnergy, ensemble: EnsembleSpec) -> Trajectory: ...
# AseDriver, JaxmdDriver, CharmmDriver
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

## 10. Open questions

- **`FFParams` boundary** — how much CHARMM FF state (exclusions, e14/vdw14,
  LJ tables) is captured as data in `MolecularSystem` vs. re-derived by each
  energy term? `cg_jaxmd` currently recomputes several of these inline.
- **Neighbor-list ownership** — does the `JaxmdDriver` own the jax-md
  `neighbor_list`, or does each `EnergyTerm` request capacities it needs
  (peptide–water slots vs. intermolecular pairs have different padding)?
- **Rescue path** — CHARMM repair/minimize is a cross-cutting concern that
  breaks purity. Model it as an explicit driver hook (`on_overlap`) rather than
  hiding it inside an energy term.
