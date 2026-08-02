# `md-system` / `cg_jaxmd`: capabilities checklist

A stakeholder-facing summary of what the experimental unified `mmml/md/` stack
(built to merge `md-system` and `cg_jaxmd` — see the
[full design doc](md-cg-unification-design.md) and
[handoff notes](md-cg-unification-handoff.md)) can do **today**, with runnable
examples and real-structure diagrams. Status marks: ✅ done & tested, 🚧 partially
done, ⬜ planned / not started.

---

## 1. Capability checklist

### Harmonization decision and remaining gap matrix

The canonical destination is **not** to copy every `cg_jaxmd` flag into a
second `md-system` implementation. The ownership decision is:

```text
cg_jaxmd JSON ─┐
               ├─ lowering → RunConfig → MolecularSystem → HybridEnergy
md-system CLI ─┘                                      ↓
                              JaxmdDriver | RigidBodySampler
```

`mmml.md` owns system state, terms, neighbor policy, drivers, samplers, and
results. `mmml md-system` becomes the supported CLI. A cg-style JSON file is a
compatibility input format lowered into `RunConfig`, not a second runtime.
`examples/cg_jaxmd.py` remains a frozen scientific reference until parity is
demonstrated, then becomes a thin compatibility wrapper or is deprecated.

| Capability | Legacy `cg_jaxmd.py` | Shared `mmml.md` core | `md-system --jaxmd-unified` | Required harmonization action |
|---|:---:|:---:|:---:|---|
| FIRE, NVE, NVT-NHC | ✅ | ✅ | ✅ | Remove the opt-in flag after the remaining acceptance gates pass. |
| NPT | — | ✅ | ✅ | Keep only the shared `JaxmdDriver` implementation. |
| Rigid-body MC | — | ✅ | ✅ | Keep as `RunConfig.sampler`, independent of front end. |
| ML intramolecular + MM nonbonded | ✅ | ✅ | ✅ | Shared terms are canonical. |
| ML peptide–water interaction | ✅ | ✅ | ⬜ | Expose the peptide-water builder and `ml_pep_water` term through `md-system`. |
| Core–water repulsive VDW wall | ✅ | ✅ | ⬜ | Expose `vdw_core` term options through the same typed term configuration. |
| SMD bias energy | ✅ | ✅ | ⬜ | Expose `smd` term configuration; keep pulling schedules/work analysis as a protocol layer. |
| φ/ψ restraint energy | ✅ | ✅ | ⬜ | Add a topology-based dihedral-index resolver, then expose the existing `dihedral` term. |
| Separate peptide/water checkpoints | ✅ | Context supports one model today | ⬜ | Generalize `EnergyContext` to named model providers keyed by molecular role. |
| Peptide/water electrostatic embedding | ✅ | ⬜ | ⬜ | Implement a registered embedding term/provider; do not hide it inside the driver. |
| ML charge-total correction | ✅ | Charge machinery exists elsewhere | ⬜ | Reuse the canonical charge-mode/neutralization primitives and record the resolved charge policy. |
| Temperature schedule | ✅ | Driver has fixed target | ⬜ | Add a serializable schedule to `EnsembleSpec`; driver consumes it without CLI-specific logic. |
| Continuous FIRE→NVT→NVE state | ✅ | Trajectory carries positions only between wrapper phases | ⬜ | Define a restart state containing positions, velocities/momenta, box, RNG, step, and integrator state. |
| H–X bond and staged CHARMM/ML repair | ✅ | Driver has an overlap-repair hook, not feature parity | ⬜ | Define a `RecoveryPolicy` hook with event records; keep CHARMM-specific repair in an adapter. |
| DCD + ASE trajectory output | ✅ | NPZ/trajectory result exists | Partial | Add output-writer plugins consuming the shared trajectory; no driver-specific writing. |
| Detailed force/energy/neighbor diagnostics | ✅ | Per-term energy composition exists | Partial | Add observers/reporters at block boundaries; diagnostics must not live in the integration loop. |
| Packmol composition builder | Different path | ✅ | ✅ | Canonical for general liquids. |
| Peptide + water builder | ✅ | ✅ | ⬜ | Wire into `md-system`; remove trialanine-specific assumptions incrementally. |
| PSF, template-PDB, PyXtal builders | Legacy `md-system` has variants | Builders exist | ⬜ | Route all through the builder registry before making unified mode default. |
| `--continue-from` handoff | Legacy paths | ⬜ | ⬜ | Implement the shared restart-state schema first; avoid format-specific shortcuts. |
| PME/ScaFaCoS in jitted dynamics | Some separate paths | ASE face only | ⬜ | Supply a genuinely JAX-traceable LR term; never silently fall back to MIC. |

The largest missing scientific capabilities are electrostatic embedding,
role-specific checkpoints, and explicit charge correction. The largest
operational gaps are restart/state continuity, recovery policy, output writers,
and observers. Builder wiring is mostly integration work.

#### Acceptance gates for retiring the duplicate runtimes

1. `md-system` can select every registered builder without leaving the shared
   `RunConfig → assemble_and_run` path.
2. cg-style energy modes lower to the same named terms and produce energy/force
   parity on frozen frames.
3. Phase transitions preserve the complete restart state, not positions alone.
4. Electrostatic embedding, charge policy, role-specific models, restraints,
   and SMD are explicit serialized configuration.
5. Recovery and diagnostics emit structured events and do not modify physics
   silently.
6. Output trajectories round-trip positions, velocities, box, energies,
   forces, term decomposition, and provenance.
7. Cross-backend tests cover FIRE/NVE/NVT/NPT, the peptide-water mixed system,
   and at least one restart at a block boundary.
8. Production comparison runs meet documented energy-drift, temperature,
   structural, and performance tolerances.

After these gates, make the shared path the default, retain a temporary
`--legacy-jaxmd` escape hatch with a removal date, point the cg workflow at the
compatibility lowering adapter, and freeze `examples/cg_jaxmd.py` against new
features.

### Shared calculator & builder architecture

The historical split — `md-system` speaking ASE `Calculator` objects,
`cg_jaxmd` hand-composing a jax-md `energy_fn` — now has one shared stack
available to both front-ends through their opt-in routes.

- ✅ One config object (`RunConfig`) for both the `md-system` CLI and the
  `cg_jaxmd` Snakemake JSON — no more drifting configs.
- ✅ One immutable system representation (`MolecularSystem` + `FFParams`) that
  every builder produces and every energy term reads — charges, LJ tables,
  exclusions resolved once by CHARMM, never recomputed at runtime.
- ✅ Four system builders share one interface: **PSF** (from an existing
  CHARMM PSF), **Packmol** (liquid boxes), **PyXtal** (molecular crystals),
  and **peptide + water** (a trialanine/TIP3 CGENFF builder).
- ✅ All six physics terms extracted into standalone, independently
  parity-tested modules: ML intramolecular, ML peptide–water dimer, classical
  MM nonbonded, repulsive core wall, steered-MD bias, backbone dihedral
  restraint.
- ✅ One `HybridEnergy` that composes selected terms and exposes both faces
  from the same definition: an ASE `Calculator` and a jittable
  `energy_fn(R)` for jax-md.
- ✅ One driver (`JaxmdDriver`) covering minimization (FIRE), NVE, NVT
  (Nosé–Hoover), and **NPT** (Nosé–Hoover barostat) — replacing two
  independently-drifting integrator implementations.
- 🚧 Both front-ends have an opt-in shared route:
  `mmml md-system --backend jaxmd --jaxmd-unified` supports only Packmol
  composition inputs, while `examples/cg_jaxmd_unified.py` supports the
  trialanine/water configuration surface documented below. The legacy routes
  remain the defaults.
- ✅ The shared modules have focused unit and optional live-CHARMM/checkpoint
  integration tests. The latter are skipped when their local dependencies are
  unavailable; do not read a test count as a deployment certification.

### Mixed systems: one ML-scored core + explicit MM/ML solvent

The headline physics capability — a peptide scored by a neural-network
potential, embedded in explicit water that is ML-scored nearby and classically
scored further out, all evaluated through the same code path as a pure-ML or
pure-MM run.

- ✅ Build a real system (CHARMM CGENFF): one ML "core" molecule + N TIP3
  waters, lowered into one `MolecularSystem`.
- ✅ The trialanine/water front end can split energy into four terms in one
  `HybridEnergy`:
  **ML intramolecular** (core, `ml_intra`) + **ML dimer** (core↔near-water,
  `ml_pep_water`) + **repulsive wall** (core↔far-water, `vdw_core`, keeps
  waters outside the ML shell from collapsing into the core unscored) +
  **classical MM** (water↔water, `mm_nonbonded`).
- ✅ The trialanine/water front end accepts a checkpoint and the documented
  energy-mode toggles. The `mixed_calculator_sweep` workflow contains the
  NVE validation setup for this route.
- 🚧 Long-range electrostatics solvers (PME, ScaFaCoS) work for one-off ASE
  evaluation but are not yet wireable into the jitted MD loop.
- ⬜ A general "any small-molecule ML core" builder — today the peptide+water
  builder is specifically trialanine-shaped.
- ⬜ Dynamic hand-off of an individual water between "ML-scored" and
  "MM-scored" as it crosses the cutoff mid-trajectory — today the ML shell is
  fixed at build time, not re-evaluated every step.

### Sampling & ensembles

- ✅ `JaxmdDriver` implements minimization, NVE, NVT, and NPT; NPT requires a
  periodic box. Mixed-system NVT/NPT production validation remains open.
- ✅ Rigid-body Monte Carlo sampler — moves whole monomers as rigid bodies
  (translation + quaternion rotation) using the *same* energy and system, no
  code fork. Selected by one config field (`sampler="rigid"`).
- ⬜ Structural validation (RDF) that rigid sampling reproduces the real
  liquid structure of flexible MD — needs a production force-field run.

### Cluster deployment & validation

- 🚧 The repository includes SLURM-oriented workflows and an automated
  `unified_backend_sweep`, but availability of CUDA, Packmol, PyCHARMM and
  the ML checkpoint is host-specific. Validate a target cluster before using
  the experimental route for production.

### GPU-native CHARMM (apocharmm) — next major piece, design-complete

- ⬜ Not started. Design is settled (device-side ML forces via a custom
  `ForceManager`, DLPack buffer transport, host-barrier sync first) — see
  design doc §4/§10 for the committed interface. Needs a CUDA build to begin.

---

## 2. What this looks like

Trialanine + TIP3 water snapshot from `examples/atoms.pdb`, colored by the
role used in the mixed-system decomposition. The role assignment is a static
distance-based illustration, not a trajectory-derived adaptive solvent shell:

![Mixed ML/MM system, zoomed](images/structures/mixed-system-zoom.png)

The same coloring over the full solvated box (200 waters), showing how small
the ML-scored shell is relative to the classically-scored bulk — this is why
the hybrid decomposition matters for cost:

![Mixed ML/MM system, full box](images/structures/mixed-system-overview.png)

For scale, the underlying CGENFF-built peptide + water box and the isolated
peptide look like this (existing `trialanine_water_box` builder output):

![Trialanine water box](images/structures/trialanine-water-box.png)
![Trialanine peptide only](images/structures/trialanine-peptide-zoom.png)

Diagrams are generated with `scripts/generate_docs_figures.py`
(`uv run python scripts/generate_docs_figures.py`), which uses ASE's
`plot_atoms`/`Matplotlib` writer under a shared style
(`mmml/utils/ase_structure_plot.py`) — orthographic projection, covalent
bonds, Jmol or role-based coloring, consistent typography. Regenerate after
any structure/builder change; CI checks staleness with `--check`.

---

## 3. Code examples

### 3.1 Lower a trialanine/water JSON-style configuration

This pure mapping is used by `examples/cg_jaxmd_unified.py`; it is also a
convenient way to inspect the exact term selection before a live build.

```python
from mmml.md.lowering import runconfig_from_cg_config

cfg = {
    "checkpoint": "examples/sppoky-epoch-0010_params.json",
    "n_waters": 4,
    "box_size": 15.0,
    "seed": 11,
    "dt_fs": 1.0,
    "fire_total_steps": 10,
    "peptide_water_ml": True,
    "peptide_water_ml_core_vdw": True,
}
run_config = runconfig_from_cg_config(cfg, phase="fire")
assert run_config.system.builder == "peptide_water"
assert run_config.terms == ("ml_intra", "mm_nonbonded", "ml_pep_water", "vdw_core")
assert run_config.ensemble.ensemble == "min"
```

Save `cfg` as JSON, then run
`uv run python examples/cg_jaxmd_unified.py --config path/to/config.json`.
Live execution also requires a working PyCHARMM installation and checkpoint.

### 3.2 Assemble and run a pre-built system

This minimal example is the tested `assemble_and_run` contract. Supplying a
pre-built `MolecularSystem` avoids a live CHARMM build; the SMD term needs only
the two selected atom indices.

```python
import numpy as np
from mmml.md.assemble import assemble_and_run
from mmml.md.config import EnsembleSpec, RunConfig
from mmml.md.system import SystemSpec

from mmml.md.system import MolecularSystem
system = MolecularSystem(
    R=np.array([[0., 0., 0.], [2., 0., 0.]]), Z=np.array([1, 1]),
    box=None, mol_id=np.array([0, 1]),
)
config = RunConfig(
    system=SystemSpec(builder="psf"), terms=("smd",), backend="jaxmd",
    ensemble=EnsembleSpec(ensemble="nve", dt_fs=0.5, n_steps=10),
)
trajectory = assemble_and_run(
    config, system=system,
    term_kwargs={"smd": {"atom_i": 0, "atom_j": 1,
                          "k_ev_per_A2": 0.5, "target": 1.5}},
)
print(trajectory.n_frames)
```

### 3.3 Expose a term composition as an ASE calculator

```python
import numpy as np
from ase import Atoms
from mmml.md.assemble import build_hybrid_energy
from mmml.md.system import MolecularSystem

system = MolecularSystem(
    R=np.array([[0., 0., 0.], [2., 0., 0.]]), Z=np.array([1, 1]),
    box=None, mol_id=np.array([0, 1]),
)

energy = build_hybrid_energy(
    system, ("smd",),
    term_kwargs={"smd": {"atom_i": 0, "atom_j": 1,
                          "k_ev_per_A2": 0.5, "target": 1.5}},
)
atoms = Atoms(numbers=system.Z, positions=system.R)
atoms.calc = energy.as_ase_calculator()
print(atoms.get_potential_energy())
```

For rigid-body Monte Carlo, set `sampler="rigid"` on `RunConfig` (or
`mmml md-system --jaxmd-unified --sampler rigid`). Geometry preservation and
neighbor-list behavior are unit-tested; structural RDF validation against
flexible MD is still open.

With `--sampler rigid`, the default intermolecular FF is **CGenFF**
(`mm_nonbonded` from Packmol/CHARMM params, no ML checkpoint). Alternatively
`--ff zbl-mbd-multipoles` freezes fragment multipoles and per-atom C6/C8/C10
once at build (from `--multipole-checkpoint` / `--mbd-checkpoint` or bundled
defaults), then evaluates classical pair electrostatics + QDO dispersion plus
intermolecular ZBL (`cuton=0.1` Å, `cutoff=0.6` Å) during MC.

### 3.4 CLI equivalent (no Python required)

`mmml md-system --backend jaxmd --jaxmd-unified` routes through this same
shared pipeline (`runconfig_from_md_system_args` → `assemble_and_run`). Its
CLI surface today wires the **Packmol composition** builder only; `--builder
pyxtal`, `--template-pdb`, and `--continue-from` raise `NotImplementedError`
rather than silently falling back to the legacy path:

```bash
# Hybrid ML/MM flexible MD (requires --checkpoint)
uv run mmml md-system --setup pbc_nve --backend jaxmd --jaxmd-unified \
  --composition "TIP3:4" --box-size 15.0 \
  --checkpoint examples/sppoky-epoch-0010_params.json \
  --dt-fs 1.0 --ps 0.01 --seed 42

# Rigid MC with CGenFF intermolecular (no checkpoint)
uv run mmml md-system --setup pbc_nvt --backend jaxmd --jaxmd-unified \
  --sampler rigid --ff cgenff \
  --composition "TIP3:4" --box-size 15.0 --ps 0.1 --seed 42

# Rigid MC with ZBL + fixed multipoles + fixed C6 dispersion
uv run mmml md-system --setup pbc_nvt --backend jaxmd --jaxmd-unified \
  --sampler rigid --ff zbl-mbd-multipoles \
  --composition "TIP3:4" --box-size 15.0 --ps 0.1 --seed 42
```

The full mixed ML-core (peptide) + MM-shell system from §3.1 is reachable
today via the `assemble_and_run` Python API and `examples/cg_jaxmd_unified.py`
(a thin JSON-config front-end); wiring the peptide-water builder into
`md-system`'s own CLI flags is tracked as open work in the
[roadmap checklist](md-cg-unification-design.md#11-roadmap-checklist).

See [`md-system` YAML configs](md-system-configs.md) for the full flag/YAML
reference and [`run`](cli/commands/run.md) for the general CLI entry point.

---

## 4. Where to look for more detail

- [Design & decisions](md-cg-unification-design.md) — the full architecture,
  every decided/open question, and the complete roadmap checklist (§11) this
  page summarizes.
- [Handoff notes](md-cg-unification-handoff.md) — implementation-level detail
  for picking the work back up.
- [Hybrid ML/MM decomposition](hybrid-mlmm-decomposition.md) — the physics of
  the term split in more depth.
- `workflows/unified_backend_sweep/README.md` and
  `workflows/mixed_calculator_sweep/` — the validation sweeps referenced above.
