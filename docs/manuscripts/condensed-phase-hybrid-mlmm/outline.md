# Manuscript outline and next steps

**Title.** Condensed-Phase Simulations Using Hybrid ML/MM Energy Functions  
**Authors.** m.meuwly *et al.* (to finalize)  
**Date.** July 2026

Judging by the current repository, methods and software maturity are ahead of
finished condensed-phase *property* tables. The paper should therefore lead with
a clear hybrid energy definition and validated infrastructure, then deliver a
**small set of liquids** with regenerable density / structure / conservation
results from Snakemake, rather than an encyclopedic solvent matrix.

---

## Recommended sections

| § | Section | Repo maturity | Primary workflows / docs |
|---|---------|---------------|--------------------------|
| 1 | **Introduction** | Narrative | Motivation: ML accuracy at short range + MM efficiency / LR in liquids |
| 2 | **Hybrid ML/MM energy function** | Strong | Methods · [hybrid-potential-regions](../../hybrid-potential-regions.md), [hybrid-mlmm-decomposition](../../hybrid-mlmm-decomposition.md), charges/LJ scales docs |
| 3 | **Periodic boundaries and long-range electrostatics** | Strong | Methods · [long-range-solver-tutorial](../../long-range-solver-tutorial.md), [pbc-super-system](../../pbc-super-system.md); `pbc_methane_ewald` |
| 4 | **Software stack and backends** | Strong | Methods · jaxmd / pycharmm / ASE; [calculator-capabilities](../../calculator-capabilities.md); `dcm5_md_benchmark`, `unified_backend_sweep` |
| 5 | **Protocols for condensed-phase MD** | Strong (design) | Methods · [liquid-box-workflow](../../liquid-box-workflow.md); Phase A MM certify → Phase B hybrid |
| 6 | **Validation of the MM / spoof layer** | Strong (bonded) | Results · `jaxmd_cgenff_spoof_smoke` + [parity report](../../jax-mm-spoof-charmm-parity.md); CGenFF clone tests |
| 7 | **Energy conservation and integrator robustness** | Partial | Results · `dcm3_nve_cutoff_sweep`, `dcm_nve_scaling`, [simulation-robustness-report](../../simulation-robustness-report.md) |
| 8 | **Pure liquids: density and structure** | Workflows ready; science **incomplete** | Results · `pbc_liquid_density_dyn`, `liquid_density_sweep`, `pbc_methane_ewald`; RDF/D analysis still thin |
| 9 | **ML vs MM vs hybrid on the same box** | Partial | Results · `cg_jaxmd_ala_water_sweep` (modes); liquid three-way still needs a dedicated paper config |
| 10 | **Solute-in-solvent / peptide illustration** | Partial | Results · `mixed_calculator_sweep`, embedding / trialanine docs; keep short |
| 11 | **Performance** | Weak → needed | Results · timings from `dcm5_md_benchmark`, `dcm_heat_scaling` |
| 12 | **Discussion & conclusions** | — | Scope, limitations (evidence policy), outlook |
| A | **SI: workflow provenance** | Strong idea | Every figure cites `workflows/<name>/` + `artifacts/...` git SHA |

Sections **6–8** are the scientific core the workflows can *own*. Sections **2–5**
are already documentable from the code. Avoid claiming validated TIP3/MEOH burst
matrices (`pbc_solvent_burst` is explicitly unverified in the evidence registry).

---

## What the paper can claim *now* vs after workflow runs

### Ready to draft (support exists)

- Hybrid energy with COM-distance ML↔MM switching and monomer decomposition.
- Long-range options (`mic`, `ewald`, `jax_pme`, …) and `mm_nonbond_mode`.
- jax-md vs PyCHARMM drivers for the same Hamiltonian.
- CGenFF bonded jax-mm-spoof vs native CHARMM ETERM (DCM/ACO, machine precision).
- NVE smoothness / cutoff-preset ranking for small DCM clusters.
- Two-phase liquid-box prep design (MM certify, then hybrid).

### Requires regenerable workflow products (next runs)

| Claim | Workflow | Deliverable for Results |
|-------|----------|-------------------------|
| Liquid density ρ(T) for DCM (and ACO) under hybrid NPT/NVT | `pbc_liquid_density_dyn`, `liquid_density_sweep` | Table + ρ(t) / ⟨ρ⟩ vs experiment |
| Methane liquid hybrid MD with Ewald at fixed liquid ρ | `pbc_methane_ewald` | Energy/T stability vs backend × checkpoint × T |
| mic vs ewald (or PME) property deltas | `liquid_density_sweep` / methane matrix | Side-by-side Δρ or energy drift |
| Backend walltime / ns day⁻¹ | `dcm5_md_benchmark` | Timing table |
| Site–site RDF for liquids | extend density workflows + analysis | RDF figures vs literature |
| Self-diffusion | new analysis on prod trajectories | D vs experiment (optional for v1) |

### Defer or SI-only

- Full validation-campaign solvent encyclopedia.
- Reaction-path / umbrella (`nh3_ch3cl_reaction_path`) unless a second paper.
- Unverified `pbc_solvent_burst` science claims.

---

## Next steps (practical order)

1. **Freeze a paper solvent set.** Suggested v1: **DCM + ACO + METH** (optional TIP3 as classical reference only). Drop broad MEOH/TIP3 burst matrix from the main text.
2. **Add a paper-facing Snakemake entry** (e.g. `workflows/manuscript_hybrid_liquids/`) that:
   - calls or includes the density / methane / spoof / benchmark workflows with **frozen YAML**;
   - writes `artifacts/manuscript_hybrid_liquids/<fig_or_table_id>/` with `metrics.json` + PNG;
   - records git SHA + checkpoint hashes in `provenance.json`.
3. **Run production legs** (not smokes): `pbc_liquid_density_dyn` for DCM/ACO at ~1.0× ρ; `pbc_methane_ewald` full T × backend × one primary checkpoint; `dcm5_md_benchmark` for timings.
4. **Fill Results placeholders** in [results.md](results.md) from those artifacts; promote numbers only when `proof.json` / metrics exist (see [evidence-policy](../../evidence-policy.md)).
5. **One ML/MM/hybrid three-way** on a single DCM box (same geometry seed) — new thin config under the manuscript workflow if no existing job does it cleanly.
6. **RDF analysis script** hooked to prod trajectories (can start as a `scripts/` + Snakemake rule).
7. **Export** Methods/Results → LaTeX when numbers stabilize (`reports/` sibling to the robustness report).

---

## Suggested figure / table plan (v1)

| ID | Content | Source workflow |
|----|---------|-----------------|
| Fig. 1 | Hybrid COM switching schematic | docs figures / `scripts/plot_mlpot_settings.py` |
| Fig. 2 | Liquid-box Phase A→B flowchart | [liquid-box-workflow](../../liquid-box-workflow.md) |
| Fig. 3 | Spoof vs CHARMM bonded parity (bar or residual table) | `jaxmd_cgenff_spoof_smoke` |
| Fig. 4 | NVE energy traces / cutoff ranking (DCM) | `dcm3_nve_cutoff_sweep` |
| Fig. 5 | ρ(t) or ⟨ρ⟩ for DCM/ACO hybrid | `pbc_liquid_density_dyn` |
| Fig. 6 | Methane hybrid: T-series or backend overlay | `pbc_methane_ewald` |
| Fig. 7 | Optional RDF (C–Cl or O–O) | density prod + analysis |
| Table 1 | Hamiltonian / switch / LR defaults | Methods |
| Table 2 | Simulation matrix (N, L, T, ensemble, backend, ckpt) | manuscript workflow config |
| Table 3 | ⟨ρ⟩ vs experiment | density workflows |
| Table 4 | Timings | `dcm5_md_benchmark` |

Full mapping: [workflow-map.md](workflow-map.md).
