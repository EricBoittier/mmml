# Workflow map (figures & tables → Snakemake)

Goal: every manuscript numerical claim regenerates from a workflow under
`workflows/`, writing versioned artifacts (metrics + plots + provenance).

---

## Paper entrypoint (recommended next code step)

Add `workflows/manuscript_hybrid_liquids/` that **includes or shells out** to the
rows below with frozen `config.paper.yaml` (no ad-hoc CLI flags). Suggested rules:

| Rule | Writes | Depends on |
|------|--------|------------|
| `parity_spoof` | Fig.3 metrics | `jaxmd_cgenff_spoof_smoke` compare |
| `nve_cutoffs` | Fig.4 | `dcm3_nve_cutoff_sweep` (paper subset) |
| `density_dcm_aco` | Fig.5, Table 3 | `pbc_liquid_density_dyn` |
| `methane_ewald` | Fig.6 | `pbc_methane_ewald` |
| `benchmark_timings` | Table 4 | `dcm5_md_benchmark` |
| `collect_manuscript` | `artifacts/manuscript_hybrid_liquids/index.json` | all of the above |

Until that meta-workflow exists, run the leaf workflows directly and paste
paths into Results.

---

## Figure ↔ workflow

| Fig. | Caption (draft) | Workflow | Config / notes | Artifact pattern |
|------|-----------------|----------|----------------|------------------|
| 1 | COM ML/MM switching | (docs/scripts) | `scripts/plot_mlpot_settings.py` | `docs/images/mlpot-settings/` |
| 2 | Phase A→B liquid box | docs | [liquid-box-workflow](../../liquid-box-workflow.md) | schematic in docs |
| 3 | Spoof vs CHARMM bonded | `jaxmd_cgenff_spoof_smoke` | `submit_compare_slurm.sh compare` | `artifacts/jaxmd_cgenff_spoof_smoke/charmm_compare/compare_report.json` |
| 4 | DCM NVE cutoff ranking | `dcm3_nve_cutoff_sweep` | paper subset of presets | `artifacts/dcm3_nve_cutoff_sweep/**` |
| 5 | DCM/ACO density | `pbc_liquid_density_dyn` | freeze \(T=300\), ρ≈1.0× | `artifacts/pbc_liquid_density_dyn/**` |
| 6 | Methane hybrid + Ewald | `pbc_methane_ewald` | `lr_solver: ewald`; T×backend×ckpt | `artifacts/pbc_methane_ewald*/**` |
| 7 | RDF (optional) | density prod + **new** analysis rule | site–site pairs | `artifacts/manuscript_hybrid_liquids/rdf/**` |

---

## Table ↔ workflow

| Table | Content | Workflow | Notes |
|-------|---------|----------|-------|
| 1 | Hamiltonian / switches / LR defaults | Methods (static) | Cite `md-system` defaults |
| 2 | Simulation matrix | manuscript config | One YAML → one table |
| 3 | \(\langle\rho\rangle\) vs experiment | `pbc_liquid_density_dyn`, `liquid_density_sweep` | Only PASS receipts |
| 4 | Timings | `dcm5_md_benchmark` | Same hardware footnote |
| S1 | Spoof ETERM residuals | `jaxmd_cgenff_spoof_smoke` | Full per-term JSON |
| S2 | Solver scan | `liquid_density_sweep` | mic vs ewald vs PME |

---

## Leaf workflow cheat sheet

| Workflow | One-line role for the paper |
|----------|----------------------------|
| `jaxmd_cgenff_spoof_smoke` | CHARMM bonded parity + driver smoke (DCM/ACO) |
| `dcm3_nve_cutoff_sweep` | Cutoff / NVE smoothness |
| `dcm_nve_scaling` | Size / INBFRQ scaling (SI) |
| `dcm5_md_benchmark` | Backend timing smoke → Table 4 seed |
| `dcm_heat_scaling` | Heat stability vs \(N\) (SI) |
| `dcm_density_setup_compare` | Prep survival at sub-bulk ρ (methods/SI) |
| `pbc_liquid_density_dyn` | Persistent liquid density MD (DCM default) |
| `liquid_density_sweep` | ρ × backend × `lr_solver` matrix |
| `pbc_methane_ewald` | CH₄ hybrid + native Ewald |
| `pbc_solvent_burst` | **Do not cite** as validated science (unverified) |
| `cg_jaxmd_ala_water_sweep` | mm/ml/hybrid modes (peptide+water) |
| `mixed_calculator_sweep` | Short mixed / water NVE illustration |
| `unified_backend_sweep` | Backend smoke (SI) |
| `validation_campaign` | Proof harness / gates — meta, not a figure source |
| `des_dimer_pair_scans` | Gas-phase DES panel (optional SI, not condensed-phase core) |
| `nh3_ch3cl_reaction_path` | Reactive path — out of scope for v1 liquids paper |

---

## Provenance checklist (per artifact)

Each directory that feeds a figure/table should contain:

1. `request.json` or `job.yaml` (frozen inputs)
2. `metrics.json` (numbers that enter the paper)
3. plot PNG/PDF (repo plot style)
4. `provenance.json` — git SHA, checkpoint hash, hostname/partition
5. optional `proof.json` if under `validation_campaign` rules

Do not copy numbers from interactive notebooks without this bundle.
