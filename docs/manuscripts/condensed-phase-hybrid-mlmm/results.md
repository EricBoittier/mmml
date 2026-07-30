# Results (draft)

*Condensed-Phase Simulations Using Hybrid ML/MM Energy Functions*  
Status: structure + completed infrastructure results; **liquid property cells are
placeholders** until manuscript workflows write metrics under
`artifacts/manuscript_hybrid_liquids/` (or the cited workflow artifact roots).

Convention: **[DONE]** = numbers already in-repo / documented; **[RUN]** = requires
workflow execution; **[TODO]** = needs a new rule or analysis script.

---

## 1. Infrastructure: bonded jax-mm-spoof matches native CHARMM  **[DONE]**

The hybrid jax-md path can run with PhysNet replaced by a JAX CGenFF bonded
clone (`jax_mm_spoof`). For DCM and ACO monomers, bonded ETERM components match
PyCHARMM to machine precision
([parity report](../../jax-mm-spoof-charmm-parity.md);
`workflows/jaxmd_cgenff_spoof_smoke/`).

| Case | \(E_\mathrm{jax}\) | \(E_\mathrm{CHARMM}\) | \(\Delta E\) (kcal mol⁻¹) |
|------|-------------------:|----------------------:|--------------------------:|
| DCM fixture | 311.3070436736117 | 311.30704367361164 | \(+5.7\times10^{-14}\) |
| DCM smoke-min monomer | 1.6671786898619212 | 1.667178689861924 | \(-2.9\times10^{-15}\) |
| ACO fixture | 7.779347176527917 | 7.779347176527915 | \(+1.8\times10^{-15}\) |
| ACO smoke-min monomer | 1.9927947535063575 | 1.9927947535063568 | \(+6.7\times10^{-16}\) |

Short vacuum/PBC NVE smokes (`DCM:4`, `ACO:4`, 0.05 ps) complete under jaxmd+spoof
(artifacts under `artifacts/jaxmd_cgenff_spoof_smoke/`). These establish that the
**driver and bonded MM layer** are consistent with CHARMM before hybrid liquid
claims.

**Figure suggestion.** Residual bar chart or table reprint (Fig. 3 in outline).

---

## 2. NVE robustness and cutoff presets  **[DONE partial] / [RUN]**

Cluster NVE studies rank ML/MM COM cutoff presets by energy smoothness
(`workflows/dcm3_nve_cutoff_sweep/`) and probe system-size / neighbor-update
scaling (`workflows/dcm_nve_scaling/`). Cross-backend short MD smokes
(`workflows/dcm5_md_benchmark/`) exercise ASE, jax-md, and PyCHARMM in vacuum and
PBC.

| Result | Status | Action |
|--------|--------|--------|
| Qualitative NVE traces / preset ranking | Partial (robustness report + sweep artifacts) | **[RUN]** freeze paper preset; export one figure |
| ΔE drift table (hybrid liquid, bulk ρ) | Missing | **[RUN]** NVE leg on certified DCM box |
| Backend timing table | Missing as paper table | **[RUN]** `dcm5_md_benchmark` → Table 4 |

Narrative may also cite the in-repo
[simulation robustness report](../../simulation-robustness-report.md) for charge
fluctuation and water-box RDF *illustrations*, without over-claiming organic
liquid agreement with experiment.

---

## 3. Pure liquids: density  **[RUN]**

### 3.1 Dichloromethane and acetone

**Protocol.** Phase A MM box certify → Phase B hybrid equilibration/production
(`pbc_npt` or `pbc_nvt`) via `workflows/pbc_liquid_density_dyn/` and/or
`workflows/liquid_density_sweep/`.

**Placeholder table (fill from `metrics.json`).**

| Solvent | \(N\) | \(L\) (Å) | \(T\) (K) | Ensemble | Backend | Checkpoint | \(\langle\rho\rangle\) (g cm⁻³) | Expt. | Status |
|---------|------:|----------:|----------:|----------|---------|------------|--------------------------------:|------:|--------|
| DCM | — | — | 300 | NPT | jaxmd | — | — | ~1.33 | **[RUN]** |
| DCM | — | — | 300 | NPT | pycharmm | — | — | ~1.33 | **[RUN]** |
| ACO | — | — | 300 | NPT | jaxmd | — | — | ~0.78 | **[RUN]** |

Existing artifact trees under `artifacts/pbc_liquid_density_dyn/` (primary clone)
may already contain candidate runs; promote to the paper only after provenance
and acceptance metrics are attached.

**Figure.** \(\rho(t)\) for production windows; optional density vs
`bulk_density_fraction` from the sweep.

### 3.2 Liquid methane with Ewald  **[RUN]**

`workflows/pbc_methane_ewald/` runs hybrid methane at fixed liquid-like density
with `lr_solver: ewald`, backends `pycharmm` and `jaxmd`, temperatures
\(\{100,200,300\}\) K, and multiple checkpoints (smoke config uses a subset).

| \(T\) (K) | Backend | Checkpoint | Status | Notes |
|----------:|---------|------------|--------|-------|
| 100 | jaxmd / pycharmm | primary | **[RUN]** | below / near melting — interpret carefully |
| 200 | … | … | **[RUN]** | |
| 300 | … | … | **[RUN]** | supercritical relative to CH₄ \(T_\mathrm{c}\) — density is imposed |

Report energy and temperature stability; density is an input for NVT liquid
methane, not an NPT observable.

---

## 4. Long-range solver comparison  **[RUN]**

Same geometry / composition with `lr_solver ∈ {mic, ewald, jax_pme, …}`
(`liquid_density_sweep` axis). Report:

- energy or force disagreement at a shared frame;
- \(\langle\rho\rangle\) or drift differences where NPT is used;
- cost (step s⁻¹) vs solver.

Until those jobs finish, state only that the software exposes the solvers
(Methods §3) without quoting unverified property deltas.

---

## 5. ML vs MM vs hybrid on one box  **[TODO] / [RUN]**

A controlled three-way comparison (pure MM, pure ML if stable, hybrid) on an
identical DCM (or ALA+water) configuration is the cleanest “Hamiltonian”
result. Partial precedent: `workflows/cg_jaxmd_ala_water_sweep/`
(`energy_modes: mm_mm, ml_mm, ml_ml`).

**Paper need.** A frozen manuscript job that writes side-by-side \(\langle\rho\rangle\),
RDF, and energy components for **one** solvent box. Status: **[TODO]** config +
**[RUN]**.

---

## 6. Solute / peptide-in-water illustration  **[DONE partial]**

Short mixed ML/MM water and trialanine+water NVE smokes
(`workflows/mixed_calculator_sweep/`) and embedding design docs show that
peptide regions can use ML while solvent remains MM. Keep this as a **brief
illustration** (one figure or SI), not a conformational free-energy claim,
unless umbrella / longer NPT legs are added deliberately.

---

## 7. Performance  **[RUN]**

| Backend | System | \(N_\mathrm{atoms}\) | Ensemble | wall time / ps | ns day⁻¹ | Status |
|---------|--------|---------------------:|----------|---------------:|---------:|--------|
| jaxmd | DCM:5 | — | NVE/NVT | — | — | **[RUN]** `dcm5_md_benchmark` |
| pycharmm | DCM:5 | — | … | — | — | **[RUN]** |
| jaxmd | DCM liquid | — | NPT | — | — | **[RUN]** production node |

---

## 8. Smoke status (engineering)

Tiny GPU smokes used while wiring manuscript workflows (not paper numbers):

| Workflow | Config / tags | Last result | Notes |
|----------|---------------|-------------|-------|
| `pbc_liquid_density_dyn` | `config.smoke.tiny.gpu.yaml` → `dcm_8` | **PASS** init+equi+prod (0.5 ps NPT); Slurm/launcher exit 0 | Handoff fmax-gate skip; density `run_job` uses direct Python + CLI mpirun re-exec; success path no longer `os._exit(0)` (lets MPI finalize so PRRTE returns 0) |
| `pbc_methane_ewald` | `meth_8_t100_l24_des_jaxmd` | **PASS** init heat + jaxmd equi/prod (0.25 ps NVT, Ewald); launcher exit 0 | Seed 89921: mini → fmax≈0.96 eV/Å, USER≈−61 kcal/mol before heat |
| `pbc_methane_ewald` | `meth_8_t100_l24_des_pycharmm` | **FAIL** pre-heat / HEAT | Root cause: DES≠methane; seed 89904 mini spike-aborts at fmax≈20.7 eV/Å, USER≈+2826; smoke had raised `max_fmax_before_dyn_ev_A=25` so heat started and CHARMM aborted (`ENERGY CHANGE TOLERANCE`, Inf-scale ΔE) by step ~85. Not an echeck wiring bug (`no_echeck_heat` / echeck=1e30 were on). Removed the 25 eV/Å ceiling so this fails at the pre-dyn gate instead. Prefer jaxmd for DES/spoof methane smoke. |
| `dcm5_md_benchmark` | needs real DCM PhysNet `MMML_CKPT` | not smoked here | Only DES / spooky JSON in `examples/ckpts_json/` |

Artifact roots: `artifacts/pbc_liquid_density_dyn_smoke_tiny/`,
`artifacts/pbc_methane_ewald_smoke_tiny/`.

---

## 9. Summary of Results status

| Block | Ready for PDF? |
|-------|----------------|
| Spoof ↔ CHARMM bonded | Yes |
| Hybrid method description | Yes (Methods) |
| NVE / cutoff engineering | Nearly (pick one frozen figure) |
| Liquid ρ vs experiment | Not until **[RUN]** + review |
| Methane Ewald matrix | Not until **[RUN]** |
| RDF / diffusion | Not until analysis hooks |
| Timings | Not until benchmark export |

Regenerate and insert numbers via the mapping in [workflow-map.md](workflow-map.md).
