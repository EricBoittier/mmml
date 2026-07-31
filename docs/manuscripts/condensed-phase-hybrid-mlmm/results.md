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
| `pbc_methane_ewald` | `meth_8_t100_l24_des_jaxmd` (spoof tiny) | **PASS** init heat + jaxmd equi/prod (0.25 ps NVT, Ewald); launcher exit 0 | Seed 89921: mini → fmax≈0.96 eV/Å, USER≈−61 kcal/mol before heat |
| `pbc_methane_ewald` | `meth_8_t100_l24_des_pycharmm` (spoof tiny) | **FAIL** pre-heat / HEAT | Root cause: DES≠methane under `jax_mm_spoof`; seed 89904 mini spike-aborts at fmax≈20.7 eV/Å, USER≈+2826; smoke had raised `max_fmax_before_dyn_ev_A=25` so heat started and CHARMM aborted (`ENERGY CHANGE TOLERANCE`) by step ~85. Removed the 25 eV/Å ceiling. |
| `pbc_methane_ewald` | `config.smoke.embeddings.tiny.yaml` (real ML, no spoof) | **PASS** 6/6 cells; Slurm exit 0 | DES + So3LR13; mechanical (`mm_charge_mode=fixed`) + electrostatic (`q0`). DES+q0 skipped (no charge head). See table below. |
| `dcm5_md_benchmark` | needs real DCM PhysNet `MMML_CKPT` | not smoked here | Only DES / spooky JSON in `examples/ckpts_json/` |

Embedding smoke tags (`artifacts/pbc_methane_ewald_smoke_embeddings/`, jobs 205234–205240):

| Tag | Checkpoint | Embedding | Backend | Result |
|-----|------------|-----------|---------|--------|
| `meth_8_t100_l24_des_mech_pycharmm` | DESdimers | mechanical / fixed | pycharmm | **PASS** (retry 205240 exit 0 after 205239 exit 1) |
| `meth_8_t100_l24_des_mech_jaxmd` | DESdimers | mechanical / fixed | jaxmd | **PASS** |
| `meth_8_t100_l24_so3lr13_mech_pycharmm` | So3LR epoch0013 | mechanical / fixed | pycharmm | **PASS** |
| `meth_8_t100_l24_so3lr13_mech_jaxmd` | So3LR epoch0013 | mechanical / fixed | jaxmd | **PASS** |
| `meth_8_t100_l24_so3lr13_es_pycharmm` | So3LR epoch0013 | electrostatic / q0 | pycharmm | **PASS** |
| `meth_8_t100_l24_so3lr13_es_jaxmd` | So3LR epoch0013 | electrostatic / q0 | jaxmd | **PASS** |

Artifact roots: `artifacts/pbc_liquid_density_dyn_smoke_tiny/`,
`artifacts/pbc_methane_ewald_smoke_tiny/`,
`artifacts/pbc_methane_ewald_smoke_embeddings/`.

---

## 9. Summary of Results status

| Block | Ready for PDF? |
|-------|----------------|
| Spoof ↔ CHARMM bonded | **Yes** (in LaTeX Table) |
| Hybrid method description | Yes (Methods) |
| NVE illustrations | **Yes** (ethanol + mixed_calculator figures in `latex/figures/`) |
| Rigid dimer scans | **Yes** (DCM/ACE/TIP3 wells + PNGs from `dimer_scan_campaign`) |
| Gas-phase umbrella/MBAR | **Yes** (barrier \(12.99\pm1.16\) kcal/mol @ ξ≈2.34 Å) |
| ADUMB WHAM PMF | Status only (exit 2; coverage figure) |
| Liquid ρ vs experiment | Not until **[RUN]** metrics (`density_g_cm3_*=null`) |
| Methane Ewald embeddings | Smoke **PASS** matrix in LaTeX (not property claims) |
| RDF / diffusion | Not until analysis hooks |
| Timings | Not until benchmark export |

Numeric extract: [latex/figures/extracted_metrics.json](latex/figures/extracted_metrics.json).  
Regenerate mapping: [workflow-map.md](workflow-map.md).
