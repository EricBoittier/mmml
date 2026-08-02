# Methods (draft)

*Condensed-Phase Simulations Using Hybrid ML/MM Energy Functions*  
Status: draft tied to MMML repository workflows. Numbers and cutoffs below match
current defaults in `md-system` / hybrid docs unless noted; freeze YAML when
promoting to the manuscript PDF.

---

## 1. Hybrid ML/MM potential

We describe condensed-phase molecular dynamics with a **hybrid** energy that
combines a short-range machine-learned many-body model with classical
CHARMM/CGenFF molecular mechanics. Interactions are partitioned by **monomer**
(typically one solvent residue). Let \(r_{IJ}\) be the center-of-mass (COM)
distance between monomers \(I\) and \(J\). Two-body (inter-monomer) contributions
are blended with COM-distance switch functions \(s_\mathrm{ML}(r)\) and
\(s_\mathrm{MM}(r)\) (sharpstep / complementary handoff; see
[hybrid-potential-regions](../../hybrid-potential-regions.md)):

\[
E = \sum_I E_I^\mathrm{ML}
  + \sum_{I<J} \Bigl[
        s_\mathrm{ML}(r_{IJ})\,E_{IJ}^\mathrm{ML}
      + s_\mathrm{MM}(r_{IJ})\,E_{IJ}^\mathrm{MM}
    \Bigr]
  + E^\mathrm{LR}
  + E^\mathrm{wall/restr}.
\]

- \(E_I^\mathrm{ML}\): intramolecular ML energy of monomer \(I\) (PhysNet / related
  checkpoint), or, in infrastructure tests, a **jax-mm-spoof** CGenFF bonded clone
  that replaces the ML slot without changing the hybrid driver
  (`jax_mm_spoof: true`).
- \(E_{IJ}^\mathrm{ML}\): ML dimer (two-body) energy, active only inside the ML
  COM window.
- \(E_{IJ}^\mathrm{MM}\): switched Lennard-Jones + Coulomb (and bonded terms as
  configured) evaluated in JAX and/or CHARMM, active in the MM handoff / tail.
- \(E^\mathrm{LR}\): optional long-range Coulomb beyond the real-space switch
  (Ewald, PME, or related), controlled by `lr_solver` and `mm_nonbond_mode`.

Default COM cutoffs in YAML (illustrative production defaults; confirm per job):

| Parameter | Symbol | Typical value |
|-----------|--------|---------------|
| `mm_switch_on` | end of ML→MM handoff | 8.0 Å |
| `ml_switch_width` | ML taper width | 1.0–1.5 Å |
| `mm_switch_width` | MM outer tail | 4.0–5.0 Å |

Charge and LJ treatments follow the hybrid MM options documented in
[hybrid-mm-charges](../../hybrid-mm-charges.md) and
[hybrid-mm-lj-scales](../../hybrid-mm-lj-scales.md) (fixed CGenFF charges;
optional trainable LJ scales — report which mode each Results row used).

**Implementation.** Production hybrid evaluation is assembled in
`mmml.interfaces.pycharmmInterface` (MLpot / calculator path) and driven by
`mmml md-system` with backends `jaxmd` or `pycharmm` (ASE for reference
smokes). Term bookkeeping follows [hybrid-mlmm-decomposition](../../hybrid-mlmm-decomposition.md).

---

## 2. Classical MM reference and jax-mm-spoof

CHARMM36 CGenFF bonded and switched nonbonded terms are available as a pure-JAX
clone for cross-checks without MLpot ([cgenff-jax-clone](../../cgenff-jax-clone.md)).
The **jax-mm-spoof** mode wires that bonded clone into the hybrid jax-md driver so
that condensed-phase pipelines can be exercised without a trained neural
checkpoint.

Bonded energy components (BOND, ANGL, DIHE, IMPR, UREY) for DCM and ACO monomers
were compared to native PyCHARMM `ENER` ETERM values
([jax-mm-spoof-charmm-parity](../../jax-mm-spoof-charmm-parity.md);
workflow `workflows/jaxmd_cgenff_spoof_smoke/`). Residual
\(|\Delta E|\lesssim 10^{-14}\) kcal mol⁻¹ on fixture and smoke-minimized
geometries. Selective bonded-only CHARMM `BLOCK` force isolation was not used on
MPI-linked builds (known hang); force-level bonded parity remains covered by
serial PyCHARMM tests where available.

---

## 3. Periodic boundaries and long-range electrostatics

Cubic periodic cells use minimum-image (MIC) pair loops and/or CHARMM IMAGE
lists depending on `mm_nonbond_mode`:

| Mode | Role |
|------|------|
| `jax_mic` | JAX real-space LJ + Coulomb under MIC (typical vacuum / modest boxes) |
| `periodic_external` | Short-range in CHARMM IMAGE; long-range via `lr_solver` |

Long-range solvers (`lr_solver`) include `mic` (truncate), `ewald`, `jax_pme`,
`nvalchemiops_pme`, and ScaFaCoS options
([long-range-solver-tutorial](../../long-range-solver-tutorial.md)).
Liquid methane production matrices in this work use **`lr_solver: ewald`** with
`mm_nonbond_mode: periodic_external` (`workflows/pbc_methane_ewald/`).

---

## 4. Box construction and equilibration protocol

Dense liquids follow a **two-phase** protocol ([liquid-box-workflow](../../liquid-box-workflow.md)):

1. **Phase A (MM certify).** Packmol placement at a start density, optional MC
   density moves, CHARMM SD/ABNR (and lattice / short NPT as configured) until
   inter-monomer contact criteria pass. Artifacts: `model.psf` / `model.crd`,
   `box.json`, prep journal.
2. **Phase B (hybrid MD).** Register MLpot (or spoof), minimize under the hybrid
   potential, then heat → equilibrate → produce under the target ensemble
   (`pbc_nvt` or `pbc_npt`).

`mmml md-system` YAML encodes composition (`DCM:N`, `ACO:N`, `METH:N`, …),
`box_size`, `bulk_density_fraction`, temperature, timestep, checkpoint path, and
backend. Campaign and sweep matrices are versioned under `workflows/*/config*.yaml`.

---

## 5. Simulation matrices (paper set)

Exact \(N\), \(L\), \(T\), and seeds are frozen in the manuscript workflow config
(see [workflow-map](workflow-map.md)). Intended v1 systems:

| System | Ensemble (prod) | Notes |
|--------|-----------------|-------|
| Dichloromethane (DCM) | NPT or NVT @ liquid-like ρ | Primary organic solvent |
| Acetone (ACO) | same | Second CGenFF solvent |
| Methane (METH) | NVT @ fixed liquid density | Ewald hybrid matrix; \(T\) sweep |

Optional classical water (TIP3) appears only as a backend / embedding control,
not as an unverified burst-campaign claim.

Checkpoints (PhysNet portable JSON) are named in job YAML (e.g. DES / Spooky
family labels in methane and density sweeps). Spoof legs omit neural weights.

---

## 6. Integrators, ensembles, and analysis

- **Integrators.** Velocity Verlet NVE for conservation tests; NVT/NPT with
  backend-native thermostats/barostats (jax-md NHC, CHARMM heat/equi/prod stages).
- **Timestep.** Typically 0.25–0.5 fs for hybrid organic liquids (job-specific;
  heat-scaling and NVE sweeps probe stability vs \(\Delta t\)).
- **Analysis.** Density \(\rho(t)\) and block averages from production
  trajectories; optional site–site RDFs; NVE drift metrics
  (\(\Delta E\), slope); wall-clock timings from benchmark suites.

---

## 7. Software and reproducibility

| Component | Role |
|-----------|------|
| MMML (`mmml`) | CLI (`md-system`, `liquid-box`, …), hybrid calculator, workflows |
| JAX / jax-md | Differentiable MM + ML evaluation; jaxmd backend |
| PyCHARMM / CHARMM | Topology, classical MM, optional dynamics backend |
| Snakemake (+ Slurm plugin) | Parameter sweeps and cluster submission |

Every Results figure/table cites a workflow directory, config hash, artifact
path, and git commit. Evidence status follows
[evidence-policy](../../evidence-policy.md): unverified campaign matrices are
not cited as scientific results.
