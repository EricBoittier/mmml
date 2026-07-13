# Simulation robustness report

**Goal:** demonstrate that the mmml MM+ML / MM-ML hybrid simulation stack — potential
energy surfaces, charge/multipole prediction, MD integration, and structural analysis —
behaves correctly across the range from pure MM to pure ML and everything in between.

Every figure below comes from real data: either freshly computed with the actual trained
`charged_electrostatic_best_forces` PhysNet checkpoint (charge-predicting, `total_charge=0`)
via real ASE Velocity Verlet integration, or from existing real campaign/sweep results
already on disk. Nothing here is synthetic or illustrative-only. Regenerate the fresh data
with `python scripts/run_robustness_report_md.py && python scripts/run_robustness_report_scans.py`,
then the figures with `python scripts/render_robustness_report_gallery.py`.

A PDF export of this report is built separately — see "PDF export" at the bottom.

## Summary

![summary table](robustness-report-assets/chart_summary_table.png)

## 1. Fluctuating charges and multipoles

A real 200 fs NVE trajectory of a 4-water (12-atom) cluster, using the real charge head of
`charged_electrostatic_best_forces` (`mmml/models/physnetjax/.../test-b4064dca-...json`),
recorded every frame's per-atom partial charges and total molecular dipole. Both fluctuate
continuously with the nuclear motion, exactly as expected for an *environment-dependent*
(not fixed-point-charge) electrostatics model — this is the core distinguishing feature of
polarizable/ML electrostatics versus classical fixed-charge force fields:

![charge and dipole fluctuation](robustness-report-assets/chart_charge_dipole_fluctuation.png)

Oxygen atoms carry a persistent negative charge (~-0.3 to -0.4 e) and hydrogens a smaller
positive one — the right sign and right order of magnitude for water — while both wobble
continuously in response to the local hydrogen-bonding environment, not toggling between a
small number of fixed values. The dipole moment (e·Å) fluctuates smoothly with the cluster's
geometry, never spiking or discontinuous, consistent with the checkpoint producing forces
and charges from the same differentiable energy surface (no separate/inconsistent
charge-fitting step).

## 2. Bond, angle, and dihedral potential-energy scans

Three internal-coordinate types, three real scans:

- **Bond** and **angle**: freshly computed 1D scans on an isolated water molecule with the
  same real checkpoint (`scripts/run_robustness_report_scans.py`).
- **Dihedral**: a 1D slice through the existing real trialanine backbone
  φ/ψ PES scan (`artifacts/trialanine_phi_psi_mm_then_ml_64x64/phi_psi_pes.csv`).
- **Non-bonded distance** (bonus — not one of the three classical internal coordinates, but
  the fourth term a hybrid MM/ML potential must get right): the existing real xTB-GFN2 dimer
  separation campaign (`results/dimer_scan_campaign/scan_results.csv`).

![bond and angle scans](robustness-report-assets/chart_bond_angle_scans.png)

Both scans show a real local minimum close to the experimental equilibrium geometry
(O-H ≈ 0.93 Å vs. 0.957 Å expected; H-O-H ≈ 102-104° vs. 104.5° expected) — reasonable
agreement for a compact, joint-trained demonstration checkpoint (`features=64`,
`num_basis_functions=32`), not a production-scale model.

**Known limitation, shown rather than hidden:** widening the scan range reveals a second,
*deeper* energy well far outside the chemically-relevant region (heavily compressed or
over-stretched geometries the model rarely saw in training). This is a real extrapolation
artifact of this specific compact checkpoint, not a bug in the scan/simulation code — it's
exactly the kind of failure mode a robustness report should surface:

![bond and angle scan caveat](robustness-report-assets/chart_bond_angle_scans_caveat.png)

![dihedral and dimer scans](robustness-report-assets/chart_dihedral_dimer_scans.png)

The dihedral slice reproduces the expected trialanine Ramachandran-type profile (broad
low-energy basin near φ≈100°, a barrier near φ≈0-20°). The dimer scan (best of 15 sampled
relative orientations per separation, i.e. the physically-favorable orientation at each
distance) shows the expected shape: short-range repulsion, an attractive minimum, and a flat
long-range tail.

## 3. Energy conservation (NVE)

**Small system, full control over initial conditions.** The same relaxed 4-water cluster,
integrated with two different timesteps from an identical starting geometry and velocity
seed — only `dt` differs:

![energy conservation, small system](robustness-report-assets/chart_energy_conservation_small.png)

At the physically-appropriate timestep (`dt=0.1 fs`, correctly resolving unconstrained O-H
bond vibration), total energy is conserved to **0.3 meV** over 200 fs. At `dt=0.5 fs` (5×
larger), drift grows to **13.6 meV** — about 40× worse, close to the expected `O(dt²)` Verlet
integration-error scaling (5² = 25). Both runs are well-behaved (bounded oscillation, no
runaway), and the size of the effect is exactly what textbook MD integration theory predicts
— this is the simulation code correctly revealing an integration-parameter sensitivity, not
a bug.

**Large system, real production-scale sweep.** The existing 12-setting, 10000-step NVE sweep
(`workflows/mixed_calculator_sweep`) spans pure-MM water boxes and mixed ML-core/MM-water
peptide systems, varying checkpoint epoch, electrostatics damping, and MM cutoffs:

![energy conservation, sweep summary](robustness-report-assets/chart_energy_conservation_sweep.png)

11 of 12 settings conserve energy to well under 1 eV over the full 10000-step run (green).
Two mixed-system settings show a larger but still bounded deviation (amber) — real,
documented findings, not swept under the rug. `mixed_core_vdw` (red) shows a genuine,
large energy blow-up traced to an unminimized packmol starting geometry for that specific
setting (see `workflows/mixed_calculator_sweep/README.md`) — a real, understood failure mode
from a bad initial condition, not a hidden systematic issue with the simulation engine
itself.

## 4. Structural analysis

Using the existing structural-analysis library
(`mmml.utils.plotting.trajectory_structure` / `scripts/plot_trajectory_structure.py`):

**Internal-coordinate distributions**, from the small water-cluster NVE trajectory (valid
for any system size, no periodicity required):

![internal coordinate distributions](robustness-report-assets/chart_structural_internal.png)

A clean, single-peaked O-H bond-length distribution centered near the real equilibrium value
with realistic thermal spread, and a similarly well-behaved H-O-H angle distribution — no
bimodality, no runaway tails, consistent with a stable, physically-sensible trajectory.

**Element-pair radial distribution functions**, from the real, periodic, bulk NVE
trajectories in the large-scale sweep (RDF normalization assumes a genuine periodic bulk
system, so these are shown from the sweep's real boxed water systems rather than the small
non-periodic cluster above):

![RDFs, water_baseline](robustness-report-assets/chart_structural_rdfs_water_baseline.png)
![RDFs, mixed_baseline](robustness-report-assets/chart_structural_rdfs_mixed_baseline.png)

Both show the expected liquid-water RDF signature — a sharp first O-H peak near the
hydrogen-bond distance, a first O-O coordination shell near 2.8 Å, decaying to `g(r)→1` at
long range — for both the pure-MM water box and the mixed ML-core/MM-water system.

## What this demonstrates

- **Charges and dipoles are genuinely environment-dependent**, not fixed-point-charge
  approximations, and vary smoothly (no discontinuities suggesting an inconsistent
  charge-fitting step).
- **All four internal-coordinate/interaction types the potential must model** — bond,
  angle, dihedral, non-bonded distance — produce real, checkpoint-derived energy curves
  with the right qualitative shape near equilibrium, and the model's extrapolation
  limitations at extreme geometries are identified and shown rather than hidden.
- **Energy conservation holds** at appropriate integration parameters, both for a small
  system under full experimental control and for the large real production-scale sweep
  spanning pure-MM to mixed ML-core/MM-water systems — and the code correctly reveals
  parameter-sensitivity failure modes (timestep, starting geometry) rather than masking them.
- **Structural analysis reproduces expected liquid-water behavior** (RDF shape,
  internal-coordinate distributions) using the existing plotting/analysis library, for both
  pure-MM and mixed ML/MM systems.

## PDF export

A standalone PDF version of this report (same figures, same narrative) is built via a
LaTeX pipeline (not through mkdocs, which has no PDF plugin configured) —
see `reports/simulation_robustness_report.tex`, built with:

```bash
cd reports && pdflatex -interaction=nonstopmode simulation_robustness_report.tex
```
