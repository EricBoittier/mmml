# Menshutkin campaign — roadmap

Written 2026-08-02, while the gas run and four solvated runs are still on the
machines. Numbers marked **preview** come from umbrella integration on partial
runs, not MBAR; they are labelled everywhere they appear and must be re-derived
before anything is published.

Companion documents: [README](README.md) (what the campaign is),
[HANDBOOK](HANDBOOK.md) (how to operate it), [RESULTS](RESULTS.md) (what we
found), [SUBMIT](SUBMIT.md) (command sequences).

---

## 0. Where we actually stand

Turan, Brickel & Meuwly (*J. Phys. Chem. B* **126**, 1951 (2022)) is the paper
this campaign reproduces. Their NH₃+MeCl 1D PMF barriers, against ours:

| solvent | Turan 1D | ours, control | ours, **+dq/dR** |
|---|---|---|---|
| gas | 35.8 | — | **35.09 ± 0.22** |
| water | **18.0 ± 0.5** | 26.17 | **18.48 ± 0.59** |
| acetonitrile | **20.6** | 26.24 | **19.64 ± 0.44** |
| benzene | 24.1 | — | 27.76 *(preview)* |
| methanol | 20.5 | — | *running* |
| cyclohexane | 33.9 ± 1.4 | — | *running* |

Solvent effects: **−16.6** (water) and **−15.5** (acetonitrile) against Turan's
−17.8 and −15.2. The control gave −8.4.

And the transition-state position, which is the independent check:

| | Turan | ours, control | ours, +dq/dR | ours, both |
|---|---|---|---|---|
| gas | +0.5 | — | — | — |
| water | **−0.1** | **+0.480** | **+0.104** | +0.046 |
| acetonitrile | — | +0.484 | +0.270 | +0.255 |

**Read those two tables together.** Our control puts the *water* TS at +0.48 —
essentially Turan's **gas-phase** value. The control is not modelling a solvent
effect on the geometry at all. Switching on dq/dR moves it to +0.10, covering
about three quarters of the distance to Turan's −0.1, and simultaneously drops
the barrier from 26.2 to 17.6 against their 18.0. Acetonitrile does the same
thing, 26.1 → 20.4 against their 20.6.

Two independent observables, two solvents, agreement inside 0.4 kcal/mol, from a
term that was switched off. That is why the whole roadmap below is written on
the assumption that **dq/dR is the production setting** and the previous runs
were the artefact — consistent with task #22, which records that
`mechanical-fluct` applies `stop_gradient` to the charges and thereby changes
the force by 101 % against finite differences. The "control" was the run whose
forces were not the gradient of its own energy.

One more consistency worth noting: Turan's MS-ARMD force field has **no explicit
polarisation**. Our dq/dR-only setting is the like-for-like comparison, and it is
the one that matches. Our `both` setting (dq/dR + induced polarisation) gives
16.06 / 19.10 — *below* Turan. That is not a failure; it is a different physical
model, and it should be compared to **experiment**, not to Turan.

### The 1D/2D trap

Turan report two water barriers: **18.0** (1D) and **21.7** (2D), against an
experimental **23.5** for NH₃+MeI. They regard the 1D value as too low. So:

* to validate our *implementation*, compare 1D to 1D → we match (17.6 vs 18.0);
* to compare to *experiment*, a 1D PMF along ξ is not enough for either of us.

This is what makes the 2D PMF (task #17) a headline deliverable rather than a
refinement, and it settles the open question in RESULTS §1 about which Turan
number to quote: **both, for different purposes, and say which.**

---

## 1. Immediate — finish and consolidate (this week)

| # | item | state |
|---|---|---|
| 1.1 | gas PMF, channel restraint + cutoff-14 model | **done** — 35.09 ± 0.22 |
| 1.2 | water & acetonitrile dq/dR | **done** — 18.48, 19.64 |
| 1.3 | methanol, benzene, cyclohexane dq/dR | running |
| 1.4 | MBAR on 1.3; figures | blocked on 1.3 |
| 1.5 | RESULTS / README / ROADMAP around dq/dR | **done** 2026-08-02 |
| 1.6 | add sampling to water's weak windows (N_eff min 8, in the barrier region) | not started |

The gate is passed: MBAR confirmed the previews to within 0.7–0.9 kcal/mol, in
the direction the previews are known to err. Everything below can proceed.

### What changed the campaign, recorded so it is not relearned

* **dq/dR was the entire discrepancy.** `mechanical-fluct` applies
  `stop_gradient` to the ML charges — a 101 % force error — and the launcher
  silently added `--freeze-charge-forces` to every `EMB=electrostatic` run, so
  dq/dR had never actually been tested. Turning it on moved the water barrier
  26.17 → 18.48 and the solvent effect −8.4 → −16.6.
* **The old diagnosis was wrong.** The 0.07 kcal/mol water/acetonitrile gap was
  read as an *embedding* limitation needing solvent → solute polarisation. It
  was not: the structure had always been right, but the charge response never
  entered the forces.
* **Polarisation is out of scope.** It raises the barrier and inverts the
  solvent ordering; Turan's MS-ARMD is non-polarisable, so dq/dR-only is the
  like-for-like comparison and it is the one that agrees.
* **The model has a validity boundary at r(C–Cl) ≈ 4.3 Å**, past which q(Cl)
  decays from −0.917 toward −0.795 instead of −1.00. The TS and CIP are inside;
  the SSIP region is not. This is a *dataset* limit (training scan ends at
  ξ = +4.1), not a sampling one.

---

## 2. Solvent analysis

### 2a. Reproducing Turan

Their inventory, and what each needs. Note the third column — most of this does
**not** need the ML model.

| analysis | Turan's protocol | what we need | cost |
|---|---|---|---|
| 2D solvent density maps (Fig 6, 7) | 2 ns NVT, solute **frozen** at MP2 R/TS/P geometry; 2000 snapshots; reference site = water O, MeOH O, MeCN N, CoM for apolar; slab \|z\|≤1 Å; 100×100 grid, 1.5 Å Gaussian; contours 90/75/50/25/10 % | frozen-solute MD, 3 states × 5 solvents | **cheap** — see below |
| solvent–solvent energy, first shell (Table 3) | same frozen runs; mean interaction energy per molecule within 5 Å, relative to reactant | same runs | free with the above |
| solvent–solvent energy vs ξ (Fig 8) | every umbrella window, mean + sd | denser trajectory output | needs re-run at `--traj-stride 1` |
| PMF per solvent | 50 ps/window | §3 below | expensive |

**The frozen-solute runs are much cheaper than they look, and this is the single
biggest efficiency win available.** With the solute frozen, the PhysNet energy is
a constant and the ML charges never change — so after one PhysNet call to get
q(R) at each of the three geometries, the simulation is **pure MM**. 2 ns × 3
states × 5 solvents = 30 ns of classical MD, which is hours, not weeks. Doing it
through the ML/MM driver instead would cost ~18 h *per run* for no physical
difference. Build a small dedicated script; do not reuse `07_solvated_pmf.py`.

Caveat to state in the writeup: Turan froze at **MP2/6-311++G(2d,2p)** optimised
geometries. Ours would be frozen at *our* model's stationary points, so the
density maps are "our model's TS", not theirs. That is the honest comparison
anyway, but it must be labelled.

The existing `diag/fig_solvent_dist.py` and `diag/solvent_energetics.py` already
implement the analysis; they are currently fed ~21 umbrella frames per window
instead of 2000, which is why their output is indicative only. They need better
input, not rewriting.

### 2b. What Turan did *not* do — our contribution

The paper explicitly omits RDFs, coordination numbers, dipole moments, charge
analysis, and any entropy/enthalpy decomposition. Our model supplies things
theirs structurally cannot, and this is where the work stops being a reproduction:

1. **Charge transfer along ξ.** We have q(R) from PhysNet at every frame.
   Report q(Cl), q(N), q(CH₃) versus ξ per solvent, and the *solvent-induced*
   difference q_solvated(ξ) − q_gas(ξ). MS-ARMD interpolates between fixed
   charge sets and cannot produce this.
2. **The dq/dR force itself.** We can now quantify what the charge-response
   force contributes: TS shift, barrier change, and its magnitude versus ξ. This
   is the campaign's main scientific claim and needs its own figure.
3. **Induced dipoles** from `ml_mm_pol` — per-atom induced moments versus ξ, and
   the polarisation energy split solute/solvent.
4. **Enthalpy–entropy decomposition.** We already have the per-term energy
   decomposition; combined with W this gives −TΔS by difference. The corrected
   water TS ledger (intra +31.78, elec −47.59, remainder +41.98, W 26.17) is a
   template. Redo it under dq/dR — the remainder term should shrink.
5. **RDFs and coordination numbers**, g(r) for Cl⁻–H_w and N–O_w versus ξ, with
   running integrals. Standard, cheap, and absent from Turan.
6. **First-shell residence times** from the frozen-solute runs — connects to
   the "correlated solvent motions" claim they make but do not quantify.

Items 1–4 are the ones that make this publishable as more than a repeat.

---

## 3. Full solvent series with dq/dR

Turan's five: water, methanol, acetonitrile, benzene, cyclohexane. We have boxes
for all five. Run each at the production setting once §1.3 confirms it.

* 46 windows × 10 ps ≈ **5 h per run** on one GPU (measured, ~380 s/window).
* 5 solvents × 2 settings (dq/dR, both) = 10 runs ≈ 50 GPU-h ≈ 2 days on 3 GPUs.

Cyclohexane is the important one and the most likely to misbehave: Turan get
33.9 ± 1.4, barely below the gas 35.8, and the largest error bar in their paper.
An apolar solvent cannot stabilise the ion pair, so the CIP may not be a minimum
at all. Expect poor overlap on the product side and budget for extra windows.

Literature comparison targets beyond Turan: Gao & Xia (*JACS* 1993),
Castejon & Wiberg (*JACS* 1999), and the polarisable-force-field study of
Menshutkin solvent effects (PMC2903038) — the last is directly relevant to our
`ml_mm_pol` results, since it asks the same question with different machinery.

---

## 4. Level of theory

Current base: **B3LYP/def2-SVPD**. The 10 % DLPNO-CCSD(T)/aug-cc-pVTZ (TightPNO)
subset — 1900 jobs, stratified over ξ = −10.16…+11.50 — is already generated in
`dataset_ccsdt10/` and unsubmitted.

Two distinct uses, and they should not be conflated:

* **Validation.** Single points on the existing geometries → how wrong is
  B3LYP/def2-SVPD along this coordinate? Answers "is the barrier error the
  functional or the sampling?" This needs only the 1900 jobs already prepared.
* **Δ-learning.** Train a correction ΔE = E_CCSD(T) − E_B3LYP on the subset. 10 %
  is plausible for a smooth correction, and it avoids retraining from scratch.
  ORCA 6.1 has **no analytic DLPNO-CCSD(T) gradients**, so there are no forces —
  an energy-only correction, which constrains the architecture.

Do validation first. If B3LYP/def2-SVPD is within ~1 kcal/mol along ξ, the
level of theory is not the bottleneck and Δ-learning can wait behind the second
system, which buys more.

---

## 5. Second system — NH₃ + CH₃I

**Why this one:** it converts a proxy comparison into a direct one. Turan compare
their NH₃+**MeCl** water barrier to the experimental **23.5 kcal/mol** for
NH₃+**MeI**, because NH₃+MeCl has no experimental value. Running the iodide
removes that substitution. Okamoto and co-workers measured CH₃I with ammonia and
the methylamines in aqueous solution, so there is a real kinetic series to hit.

Work required:
* training data with iodine — new scan + NMS sampling, new PhysNet training;
* basis/ECP: def2-SVPD carries an ECP for I; CCSD(T) needs aug-cc-pVTZ-**PP**;
* scalar-relativistic effects matter for I and must at least be checked;
* CGenFF has no I in the current setup — check `MMML_CGENFF_EXTRA_*`;
* everything downstream (boxes, walls, channel file) regenerates from the scan.

Risk: iodine is the first element in this campaign where the ECP/relativistic
treatment could quietly dominate the error. Validate the gas-phase barrier
against CCSD(T) *before* running any solvent.

## 6. Third system — pyridine + CH₃Br

**This is the strongest experimental anchor available**, and it should probably
be prioritised over §5.

Turan already ran it, so we get both a method comparison *and* experiment:

| | Turan | experiment |
|---|---|---|
| gas | 29.7 | — |
| water | 17.9 | — |
| methanol | 22.1 | — |
| acetonitrile | 23.2 | **22.5** |
| benzene | 22.2 | — |
| cyclohexane | 28.1 | **27.6** |

Two solvents with experimental barriers, spanning polar aprotic to apolar — the
widest span in the campaign. Castejon & Wiberg measured pyridine + MeBr at 25 °C
in cyclohexane, di-n-butyl ether and acetonitrile.

Cost is higher than NH₃+MeCl: pyridine is 11 atoms, so the solute is 17 atoms
rather than 9, and the aromatic ring needs sampling that the current NMS
protocol was not tuned for. But there is no new element (Br is in the tables) and
no ECP question.

**Recommended order: §6 before §5.** Pyridine+MeBr gives two experimental points
and reuses known elements; NH₃+MeI gives one experimental point and introduces
iodine, ECPs and relativity at the same time.

---

## 7. Documentation

* **README** — restate the campaign around the embedding ladder, and record that
  dq/dR is production. Currently describes `mechanical-fluct` as the default.
* **RESULTS** — rewrite after MBAR (§1.3). Must correct: the water/acetonitrile
  "0.07 kcal/mol gap" section, the 99.6 % compensation figure (arithmetic error;
  the correct ledger gives 74 %), and the 1D/2D comparison question. Styling pass
  to follow the content rewrite, not precede it.
* **HANDBOOK** — add the CPU-on-a-GPU-host procedure (`MENSH_DEVICE=cpu` *and*
  `MMML_MLPOT_DEVICE=cpu`; the second is not optional and the failure mode is
  silently taking a production GPU), and `MENSH_GAS_OUT`.

---

## 8. Open risks

1. **The preview barriers might not survive MBAR.** Everything here depends on
   §1.3. Umbrella integration and MBAR agreed to 0.00/0.13 on the control runs,
   which is reassuring but not proof for the dq/dR runs.
2. **Task #23 — `ml_mm_pol` uses covalent α for chloride** (2.315 Å³ rather than
   approaching 3.760), because nothing passes `ml_charges`. **Measured cost:
   0.17 kcal/mol on the barrier** — below the ±0.3–0.4 MBAR error bars. Small
   because the barrier is a TS-minus-reactant difference where Cl is nearly
   neutral at the reactant, and because Cl carries only ~6 % of E_pol; the
   hydrogens in close solvent contact dominate.

   **Decision: do not fix mid-campaign.** Run the whole `both` series with the
   same behaviour. A uniform small offset is harmless; a series where some
   solvents used corrected α and others did not would inject a systematic ~0.2
   into precisely the cross-solvent comparison the series exists to make. Label
   the results *"induced polarisation with covalent-atom polarisabilities"*.

   The fix is not one line: `HybridEnergy` broadcasts identical kwargs to every
   term, but nothing computes q(R) outside `ml_mm_elec`'s own forward pass, so
   it needs either shared charges or a second PhysNet pass (double ML cost).
   Not something to land while five jobs run against the energy assembly.

   Not a growing risk for the halide series — the absolute α correction is
   similar for Br (+1.64 Å³) and I (+1.55) as for Cl (+1.44).

   **All dq/dR runs are unaffected**; they do not include `ml_mm_pol`.
3. **The channel restraint makes our barrier a channel-restricted free energy.**
   Turan's is not restrained the same way. The comparison is close enough that
   this is evidently not dominant, but a wall-sensitivity check (task #18) is
   owed before publication.
4. **~~Gas reference~~** — done, 35.09 ± 0.22, and it is clean (N_eff min 241).
5. **Cyclohexane may not have a well-defined product state.** See §3.
6. **The charge decay past r(C–Cl) ≈ 4.3 Å bounds every CIP→SSIP statement.**
   Training data stops at ξ = +4.1; beyond it the model reverts the ion pair
   toward neutrality, so the desolvation bump is largely artifact. Extending the
   umbrella ladder would make this worse, not better — the fix is training data,
   and it should be scoped with the CCSD(T) work in §4 rather than separately.
7. **The CIP/SSIP distinction does not survive to large solvents.** Water gives
   a clean answer (transient bridge, n_bridge ≈ 0.5, no stable SSIP);
   acetonitrile and larger are ill-posed because the molecule is as long as the
   ion separation. Do not report an SSIP for them without a different criterion.
