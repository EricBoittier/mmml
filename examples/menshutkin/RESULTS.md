# Menshutkin NH₃ + CH₃Cl — results

PhysNet (B3LYP/def2-SVPD) on the 9-atom solute, CGenFF solvent, umbrella
sampling in JAX-MD, unbiased with MBAR. Benchmarked against Turan, Brickel &
Meuwly, *J. Phys. Chem. B* **126**, 1951 (2022).

Updated 2026-08-02. **preview** = umbrella integration on a run still going,
good to ~±0.7 kcal/mol. Everything else is MBAR on a complete run.

How to run any of this: **[SUBMIT.md](SUBMIT.md)**. Operating notes and traps:
**[HANDBOOK.md](HANDBOOK.md)**. What comes next: **[ROADMAP.md](ROADMAP.md)**.

---

## 1. The headline — dq/dR was switched off, and it was the whole discrepancy

![PMF](../../artifacts/menshutkin/figures/conference/fig_pmf.png)

| | ours | Turan | Δ |
|---|---|---|---|
| gas | **35.09 ± 0.22** | 35.8 | −0.7 |
| water | **18.48 ± 0.59** | 18.0 | +0.5 |
| acetonitrile | **19.64 ± 0.44** | 20.6 | −1.0 |
| benzene | 27.76 *(preview)* | 24.1 | +3.7 |
| methanol | *running* | 20.5 | |
| cyclohexane | *running* | 33.9 | |

The quantity that tests the physics, because model error largely cancels:

| solvent effect (gas → solvent) | ours | Turan |
|---|---|---|
| water | **−16.6** | −17.8 |
| acetonitrile | **−15.5** | −15.2 |

The same campaign with dq/dR **off** gave **−8.4**, and put water and
acetonitrile within **0.07 kcal/mol** of each other — no solvent model should
produce that. With dq/dR on they separate by 1.2 in the correct order.

### The control was wrong, not merely different

`mechanical-fluct` applies `stop_gradient` to the ML charges. Against central
finite differences of the same energy that changes the force by **101 %**, so
the control's forces were not the gradient of its own energy and its dynamics
did not sample exp(−βU). dq/dR is not an enhancement on top of correct physics —
it **is** the correct physics (task #22).

### The transition state moves: the independent check

Turan report the TS shifting from ξ = +0.5 (gas) to **−0.1** (water).

| water TS | Turan | control | + dq/dR |
|---|---|---|---|
| ξ | **−0.1** | **+0.48** | **+0.10** |

The control put the *water* TS essentially at Turan's **gas-phase** value — it
was modelling no geometric solvent effect at all. dq/dR covers three quarters of
the gap while dropping the barrier 26.2 → 18.5. Two observables, one term.

### Why matching Turan is the right target for dq/dR-only

MS-ARMD is **non-polarisable**, so dq/dR-only is the like-for-like comparison,
and it is the one that agrees. Adding induced polarisation moves *away* from
Turan (§4) — a different physical model, to be judged against experiment.

### The wider literature

| | barrier | solvent effect |
|---|---|---|
| **this work (dq/dR)** | gas 35.09 → water 18.48 | **−16.6** |
| Turan, MS-ARMD | 35.8 → 18.0 | −17.8 |
| QM/MM umbrella, AM1 | 50.7 → 30.6 | −20.1 |
| QM/MM, another study | → 25.8 | — |
| experiment, NH₃ + **MeI** | 23.5 | — |
| Truong, GCOSMO (BH&HLYP/MP4/MP2) | 23.5 / 24.8 / 31.4 | — |

Cautions: AM1's gas barrier is 50.7 — it overestimates Sₙ2 badly, so its total
agreeing with anything is compensating error. Truong is a continuum calculation;
our number compares to their ΔV‡, not ΔG‡. Experiment is for methyl **iodide**,
which is why §6 of the roadmap proposes running the iodide directly.

---

## 2. The gas reference

**35.09 ± 0.22** at ξ = +0.80; ΔG_rxn +31.92 at ξ = +1.50, strongly endothermic
as gas-phase Menshutkin must be.

N_eff min **241**, median 383; overlap min 0.347, median 0.451; 3600
frames/window; the channel restraint held r(C–Cl)+r(C–N) to 4.35–4.99 Å, so the
run is genuinely on the reaction path.

This replaces an earlier profile with two defects: no channel restraint (it
drifted 0.55–0.92 Å off path over ξ = +0.4…+1.0) and the wrong checkpoint
(cutoff 8, 7× worse on held-out energies and degrading most toward products —
exactly at the gas TS).

**Open:** our gas TS sits at ξ ≈ +0.70…+0.80 against Turan's +0.5. The top is
flat (35.089 at +0.70 vs 35.094 at +0.80) so the position is poorly determined,
but it is later than theirs. More likely a PES difference than sampling.

---

## 3. The ion pair, and where the model stops working

### Contact ion pair — solidly located

| | ξ | r(C–Cl) | PMF |
|---|---|---|---|
| water | +2.35 | 3.85 Å | **−19.88** |
| acetonitrile | +2.10 | 3.60 Å | **−3.42** |

Trustworthy for three reasons: it is a **free-energy feature** (the PMF minimum),
so it depends on no cutoff choice; the **structure agrees independently** — at
the water CIP n(Cl) = 5.8 against a bulk plateau of 7.0, i.e. the cation is
inside the chloride's shell displacing solvent, which is the definition; and it
sits **inside** the model's validity boundary (below).

The 16 kcal/mol difference in CIP depth between water and acetonitrile is a
strong, defensible solvent result — the protic/aprotic distinction in a quantity
we can stand behind.

Water's depth is consistent with an independent thermodynamic cycle: ΔG_gas ≈
+127 (BDE 83.7 + IE(CH₃) 227 − EA(Cl) 83.4 − methyl cation affinity of NH₃ ≈
100) plus ΔG_solv(Cl⁻) ≈ −75 and ΔG_solv(CH₃NH₃⁺) ≈ −70 gives ≈ **−13 to −18**.
The control's +12.8 was badly inconsistent with that; −19.9 is not.

### No stable SSIP is observed — and for the larger solvents the question is ill-posed

Shell radii from the **first minimum of g(r)**, computed per solvent, counting
per molecule via its nearest atom so no coordinating site has to be chosen
(`diag/solvation_shells.py`):

| solvent | first peak | shell radius | max n_bridge |
|---|---|---|---|
| water | 2.25 Å | **3.05** | **0.50** |
| methanol | 3.35 | 4.25 | — |
| acetonitrile | 3.05 | 4.35 | unusable |
| benzene | 3.25 | 5.25 | — |

Water's 3.05 Å reproduces the literature Cl⁻–H(water) shell radius with nothing
typed in, which validates the method.

**Water shows a transient bridge, not an SSIP**: n_bridge ≈ 0.5 means one water
lies between the ions in about half the frames near r(C–Cl) ≈ 4.6 Å.

**Acetonitrile cannot be answered this way**: the molecule is 4–5 Å long,
comparable to the entire ion separation, so "is one solvent molecule between the
ions" is ill-posed. Adding a between-ness projection test barely moved the
counts (3.86 → 3.28). The CIP/SSIP distinction is well defined for water and
progressively meaningless for larger solvents — a result about the system, not a
gap in the analysis.

> **Retracted.** An earlier note put a water SSIP at ξ = +3.26. That came from
> hand-set cutoffs (3.8/4.5 Å); with g(r)-derived cutoffs (3.05 Å) the bridging
> count peaks at 0.50, not 1.17. No SSIP is claimed or plotted.

### The charges decay — this bounds everything past the CIP

| r(C–Cl) | q(Cl) |
|---|---|
| 3.60 Å | −0.891 |
| **4.28 Å** | **−0.917** ← maximum |
| 5.80 Å | −0.850 |
| 7.16 Å | −0.795 |

A separated chloride must approach **−1.00**. Instead q peaks at 4.28 Å and runs
backwards, losing 13 % by 7.16 Å. Total charge stays ~0 — the charge flows back
onto the methylammonium. The model un-ionizes the pair as it separates.

Solvation scales as q². A Born estimate for Cl⁻ (r_ion 1.9 Å, ε 37.5) gives −71
kcal/mol at −0.917 and −54 at −0.795: a loss of ~18 against the ~20.5 rise
observed in acetonitrile. **The "desolvation bump" is largely the model
discharging the ions, not solvent structure.**

Cause: the training scan covers ξ = −2.2…+4.1; at ξ = +5.6 we are 1.5 Å outside
it. **Validity boundary: r(C–Cl) ≲ 4.3 Å (ξ ≲ +2.8) — 34 of 46 windows.** The TS
(r ≈ 2.4 Å) and the CIP (3.85 Å) are both well inside. Contaminated: the climb
past the CIP, which is where SSIP energetics would live.

---

## 4. Induced polarisation — measured, then scoped out

| | control | + polarisation | Δ |
|---|---|---|---|
| water barrier | 26.17 | 27.71 | **+1.54** |
| acetonitrile barrier | 26.24 | 26.34 | +0.09 |

Polarisation *raises* the barrier and inverts the solvent ordering — away from
Turan and from experiment. Since dq/dR alone reproduces the reference,
polarisation is out of scope; its runs were stopped (see `README_PARTIAL.md` in
each `*_both` directory).

Known defect (task #23): `ml_mm_pol` never receives `ml_charges`, so chloride
uses the covalent α (2.315 Å³) instead of approaching the anionic 3.760.
**Measured cost: 0.17 kcal/mol** — below the MBAR error bars, because the
barrier is a TS-minus-reactant difference where Cl is nearly neutral at the
reactant, and because Cl carries only ~6 % of E_pol.

---

## 5. Enthalpy–entropy compensation (control runs)

At the water TS, against acetonitrile:

| | water | acetonitrile | Δ |
|---|---|---|---|
| ⟨E_intra⟩ | +31.78 | +25.85 | +5.93 |
| ⟨E_ml_mm_elec⟩ | −47.59 | −24.11 | **−23.48** |
| reorganisation + entropy | +41.98 | +24.50 | **+17.48** |
| **W** | 26.17 | 26.24 | **−0.07** |

Water gains 23.5 of electrostatic stabilisation and gives back 17.5 — **74 %
compensation**.

> **Corrected.** Previously reported as 99.6 %. That subtracted ⟨E_elec⟩ from W
> without removing ⟨E_intra⟩, double-counting the intramolecular barrier. 74 %
> is not implausible — aqueous reorganisation energies run 20–40 kcal/mol.

To be redone under dq/dR, where the residual term should shrink.

---

## 6. Caveats for any slide

1. **Water's statistics are weak** — N_eff median 42, min 8, and the min-8 window
   is in the *barrier* region. ±0.59 is probably optimistic.
2. **The barrier is a channel-restricted free energy.** Turan's is not
   restrained the same way. A wall-sensitivity check is owed (task #18).
3. **No standard-state correction** (task #24) — 1–3 kcal/mol, affects claims of
   agreement with *experiment*, not the Turan comparison, and largely cancels in
   solvent-to-solvent differences.
4. **1D vs 2D.** Turan report water at 18.0 (1D) and 21.7 (2D) against
   experiment 23.5, considering their 1D too low. Our 1D reproduces their 1D —
   validating the implementation — but no 1D PMF along ξ compares to experiment
   (task #17).
5. **Benzene is 3.7 above Turan**, the only solvent outside 1.0, and still a
   preview. Watch whether cyclohexane is also high.

---

## 7. Settings, and what they cost to establish

| setting | why |
|---|---|
| `--channel-k 50 --channel-tol 0.60 --channel-cn-tol 0.80` | ξ fixes only the *difference* of two distances; without a channel the methyl leaves both partners while reporting a perfect reaction coordinate. Tolerances from Truong *et al.* (water elongates C–N by 0.42 Å at the TS), not from our own clamped runs, which would be circular |
| `--angle-min-deg 130` | with fluctuating charges the chloride solvates side-on; a run without this sampled N–C–Cl at 87–93° across the reactant side while bond lengths looked healthy |
| `--dt-fs 1.0` | NVE drift +0.018 meV/atom/ps against −0.006 at 0.25 fs |
| `--prod-ps 10` | 1 ps gave zero overlap between adjacent windows |
| `--lr-solver ewald` | the reaction creates a ±1 ion pair; truncated Coulomb under-stabilises it |
| k = 150 kcal/mol/Å², ξ = −1.3…+5.6, 46 windows | Turan's k and range, extended through the CIP |

Boxes follow Turan: water 30 Å (900 TIP3P), methanol 25 (233), acetonitrile 28
(253), benzene 27 (133), cyclohexane 30 (150).

---

## 8. Figures

```bash
python artifacts/menshutkin/diag/fig_progress.py
```

| file | contents |
|---|---|
| `fig_pmf.png` | the result — gas + dq/dR solvents, Turan bars at each TS, CIP marked |
| `fig_pmf_controls.png` | plus the superseded no-dq/dR controls |
| `fig_pmf_midterm.png` | plus runs still going, labelled *n*/46 |
