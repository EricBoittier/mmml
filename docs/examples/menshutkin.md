# Menshutkin reaction: reactive ML/MM free-energy campaign

!!! note "Source"
    This page mirrors
    [`examples/menshutkin/README.md`](https://github.com/EricBoittier/mmml/blob/main/examples/menshutkin/README.md).
    That README is the hub of a five-document set; its siblings —
    [`RESULTS.md`](https://github.com/EricBoittier/mmml/blob/main/examples/menshutkin/RESULTS.md),
    [`SUBMIT.md`](https://github.com/EricBoittier/mmml/blob/main/examples/menshutkin/SUBMIT.md),
    [`HANDBOOK.md`](https://github.com/EricBoittier/mmml/blob/main/examples/menshutkin/HANDBOOK.md)
    and
    [`ROADMAP.md`](https://github.com/EricBoittier/mmml/blob/main/examples/menshutkin/ROADMAP.md)
    — live next to it in the repository and are not mirrored here.
    Paths below are relative to the repository root, and host names are the
    machines this campaign was run on — substitute your own.

NH₃ + CH₃Cl → CH₃NH₃⁺ ··· Cl⁻ in the gas phase and in solution, with a PhysNet
potential on the 9-atom reacting solute and CGenFF for the solvent, sampled by
umbrella sampling in JAX-MD and unbiased with MBAR.

**Goal:** the gas-phase PMF, the same PMF in Turan's five solvents, and the
figures comparing them.

Reference: Turan, Brickel & Meuwly, *J. Phys. Chem. B* **126**, 1951 (2022) —
`jp1c09710.pdf` in the example directory. Also `truong1997.pdf` (GCOSMO
continuum), which supplies experimental anchors.

## Where to find what

| document | contents |
|---|---|
| [`RESULTS.md`](https://github.com/EricBoittier/mmml/blob/main/examples/menshutkin/RESULTS.md) | **the findings** — barriers, solvent effects, ion-pair states, limits |
| [`SUBMIT.md`](https://github.com/EricBoittier/mmml/blob/main/examples/menshutkin/SUBMIT.md) | **how to run each stage** — prerequisites, one command, how to verify |
| [`HANDBOOK.md`](https://github.com/EricBoittier/mmml/blob/main/examples/menshutkin/HANDBOOK.md) | operating notes: machines, traps, diagnostics, what is settled |
| [`ROADMAP.md`](https://github.com/EricBoittier/mmml/blob/main/examples/menshutkin/ROADMAP.md) | what comes next and in what order |
| this page | what the campaign is, and the physics you must not break |

Two further pages on this site rather than in the repository:

- [Campaign record to 2026-08-02 and the general recipe](menshutkin-campaign-record.md)
  — the page this one replaced, kept in full. It holds the system-agnostic
  recipe for reproducing this workflow on another reaction, the long
  bibliography, the box-building and training-manifold diagnostics, and the
  record of every bug found on the way. Its *conclusions* are superseded by
  this page; its *methods* are not.
- [Batched umbrella sampling](../umbrella.md) — the sampling machinery
  underneath: window schedules, walls, the hybrid engine, and the knobs named
  below.

---

## Status — 2026-08-02

**dq/dR (the charge-response force) is the production setting.** It was
switched off, and switching it on reproduces the reference.

| | ours | Turan | |
|---|---|---|---|
| gas | **35.09 ± 0.22** | 35.8 | ✓ |
| water | **18.48 ± 0.59** | 18.0 | ✓ |
| acetonitrile | **19.64 ± 0.44** | 20.6 | ✓ |
| benzene | 27.76 *(preview)* | 24.1 | watch |
| methanol, cyclohexane | *running* | 20.5, 33.9 | |

**Solvent effect: −16.6 (water), −15.5 (acetonitrile)** against Turan's −17.8
and −15.2. The earlier campaign gave −8.4 and put water and acetonitrile
0.07 kcal/mol apart; both were artifacts of running with `stop_gradient` on the
ML charges, which changes the force by 101 % against finite differences.

**Bounds of validity:** r(C–Cl) ≲ 4.3 Å (ξ ≲ +2.8). Past that the model's
charges decay — q(Cl) peaks at −0.917 and falls to −0.795 by 7.2 Å — so the
apparent desolvation bump is largely the ions discharging. The transition state
(2.4 Å) and the contact ion pair (3.85 Å) are both well inside.

All five solvent boxes build at Turan's box sizes and correct density.

---

## The one thing to understand before changing anything

**ξ = r(C–Cl) − r(C–N) fixes the difference of two distances and leaves their
sum free.** On a fitted potential the dissociated branch is downhill, so the
methyl walks away from *both* partners while reporting a perfect reaction
coordinate. Measured, with no bound: r(C–Cl) 3.2–5.3 Å **and** r(C–N) 3.1–3.7 Å
at the same time. The run did not crash, the histograms were smooth, and MBAR
returned a clean profile with respectable overlap. The "barrier" was the cost of
tearing the methyl off.

A *constant* bound is not enough either — it is a box, and the path is a line
inside it. Two runs died mid-flight while sitting inside a `min(r) ≤ 2.25 Å`
wall the whole time, 0.9 Å off the path in the sum.

So three restraints, and each one earns its place:

| restraint | why | range |
|---|---|---|
| `min(r) ≤ 2.25 Å` | at least one bond always formed | everywhere (training max 2.18 in-range, 1.57 beyond) |
| channel on the **sum** and on **r(C–N)** | follows the reference path as a function of the *current* ξ | everywhere |
| `angle(N–C–Cl) ≥ 130°` | without it the fluctuating-charge run sampled 87–93°, i.e. side-on attack, with healthy bond lengths and nothing else flagging it | **everywhere** |

### The angle wall stays ON everywhere — including past the ion pair

Earlier revisions of this page said the opposite: that it "must come off past the
ion pair" because the training angle falls to 85–88° once the chloride is far.
**That was read off the wrong file.** The two training sources disagree, and both
are correct about different things:

| | ξ +1.5…+2.0 | ξ +4.0…+4.5 |
|---|---|---|
| `scan_nh3_ch3cl.npz` — relaxed scan, **the reaction path** | 174.7° | **177.0°** (min 175.3) |
| `nh3_ch3cl_filtered.npz` — NMS + **orientational** sampling | 90.0° | 83.1° (min 3.8) |

The second file deliberately walks the chloride around the methylammonium at
fixed separation. Legitimate *training* data; not the path. On the path the
angle is 175–177° at ξ = +4, so a 130° floor **never binds and forbids
nothing**.

Run with the wall off (2026-08-01), the extension sampled **mean 65.6°** at
ξ = +1.60 where the main run held 141.9° — side-on, not an Sₙ2 — and three of
six windows went non-finite. The three that survived were worse: clean ⟨ξ⟩, sd,
`minr` and 0.000 restraint contact, all for the wrong geometry. With the wall
on, all 17 windows completed.

It is also required for **stitching**: joining an extension to the main run
through their overlap is only defined if both carry the same Hamiltonian.
Different walls there means it is not an overlap at all, and the offset between
the two profiles is undetermined.

### The channel restraint, and why it keeps MBAR one-dimensional

It pulls the sum (and r(C–N)) toward the reference path evaluated at the
configuration's **own** ξ, not at a window's target. That makes it one fixed
function of the coordinates, identical in every window, so it cancels in the
MBAR reduced-potential differences exactly as the walls do. Aimed at each
window's ξ₀ instead, it would not cancel and would force a 2D analysis.

Tolerances come from the literature, not from our own runs: Truong measure water
elongating C–N by **0.42 Å** at the transition state. Our own estimate of that
shift (+0.106 Å) was measured in runs already clamping at 0.10 Å — the
suppressed value — so sizing the restraint from it was circular.

Watch `CHAN n%` in the log. A few percent is fine; a window spending most of its
time against the restraint is reporting a property of the restraint.

The implementation is `ReactionChannelRestraint`
(`mmml/md/restraints/linear_distance.py`), reachable from the CLI as
`mmml umbrella-sample --wall-channel`; see
[Reaction-channel restraints](../umbrella.md#reaction-channel-restraints).

---

## Files

**Pipeline.** `04`–`06` are in `legacy/` — the retired CHARMM box path, kept
because the atom-order mapping is documented there.

| | |
|---|---|
| `01_seed_windows.py` | window seeds from the reaction scan |
| `02_gas_pmf.sh` | gas PMF: seeds → replicas → merge → MBAR → report |
| `03_gas_report.py` | gas profile and figures |
| `07_solvated_pmf.py` | **solvated PMF** — box, restraints, staged MD, windows |
| `08_solvated_mbar.py` | MBAR for the solvated windows |
| `10_extract_solvent_params.py`, `11_extract_all_solvents.sh` | CHARMM parameter extraction, offline |

**Libraries** (imported, not run): `solute.py` (atom order, CV indices, model
loading), `solvent_models.py`, `jaxmd_box.py` (CHARMM-free box builder),
`cutoff_pairs.py` (12 Å pair list), `merge_replicas.py`.

**Launchers.** All take `GPU=`.

| | |
|---|---|
| `run_solvated_production.sh` | **the entry point** — takes `SOLVENT`, `EMB`, `XI_MAX`, `PROD_PS` |
| `run_gas_gpu.sh` | gas phase |
| `run_train_gpu.sh` | PhysNet training |

**Diagnostics** (`artifacts/menshutkin/diag/`): `figstyle.py`,
`make_all_figures.py`, `fig_pmf.py`, `fig2_solvents.py`, `methods_table.py`,
`model_bias.py`, `validate_model.py`, `split_half.py`, and two for when a run
breaks — `inspect_blowup.py`, `trace_nan_forces.py`.

---

## Traps, in the order they cost time

**ssh starts in `$HOME`, not the repo.** Remote launches need absolute paths for
both the script and the redirect, or the job silently does nothing and leaves no
log.

**Two atom orderings exist.** `load_scan` returns **Cl, N, C** (0, 1, 2); the
simulation uses **N, C, Cl** (0, 4, 5). Using the wrong one silently analyses
the wrong distances — it produced a training table showing r(C–Cl) constant at
1.70 Å across the entire reaction before anyone noticed.

**The model's radial cutoff is not a step function.** e3x's envelope attenuates
smoothly: at cutoff 8 Å the C–Cl pair is 72 % damped at 6 Å and effectively
invisible at 7 Å. Fine for the contact ion pair (envelope ≥ 0.85 within Turan's
range), unusable for the SSIP. Use the long-cutoff checkpoint (`epoch-1436`,
cutoff 14) — it is also 7–10× more accurate on held-out energies and has no
accuracy gradient along ξ.

**The seed scan stops at ξ = +4.12.** Past that `solute_geometry_at_xi` silently
returns the same frame for every window.

**Solvent equilibration matters more than it looks.** A freshly packed box is at
the right density but has no liquid structure — it drops ~850 eV in its first
heat stage. The 100 ps equilibrated box is cached per (solvent, box, seed); the
sampler knob is `pre_equilibrate_ps`, described under
[Pre-equilibration and window chaining](../umbrella.md#pre-equilibration-and-window-chaining).

**A diagnostic must never be able to end the run it reports on.** A `NameError`
in a warning f-string that only evaluated when a window dipped below the
training-energy floor destroyed a healthy 16-window run. The per-window report
block is now wrapped.

**The neighbour-list rebuild cadence was silently tied to `record_every`.**
`JaxmdDriver.block_size` defaults to `record_every`, and the driver refreshes the
pair list once per *block*. Production legs pass 20, so they were fine. The
box-equilibration legs passed `chunk = 10000` steps — to avoid storing 100 ps of
frames — and thereby ran **10 ps of dynamics on one stale list**. The list is
built at 12 Å while the switching function reaches zero near 10 Å, so there is
~2 Å of skin; water diffuses ~3.7 Å RMS in 10 ps. Pairs entered the cutoff
unseen, and the energy jumped 6 eV at the next rebuild before going non-finite.
**Both fresh-box failures in this campaign (water seed1, acetonitrile) were
this** — neither had anything to do with packing or the timestep. Fixed by
capping the block at the production cadence; costs ~15 %.

**Two training files disagree about the N–C–Cl angle, and both are right.**
`examples/m/scan_nh3_ch3cl.npz` (the relaxed scan = the reaction path) stays at
168–177° across the whole range, including 177.0° at ξ = +4.0…+4.5.
`examples/m/nh3_ch3cl_filtered.npz` (NMS + *orientational* sampling) falls to
~83°, because it deliberately walks the chloride around the methylammonium at
fixed separation. Those are legitimate training data but they are not the path.
Reading the angle off the wrong file produced the instruction to disable the
angle wall past the ion pair; run that way, the extension sampled **65.6°** at
ξ = +1.60 where the main run held 141.9° — side-on, not an Sₙ2 — and three of
six windows went non-finite. On the path a 130° floor never binds.

---

## Measured settings, and what they cost to establish

| setting | value | evidence |
|---|---|---|
| timestep | **1 fs** | NVE drift +0.018 meV/atom/ps from the equilibrated box, vs −0.006 at 0.25. Geometry matches 0.25 fs to 0.013 Å in ⟨ξ⟩, inside the 0.06–0.11 Å thermal spread. |
| pair list | **12 Å cutoff** | exact to 3×10⁻⁶ eV; 10 Å is **not** (+0.13 eV). The complete O(N²) list is 4× slower on CPU and ~equal on GPU. |
| Ewald α | **leave alone** | overriding it breaks cancellation with the self-energy and exclusion corrections: +12.2 eV at α = 0.29, +307 eV at 0.25. |
| window spacing / k | **0.1 Å, 6.505 eV/Å²** | Turan's. Softening k to use fewer windows preserves *overlap* but loses *position* — offset = F/k drove windows 0.43 Å off target. |
| production | **10 ps/window** | 1 ps gave overlap 0.000 and a 100 %-one-sided drift. Longer helps little: discarding 20 % or 40 % of a 10 ps window changes the drift not at all, so it is a slow mode, not a startup transient. Replicas beat longer windows. |

---

## Embedding

| mode | charges follow reaction | dq/dR in forces | solvent polarises solute | forces = ∇E |
|---|---|---|---|---|
| `mechanical` | no (frozen at reactants) | — | no | yes |
| `mechanical-fluct` | yes | **no** | no | **no** — 101 % force error |
| `electrostatic` **(production)** | yes | **yes** | no | **yes** |
| `+ --polarisation` | yes | yes | **yes** | yes |

**`electrostatic` without `--freeze-charge-forces` is the production setting**
and is what every result in `RESULTS.md` uses. The historical note that it
"runs away (q(Cl) −0.80 → −1.03 in 50 fs)" came from a probe seeded at the wrong
ξ — it restrained a box equilibrated at ξ ≈ 0 to ξ = +0.50 against a
150 kcal/mol/Å² spring, and destabilised its own λ = 0 control. Re-tested from
properly equilibrated frames, undamped dq/dR is stable, and five 46-window
production runs have since completed on it without a single divergence.

A naming caveat for the writeup: `electrostatic` here is **not** electrostatic
embedding in the QM/MM sense. PhysNet's charges depend only on solute geometry
and never see the solvent field; dq/dR makes the forces the true gradient and
captures charge response to *geometry*. Only `ml_mm_pol` lets the solvent field
polarise the solute.

**`mechanical-fluct` forces are NOT the gradient of its energy.** Earlier
revisions claimed they were — "because there is no dq/dR term to drop" — which
is true of `mechanical` (charges genuinely frozen, dq/dR identically zero) and
false of `mechanical-fluct`, where q depends on R. Measured on an acetonitrile
box at ξ = +0.30, analytic gradient against central finite differences of the
*same* energy function:

| | discrepancy |
|---|---|
| `charge_gradient=True` | 0.00007 eV/Å — **0.01 %** |
| `charge_gradient=False` ← mechanical-fluct | 1.456 eV/Å — **101 %** |

The dropped term is 21–139 % of the retained force across ξ = −1.3…+2.25, so it
never switches off. The consequence is that the force field is not curl-free, so
Langevin does not sample exp(−βE) for the energy being evaluated. It does not
affect MBAR's algebra, and it cancels in any comparison of two runs that share
the setting — which is every comparison this campaign relies on — but it is not
a well-defined Hamiltonian.

`MLMMElectrostaticTerm(charge_gradient_scale=λ)` allows a damped response:
`q_eff = λq + (1−λ)·sg(q)` leaves the value exactly q and scales the gradient to
λ·dq/dR. Not exercised in production, and no longer needed — undamped dq/dR is
stable.

!!! warning "Fixed 2026-08-02"
    `run_solvated_production.sh` used to add `--freeze-charge-forces`
    automatically whenever `EMB=electrostatic`, which set
    `charge_gradient=False` — so every "electrostatic" run ever launched through
    it had `mechanical-fluct` forces and never tested dq/dR at all. It is now
    opt-in via `FREEZE_CHARGE_FORCES=1` and defaults **off**.

### What dropping dq/dR cost, measured

The campaign ran for weeks on `mechanical-fluct`. The consequences, all resolved
by turning the term on:

| | without dq/dR | with dq/dR | Turan |
|---|---|---|---|
| water barrier | 26.17 | **18.48** | 18.0 |
| acetonitrile barrier | 26.24 | **19.64** | 20.6 |
| water − acetonitrile | 0.07 | **1.16** | 2.6 |
| solvent effect (water) | −8.4 | **−16.6** | −17.8 |
| water TS position | +0.48 | **+0.10** | −0.1 |

The old diagnosis — that this was an *embedding* limitation requiring solvent →
solute polarisation — was wrong. The structure had always been right (water puts
4.7 hydrogens inside 3 Å of the developing chloride; acetonitrile, no donor,
none); what was missing was that the charge response never entered the forces,
so those hydrogens could not do any work on the reaction coordinate.

### `--polarisation`

Adds the `ml_mm_pol` energy term (`MLMMPolarisationTerm`):
`E = −½ Σ αᵢ|Eᵢ|²`, the induced component EMLE (J. Chem. Theory Comput. 2023,
19, 1417) obtains from a Thole model. Its point is that this needs **no QM/MM
training data** — in-vacuo atomic properties suffice.

!!! note "Out of scope for this campaign"
    Run as a controlled A/B, polarisation *raises* the barrier (+1.54 water,
    +0.09 acetonitrile) and inverts the solvent ordering — away from Turan and
    from experiment. Since dq/dR alone reproduces the reference, and since
    Turan's MS-ARMD is itself non-polarisable so dq/dR-only is the like-for-like
    comparison, the polarisation runs were stopped. Kept here because the term
    is implemented, tested, and worth revisiting when the target is experiment
    rather than Turan.

It does **not** double-count. PhysNet is trained on *gas-phase* B3LYP/def2-SVPD
so its charges carry no condensed-phase pre-polarisation; `ml_mm_elec` is pure
static Coulomb; TIP3P's enhanced charges (2.35 D vs 1.85 D gas) represent water
polarising *water*, not the solute. Only the 9 solute atoms are polarised.

Thole damping is not optional: undamped, `−α|E|²` diverges *downward* as an MM
hydrogen approaches — an attractive singularity the integrator falls into.

### Where this sits — and why the QM/MM ladder does not transfer

In **QM/MM** electrostatic embedding the MM point charges are added to the
one-electron Hamiltonian and the wavefunction relaxes self-consistently in that
external potential. The response is an electronic-structure result you get for
free once the charges are in.

**ML/MM has no wavefunction to polarise.** PhysNet is a function of nuclear
coordinates alone; there is no Hamiltonian to perturb. Electrostatic embedding
therefore has to be *reconstructed*, not inherited — the point EMLE makes: ML
potentials are trained to reproduce whole-system energies without accounting for
the response to external electric fields, so ML cannot be combined with MM in an
electrostatic embedding scheme unless the ML architecture itself is changed.

The ladder for ML/MM is therefore its own:

| rung | how the solute responds to the solvent |
|---|---|
| 1. mechanical, fixed FF charges | not at all |
| 1.5 `mechanical-fluct` | charges follow its **own geometry** only |
| 2. **`--polarisation`** ← here | classical induced dipoles from the real MM field, in-vacuo α |
| 3. field-conditioned ML | the network **takes the MM field as input** and predicts the polarised energy/charges (`mmml/models/efield`, NepoIP/MM) |
| 4. + polarisable MM | mutual; the solvent responds back |

Rung 3 is the ML/MM analogue of QM/MM electrostatic embedding — the model
*learns* the response instead of having it added classically. Rung 2
approximates what rung 3 would learn, which is why it needs no QM/MM reference
data. Do not describe rung 2 as "electrostatic embedding" without saying which.

Note rung 1.5 is already past textbook mechanical embedding: q(Cl) runs
−0.20 → −0.92 along the coordinate rather than being a fixed force-field
parameter. That responsiveness to *geometry* is what distinguishes this work
from Turan's fixed-charge MS-ARMD; what it lacked was responsiveness to
*environment*.

---

## Literature

- **Turan, Brickel & Meuwly**, *Solvent Effects on the Menshutkin Reaction*,
  J. Phys. Chem. B **126**, 1951 (2022) — `jp1c09710.pdf`.
  MS-ARMD fitted to MP2/6-311++G(2d,2p); umbrella sampling on the same ξ,
  30 windows over [−1.3, 1.6] Å, k = 150 kcal/mol/Å², **50 ps/window**, WHAM.
  Barriers: gas 35.8; water 18.0 ± 0.5, methanol 20.5, acetonitrile 20.6,
  benzene 24.1, cyclohexane 33.9 ± 1.4.
  **Experimental** (NH₃ + MeI): 23.5 water, 20.8 methanol.

- **Truong, Truong & Stefanovich**, J. Chem. Phys. **107**, 1881 (1997) —
  `truong1997.pdf`. GCOSMO continuum. Gas ΔV‡ 32.1 (MP4) / 32.6 (BH&HLYP),
  ΔG‡ 45.1 / 45.7; aqueous ΔG‡ 24.8 / 23.5 / 31.4 (MP2).
  Solvent elongates C–N by 0.42 Å and shortens C–Cl by 0.30 Å at the TS.

  **Our barrier is comparable to their ΔV‡, not their ΔG‡** — the 13 kcal/mol
  gap is association entropy that a 1D PMF in a fixed box does not capture.
  Turan's number sits in the same place, which is why theirs is the
  like-for-like comparison.

- **Solà et al.** / **Tucker & Truhlar**, JACS **112**, 3338 (1990) — on when
  the antisymmetric stretch is *not* a sufficient reaction coordinate. This is
  our restraint problem, already known in 1990.
